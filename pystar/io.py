import dask.array as da
import dask
import numpy as np
import tifffile
import xarray as xr
from pathlib import Path
from typing import List, Optional
from .infrastructure import ExperimentConfig


def get_fov_output_structure(base_dir: Path, fov_id: int) -> dict:
    """
    统一管理目录结构的逻辑。
    Good Taste: 如果你想改文件夹名，只改这里一行，全项目生效。
    """
    # 构造 PositionX/output_pystar
    fov_root = base_dir / f"Position{fov_id}" / "output_pystar"

    # 定义所有子目录
    dirs = {
        "root": fov_root,
        "transforms": fov_root / "transforms",
        "spots": fov_root / "spots",
        "extraction": fov_root / "extraction",
        "decoded": fov_root / "decoded",
        "qc": fov_root / "qc_reports",
        "cleaned": fov_root / "clean_data",
    }

    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)

    return dirs


class ImageLoader:
    def __init__(self, config: ExperimentConfig):
        self.cfg = config
        self.raw_path = self.cfg.dataset.raw_data_path
        self.dims = self.cfg.dataset.dimensions
        self.pattern = self.cfg.dataset.filename_pattern

    def _get_path(self, fov: int, round_id: int, channel_id: int) -> Path:
        """
        定位文件路径。
        """
        # 尝试两种补零格式：ch00 (常见) 和 ch0 (偶尔见)
        # 这是一个实用的 hack，避免因为文件名格式不对就崩溃
        candidates = []

        for ch_str in [f"{channel_id:02d}", f"{channel_id}"]:
            glob_pattern = self.pattern.format(round=round_id, fov=fov, ch=ch_str)
            found = list(self.raw_path.glob(glob_pattern))
            if found:
                candidates.extend(found)
                break  # 找到了就停止

        if not candidates:
            # 构建一个失败时的提示路径
            debug_path = self.raw_path / self.pattern.format(
                round=round_id, fov=fov, ch=f"{channel_id}"
            )
            raise FileNotFoundError(
                f" Data missing!\n"
                f"Looking for: R{round_id} / FOV{fov} / CH{channel_id}\n"
                f"Pattern tried: {debug_path}\n"
                f"Check your 'raw_data_path' and 'filename_pattern' in yaml."
            )

        if len(candidates) > 1:
            raise ValueError(
                f"Ambiguous pattern! Found multiple files for one channel:\n{candidates}"
            )

        return candidates[0]

    def get_clean_path(self, fov_id: int, round_id: int, channel_id: int) -> Path:
        """获取 Clean Data 的路径"""
        base_dir = Path(self.cfg.pipeline.output.directory)
        paths = get_fov_output_structure(base_dir, fov_id)
        clean_dir = paths["cleaned"]
        return clean_dir / f"clean_fov_{fov_id}_round_{round_id}_ch_{channel_id}.tif"

    def load_clean_image(
        self, fov_id: int, round_id: int, channel_id: int
    ) -> np.ndarray:
        """直接读取处理好的 Clean Tiff"""
        path = self.get_clean_path(fov_id, round_id, channel_id)
        if not path.exists():
            raise FileNotFoundError(
                f"Clean image not found: {path}. Run preprocessing first!"
            )
        return tifffile.imread(path)

    def _lazy_load_tiff(self, path: Path) -> da.Array:
        # 使用 delayed 读取，不立即加载进内存
        def loader(p):
            return tifffile.imread(p).squeeze()

        shape = (self.dims["z"], self.dims["height"], self.dims["width"])
        dtype = np.uint8  # 我们目前数据是 8-bit

        sample = dask.delayed(loader)(path)
        # 从 Config 读取 chunk size
        chunks = (
            self.cfg.dataset.io_chunk_size["z"],
            self.cfg.dataset.io_chunk_size["y"],
            self.cfg.dataset.io_chunk_size["x"],
        )

        # 告诉 Dask 怎么切，不要让它瞎猜
        arr = da.from_delayed(sample, shape=shape, dtype=dtype)
        return arr.rechunk(chunks)

    def _create_zero_array(self) -> da.Array:
        """创建一个与图像尺寸匹配的全零数组（用于缺失的channel）"""
        shape = (self.dims["z"], self.dims["height"], self.dims["width"])
        chunks = (
            self.cfg.dataset.io_chunk_size["z"],
            self.cfg.dataset.io_chunk_size["y"],
            self.cfg.dataset.io_chunk_size["x"],
        )
        # 创建全零数组
        arr = da.zeros(shape, dtype=np.uint8, chunks=chunks)
        return arr

    def load_fov(self, fov_id: int) -> xr.DataArray:
        """
        加载单个 FOV。

        严格按 round_structure 配置加载，配置外的通道不处理。
        所有轮次的通道配置必须一致，否则报错。
        文件缺失时自动用零值填充，不报错。
        """
        rounds_cfg = self.cfg.dataset.round_structure
        all_rounds = sorted(rounds_cfg.keys())

        # 验证：所有轮次的通道配置必须一致
        first_round_channels = sorted(rounds_cfg[all_rounds[0]])
        for r_id in all_rounds[1:]:
            if sorted(rounds_cfg[r_id]) != first_round_channels:
                raise ValueError(
                    f"Round {r_id} 的通道配置与 Round {all_rounds[0]} 不一致。\n"
                    f"期望: {first_round_channels}, "
                    f"实际: {sorted(rounds_cfg[r_id])}。\n"
                    f"所有轮次的通道配置必须完全一致。"
                )

        all_channels = first_round_channels
        round_stacks = []
        zero_fill_count = 0

        print(f"DEBUG: Loading FOV {fov_id}...", end="", flush=True)

        for r_id in all_rounds:
            channel_stacks = []

            # 只加载 round_structure 中配置的通道
            for c_id in rounds_cfg[r_id]:
                try:
                    fpath = self._get_path(fov_id, r_id, c_id)
                    arr = self._lazy_load_tiff(fpath)
                    channel_stacks.append(arr)
                except FileNotFoundError:
                    # 文件缺失，用零值填充
                    arr = self._create_zero_array()
                    channel_stacks.append(arr)
                    zero_fill_count += 1
                    print(f"\n   [WARNING] Missing file for FOV{fov_id} R{r_id} CH{c_id}, filling with zeros.")

            # Stack channels -> (C, Z, Y, X)
            round_stacks.append(da.stack(channel_stacks))
        
        if zero_fill_count > 0:
            print(f"\n   Total zero-filled channels: {zero_fill_count}")

        # Stack rounds -> (R, C, Z, Y, X)
        final_dask = da.stack(round_stacks)

        # 物理坐标
        z_coords = np.arange(self.dims["z"]) * self.cfg.dataset.pixel_size_z_nm
        y_coords = np.arange(self.dims["height"]) * self.cfg.dataset.pixel_size_xy_nm
        x_coords = np.arange(self.dims["width"]) * self.cfg.dataset.pixel_size_xy_nm

        xarr = xr.DataArray(
            final_dask,
            coords={
                "round": all_rounds,
                "channel": all_channels,
                "z": z_coords,
                "y": y_coords,
                "x": x_coords,
            },
            dims=("round", "channel", "z", "y", "x"),
            name=f"fov_{fov_id}",
            attrs={
                "fov_id": fov_id,
                "valid_channels_map": rounds_cfg,
                "channel_roles": self.cfg.dataset.channel_roles,
            },
        )
        print(" Done.")
        return xarr
