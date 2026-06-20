# pyright: reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownVariableType=false

"""Image loading implementation behind the stable :mod:`pystar.io` facade.

Userspace should continue importing ``ImageLoader`` from ``pystar.io``.  This
private module owns the concrete raw/clean image loading behavior so the public
facade can remain a compatibility surface without carrying the implementation.
"""

from pathlib import Path
from typing import Any, cast

import dask
import dask.array as da
import numpy as np
import tifffile
import xarray as xr
from numpy.typing import NDArray

from ._io_paths import get_fov_output_structure
from .infrastructure import ExperimentConfig


class ImageLoader:
    """Load raw and cleaned image volumes under the PyStar path contract.

    Raw images are resolved from ``dataset.raw_data_path`` and
    ``dataset.filename_pattern``. Clean images are resolved from the canonical
    per-FOV output tree created by ``get_fov_output_structure``. The public
    ``load_fov`` method returns a lazy xarray volume with dimensions
    ``(round, channel, z, y, x)`` and physical coordinates in nanometers; missing
    channels declared absent from a round are represented as zero volumes so that
    downstream stages see a rectangular tensor.
    """

    def __init__(self, config: ExperimentConfig):
        self.cfg = config
        self.raw_path = self.cfg.dataset.raw_data_path
        self.dims = self.cfg.dataset.dimensions
        self.pattern = self.cfg.dataset.filename_pattern

    def _get_path(self, fov: int, round_id: int, channel_id: int) -> Path:
        """
        Resolve one raw TIFF path from the YAML filename pattern.

        ``filename_pattern`` receives ``round``, ``fov`` and ``ch`` placeholders.
        The loader tries both zero-padded and plain channel strings because Leica
        exports are not always consistent across experiments. Missing or
        ambiguous matches fail loudly; raw input paths must come from the config,
        not from hard-coded local overrides.
        """
        # 尝试两种补零格式：ch00 (常见) 和 ch0 (偶尔见)
        # 这是一个实用的 hack，避免因为文件名格式不对就崩溃
        candidates = []

        for ch_str in [f"{channel_id:02d}", f"{channel_id}"]:
            glob_pattern = self.pattern.format(
                round=round_id,
                fov=fov,
                ch=ch_str,
            )
            found = list(self.raw_path.glob(glob_pattern))
            if found:
                candidates.extend(found)
                break # 找到了就停止

        if not candidates:
            # 构建一个失败时的提示路径
            debug_path = self.raw_path / self.pattern.format(round=round_id, fov=fov, ch=f"{channel_id}")
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
        """Return the canonical clean-image path for one FOV/round/channel."""
        base_dir = Path(self.cfg.pipeline.output.directory)
        paths = get_fov_output_structure(base_dir, fov_id)
        clean_dir = paths["cleaned"]
        return clean_dir / f"clean_fov_{fov_id}_round_{round_id}_ch_{channel_id}.tif"

    def load_clean_image(self, fov_id: int, round_id: int, channel_id: int) -> NDArray[Any]:
        """Read a preprocessed clean 3-D TIFF from the PyStar output tree."""
        path = self.get_clean_path(fov_id, round_id, channel_id)
        if not path.exists():
            raise FileNotFoundError(f"Clean image not found: {path}. Run preprocessing first!")
        return tifffile.imread(path)

    def _lazy_load_tiff(self, path: Path) -> da.Array:
        """Create a Dask array for one raw 3-D TIFF without loading pixels now."""
        # 使用 delayed 读取，不立即加载进内存
        def loader(p):
            return tifffile.imread(p).squeeze()

        shape = (self.dims['z'], self.dims['height'], self.dims['width'])
        dtype = np.uint8 # 我们目前数据是 8-bit

        sample = dask.delayed(loader)(path)
        # 从 Config 读取 chunk size
        chunks = (
            self.cfg.dataset.io_chunk_size['z'],
            self.cfg.dataset.io_chunk_size['y'],
            self.cfg.dataset.io_chunk_size['x']
        )

        # 告诉 Dask 怎么切，不要让它瞎猜
        arr = da.from_delayed(sample, shape=shape, dtype=dtype)
        return arr.rechunk(cast(Any, chunks))

    def load_fov(self, fov_id: int) -> xr.DataArray:
        """
        Load one FOV as a lazy rectangular ``round/channel/z/y/x`` array.

        The dataset config may declare that different rounds have different
        channel sets. Valid channels are loaded lazily from disk; unavailable
        channels are zero-padded so registration/preprocessing can index by the
        global channel list without special cases. Coordinate arrays are physical
        distances in nanometers, while pixel-level algorithms downstream still
        use integer ``z, y, x`` indices.
        """
        rounds_cfg = self.cfg.dataset.round_structure
        all_rounds = sorted(rounds_cfg.keys())
        all_channels = sorted(self.cfg.dataset.channel_roles.keys())

        round_stacks = []

        print(f"DEBUG: Loading FOV {fov_id} structure...", end="", flush=True)

        for r_id in all_rounds:
            valid_channels = rounds_cfg[r_id]
            channel_stacks = []

            for c_id in all_channels:
                if c_id in valid_channels:
                    # 真实加载
                    fpath = self._get_path(fov_id, r_id, c_id)
                    arr = self._lazy_load_tiff(fpath)
                else:
                    # 虚拟填充 (Padding)
                    arr = da.zeros(
                        (self.dims['z'], self.dims['height'], self.dims['width']),
                        dtype=np.uint8
                    )

                channel_stacks.append(arr)

            # Stack channels -> (C, Z, Y, X)
            round_stacks.append(da.stack(channel_stacks))

        # Stack rounds -> (R, C, Z, Y, X)
        final_dask = da.stack(round_stacks)

        # 物理坐标
        z_coords = np.arange(self.dims['z']) * self.cfg.dataset.pixel_size_z_nm
        y_coords = np.arange(self.dims['height']) * self.cfg.dataset.pixel_size_xy_nm
        x_coords = np.arange(self.dims['width']) * self.cfg.dataset.pixel_size_xy_nm

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
                "channel_roles": self.cfg.dataset.channel_roles
            }
        )
        print(" Done.")
        return xarr
