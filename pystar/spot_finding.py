import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Optional, cast
from numpy.typing import NDArray
from scipy import ndimage
from skimage.feature import blob_dog
from skimage.morphology import local_maxima
from .io import ImageLoader
from .io import get_fov_output_structure, get_matlab_stage_contract
from .matlab_engine_bootstrap import (
    merge_matlab_session_lifecycle_summaries,
    summarize_matlab_boundary_traces,
)
from .matlab_spot_finding import MATLABSpotFindingBackend
from .serialization import write_backend_metadata
from .visualization import inspect_spots_interactive


def _empty_spots_dataframe() -> pd.DataFrame:
    """Return an empty spot table with the canonical numeric columns."""
    return pd.DataFrame({
        'z': pd.Series(dtype=np.float32),
        'y': pd.Series(dtype=np.float32),
        'x': pd.Series(dtype=np.float32),
        'intensity': pd.Series(dtype=np.float32),
    })


def _threshold_reference_value(vol_3d: NDArray[np.generic]) -> float:
    """Choose the intensity scale used by relative Max3D thresholds.

    Integer images use the physical dtype ceiling so `threshold_rel=0.1` means
    25.5 for uint8 and 6553.5 for uint16. Floating images use their observed
    maximum, which keeps notebook experiments and normalized arrays usable
    without forcing a dtype conversion.
    """
    if vol_3d.dtype == np.uint8:
        return 255.0
    if vol_3d.dtype == np.uint16:
        return 65535.0
    return float(vol_3d.max())


def _detect_max3d_regional_maxima(vol_3d: NDArray[np.generic], threshold_rel: float) -> pd.DataFrame:
    """Detect native Max3D spots as 26-connected regional maxima.

    The detector returns one spot per connected plateau rather than one spot per
    bright voxel. The plateau is summarized by a geometric centroid in `z, y, x`
    pixel coordinates and by the maximum voxel intensity inside that plateau.
    This mirrors the STATE/MATLAB max3d interpretation closely enough for the
    Python-native baseline while preserving PyStar's canonical spot table schema.
    """
    abs_thresh = threshold_rel * _threshold_reference_value(vol_3d)
    regional_max_mask = np.asarray(local_maxima(vol_3d, connectivity=3, allow_borders=True), dtype=bool)
    threshold_mask = np.asarray(np.greater(vol_3d, abs_thresh), dtype=bool)
    mask = regional_max_mask & threshold_mask
    structure = np.ones((3, 3, 3), dtype=bool)
    labeled, n_spots = cast(tuple[object, int], ndimage.label(mask, structure=structure))

    if n_spots == 0:
        return _empty_spots_dataframe()

    indices = np.arange(1, n_spots + 1)
    geometric_centroids = ndimage.center_of_mass(mask.astype(np.uint8), labeled, indices)
    max_intensities = ndimage.maximum(vol_3d, labeled, indices)

    coords = np.asarray(geometric_centroids, dtype=np.float32)
    df = pd.DataFrame({'z': coords[:, 0], 'y': coords[:, 1], 'x': coords[:, 2]})
    df['intensity'] = np.asarray(max_intensities, dtype=np.float32)
    return df


class SpotFinder:
    """Convert cleaned reference-round images into candidate spot coordinates.

    `SpotFinder` reads one clean 3D TIFF per sequencing channel from the
    canonical `clean_data/` directory and writes `spots/spots_fov_<id>.csv`.
    Spot coordinates are always `z, y, x` pixel coordinates in the reference
    round frame. The output also carries `channel`, `fov`, and `algo` columns so
    downstream extraction and QC can recover how each coordinate was produced.

    The `provider` switch is the boundary between Python-native detection and
    MATLAB-backed detection. Native algorithms run in this class; MATLAB mode
    delegates the image volume to `MATLABSpotFindingBackend` and then normalizes
    the returned table into the same PyStar schema. No provider silently falls
    back to another implementation.
    """

    def __init__(self, config):
        self.cfg = config
        self.spot_cfg = config.pipeline.spot_finding
        self.loader = ImageLoader(config)
        
        # 预留模型槽位，不要在初始化时乱占显存
        self._model = None
        self._matlab_backend: Optional[MATLABSpotFindingBackend] = None

    def close(self) -> None:
        """Release the optional MATLAB backend session owned by this finder."""
        if self._matlab_backend is None:
            return
        try:
            self._matlab_backend.close()
        finally:
            self._matlab_backend = None

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self.close()
        except Exception:
            pass

    def _get_matlab_backend(self) -> MATLABSpotFindingBackend:
        if self._matlab_backend is None:
            self._matlab_backend = MATLABSpotFindingBackend(self.cfg)
        return self._matlab_backend

    def _get_spotiflow_model(self):
        """延迟加载模型：只有真正开始挖矿时，才启动大型机械。"""
        if self._model is None:
            try:
                from spotiflow.model import Spotiflow  # pyright: ignore[reportMissingImports]
                model_name = self.spot_cfg.spotiflow.model_name
                print(f" [SpotFinder] 正在从硬盘调取 Spotiflow 模型: {model_name}...")
                self._model = Spotiflow.from_pretrained(model_name)
            except ImportError:
                print(" 错误: 未安装Spotiflow库")
                print(" 运行: pip install spotiflow")
                raise
        return self._model

    def find_spots_in_fov(self, fov_id: int):
        """
        Detect all sequencing-channel spots for one FOV.

        The method reads the configured reference round from `clean_data/`, runs
        the selected detector independently on each sequencing channel, injects
        provenance columns, and persists the merged result to the canonical
        spots CSV. Missing clean channel files are skipped loudly at the channel
        level; a FOV with no detected channel data returns an empty table.
        """
        base_dir = Path(self.cfg.pipeline.output.directory)
        paths = get_fov_output_structure(base_dir, fov_id)
        
        ref_round = self.spot_cfg.reference_round
        # 获取所有由 'seq' 定义的通道 (通常 0,1,2,3)
        roles = self.cfg.dataset.channel_roles
        channels = sorted([c for c, role in roles.items() if role == 'seq'])
        
        print(f" [SpotFinding] Mining FOV {fov_id} using Clean Data (Ref Round {ref_round})...")
        print(f" [SpotFinding] Target Channels: {channels}")
        print(f" [SpotFinding] Provider: {self.spot_cfg.provider}")

        all_spots_dfs = []
        backend_records: list[dict[str, Any]] = []
        algo = self.spot_cfg.algorithm
        final_algo = f"matlab_{algo}" if self.spot_cfg.provider == "matlab" else algo
        
        # 用于 QC 可视化的容器 (Channel -> Image)
        qc_images = {}

        for c in channels:
            # 1. 加载 Clean Data (单通道)
            try:
                # 使用 loader 从 clean_data 目录读取
                vol = self.loader.load_clean_image(fov_id, ref_round, c)
                print(f"Data type: {vol.dtype}")
                print(f"Value range: [{vol.min():.2f}, {vol.max():.2f}]")
                print(f"Mean: {vol.mean():.2f}, Std: {vol.std():.2f}")
            except FileNotFoundError:
                print(f" !!! Skip Channel {c}: Clean data not found. Run Sanitizer first!")
                continue

            # 2. 归一化 (Normalization)
            # 我们的 Clean Data 是 float32，范围大概在 0.0 - 255.0 之间 (取决于 Gain)
            # 算法通常喜欢 0-1 范围
            # 注意：如果你的 Gain 很大，值可能超过 255。这里简单除以 255 是一种折中，
            # 保证参数 (threshold) 的物理意义和之前 RAW 流程保持一致。
            #vol_norm = vol.astype(np.float32) / 255.0

            # 3. 收集 QC 图像 (只存中间层，节省内存)
            z_mid = vol.shape[0] // 2
            qc_images[c] = vol[z_mid].copy()

            # 4. 选择算法
            if self.spot_cfg.provider == "matlab":
                result = self._get_matlab_backend().find_spots(
                    vol,
                    fov_id=fov_id,
                    round_id=ref_round,
                    channel_id=c,
                )
                df_c = result["spots"].copy()
                backend_metadata = result.get("backend_metadata")
                if isinstance(backend_metadata, dict):
                    backend_records.append(backend_metadata)
            else:
                # 运行具体算法
                if algo == "spotiflow":
                    df_c = self._run_spotiflow(vol)
                elif algo == "blob_dog":
                    df_c = self._run_blob_dog(vol)
                elif algo == "peak_local_max":
                    df_c = self._run_peak_local_max(vol)
                else:
                    raise ValueError(f"Unknown algorithm: {algo}")
            
            # 5.标签注入
            df_c['channel'] = c
            all_spots_dfs.append(df_c)
            print(f"   > Channel {c}: found {len(df_c)} spots")
            
        # 4. 合并结果
        if not all_spots_dfs:
            if self.spot_cfg.provider == "matlab" and backend_records:
                boundary_traces = [
                    trace
                    for record in backend_records
                    if isinstance(record, dict)
                    for trace in [record.get("boundary_instrumentation")]
                    if isinstance(trace, dict)
                ]
                session_summaries = [
                    summary
                    for record in backend_records
                    if isinstance(record, dict)
                    for summary in [record.get("session_lifecycle_summary")]
                    if isinstance(summary, dict)
                ]
                write_backend_metadata(
                    paths["qc"] / f"spot_finding_backend_fov_{fov_id}.json",
                    {
                        "provider": self.spot_cfg.provider,
                        "matlab_stage_contract": get_matlab_stage_contract(self.cfg, "spot_finding"),
                        "fov_id": int(fov_id),
                        "reference_round": int(ref_round),
                        "algorithm": final_algo,
                        "channel_results": backend_records,
                        "boundary_instrumentation_summary": summarize_matlab_boundary_traces(boundary_traces) if boundary_traces else None,
                        "session_lifecycle_summary": merge_matlab_session_lifecycle_summaries(session_summaries) if session_summaries else None,
                    },
                )
            print(" [SpotFinding] No spots found in any channel!")
            return pd.DataFrame()

        df = pd.concat(all_spots_dfs, ignore_index=True)

        # 注入元数据
        df['fov'] = fov_id
        df['algo'] = final_algo

        # 固化结果
        out_csv = paths["spots"] / f"spots_fov_{fov_id}.csv"
        persistence_started = time.perf_counter()
        df.to_csv(out_csv, index=False)
        if self.spot_cfg.provider == "matlab" and backend_records:
            boundary_traces = [
                trace
                for record in backend_records
                if isinstance(record, dict)
                for trace in [record.get("boundary_instrumentation")]
                if isinstance(trace, dict)
            ]
            session_summaries = [
                summary
                for record in backend_records
                if isinstance(record, dict)
                for summary in [record.get("session_lifecycle_summary")]
                if isinstance(summary, dict)
            ]
            boundary_summary = summarize_matlab_boundary_traces(boundary_traces) if boundary_traces else None
            persistence_ms = round((time.perf_counter() - persistence_started) * 1000.0, 3)
            if boundary_summary is not None:
                boundary_summary["fov_canonical_persistence_ms"] = persistence_ms
            write_backend_metadata(
                paths["qc"] / f"spot_finding_backend_fov_{fov_id}.json",
                {
                    "provider": self.spot_cfg.provider,
                    "matlab_stage_contract": get_matlab_stage_contract(self.cfg, "spot_finding"),
                    "fov_id": int(fov_id),
                    "reference_round": int(ref_round),
                    "algorithm": final_algo,
                    "channel_results": backend_records,
                    "boundary_instrumentation_summary": boundary_summary,
                    "session_lifecycle_summary": merge_matlab_session_lifecycle_summaries(session_summaries) if session_summaries else None,
                },
            )
        print(f" [SpotFinding] Finished. Total: {len(df)} spots. Saved to {out_csv.name}")
        
        if self.cfg.pipeline.qc_images_enabled():
                qc_dir = paths["qc"]
                
                # 构造一个伪 4D 数组 (C, 1, H, W)
                h, w = list(qc_images.values())[0].shape
                viz_stack = np.zeros((len(channels), 1, h, w), dtype=np.float32)
                
                for idx, c in enumerate(channels):
                    if c in qc_images:
                        viz_stack[idx, 0, :, :] = qc_images[c]
                inspect_spots_interactive(
                    viz_stack, df, 
                    z_plane=0, 
                    roi_size=128,
                    output_path=qc_dir / f"spot_finding_qc_fov_{fov_id}.png"
                )
        
        return df

    def _run_spotiflow(self, vol_3d):
        """Run Spotiflow on one 3D volume and normalize its coordinate table."""
        model = self._get_spotiflow_model()
        params = self.spot_cfg.spotiflow
        
        print(f" [Spotiflow] 正在预测，概率阈值: {params.prob_thresh}")
        # Spotiflow 的 predict 非常简洁，直接返回亚像素坐标
        coords, details = model.predict(
            vol_3d, 
            prob_thresh=params.prob_thresh,
            subpix=True
        )
        
        # 修正：维度自适应 (Good Taste: 消除特殊情况)
        # 无论返回 (N, 3) 还是 (N, 2)，这里的逻辑都能跑
        ndim = coords.shape[1] 
        cols = ['z', 'y', 'x'][-ndim:] # 自动取后N个标签
        
        df = pd.DataFrame({col: coords[:, idx] for idx, col in enumerate(cols)})
        
        # 修正：防御性地获取 details
        # 有些版本返回对象，有些可能是字典，我们做一个简单的兼容处理
        # (假设这里它是对象，和你的原始代码一致，但请在运行时确认)
        if hasattr(details, 'intens'):
            df['intensity'] = details.intens
            df['prob'] = details.prob
        else:
            # 如果是字典的情况
            df['intensity'] = details.get('intens', 0)
            df['prob'] = details.get('prob', 0)
            
        return df

    def _run_blob_dog(self, vol_3d):
        """Run Difference-of-Gaussians blob detection on one 3D volume."""
        params = self.spot_cfg.blob_dog
        blobs = blob_dog(
            vol_3d,
            min_sigma=params.min_sigma,
            max_sigma=params.max_sigma,
            threshold=params.threshold,
            overlap=params.overlap
        )
        if blobs.shape[1] == 4:
            cols = ['z', 'y', 'x', 'sigma']
        elif blobs.shape[1] == 6:
            cols = ['z', 'y', 'x', 'sigma_z', 'sigma_y', 'sigma_x']
        else:
            # 万一以后有更奇怪的输出，直接按索引给个默认列名，而不是崩溃
            cols = [f'col_{i}' for i in range(blobs.shape[1])]

        df = pd.DataFrame({col: blobs[:, idx] for idx, col in enumerate(cols)})
        return df

    def _run_peak_local_max(self, vol_3d):
        """Run the native Max3D regional-maxima baseline on one 3D volume."""
        params = self.spot_cfg.peak_local_max
        return _detect_max3d_regional_maxima(vol_3d, params.threshold_rel)
        

    
def _run_algo_on_channels(vol, run_fn):
    """
    Run a notebook helper on either a single 3D volume or a channel stack.

    Input may be `(z, y, x)` or `(channel, z, y, x)`. The returned table always
    includes a `channel` column, even for single-volume input, so ad-hoc
    notebooks have the same shape as pipeline spot tables.
    """
    # 1. 维度归一化
    if vol.ndim == 3:
        # (Z, Y, X) -> (1, Z, Y, X)
        vol_4d = vol[np.newaxis, ...]
    elif vol.ndim == 4:
        # (C, Z, Y, X)
        vol_4d = vol
    else:
        raise ValueError(f"Input must be 3D or 4D, got {vol.ndim}D")

    all_dfs = []
    n_ch = vol_4d.shape[0]

    # 2. 遍历通道
    for c in range(n_ch):
        # 提取单通道 3D 数据
        vol_3d = vol_4d[c]
        
        # 运行具体的算法函数
        df = run_fn(vol_3d)
        
        # 注入 Channel ID (这在 Notebook 调试多通道时至关重要)
        df['channel'] = c
        all_dfs.append(df)
    
    # 3. 合并
    return pd.concat(all_dfs, ignore_index=True)

def detect_spots_max3d(vol_3d, threshold=0.05, min_dist=3):
    """
    Notebook helper for native Max3D regional-maxima detection.

    Pass an in-memory `(z, y, x)` volume or `(channel, z, y, x)` stack and get a
    `z, y, x, channel` table back. `min_dist` is kept as a historical call
    parameter; the current Max3D implementation uses 26-connected regional
    maxima and therefore does not suppress neighboring plateaus by distance.
    """
    _ = min_dist

    def _logic(vol_3d):
        df = _detect_max3d_regional_maxima(vol_3d, threshold)
        return df[['z', 'y', 'x']]

    return _run_algo_on_channels(vol_3d, _logic)

def detect_spots_blob_dog(vol_3d, min_sigma=(0.5, 0.5, 0.5), max_sigma=3, threshold=0.05, overlap=0.5):
    """
    Notebook helper for DoG spot detection on a volume or channel stack.

    Sigma values are passed straight to `blob_dog`; anisotropic 3D sigmas may be
    supplied as three-element tuples when z resolution differs from xy.
    """
    def _logic(vol_3d):
        blobs = blob_dog(
            vol_3d,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            threshold=threshold,
            overlap=overlap
        )
        if blobs.shape[1] == 4:
            cols = ['z', 'y', 'x', 'sigma']
        elif blobs.shape[1] == 6:
            cols = ['z', 'y', 'x', 'sigma_z', 'sigma_y', 'sigma_x']
        else:
            cols = [f'col_{i}' for i in range(blobs.shape[1])]

        return pd.DataFrame({col: blobs[:, idx] for idx, col in enumerate(cols)})

    return _run_algo_on_channels(vol_3d, _logic)

def detect_spots_spotiflow(vol_3d, model_name="general", prob_thresh=0.5, use_gpu=True):
    """
    Notebook helper for running a Spotiflow pretrained model.

    This helper loads the model immediately and is intended for interactive
    experiments, not for pipeline-scale batch execution. The pipeline class
    lazy-loads the model instead.
    """
    try:
        from spotiflow.model import Spotiflow  # pyright: ignore[reportMissingImports]
    except ImportError:
        print(" 错误: 未安装Spotiflow库")
        print(" 运行: pip install spotiflow")
        raise
    
    print(f" [Spotiflow] 正在加载模型: {model_name}...")
    model = Spotiflow.from_pretrained(model_name)
    
    def _logic(vol_3d):
        print(f" [Spotiflow] 正在预测，概率阈值: {prob_thresh}")
        coords, details = model.predict(
            vol_3d, 
            prob_thresh=prob_thresh,
            subpix=True
        )
        
        ndim = coords.shape[1] 
        cols = ['z', 'y', 'x'][-ndim:] 
        
        df = pd.DataFrame({col: coords[:, idx] for idx, col in enumerate(cols)})
        
        if hasattr(details, 'intens'):
            df['intensity'] = details.intens
            df['prob'] = details.prob
        else:
            df['intensity'] = details.get('intens', 0)
            df['prob'] = details.get('prob', 0)
            
        return df
    return _run_algo_on_channels(vol_3d, _logic)
