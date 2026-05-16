# pystar/registration.py

import hashlib
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
import numpy as np
import xarray as xr
import SimpleITK as sitk
from datetime import datetime, timezone
from typing import Tuple, Optional, Dict, Any, cast
from pathlib import Path
from numpy.typing import NDArray
from scipy.ndimage import shift as scipy_shift
from scipy.ndimage import map_coordinates
from skimage.registration import phase_cross_correlation, optical_flow_tvl1
from skimage.transform import resize
from skimage.filters import gaussian
import warnings

from .infrastructure import ExperimentConfig
from .io import ImageLoader
from .io import get_fov_output_structure
from .io import PROVENANCE_VERSION, build_execution_envelope, build_release_contract, save_transform_manifest, persist_flow_3d_sidecar, load_transform_manifest
from .matlab_engine_bootstrap import MatlabSharedSessionOwner
from .matlab_registration import MATLABGlobalRegistrationBackend
from ._registration_performance import (
    RegistrationPerformanceRecorder,
    build_registration_provider_summary,
    diagnostics_timer_start,
    elapsed_ms_since,
)
from .tiling import (
    TileLayout,
    TileSpec,
    build_matlab_subtile_layout,
    build_yx_tile_layout,
    extract_tile,
    extract_tile_write_window,
    resolve_grid_shape_yx,
    stitch_tiles,
)
from .visualization import save_registration_qc


FloatArray = NDArray[np.float32]
BoolArray = NDArray[np.bool_]


@dataclass(slots=True)
class _LocalRegistrationContext:
    fov_id: int
    round_id: int
    ref_round: int
    ref_scope_3d: FloatArray
    ref_mip_clean: FloatArray
    mov_scope_3d: FloatArray
    mov_mip_clean: FloatArray
    mov_mip_shifted: FloatArray
    shift_2d: FloatArray
    global_shift_3d: FloatArray
    corr_after_global: float
    scope_descriptor: Dict[str, Any]
    backend_metadata: Optional[Dict[str, Any]]
    reject_if_worse: bool
    mask: BoolArray
    local_provider: str
    local_method: str
    local_handler_name: Optional[str]


@dataclass(slots=True)
class _LocalRegistrationOutcome:
    flow_2d: Optional[FloatArray]
    flow_3d: Optional[FloatArray]
    final_corr: float
    final_img_qc: FloatArray
    backend_metadata: Optional[Dict[str, Any]]


@dataclass(frozen=True, slots=True)
class _RoundRegistrationStrategy:
    global_provider: str
    local_provider: str
    local_method: str
    enable_local: bool
    local_skip_if_global_corr_below: float
    global_handler_name: str
    local_handler_name: Optional[str]
    local_default_mode: Optional[str]


_GLOBAL_REGISTRATION_HANDLER_NAMES: Dict[str, str] = {
    'native': '_run_global_registration_native',
    'matlab': '_run_global_registration_matlab',
}


_LOCAL_REGISTRATION_HANDLER_NAMES: Dict[Tuple[str, str], str] = {
    ('native', 'demons_3d'): '_run_local_native_demons_3d',
    ('native', 'optical_flow'): '_run_local_native_optical_flow',
    ('native', 'bspline'): '_run_local_native_bspline',
    ('matlab', 'demons_3d'): '_run_local_matlab_demons_3d',
}


_LOCAL_PROVIDER_DEFAULT_MODES: Dict[str, str] = {
    'native': 'native_local_registration',
    'matlab': 'experimental_local_kernel_swap',
}


def _ensure_supported_matlab_local_method(local_method: str) -> None:
    if local_method != 'demons_3d':
        raise ValueError(
            "registration.local.provider='matlab' currently supports only local_method='demons_3d'; "
            f"refusing to skip local registration silently for local_method={local_method!r}"
        )

# ==============================================================================
# SECTION A: Data Extraction Layer
# ==============================================================================

def compute_overlap_roi(shape: Tuple[int, int], shift_2d: FloatArray) -> BoolArray:
    """
    [Helper] 计算全局移动后的有效重叠区域掩码。
    shift_2d = [dy, dx]
    """
    h, w = shape
    dy, dx = int(shift_2d[0]), int(shift_2d[1])
    
    mask = np.ones((h, w), dtype=bool)
    
    # 如果向下移 (dy > 0)，顶部是无效的
    if dy > 0:
        mask[:dy, :] = False
    # 如果向上移 (dy < 0)，底部是无效的
    elif dy < 0:
        mask[dy:, :] = False
        
    # 如果向右移 (dx > 0)，左边是无效的
    if dx > 0:
        mask[:, :dx] = False
    # 如果向左移 (dx < 0)，右边是无效的
    elif dx < 0:
        mask[:, dx:] = False
        
    return mask
# ==============================================================================
# SECTION B: Global Registration (3D FFT-based)
# ==============================================================================

def compute_global_shift_3d(
    ref_3d_clean: FloatArray,
    mov_3d_clean: FloatArray,
    downsample_factor: int = 1,
    max_shift: int = 200
) -> Tuple[FloatArray, float]:
    """
    [Core] 计算 3D 全局刚性位移。
    
    KEY IMPROVEMENT:
    输入必须已经是 Clean Image
    我们在计算位移时使用经过去背景的图像，
    但这个去背景只用于计算位移 (calc)和寻点（spot finding），不修改原始数据。

    Parameters
    ----------
    ref_3d_clean : np.ndarray
        经过预处理的参考图像 (Z, Y, X)
    mov_3d_clean : np.ndarray
        经过预处理的移动图像 (Z, Y, X)
    downsample_factor : int
        加速倍数
    max_shift : int
        最大允许位移

    Returns
    -------
    shift : np.ndarray
        [dz, dy, dx]
    correlation : float
        Correlation score
    """
    if ref_3d_clean.ndim != 3 or mov_3d_clean.ndim != 3:
        raise ValueError(f"Expected 3D images, got {ref_3d_clean.ndim}D and {mov_3d_clean.ndim}D")

    # 1. 下采样加速 (Downsample FIRST to save time on preprocessing)
    if downsample_factor > 1:
        # 注意: 步长是 [1, factor, factor] 还是 [factor, factor, factor]?
        # 考虑到 Z 轴层数通常很少 (30层)，Z轴最好不要降采样，否则没法对齐了。
        # 策略：Z轴保持原样，XY轴降采样。
        slice_s = slice(None, None, downsample_factor)
        # Z轴全取 (::1)，XY下采样
        ref_s = ref_3d_clean[:, slice_s, slice_s]
        mov_s = mov_3d_clean[:, slice_s, slice_s]        

    else:
        ref_s, mov_s = ref_3d_clean, mov_3d_clean 

    # 3. FFT 相位相关
    # 注意: normalization=None 因为我们已经手动归一化了
    shift_s, error, _ = phase_cross_correlation(
        ref_s, mov_s,
        upsample_factor=10,
        normalization=cast(Any, None)
    )
    
    # 4. 恢复真实尺度
    # shift_s 是 [dz, dy, dx]
    # Z 轴没缩放，XY 轴缩放了
    real_shift = np.array(shift_s)
    if downsample_factor > 1:
        real_shift[1] *= downsample_factor # dy
        real_shift[2] *= downsample_factor # dx
        
    # 计算相关性得分 (CC)
    # 这是一个反向指标，error越小越好，但在 skimage 里它不是直接的 Pearson Correlation。
    # 为了拿到真正的 Pearson Corr，我们需要手动算一下 (用对齐后的图)
    
    # 这里我们做一个快速验证
    correlation = 1.0 - error # 粗略估计
    
    # 安全检查
    if np.linalg.norm(real_shift) > max_shift:
        warnings.warn(
            f"Global shift {real_shift} exceeds max_shift={max_shift}. "
            f"Registration may be unreliable!"
        )
        
    return real_shift.astype(np.float32), float(correlation)

def apply_rigid_shift_3d(img_3d: FloatArray, shift_3d: FloatArray) -> FloatArray:
    """
    [Transform] 应用 3D 刚性位移。
    
    Parameters
    ----------
    img_3d : np.ndarray
        (Z, Y, X)
    shift_3d : np.ndarray
        [dz, dy, dx]
        
    Returns
    -------
    shifted : np.ndarray
        (Z, Y, X)
    """
    return np.asarray(
        scipy_shift(img_3d, shift_3d, order=1, mode='constant', cval=0.0),
        dtype=np.float32,
    )

# ==============================================================================
# SECTION C: Local Registration (2D Optical Flow with Quality Masking)
# ==============================================================================

def create_quality_mask(
    img_2d: FloatArray,
    valid_roi_mask: Optional[BoolArray] = None,
    edge_margin: int = 50,
    threshold: float = 1e-5 
) -> BoolArray:
    """
    [Helper] 基于 Clean MIP 创建掩码。
    """
    h, w = img_2d.shape
    
    # 1. 核心逻辑：有信号的地方
    mask = (img_2d > threshold)

    # 2. 如果有通过全局位移计算出的有效ROI，取交集
    if valid_roi_mask is not None:
        mask = mask & valid_roi_mask

    # 3. 排除边缘 (防止光流在边界处发疯)
    if edge_margin > 0:
        mask[:edge_margin, :] = False
        mask[-edge_margin:, :] = False
        mask[:, :edge_margin] = False
        mask[:, -edge_margin:] = False

    return mask

def compute_optical_flow_masked(
    ref_2d: FloatArray,
    mov_2d: FloatArray,
    mask: Optional[BoolArray],
    config_obj: Any
) -> Optional[FloatArray]:
    """
    [Core] 计算带掩码的 2D 光流。
    使用降采样策略强制算法只关注宏观形变，忽略稀疏斑点的微小错位。
    
    Parameters
    ----------
    ref_2d : np.ndarray
        参考图像 (Y, X)
    mov_2d : np.ndarray
        移动图像 (Y, X)，已经过全局位移校正
    mask : np.ndarray (bool), optional
        高质量区域掩码
    config_obj : object
        光流参数配置对象
        
    Returns
    -------
    flow : np.ndarray or None
        Shape (2, Y, X)，[dy, dx] 或失败返回 None
    """
    if ref_2d.ndim != 2 or mov_2d.ndim != 2:
        raise ValueError("Optical flow requires 2D images")

    
    # ---  Downsampling (The Magic Trick) ---
    # 不要在大图上算！把图缩小。
    # 默认缩放到 0.25 (即 1/4 尺寸)，相当于金字塔的某一层。
    # 这比单纯的 blur 更有效，因为它物理上消除了高频噪声。
    scale_factor = float(getattr(config_obj, 'coarse_scale', 0.25))
    
    h, w = ref_2d.shape
    
    # 简单的尺寸保护，防止缩得太小
    if h * scale_factor < 128 or w * scale_factor < 128:
        scale_factor = 1.0 # 图像太小就不缩了

    small_shape = (int(h * scale_factor), int(w * scale_factor))
    
    # 应用 Mask (如果有)
    if mask is not None:
        ref_2d *= mask
        mov_2d *= mask

    # 显式缩放 + 模糊 (双重保险)
    # resize 本身带有抗锯齿(anti_aliasing=True)，这就相当于一次低通滤波
    ref_small = resize(ref_2d, small_shape, anti_aliasing=True)
    mov_small = resize(mov_2d, small_shape, anti_aliasing=True)

    # 在小图上再加一点模糊，确保平滑
    blur_sigma = getattr(config_obj, 'blur_sigma', 1.0) # 在小图上，sigma=1已经很大了
    ref_small = gaussian(ref_small, sigma=blur_sigma)
    mov_small = gaussian(mov_small, sigma=blur_sigma)

    try:
        # ---  Compute Flow on Small Image ---
        flow_small = optical_flow_tvl1(
            ref_small, mov_small,
            attachment=cast(Any, getattr(config_obj, 'attachment', 15.0)), # 强力贴合
            tightness=getattr(config_obj, 'tightness', 0.2),    # 允许形变
            num_warp=getattr(config_obj, 'num_warp', 5),
            num_iter=getattr(config_obj, 'num_iter', 20),
            tol=getattr(config_obj, 'tol', 0.0001),
            prefilter=False # 我们已经手动 blur 了，这里关掉
        )
        
        # ---  Upscale Flow (Restore) ---
        # flow shape is (2, small_h, small_w)
        # 我们必须把它放大回 (2, h, w)
        
        flow_large = np.zeros((2, h, w), dtype=np.float32)
        
        # 关键数学细节：
        # 如果图像放大了 N 倍，位移量(像素数)也要放大 N 倍！
        correction_factor = float(1.0 / scale_factor)
        
        # Resize Y component
        flow_large[0] = np.asarray(resize(flow_small[0], (h, w), order=1), dtype=np.float32) * correction_factor
        # Resize X component
        flow_large[1] = np.asarray(resize(flow_small[1], (h, w), order=1), dtype=np.float32) * correction_factor
        
        # ---  Final Polish ---
        # 放大插值可能会带来网格效应，最后做一次平滑
        flow_large[0] = gaussian(flow_large[0], sigma=3.0)
        flow_large[1] = gaussian(flow_large[1], sigma=3.0)
        
        return flow_large

    except Exception as e:
        warnings.warn(f"Optical flow crashed: {e}")
        return None

def register_local_bspline(
    ref_2d: FloatArray, 
    mov_2d: FloatArray, 
    mask: Optional[BoolArray],
    config_obj: Any
) -> Optional[FloatArray]:
    """
    [Core] 使用 SimpleITK B-Spline 进行局部非刚性配准。
    
    Returns
    -------
    flow : np.ndarray (2, H, W) -> [dy, dx]
    """
    h, w = ref_2d.shape
    
    # 1. 类型转换 (SimpleITK 需要 float32)
    # SimpleITK 的图像坐标是 (X, Y)，而 Numpy 是 (Y, X)。
    # 我们直接 GetImageFromArray，它会把 Numpy 的 (Y, X) 变成 SimpleITK 的 Size(X, Y)
    fixed_sitk = sitk.GetImageFromArray(ref_2d.astype(np.float32))
    moving_sitk = sitk.GetImageFromArray(mov_2d.astype(np.float32))
    
    # 显式声明我们不在乎物理尺寸，只在乎像素
    # 但必须要一致。
    fixed_sitk.SetSpacing([1.0, 1.0])
    moving_sitk.SetSpacing([1.0, 1.0])
    
    # 2. 处理 Mask
    # SimpleITK metric mask 需要是 uint8，且 1=valid, 0=invalid
    if mask is not None:
        mask_sitk = sitk.GetImageFromArray(mask.astype(np.uint8))
    else:
        mask_sitk = None

    # 3. 初始化 B-Spline
    # grid_spacing: 网格间距。越大越平滑(rigid)，越小越柔软(flexible)。
    # 默认 50 像素（约 5um），适合捕捉组织大尺度形变，忽略单个 RNA 点的抖动。
    grid_spacing = getattr(config_obj, 'grid_spacing', 50)
    
    # 防止 grid 太密导致过拟合或崩溃
    transform_domain_mesh_size = [
        int(w / grid_spacing), 
        int(h / grid_spacing)
    ]
    
    # 确保至少有 3x3 个网格，否则没法弯曲
    transform_domain_mesh_size = [max(3, x) for x in transform_domain_mesh_size]

    try:
        initial_tx = sitk.BSplineTransformInitializer(
            fixed_sitk, 
            transform_domain_mesh_size
        )

        # 4. 设置配准方法
        R = sitk.ImageRegistrationMethod()
        
        # 使用相关性作为指标 (Correlation) - 类似 MATLAB 的互相关
        # 注意：这里我们只计算 Mask 区域内的 Metric
        R.SetMetricAsCorrelation()
        if mask_sitk is not None:
            R.SetMetricFixedMask(mask_sitk)

        # 优化器 LBFGSB (有限内存拟牛顿法) - 适合高维优化
        R.SetOptimizerAsLBFGSB(
            gradientConvergenceTolerance=1e-5,
            numberOfIterations=getattr(config_obj, 'num_iter', 50),
            maximumNumberOfCorrections=5,
            maximumNumberOfFunctionEvaluations=1000,
            costFunctionConvergenceFactor=1e+7
        )
        
        R.SetInitialTransform(initial_tx, inPlace=True)
        R.SetInterpolator(sitk.sitkLinear)
        
        # 多分辨率策略 (金字塔) - 自动处理降采样
        # [4, 2, 1] 意味着先在 1/4 尺寸跑，再 1/2，最后全尺寸微调
        R.SetShrinkFactorsPerLevel(shrinkFactors = [4, 2, 1])
        R.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
        R.SetSmoothingSigmasAreSpecifiedInPhysicalUnits(False)
        
        # 5. 执行
        # print("  [B-Spline] Starting SimpleITK optimization...")
        final_tx = R.Execute(fixed_sitk, moving_sitk)
        
        # print(f"  [B-Spline] Final Metric: {R.GetMetricValue():.4f}, "
        #       f"Stop Condition: {R.GetOptimizerStopConditionDescription()}")

        # 6. 生成位移场 (Displacement Field)
        # 我们需要把它转回 Numpy 的 [dy, dx] 格式以便应用
        displacement_filter = sitk.TransformToDisplacementFieldFilter()
        displacement_filter.SetReferenceImage(fixed_sitk)
        displacement_field = displacement_filter.Execute(final_tx)
        
        # sitk field 是 (X, Y, 2) 的矢量图
        # 转回 Numpy 变成 (Y, X, 2)
        field_np = sitk.GetArrayFromImage(displacement_field)
        
        # SimpleITK 的 vector component 0 是 X (dx), 1 是 Y (dy)
        dx = field_np[..., 0]
        dy = field_np[..., 1]
        
        # 我们的格式是 (2, Y, X) -> [dy, dx]
        flow = np.stack([dy, dx], axis=0)
        
        return flow

    except Exception as e:
        warnings.warn(f"SimpleITK B-Spline failed: {e}")
        return None

def register_local_demons_3d(
    ref_3d: FloatArray,
    mov_3d: FloatArray,
    config_obj: Any
) -> Optional[FloatArray]:
    """
    3D Demons 非刚性配准
    
    Returns
    -------
    displacement_field : np.ndarray (3, Z, Y, X) -> [dz, dy, dx]
    """
    
    # 1. 转换为 SimpleITK 格式
    fixed_sitk = sitk.GetImageFromArray(ref_3d.astype(np.float32))
    moving_sitk = sitk.GetImageFromArray(mov_3d.astype(np.float32))
    
    # 2. 设置物理空间（保持一致）
    fixed_sitk.SetSpacing([1.0, 1.0, 1.0])
    moving_sitk.SetSpacing([1.0, 1.0, 1.0])
    
    # 3. 初始化 Demons 配准
    demons = sitk.DemonsRegistrationFilter()
    
    # 参数对应 MATLAB 的设置
    demons.SetNumberOfIterations(getattr(config_obj, 'num_iter', 50))
    demons.SetStandardDeviations(getattr(config_obj, 'smoothing_sigma', 1.0))
    
    # 4. 多分辨率策略（对应 MATLAB 的 PyramidLevels）
    # MATLAB: pyd_level = floor(log2(obj.dimZ))
    pyd_level = int(np.floor(np.log2(ref_3d.shape[0])))
    if pyd_level == 0:
        pyd_level = 1
    
    # 使用 MultiResolution 包装器
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetShrinkFactorsPerLevel([2**i for i in range(pyd_level, 0, -1)])
    registration_method.SetSmoothingSigmasPerLevel([2.0*i for i in range(pyd_level, 0, -1)])
    
    try:
        # 5. 执行配准
        print(f"  [Demons 3D] Starting registration (pyramid levels: {pyd_level})...")
        displacement_field_sitk = demons.Execute(fixed_sitk, moving_sitk)
        
        # 6. 转换为 numpy 格式
        # SimpleITK 返回的是 (Z, Y, X, 3) 的向量场
        field_np = sitk.GetArrayFromImage(displacement_field_sitk)
        
        # 重新排列为 (3, Z, Y, X)
        dz = field_np[..., 2]  # SimpleITK 的 Z 分量
        dy = field_np[..., 1]  # Y 分量
        dx = field_np[..., 0]  # X 分量
        
        flow_3d = np.stack([dz, dy, dx], axis=0)
        
        print(f"  [Demons 3D] Finished. Mean displacement: {np.abs(flow_3d).mean():.2f} px")
        return flow_3d
        
    except Exception as e:
        warnings.warn(f"Demons 3D registration failed: {e}")
        return None

def apply_warp_field_2d(img_2d: FloatArray, flow: FloatArray) -> FloatArray:
    """
    [Transform] 应用 2D 变形。
    
    Parameters
    ----------
    img_2d : np.ndarray
        (Y, X)
    flow : np.ndarray
        (2, Y, X)
        
    Returns
    -------
    warped : np.ndarray
        (Y, X)
    """
    if img_2d.ndim != 2:
        raise ValueError("Warp requires 2D image")

    nr, nc = img_2d.shape
    row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')

    # Inverse mapping
    new_row = row_coords + flow[0]
    new_col = col_coords + flow[1]

    warped = map_coordinates(
        img_2d,
        np.array([new_row, new_col]),
        order=1, 
        mode='constant', 
        cval=0,
        prefilter=False
    )
    return warped.astype(img_2d.dtype)

def composite_transform_2d(
    img_2d: FloatArray,
    shift_2d: FloatArray,
    flow: Optional[FloatArray]
) -> FloatArray:
    """
    [Helper] 一键应用 2D 位移场。
    
    Parameters
    ----------
    img_2d : np.ndarray
        (Y, X)
    shift_2d : np.ndarray
        [dy, dx]
    flow : np.ndarray, optional
        (2, Y, X)
        
    Returns
    -------
    transformed : np.ndarray
        (Y, X)
    """
    # Step 1: 刚性平移
    shifted = np.asarray(
        scipy_shift(img_2d, shift_2d, order=1, mode='constant', cval=0.0),
        dtype=np.float32,
    )

    # Step 2: 光流变形
    if flow is not None:
        return apply_warp_field_2d(shifted, flow)

    return shifted.astype(np.float32)

def apply_warp_field_3d(img_3d: FloatArray, flow_3d: FloatArray) -> FloatArray:
    """
    应用 3D 变形场
    
    Parameters
    ----------
    img_3d : np.ndarray (Z, Y, X)
    flow_3d : np.ndarray (3, Z, Y, X) -> [dz, dy, dx]
    
    Returns
    -------
    warped : np.ndarray (Z, Y, X)
    """
    nz, ny, nx = img_3d.shape
    
    z_coords, y_coords, x_coords = np.meshgrid(
        np.arange(nz, dtype=np.float32), np.arange(ny, dtype=np.float32), np.arange(nx, dtype=np.float32), indexing='ij'
    )
    
    # 应用位移（逆向映射）
    new_z = z_coords + flow_3d[0]
    new_y = y_coords + flow_3d[1]
    new_x = x_coords + flow_3d[2]
    
    # 插值
    warped = map_coordinates(
        img_3d,
        np.array([new_z, new_y, new_x], dtype=np.float32),
        order=1,
        mode='constant',
        cval=0,
        prefilter=False
    )
    
    return warped.astype(img_3d.dtype)

# ==============================================================================
# SECTION D: Quality Metrics
# ==============================================================================

def simple_correlation(img1: FloatArray, img2: FloatArray) -> float:
    """
    计算两张图像的皮尔逊相关系数（中心裁剪加速）。
    
    Parameters
    ----------
    img1, img2 : np.ndarray
        必须是 2D 图像
        
    Returns
    -------
    corr : float
    """
    if img1.ndim == 3:
        img1 = img1.max(axis=0)
    if img2.ndim == 3:
        img2 = img2.max(axis=0)

    h, w = img1.shape

    # 只算中心区域加速
    if h > 1024:
        cy, cx = h // 2, w // 2
        crop = 512
        i1 = img1[cy - crop:cy + crop, cx - crop:cx + crop].flatten()
        i2 = img2[cy - crop:cy + crop, cx - crop:cx + crop].flatten()
    else:
        i1, i2 = img1.flatten(), img2.flatten()

    return np.corrcoef(i1, i2)[0, 1]


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_available_memory_bytes() -> Optional[int]:
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None
    return page_size * available_pages


def _compute_environment_hash(config_hash: str, execution_envelope: Dict[str, str], software_versions: Dict[str, str]) -> str:
    payload = {
        "config_hash": config_hash,
        "execution_envelope": execution_envelope,
        "software_versions": software_versions,
    }
    digest = hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _resolve_demons_tiling_layout(volume_shape_zyx: Tuple[int, int, int], config_obj: Any) -> Optional[TileLayout]:
    if not bool(config_obj.use_tiling):
        return None

    grid_shape_yx, grid_source = resolve_grid_shape_yx(
        volume_shape_zyx,
        tile_size=int(config_obj.tile_size),
        sqrt_pieces=(None if config_obj.sqrt_pieces is None else int(config_obj.sqrt_pieces)),
        tile_grid_shape_yx=config_obj.tile_grid_shape_yx,
    )
    if grid_shape_yx == (1, 1):
        raise ValueError("registration.local.params.demons_3d.use_tiling=true requires at least a 2D multi-tile layout")

    overlap_yx = None
    if config_obj.tile_overlap is not None:
        overlap_value = int(config_obj.tile_overlap)
        overlap_yx = (overlap_value, overlap_value)

    if config_obj.tiling_layout_policy == 'matlab_subtile':
        return build_matlab_subtile_layout(
            volume_shape_zyx,
            sqrt_pieces=int(grid_shape_yx[0]),
            overlap_yx=overlap_yx,
            grid_source=grid_source,
        )

    overlap_value = 0 if overlap_yx is None else int(overlap_yx[0])
    return build_yx_tile_layout(
        volume_shape_zyx,
        grid_shape_yx=grid_shape_yx,
        overlap_yx=(overlap_value, overlap_value),
        grid_source=grid_source,
    )


def _build_tiling_summary(layout: TileLayout, tile_summaries: list[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "enabled": True,
        "grid_shape_yx": [int(layout.grid_shape_yx[0]), int(layout.grid_shape_yx[1])],
        "overlap_yx": [int(layout.overlap_yx[0]), int(layout.overlap_yx[1])],
        "grid_source": layout.grid_source,
        "tile_count": int(layout.tile_count),
        "tiles": tile_summaries,
    }


def _run_tiled_native_demons_registration(
    ref_volume_zyx: FloatArray,
    mov_volume_zyx: FloatArray,
    config_obj: Any,
    layout: TileLayout,
) -> Tuple[Optional[FloatArray], Dict[str, Any]]:
    print(
        "     [Tiling] Native demons_3d on "
        f"{layout.tile_count} tiles ({layout.grid_shape_yx[0]}x{layout.grid_shape_yx[1]}, source={layout.grid_source})"
    )

    tile_outputs: list[tuple[TileSpec, FloatArray]] = []
    tile_summaries: list[Dict[str, Any]] = []
    for tile in layout.tiles:
        tile_label = f"tile {tile.tile_index}/{layout.tile_count}"
        print(
            f"       [Tiling] {tile_label} | origin={tile.region_origin_zyx} shape={tile.region_shape_zyx}"
        )
        ref_tile = np.asarray(extract_tile(ref_volume_zyx, tile), dtype=np.float32)
        mov_tile = np.asarray(extract_tile(mov_volume_zyx, tile), dtype=np.float32)
        flow_tile = register_local_demons_3d(ref_tile, mov_tile, config_obj)
        if flow_tile is None:
            raise RuntimeError(
                "Tiled native demons_3d returned no flow for "
                f"{tile_label}; refusing to silently degrade local registration."
            )

        tile_outputs.append((tile, np.asarray(flow_tile, dtype=np.float32)))
        tile_summaries.append(
            {
                **tile.as_dict(),
                "provider": "native",
                "status": "completed",
                "mean_abs_displacement": float(np.abs(flow_tile).mean()),
            }
        )

    stitched = np.asarray(
        stitch_tiles(tile_outputs, full_shape_zyx=layout.full_volume_shape_zyx),
        dtype=np.float32,
    )
    return stitched, _build_tiling_summary(layout, tile_summaries)


def _run_tiled_matlab_demons_registration(
    *,
    backend: MATLABGlobalRegistrationBackend,
    ref_volume_zyx: FloatArray,
    mov_volume_zyx: FloatArray,
    fov_id: int,
    round_id: int,
    reference_round: int,
    scope_descriptor: Dict[str, Any],
    layout: TileLayout,
    diagnostics: Optional[RegistrationPerformanceRecorder] = None,
) -> Tuple[FloatArray, Dict[str, Any]]:
    print(
        "     [Tiling] MATLAB demons_3d on "
        f"{layout.tile_count} tiles ({layout.grid_shape_yx[0]}x{layout.grid_shape_yx[1]}, source={layout.grid_source})"
    )

    tile_outputs: list[tuple[TileSpec, FloatArray]] = []
    tile_summaries: list[Dict[str, Any]] = []
    for tile in layout.tiles:
        tile_total_started = diagnostics_timer_start()
        tile_label = f"tile {tile.tile_index}/{layout.tile_count}"
        print(
            f"       [Tiling] {tile_label} | origin={tile.region_origin_zyx} shape={tile.region_shape_zyx}"
        )
        tile_extract_started = diagnostics_timer_start()
        ref_tile = np.asarray(extract_tile(ref_volume_zyx, tile), dtype=np.float32)
        mov_tile = np.asarray(extract_tile(mov_volume_zyx, tile), dtype=np.float32)
        tile_extract_ms = elapsed_ms_since(tile_extract_started)
        backend_call_started = diagnostics_timer_start()
        local_result = backend.compute_local_flow(
            ref_tile,
            mov_tile,
            fov_id=fov_id,
            round_id=int(round_id),
            reference_round=int(reference_round),
            scope_descriptor=scope_descriptor,
            compute_tile=tile.as_dict(),
        )
        backend_call_ms = elapsed_ms_since(backend_call_started)
        validation_started = diagnostics_timer_start()
        flow_tile = np.asarray(local_result["flow_3d"], dtype=np.float32)
        if flow_tile.shape != (3, *tile.region_shape_zyx):
            raise ValueError(
                "MATLAB tiled local registration returned flow_3d with incompatible tile shape: "
                f"expected {(3, *tile.region_shape_zyx)}, got {flow_tile.shape} for tile {tile.tile_index}"
            )
        flow_validation_ms = elapsed_ms_since(validation_started)

        tile_outputs.append((tile, flow_tile))
        tile_backend_metadata = cast(Dict[str, Any], local_result.get("backend_metadata", {}))
        tile_normalized = cast(Dict[str, Any], tile_backend_metadata.get("normalized_result", {}))
        tile_summaries.append(
            {
                **tile.as_dict(),
                "provider": "matlab",
                "status": "completed",
                "mean_abs_displacement": float(tile_normalized.get("mean_abs_displacement", float(np.abs(flow_tile).mean()))),
            }
        )
        if diagnostics is not None:
            boundary = tile_backend_metadata.get("boundary_instrumentation")
            session_lifecycle = tile_backend_metadata.get("session_lifecycle")
            session_lifecycle_summary = tile_backend_metadata.get("session_lifecycle_summary")
            diagnostics.record_tiled_local_tile(
                round_id,
                tile_identity=tile.as_dict(),
                total_elapsed_wall_ms=elapsed_ms_since(tile_total_started),
                extraction_elapsed_wall_ms=tile_extract_ms,
                backend_call_elapsed_wall_ms=backend_call_ms,
                flow_validation_elapsed_wall_ms=flow_validation_ms,
                boundary_instrumentation=(cast(Mapping[str, Any], boundary) if isinstance(boundary, Mapping) else None),
                session_lifecycle=(
                    cast(Mapping[str, Any], session_lifecycle)
                    if isinstance(session_lifecycle, Mapping)
                    else None
                ),
                session_lifecycle_summary=(
                    cast(Mapping[str, Any], session_lifecycle_summary)
                    if isinstance(session_lifecycle_summary, Mapping)
                    else None
                ),
                normalized_result=tile_normalized,
            )

    stitch_started = diagnostics_timer_start()
    stitched = np.asarray(
        stitch_tiles(tile_outputs, full_shape_zyx=layout.full_volume_shape_zyx),
        dtype=np.float32,
    )
    stitch_ms = elapsed_ms_since(stitch_started)
    tiling_summary = _build_tiling_summary(layout, tile_summaries)
    if diagnostics is not None:
        diagnostics.record_tiled_local_summary(
            round_id,
            layout_summary=tiling_summary,
            stitch_elapsed_wall_ms=stitch_ms,
        )
    return stitched, tiling_summary


def _build_scope_descriptor(ref_clean_3d: FloatArray, scope_mode: str) -> Dict[str, Any]:
    z_dim, y_dim, x_dim = (int(value) for value in ref_clean_3d.shape)
    full_volume_shape = [z_dim, y_dim, x_dim]

    if scope_mode == "full_fov":
        return {
            "coverage_mode": "full_fov",
            "region_origin_zyx": [0, 0, 0],
            "region_shape_zyx": full_volume_shape,
            "full_volume_shape_zyx": full_volume_shape,
        }

    if scope_mode != "tile_local":
        raise ValueError(f"Unsupported scope_mode: {scope_mode!r}")

    layout = build_matlab_subtile_layout(
        (z_dim, y_dim, x_dim),
        sqrt_pieces=4,
        grid_source="scope_tile_local_fixed_4x4",
    )
    tile_scores: list[float] = []
    for tile in layout.tiles:
        tile_scores.append(float(np.asarray(extract_tile_write_window(ref_clean_3d, tile), dtype=np.float32).sum()))

    if not tile_scores:
        raise ValueError("Unable to derive tile_local scope descriptor from an empty reference volume")

    selected_index = int(np.argmax(np.asarray(tile_scores, dtype=np.float32)))
    selected_tile = layout.tiles[selected_index]
    _, y0, x0 = selected_tile.write_origin_zyx
    _, dy, dx = selected_tile.write_shape_zyx
    return {
        "coverage_mode": "tile_local",
        "region_origin_zyx": [0, int(y0), int(x0)],
        "region_shape_zyx": [z_dim, int(dy), int(dx)],
        "full_volume_shape_zyx": full_volume_shape,
        "tile_grid_shape_yx": [int(layout.grid_shape_yx[0]), int(layout.grid_shape_yx[1])],
        "tile_index": selected_index + 1,
    }


def _crop_volume_to_scope(img_3d: FloatArray, scope_descriptor: Dict[str, Any]) -> FloatArray:
    z0, y0, x0 = (int(value) for value in scope_descriptor["region_origin_zyx"])
    dz, dy, dx = (int(value) for value in scope_descriptor["region_shape_zyx"])
    z1, y1, x1 = z0 + dz, y0 + dy, x0 + dx
    return img_3d[z0:z1, y0:y1, x0:x1]


def _format_scope_descriptor(scope_descriptor: Dict[str, Any]) -> str:
    coverage_mode = scope_descriptor["coverage_mode"]
    z0, y0, x0 = scope_descriptor["region_origin_zyx"]
    dz, dy, dx = scope_descriptor["region_shape_zyx"]
    region_summary = f"z[{z0}:{z0 + dz}) y[{y0}:{y0 + dy}) x[{x0}:{x0 + dx})"
    if coverage_mode == "full_fov":
        return f"full_fov ({region_summary})"

    tile_grid_shape = scope_descriptor.get("tile_grid_shape_yx", [1, 1])
    tile_index = scope_descriptor.get("tile_index", "?")
    return f"tile_local tile {tile_index}/{int(tile_grid_shape[0]) * int(tile_grid_shape[1])} ({region_summary})"


def _attach_local_flow_metadata(
    backend_metadata: Optional[Dict[str, Any]],
    *,
    provider: str,
    local_method: str,
    status: str,
    corr_after_global: float,
    corr_after_local: Optional[float] = None,
    reject_if_worse: Optional[bool] = None,
    default_mode: Optional[str] = None,
) -> Dict[str, Any]:
    metadata = dict(backend_metadata) if backend_metadata is not None else {}
    if default_mode is not None:
        metadata['mode'] = default_mode
    metadata.setdefault('provider', provider)

    local_flow_metadata: Dict[str, Any] = {
        'provider': provider,
        'requested_local_method': local_method,
        'status': status,
        'diagnostic_corr_after_global': float(corr_after_global),
    }
    if corr_after_local is not None:
        local_flow_metadata['diagnostic_corr_after_local'] = float(corr_after_local)
        local_flow_metadata['diagnostic_corr_delta'] = float(corr_after_local - corr_after_global)
    if reject_if_worse is not None:
        local_flow_metadata['reject_if_correlation_worse_configured'] = bool(reject_if_worse)
        local_flow_metadata['correlation_gate_effective'] = 'diagnostic_only'

    metadata['local_flow'] = local_flow_metadata
    return metadata

# ==============================================================================
# SECTION E: Registration Engine (Orchestration)
# ==============================================================================

class RegistrationEngine:
    """Orchestrate per-FOV registration and persist transform manifests.

    The engine reads preprocessed clean volumes from the canonical PyStar output
    tree, computes each moving round's transform relative to
    ``registration.reference_round``, and writes ``transforms_fov_<id>.npy`` plus
    provenance. Transform dictionaries use image coordinates in ``z, y, x``:
    ``global_shift_3d`` is the rigid moving-to-reference shift, while local
    ``flow_3d`` fields are residual displacements after that rigid shift. The
    ``_semantics`` and ``_scope`` annotations are part of the runtime contract
    consumed later by extraction; they prevent applying tile-local or differently
    composed fields as if they were full-FOV total transforms.

    Native and MATLAB providers share the same manifest schema. Provider choice
    only changes how global shifts and local flows are computed; missing or
    unsupported provider contracts fail explicitly instead of falling back.
    """

    def __init__(self, config: ExperimentConfig, matlab_session_owner: Optional[MatlabSharedSessionOwner] = None):
        self.cfg = config
        self.reg_cfg = config.pipeline.registration
        self._matlab_session_owner = matlab_session_owner
        self._matlab_backend: Optional[MATLABGlobalRegistrationBackend] = None
        self._registration_diagnostics: Optional[RegistrationPerformanceRecorder] = None

    def close(self) -> None:
        """Release the lazily-created MATLAB registration backend, if any."""
        if self._matlab_backend is None:
            return
        self._matlab_backend.close()
        self._matlab_backend = None

    def __del__(self):  # pragma: no cover - best-effort cleanup only
        try:
            self.close()
        except Exception:
            pass

    def _get_matlab_backend(self) -> MATLABGlobalRegistrationBackend:
        if self._matlab_backend is None:
            self._matlab_backend = MATLABGlobalRegistrationBackend(
                self.cfg,
                matlab_session_owner=self._matlab_session_owner,
            )
        return self._matlab_backend

    def _build_provenance(
        self,
        transforms: Dict[Any, Any],
        round_summary: Dict[int, Dict[str, str]],
        started_at: str,
        ended_at: str,
        backend_round_metadata: Optional[Dict[int, Dict[str, Any]]] = None,
        registration_backend_details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build the transform-manifest provenance record.

        The provenance captures the requested registration intent, provider
        boundaries, field semantics, tiling settings, software versions, hardware
        context, and per-round backend metadata. Downstream stages use this
        record as evidence that a transform bundle was produced under the same
        contract they are about to consume.
        """
        execution_envelope = build_execution_envelope(self.cfg)
        release_contract = build_release_contract(self.cfg, transforms)

        software_versions: Dict[str, str] = {
            "Python": sys.version.split()[0],
            "NumPy": np.__version__,
        }
        sitk_version = sitk.Version_VersionString() if hasattr(sitk, "Version_VersionString") else None
        if sitk_version:
            software_versions["SimpleITK"] = sitk_version
        if registration_backend_details is not None:
            matlab_version = registration_backend_details.get("matlab_version")
            if isinstance(matlab_version, str) and matlab_version.strip():
                software_versions["MATLAB"] = matlab_version

        hardware_context: Dict[str, Any] = {
            "cpu_count": os.cpu_count() or 1,
        }
        available_memory_bytes = _get_available_memory_bytes()
        if available_memory_bytes is not None:
            hardware_context["memory_available_bytes"] = available_memory_bytes

        config_source_path = self.cfg.config_source_path
        config_hash = self.cfg.config_sha256 or "sha256:unknown"
        profile = release_contract["requested_intent"]["registration_profile"]
        expected_field_semantics = self.cfg.pipeline.field_semantics.as_dict()
        declared_field_semantics = self.cfg.pipeline.registration.field_semantics.as_dict()
        demons_tiling = self.cfg.pipeline.registration.demons_3d
        key_parameters = {
            "scope_mode": self.cfg.pipeline.scope_mode,
            "transform_application_mode": self.cfg.pipeline.extraction.transform_application_mode,
            "application_intent": release_contract["requested_intent"]["application_intent"],
            "preprocessing_provider_mode": release_contract["requested_intent"].get("preprocessing_provider_mode"),
            "preprocessing_steps": release_contract["requested_intent"].get("preprocessing_steps"),
            "spot_finding_provider": release_contract["requested_intent"].get("spot_finding_provider"),
            "extraction_provider": release_contract["requested_intent"].get("extraction_provider"),
            "matlab_stage_contracts": release_contract["requested_intent"].get("matlab_stage_contracts"),
            "local_acceptance_mode": release_contract["requested_intent"]["local_acceptance_mode"],
            "final_corr_metric": release_contract["requested_intent"]["final_corr_metric"],
            "final_corr_diagnostic_only": release_contract["requested_intent"]["final_corr_diagnostic_only"],
            "final_corr_release_gate": release_contract["requested_intent"]["final_corr_release_gate"],
            "registration_profile": profile["name"],
            "registration_global_method": profile.get("global_method"),
            "registration_global_provider": profile.get("global_provider"),
            "registration_local_method": profile.get("local_method"),
            "registration_local_provider": profile.get("local_provider"),
            "declared_transform_capabilities": profile["declared_transform_capabilities"],
            "preprocessing_backend": execution_envelope["preprocessing_backend"],
            "registration_backend": execution_envelope["registration_backend"],
            "accelerator": self.cfg.pipeline.accelerator,
            "backend_mode_status": release_contract["requested_intent"].get("backend_mode_status"),
            "field_semantics": expected_field_semantics,
            "registration_field_semantics": declared_field_semantics,
            "local_tiling_enabled": bool(demons_tiling.use_tiling),
            "local_tiling_layout_policy": demons_tiling.tiling_layout_policy,
            "local_tiling_tile_size": int(demons_tiling.tile_size),
            "local_tiling_tile_overlap": (
                None if demons_tiling.tile_overlap is None else int(demons_tiling.tile_overlap)
            ),
            "local_tiling_sqrt_pieces": (
                None if demons_tiling.sqrt_pieces is None else int(demons_tiling.sqrt_pieces)
            ),
            "local_tiling_grid_shape_yx": (
                None
                if demons_tiling.tile_grid_shape_yx is None
                else [
                    int(demons_tiling.tile_grid_shape_yx[0]),
                    int(demons_tiling.tile_grid_shape_yx[1]),
                ]
            ),
        }
        if self.reg_cfg.global_provider == "matlab":
            key_parameters["matlab_registration_entrypoint"] = self.cfg.providers.matlab.registration.entrypoint
        if self.reg_cfg.local_provider == "matlab":
            key_parameters["matlab_local_registration_entrypoint"] = self.cfg.providers.matlab.registration.local_entrypoints.get(
                self.reg_cfg.local_method
            )

        runtime_context = {
            "pipeline_version": "0+unknown",
            "execution_timestamp": ended_at,
            "environment_hash": _compute_environment_hash(config_hash, execution_envelope, software_versions),
            "start_time": started_at,
            "end_time": ended_at,
            "duration_seconds": (
                datetime.fromisoformat(ended_at).timestamp() - datetime.fromisoformat(started_at).timestamp()
            ),
            "software_versions": software_versions,
            "hardware_context": hardware_context,
            "config_reference": {
                "config_path": str(config_source_path) if config_source_path is not None else "inline-config",
                "config_hash": config_hash,
                "key_parameters": key_parameters,
            },
        }
        if registration_backend_details is not None:
            runtime_context["registration_backend_details"] = registration_backend_details

        stage_outcomes: Dict[str, Any] = {"round_summary": round_summary}
        if backend_round_metadata:
            stage_outcomes["registration_backend"] = {
                "name": execution_envelope["registration_backend"],
                "experimental": execution_envelope["registration_backend"] != "native_pystar",
                "round_results": backend_round_metadata,
            }

        return {
            "provenance_version": PROVENANCE_VERSION,
            "execution_envelope": execution_envelope,
            "runtime_context": runtime_context,
            "stage_outcomes": stage_outcomes,
            "release_contract": release_contract,
        }

    def _save_transforms(self, transforms: Dict[int, Dict[str, Any]], fov_id: int, provenance: Optional[Dict[str, Any]] = None):
        """Persist the full per-round transform manifest for one FOV."""
        base_dir = Path(self.cfg.pipeline.output.directory)
        save_transform_manifest(base_dir, fov_id, transforms, provenance=provenance)

    def _spill_round_flow_3d(self, fov_id: int, round_id: int, round_transform: Dict[str, Any]) -> None:
        """Move large 3-D displacement fields to sidecar files.

        ``flow_3d`` arrays have shape ``(3, z, y, x)`` and can dominate manifest
        size. This helper writes the array under the FOV transform directory and
        replaces the in-memory array with a descriptor that can be materialized
        later by ``load_transform_manifest``.
        """
        flow_3d = round_transform.get('flow_3d')
        if flow_3d is None or not isinstance(flow_3d, np.ndarray):
            if self._registration_diagnostics is not None:
                self._registration_diagnostics.record_flow_sidecar_persistence(
                    round_id,
                    elapsed_wall_ms=0.0,
                    descriptor=None,
                    sidecar_path=None,
                )
            return

        base_dir = Path(self.cfg.pipeline.output.directory)
        persist_started = diagnostics_timer_start()
        descriptor = persist_flow_3d_sidecar(base_dir, fov_id, int(round_id), np.asarray(flow_3d))
        elapsed_ms = elapsed_ms_since(persist_started)
        round_transform['flow_3d'] = descriptor
        if self._registration_diagnostics is not None:
            paths = get_fov_output_structure(base_dir, fov_id)
            descriptor_path = descriptor.get("path") if isinstance(descriptor, dict) else None
            sidecar_path = paths["transforms"] / str(descriptor_path) if isinstance(descriptor_path, str) else None
            self._registration_diagnostics.record_flow_sidecar_persistence(
                round_id,
                elapsed_wall_ms=elapsed_ms,
                descriptor=descriptor,
                sidecar_path=sidecar_path,
            )

    def _annotate_transform_semantics(self, transforms: Dict[int, Dict[str, Any]]) -> None:
        """Attach declared field-composition metadata to every round transform."""
        semantics = self.cfg.pipeline.registration.field_semantics.as_dict()
        recorded_at = _iso_utc_now()
        for round_id, transform_data in transforms.items():
            if not isinstance(transform_data, dict) or 'global_shift_3d' not in transform_data:
                continue
            transform_data['_semantics'] = {
                **semantics,
                'recorded_at': recorded_at,
            }

    def _annotate_transform_scope(self, transforms: Dict[int, Dict[str, Any]], scope_descriptor: Dict[str, Any]) -> None:
        """Attach full-FOV or tile-local coverage metadata to every transform."""
        for _, transform_data in transforms.items():
            if not isinstance(transform_data, dict) or 'global_shift_3d' not in transform_data:
                continue
            transform_data['_scope'] = dict(scope_descriptor)
        
    def _load_combined_clean_volume(self, fov_id: int, round_id: int) -> FloatArray:
        """
        加载该轮次用于 registration 的 clean data 体积。

        - method='mip_all_channels'：读取指定 seq 通道并在 channel 轴做 max projection
        - method='single_channel'：只读取单个 seq 通道，避免不必要的堆叠峰值内存

        Returns:
            volume_3d: (Z, Y, X)
        """
        loader = ImageLoader(self.cfg)
        round_structure = self.cfg.dataset.round_structure.get(int(round_id))
        if round_structure is None:
            raise KeyError(f"Round {round_id} is missing from dataset.round_structure")

        roles = self.cfg.dataset.channel_roles
        seq_channels = [c for c in round_structure if roles.get(c) == 'seq']
        if not seq_channels:
            raise ValueError(f"Round {round_id} has no SEQ channels for registration!")

        source_cfg = self.reg_cfg.source
        if source_cfg.method == 'single_channel':
            single_channel_id = source_cfg.single_channel_id
            if single_channel_id is None:
                raise ValueError(
                    "registration.source.single_channel_id is required when method='single_channel'"
                )
            target_channel = int(single_channel_id)
            if target_channel not in seq_channels:
                raise ValueError(
                    f"registration.source.single_channel_id={target_channel} is not a seq channel available in round {round_id}: {seq_channels}"
                )
            return loader.load_clean_image(fov_id, round_id, target_channel)

        if source_cfg.method != 'mip_all_channels':
            raise ValueError(f"Unsupported registration.source.method: {source_cfg.method!r}")

        requested_channels = [int(c) for c in (source_cfg.mip_channels or [])]
        target_channels = [c for c in requested_channels if c in seq_channels]
        if not target_channels:
            raise ValueError(
                f"registration.source.mip_channels={requested_channels} does not overlap with available seq channels in round {round_id}: {seq_channels}"
            )

        if len(target_channels) == 1:
            return loader.load_clean_image(fov_id, round_id, target_channels[0])

        first_channel, *remaining_channels = target_channels
        combined_vol = np.array(
            loader.load_clean_image(fov_id, round_id, first_channel),
            copy=True,
        )
        for channel_id in remaining_channels:
            np.maximum(
                combined_vol,
                loader.load_clean_image(fov_id, round_id, channel_id),
                out=combined_vol,
            )
        return combined_vol

    def _resolve_global_registration_handler_name(self, provider: str) -> str:
        try:
            return _GLOBAL_REGISTRATION_HANDLER_NAMES[provider]
        except KeyError as exc:
            raise ValueError(f"Unsupported registration.global.provider: {provider!r}") from exc

    def _resolve_local_registration_handler_name(self, *, local_provider: str, local_method: str) -> str:
        try:
            return _LOCAL_REGISTRATION_HANDLER_NAMES[(local_provider, local_method)]
        except KeyError as exc:
            if local_provider == 'native':
                raise ValueError(
                    f"Unsupported registration.local.method for provider 'native': {local_method!r}"
                ) from exc
            if local_provider == 'matlab':
                _ensure_supported_matlab_local_method(local_method)
            raise ValueError(f"Unsupported registration.local.provider: {local_provider!r}") from exc

    def _build_round_registration_strategy(
        self,
        *,
        global_provider: Optional[str] = None,
        local_provider: Optional[str] = None,
        local_method: Optional[str] = None,
        enable_local: Optional[bool] = None,
        local_skip_if_global_corr_below: Optional[float] = None,
    ) -> _RoundRegistrationStrategy:
        resolved_global_provider = cast(
            str,
            self.reg_cfg.global_provider if global_provider is None else global_provider,
        )
        resolved_local_provider = cast(
            str,
            self.reg_cfg.local_provider if local_provider is None else local_provider,
        )
        resolved_local_method = cast(
            str,
            self.reg_cfg.local_method if local_method is None else local_method,
        )
        resolved_enable_local = bool(self.reg_cfg.enable_local if enable_local is None else enable_local)
        resolved_local_threshold = float(
            self.reg_cfg.guards.skip_if_global_corr_below
            if local_skip_if_global_corr_below is None
            else local_skip_if_global_corr_below
        )

        global_handler_name = self._resolve_global_registration_handler_name(resolved_global_provider)
        local_handler_name: Optional[str] = None
        local_default_mode: Optional[str] = None
        if resolved_enable_local:
            try:
                local_default_mode = _LOCAL_PROVIDER_DEFAULT_MODES[resolved_local_provider]
            except KeyError as exc:
                raise ValueError(
                    f"Unsupported registration.local.provider: {resolved_local_provider!r}"
                ) from exc
            local_handler_name = self._resolve_local_registration_handler_name(
                local_provider=resolved_local_provider,
                local_method=resolved_local_method,
            )

        return _RoundRegistrationStrategy(
            global_provider=resolved_global_provider,
            local_provider=resolved_local_provider,
            local_method=resolved_local_method,
            enable_local=resolved_enable_local,
            local_skip_if_global_corr_below=resolved_local_threshold,
            global_handler_name=global_handler_name,
            local_handler_name=local_handler_name,
            local_default_mode=local_default_mode,
        )

    def _run_global_registration_native(
        self,
        *,
        ref_scope_3d: FloatArray,
        mov_scope_3d: FloatArray,
        **_: Any,
    ) -> Tuple[FloatArray, float, Optional[Dict[str, Any]]]:
        global_shift_3d, global_corr = compute_global_shift_3d(
            ref_scope_3d,
            mov_scope_3d,
            downsample_factor=self.reg_cfg.downsample_factor,
            max_shift=self.reg_cfg.global_max_shift,
        )
        print(f"     Global Shift (3D): {global_shift_3d}, Corr (Est): {global_corr:.4f}")
        return global_shift_3d, global_corr, None

    def _run_global_registration_matlab(
        self,
        *,
        fov_id: int,
        round_id: int,
        ref_round: int,
        ref_scope_3d: FloatArray,
        mov_scope_3d: FloatArray,
        scope_descriptor: Dict[str, Any],
        **_: Any,
    ) -> Tuple[FloatArray, float, Optional[Dict[str, Any]]]:
        matlab_backend = self._get_matlab_backend()
        result = matlab_backend.compute_global_shift(
            ref_scope_3d,
            mov_scope_3d,
            fov_id=fov_id,
            round_id=int(round_id),
            reference_round=int(ref_round),
            scope_descriptor=scope_descriptor,
        )
        global_shift_3d = np.asarray(result['global_shift_3d'], dtype=np.float32)
        global_corr = float(result['global_corr'])
        backend_metadata = cast(Optional[Dict[str, Any]], result.get('backend_metadata'))
        print(f"     MATLAB Global Shift (3D): {global_shift_3d}, Corr (Kernel): {global_corr:.4f}")
        return global_shift_3d, global_corr, backend_metadata

    def _run_global_registration(
        self,
        *,
        handler_name: str,
        provider: Optional[str] = None,
        method: Optional[str] = None,
        fov_id: int,
        round_id: int,
        ref_round: int,
        ref_scope_3d: FloatArray,
        mov_scope_3d: FloatArray,
        scope_descriptor: Dict[str, Any],
    ) -> Tuple[FloatArray, float, Optional[Dict[str, Any]]]:
        handler = cast(Any, getattr(self, handler_name))
        global_started = diagnostics_timer_start()
        result = handler(
            fov_id=fov_id,
            round_id=round_id,
            ref_round=ref_round,
            ref_scope_3d=ref_scope_3d,
            mov_scope_3d=mov_scope_3d,
            scope_descriptor=scope_descriptor,
        )
        elapsed_ms = elapsed_ms_since(global_started)
        global_shift_3d, global_corr, backend_metadata = result
        if self._registration_diagnostics is not None and round_id >= 0:
            self._registration_diagnostics.record_global_registration(
                round_id,
                elapsed_wall_ms=elapsed_ms,
                provider=provider,
                method=method,
                global_shift_3d=global_shift_3d,
                global_corr=float(global_corr),
                backend_metadata=backend_metadata,
            )
        return result

    def _compute_post_global_registration_state(
        self,
        *,
        ref_mip_clean: FloatArray,
        mov_scope_3d: FloatArray,
        global_shift_3d: FloatArray,
    ) -> Tuple[FloatArray, FloatArray, FloatArray, float]:
        mov_mip_clean = mov_scope_3d.max(axis=0)
        shift_2d = global_shift_3d[1:]
        mov_mip_shifted = np.asarray(
            scipy_shift(mov_mip_clean, shift_2d, order=1),
            dtype=np.float32,
        )
        corr_after_global = simple_correlation(ref_mip_clean, mov_mip_shifted)
        print(f"     After Global (Clean MIP): Corr = {corr_after_global:.4f}")
        return mov_mip_clean, shift_2d, mov_mip_shifted, corr_after_global

    def _build_local_registration_context(
        self,
        *,
        fov_id: int,
        round_id: int,
        ref_round: int,
        ref_scope_3d: FloatArray,
        ref_mip_clean: FloatArray,
        mov_scope_3d: FloatArray,
        mov_mip_clean: FloatArray,
        mov_mip_shifted: FloatArray,
        shift_2d: FloatArray,
        global_shift_3d: FloatArray,
        corr_after_global: float,
        scope_descriptor: Dict[str, Any],
        backend_metadata: Optional[Dict[str, Any]],
        local_provider: str,
        local_method: str,
        local_handler_name: Optional[str] = None,
    ) -> _LocalRegistrationContext:
        reject_if_worse = bool(self.reg_cfg.guards.reject_if_correlation_worse)
        overlap_mask = compute_overlap_roi(
            (int(ref_mip_clean.shape[0]), int(ref_mip_clean.shape[1])),
            shift_2d,
        )
        mask = create_quality_mask(ref_mip_clean, valid_roi_mask=overlap_mask)
        return _LocalRegistrationContext(
            fov_id=fov_id,
            round_id=round_id,
            ref_round=ref_round,
            ref_scope_3d=ref_scope_3d,
            ref_mip_clean=ref_mip_clean,
            mov_scope_3d=mov_scope_3d,
            mov_mip_clean=mov_mip_clean,
            mov_mip_shifted=mov_mip_shifted,
            shift_2d=shift_2d,
            global_shift_3d=global_shift_3d,
            corr_after_global=float(corr_after_global),
            scope_descriptor=scope_descriptor,
            backend_metadata=backend_metadata,
            reject_if_worse=reject_if_worse,
            mask=mask,
            local_provider=local_provider,
            local_method=local_method,
            local_handler_name=local_handler_name,
        )

    def _build_round_transform(
        self,
        *,
        global_shift_3d: FloatArray,
        global_corr: float,
        flow_2d: Optional[FloatArray],
        flow_3d: Optional[FloatArray],
        final_corr: float,
    ) -> Dict[str, Any]:
        return {
            'global_shift_3d': np.asarray(global_shift_3d, dtype=np.float32),
            'global_corr': float(global_corr),
            'flow_2d': flow_2d,
            'flow_3d': flow_3d,
            'final_corr': float(final_corr),
            'is_reference_round': False,
        }

    def _dispatch_local_registration(self, context: _LocalRegistrationContext) -> _LocalRegistrationOutcome:
        resolved_handler_name = context.local_handler_name
        if resolved_handler_name is None:
            resolved_handler_name = self._resolve_local_registration_handler_name(
                local_provider=context.local_provider,
                local_method=context.local_method,
            )
        handler = cast(Any, getattr(self, resolved_handler_name))
        return handler(context)

    def _run_local_native_demons_3d(self, context: _LocalRegistrationContext) -> _LocalRegistrationOutcome:
        rigid_started = diagnostics_timer_start()
        mov_shifted_3d = apply_rigid_shift_3d(context.mov_scope_3d, context.global_shift_3d)
        if self._registration_diagnostics is not None and context.round_id >= 0:
            self._registration_diagnostics.record_local_internal_phase(
                context.round_id,
                "local_rigid_shift_preparation",
                elapsed_ms_since(rigid_started),
            )
        tiling_started = diagnostics_timer_start()
        tiling_layout = _resolve_demons_tiling_layout(
            cast(Tuple[int, int, int], tuple(int(value) for value in mov_shifted_3d.shape)),
            self.cfg.pipeline.registration.demons_3d,
        )
        if self._registration_diagnostics is not None and context.round_id >= 0:
            self._registration_diagnostics.record_local_internal_phase(
                context.round_id,
                "local_tiling_layout_resolution",
                elapsed_ms_since(tiling_started),
                details={"tiling_enabled": tiling_layout is not None},
            )
        tiling_summary: Optional[Dict[str, Any]] = None
        native_demons_started = diagnostics_timer_start()
        if tiling_layout is None:
            flow_3d = register_local_demons_3d(
                context.ref_scope_3d,
                mov_shifted_3d,
                self.cfg.pipeline.registration.demons_3d,
            )
        else:
            flow_3d, tiling_summary = _run_tiled_native_demons_registration(
                context.ref_scope_3d,
                mov_shifted_3d,
                self.cfg.pipeline.registration.demons_3d,
                tiling_layout,
            )
        if self._registration_diagnostics is not None and context.round_id >= 0:
            self._registration_diagnostics.record_local_internal_phase(
                context.round_id,
                "native_demons_registration_compute",
                elapsed_ms_since(native_demons_started),
            )

        if flow_3d is None:
            backend_metadata = _attach_local_flow_metadata(
                context.backend_metadata,
                provider='native',
                local_method=context.local_method,
                status='no_flow_returned',
                corr_after_global=float(context.corr_after_global),
                reject_if_worse=context.reject_if_worse,
                default_mode='native_local_registration',
            )
            if tiling_summary is not None:
                backend_metadata['local_flow']['tiling'] = tiling_summary
            return _LocalRegistrationOutcome(
                flow_2d=None,
                flow_3d=None,
                final_corr=float(context.corr_after_global),
                final_img_qc=context.mov_mip_shifted,
                backend_metadata=backend_metadata,
            )

        final_qc_started = diagnostics_timer_start()
        final_img_qc_clean = apply_warp_field_3d(mov_shifted_3d, flow_3d).max(axis=0)
        rec_corr = simple_correlation(context.ref_mip_clean, final_img_qc_clean)
        if self._registration_diagnostics is not None and context.round_id >= 0:
            self._registration_diagnostics.record_final_qc(
                context.round_id,
                elapsed_wall_ms=elapsed_ms_since(final_qc_started),
                final_corr=float(rec_corr),
            )
        status = 'accepted'
        if rec_corr < context.corr_after_global:
            print(
                "  [Diagnostic] Correlation decreased, but keeping local 3D flow; "
                "final_corr remains diagnostic-only "
                f"({rec_corr:.4f} < {context.corr_after_global:.4f})"
            )
            status = 'accepted_correlation_decrease'
        else:
            print(
                "  [Diagnostic] After Local 3D: "
                f"Corr = {rec_corr:.4f} (Δ = {rec_corr - context.corr_after_global:+.4f})"
            )
        backend_metadata = _attach_local_flow_metadata(
            context.backend_metadata,
            provider='native',
            local_method=context.local_method,
            status=status,
            corr_after_global=float(context.corr_after_global),
            corr_after_local=float(rec_corr),
            reject_if_worse=context.reject_if_worse,
            default_mode='native_local_registration',
        )
        if tiling_summary is not None:
            backend_metadata['local_flow']['tiling'] = tiling_summary
        return _LocalRegistrationOutcome(
            flow_2d=None,
            flow_3d=flow_3d,
            final_corr=float(rec_corr),
            final_img_qc=final_img_qc_clean,
            backend_metadata=backend_metadata,
        )

    def _run_local_native_optical_flow(self, context: _LocalRegistrationContext) -> _LocalRegistrationOutcome:
        compute_started = diagnostics_timer_start()
        flow_2d = compute_optical_flow_masked(
            context.ref_mip_clean,
            context.mov_mip_shifted,
            context.mask,
            self.cfg.pipeline.registration.optical_flow,
        )
        if self._registration_diagnostics is not None and context.round_id >= 0:
            self._registration_diagnostics.record_local_internal_phase(
                context.round_id,
                "native_optical_flow_compute",
                elapsed_ms_since(compute_started),
            )
        if flow_2d is None:
            backend_metadata = _attach_local_flow_metadata(
                context.backend_metadata,
                provider='native',
                local_method=context.local_method,
                status='no_flow_returned',
                corr_after_global=float(context.corr_after_global),
                reject_if_worse=context.reject_if_worse,
                default_mode='native_local_registration',
            )
            return _LocalRegistrationOutcome(
                flow_2d=None,
                flow_3d=None,
                final_corr=float(context.corr_after_global),
                final_img_qc=context.mov_mip_shifted,
                backend_metadata=backend_metadata,
            )

        final_qc_started = diagnostics_timer_start()
        final_img_qc_clean = composite_transform_2d(context.mov_mip_clean, context.shift_2d, flow_2d)
        rec_corr = simple_correlation(context.ref_mip_clean, final_img_qc_clean)
        if self._registration_diagnostics is not None and context.round_id >= 0:
            self._registration_diagnostics.record_final_qc(
                context.round_id,
                elapsed_wall_ms=elapsed_ms_since(final_qc_started),
                final_corr=float(rec_corr),
            )
        diff = rec_corr - context.corr_after_global
        status = 'accepted'
        if rec_corr < context.corr_after_global:
            print(
                "     [Diagnostic] Correlation decreased, but keeping local flow; "
                "final_corr remains diagnostic-only "
                f"({rec_corr:.4f} < {context.corr_after_global:.4f})"
            )
            status = 'accepted_correlation_decrease'
        else:
            print(
                "     [Diagnostic] After Local:  "
                f"Corr = {rec_corr:.4f} (Δ = {diff:+.4f})"
            )
        backend_metadata = _attach_local_flow_metadata(
            context.backend_metadata,
            provider='native',
            local_method=context.local_method,
            status=status,
            corr_after_global=float(context.corr_after_global),
            corr_after_local=float(rec_corr),
            reject_if_worse=context.reject_if_worse,
            default_mode='native_local_registration',
        )
        return _LocalRegistrationOutcome(
            flow_2d=flow_2d,
            flow_3d=None,
            final_corr=float(rec_corr),
            final_img_qc=final_img_qc_clean,
            backend_metadata=backend_metadata,
        )

    def _run_local_native_bspline(self, context: _LocalRegistrationContext) -> _LocalRegistrationOutcome:
        compute_started = diagnostics_timer_start()
        flow_2d = register_local_bspline(
            context.ref_mip_clean,
            context.mov_mip_shifted,
            context.mask,
            self.cfg.pipeline.registration.bspline,
        )
        if self._registration_diagnostics is not None and context.round_id >= 0:
            self._registration_diagnostics.record_local_internal_phase(
                context.round_id,
                "native_bspline_compute",
                elapsed_ms_since(compute_started),
            )
        if flow_2d is None:
            backend_metadata = _attach_local_flow_metadata(
                context.backend_metadata,
                provider='native',
                local_method=context.local_method,
                status='no_flow_returned',
                corr_after_global=float(context.corr_after_global),
                reject_if_worse=context.reject_if_worse,
                default_mode='native_local_registration',
            )
            return _LocalRegistrationOutcome(
                flow_2d=None,
                flow_3d=None,
                final_corr=float(context.corr_after_global),
                final_img_qc=context.mov_mip_shifted,
                backend_metadata=backend_metadata,
            )

        final_qc_started = diagnostics_timer_start()
        final_img_qc_clean = composite_transform_2d(context.mov_mip_clean, context.shift_2d, flow_2d)
        rec_corr = simple_correlation(context.ref_mip_clean, final_img_qc_clean)
        if self._registration_diagnostics is not None and context.round_id >= 0:
            self._registration_diagnostics.record_final_qc(
                context.round_id,
                elapsed_wall_ms=elapsed_ms_since(final_qc_started),
                final_corr=float(rec_corr),
            )
        diff = rec_corr - context.corr_after_global
        status = 'accepted'
        if rec_corr < context.corr_after_global:
            print(
                "     [Diagnostic] Correlation decreased, but keeping local flow; "
                "final_corr remains diagnostic-only "
                f"({rec_corr:.4f} < {context.corr_after_global:.4f})"
            )
            status = 'accepted_correlation_decrease'
        else:
            print(
                "     [Diagnostic] After Local:  "
                f"Corr = {rec_corr:.4f} (Δ = {diff:+.4f})"
            )
        backend_metadata = _attach_local_flow_metadata(
            context.backend_metadata,
            provider='native',
            local_method=context.local_method,
            status=status,
            corr_after_global=float(context.corr_after_global),
            corr_after_local=float(rec_corr),
            reject_if_worse=context.reject_if_worse,
            default_mode='native_local_registration',
        )
        return _LocalRegistrationOutcome(
            flow_2d=flow_2d,
            flow_3d=None,
            final_corr=float(rec_corr),
            final_img_qc=final_img_qc_clean,
            backend_metadata=backend_metadata,
        )

    def _run_local_matlab_demons_3d(self, context: _LocalRegistrationContext) -> _LocalRegistrationOutcome:
        _ensure_supported_matlab_local_method(context.local_method)
        matlab_backend = self._get_matlab_backend()
        rigid_started = diagnostics_timer_start()
        mov_shifted_3d = apply_rigid_shift_3d(context.mov_scope_3d, context.global_shift_3d)
        if self._registration_diagnostics is not None and context.round_id >= 0:
            self._registration_diagnostics.record_local_internal_phase(
                context.round_id,
                "local_rigid_shift_preparation",
                elapsed_ms_since(rigid_started),
            )
        tiling_started = diagnostics_timer_start()
        tiling_layout = _resolve_demons_tiling_layout(
            cast(Tuple[int, int, int], tuple(int(value) for value in mov_shifted_3d.shape)),
            self.cfg.pipeline.registration.demons_3d,
        )
        if self._registration_diagnostics is not None and context.round_id >= 0:
            self._registration_diagnostics.record_local_internal_phase(
                context.round_id,
                "local_tiling_layout_resolution",
                elapsed_ms_since(tiling_started),
                details={"tiling_enabled": tiling_layout is not None},
            )
        tiling_summary: Optional[Dict[str, Any]] = None
        local_result: Optional[Dict[str, Any]] = None
        local_backend_details: Dict[str, Any]

        if tiling_layout is None:
            matlab_local_started = diagnostics_timer_start()
            local_result = matlab_backend.compute_local_flow(
                context.ref_scope_3d,
                mov_shifted_3d,
                fov_id=context.fov_id,
                round_id=int(context.round_id),
                reference_round=int(context.ref_round),
                scope_descriptor=context.scope_descriptor,
            )
            if self._registration_diagnostics is not None and context.round_id >= 0:
                self._registration_diagnostics.record_local_internal_phase(
                    context.round_id,
                    "matlab_local_backend_call",
                    elapsed_ms_since(matlab_local_started),
                )
            validation_started = diagnostics_timer_start()
            flow_3d = np.asarray(local_result['flow_3d'], dtype=np.float32)
            local_backend_details = cast(Dict[str, Any], local_result['backend_metadata'])
            if self._registration_diagnostics is not None and context.round_id >= 0:
                self._registration_diagnostics.record_local_internal_phase(
                    context.round_id,
                    "matlab_local_flow_array_normalization",
                    elapsed_ms_since(validation_started),
                )
        else:
            flow_3d, tiling_summary = _run_tiled_matlab_demons_registration(
                backend=matlab_backend,
                ref_volume_zyx=context.ref_scope_3d,
                mov_volume_zyx=mov_shifted_3d,
                fov_id=context.fov_id,
                round_id=int(context.round_id),
                reference_round=int(context.ref_round),
                scope_descriptor=context.scope_descriptor,
                layout=tiling_layout,
                diagnostics=self._registration_diagnostics,
            )
            local_backend_details = {
                'provider': 'matlab',
                'mode': 'experimental_local_kernel_swap',
                'runtime': {
                    'runtime_path': str(matlab_backend.runtime_dir),
                    'runtime_manifest': str(matlab_backend.runtime_dir / 'runtime_manifest.json'),
                    'entrypoint': matlab_backend.local_entrypoint,
                },
                'normalized_result': {
                    'flow_3d_shape': [int(value) for value in flow_3d.shape],
                    'flow_3d_dtype': str(flow_3d.dtype),
                    'mean_abs_displacement': float(np.abs(flow_3d).mean()),
                },
            }

        shape_validation_started = diagnostics_timer_start()
        expected_shape = (3, *mov_shifted_3d.shape)
        if flow_3d.shape != expected_shape:
            raise ValueError(
                "MATLAB local registration returned flow_3d with incompatible shape: "
                f"expected {expected_shape}, got {flow_3d.shape}"
            )
        if self._registration_diagnostics is not None and context.round_id >= 0:
            self._registration_diagnostics.record_local_internal_phase(
                context.round_id,
                "matlab_local_flow_shape_validation",
                elapsed_ms_since(shape_validation_started),
                details={"expected_shape": list(expected_shape), "actual_shape": [int(value) for value in flow_3d.shape]},
            )

        final_qc_started = diagnostics_timer_start()
        final_img_qc_clean = apply_warp_field_3d(mov_shifted_3d, flow_3d).max(axis=0)
        rec_corr = simple_correlation(context.ref_mip_clean, final_img_qc_clean)
        if self._registration_diagnostics is not None and context.round_id >= 0:
            self._registration_diagnostics.record_final_qc(
                context.round_id,
                elapsed_wall_ms=elapsed_ms_since(final_qc_started),
                final_corr=float(rec_corr),
            )
        local_backend_metadata = (
            dict(cast(Dict[str, Any], local_result['backend_metadata']))
            if local_result is not None
            else dict(local_backend_details)
        )
        local_backend_metadata['diagnostic_corr_after_global'] = float(context.corr_after_global)
        local_backend_metadata['diagnostic_corr_after_local'] = float(rec_corr)

        if rec_corr < context.corr_after_global:
            print(
                "  [Diagnostic] MATLAB local correlation decreased, but keeping local 3D flow; "
                "final_corr remains diagnostic-only "
                f"({rec_corr:.4f} < {context.corr_after_global:.4f})"
            )
            status = 'accepted_correlation_decrease'
        else:
            print(
                "  [Diagnostic] After MATLAB Local 3D: "
                f"Corr = {rec_corr:.4f} (Δ = {rec_corr - context.corr_after_global:+.4f})"
            )
            status = 'accepted'
        final_corr = float(rec_corr)
        final_img_qc = final_img_qc_clean

        backend_metadata = _attach_local_flow_metadata(
            context.backend_metadata,
            provider='matlab',
            local_method=context.local_method,
            status=status,
            corr_after_global=float(context.corr_after_global),
            corr_after_local=float(rec_corr),
            reject_if_worse=context.reject_if_worse,
            default_mode='experimental_local_kernel_swap',
        )
        if backend_metadata is not None:
            backend_metadata['local_flow'].update(local_backend_metadata)
            if tiling_summary is not None:
                backend_metadata['local_flow']['tiling'] = tiling_summary

        return _LocalRegistrationOutcome(
            flow_2d=None,
            flow_3d=flow_3d,
            final_corr=final_corr,
            final_img_qc=final_img_qc,
            backend_metadata=backend_metadata,
        )

    def _register_round_orchestrated(
        self,
        *,
        fov_id: int,
        round_id: int,
        ref_round: int,
        ref_scope_3d: FloatArray,
        ref_mip_clean: FloatArray,
        mov_scope_3d: FloatArray,
        scope_descriptor: Dict[str, Any],
        strategy: Optional[_RoundRegistrationStrategy] = None,
    ) -> Tuple[Dict[str, Any], FloatArray, Optional[Dict[str, Any]]]:
        resolved_strategy = strategy or self._build_round_registration_strategy()

        global_shift_3d, global_corr, backend_metadata = self._run_global_registration(
            handler_name=resolved_strategy.global_handler_name,
            provider=resolved_strategy.global_provider,
            method=getattr(getattr(self.reg_cfg, 'global_stage', None), 'method', None),
            fov_id=fov_id,
            round_id=round_id,
            ref_round=ref_round,
            ref_scope_3d=ref_scope_3d,
            mov_scope_3d=mov_scope_3d,
            scope_descriptor=scope_descriptor,
        )
        post_global_started = diagnostics_timer_start()
        mov_mip_clean, shift_2d, mov_mip_shifted, corr_after_global = self._compute_post_global_registration_state(
            ref_mip_clean=ref_mip_clean,
            mov_scope_3d=mov_scope_3d,
            global_shift_3d=global_shift_3d,
        )
        if self._registration_diagnostics is not None and round_id >= 0:
            self._registration_diagnostics.record_post_global_qc(
                round_id,
                elapsed_wall_ms=elapsed_ms_since(post_global_started),
                corr_after_global=float(corr_after_global),
            )

        flow_2d = None
        flow_3d = None
        final_corr = float(corr_after_global)
        final_img_qc = mov_mip_shifted

        if resolved_strategy.enable_local:
            context = self._build_local_registration_context(
                fov_id=fov_id,
                round_id=int(round_id),
                ref_round=int(ref_round),
                ref_scope_3d=ref_scope_3d,
                ref_mip_clean=ref_mip_clean,
                mov_scope_3d=mov_scope_3d,
                mov_mip_clean=mov_mip_clean,
                mov_mip_shifted=mov_mip_shifted,
                shift_2d=shift_2d,
                global_shift_3d=global_shift_3d,
                corr_after_global=float(corr_after_global),
                scope_descriptor=scope_descriptor,
                backend_metadata=backend_metadata,
                local_provider=resolved_strategy.local_provider,
                local_method=resolved_strategy.local_method,
                local_handler_name=resolved_strategy.local_handler_name,
            )

            if corr_after_global < resolved_strategy.local_skip_if_global_corr_below:
                warnings.warn(f"Low correlation {corr_after_global:.3f}, skipping local.")
                local_skip_started = diagnostics_timer_start()
                backend_metadata = _attach_local_flow_metadata(
                    backend_metadata,
                    provider=resolved_strategy.local_provider,
                    local_method=resolved_strategy.local_method,
                    status='skipped_low_global_corr',
                    corr_after_global=float(corr_after_global),
                    reject_if_worse=context.reject_if_worse,
                    default_mode=cast(str, resolved_strategy.local_default_mode),
                )
                if self._registration_diagnostics is not None and round_id >= 0:
                    self._registration_diagnostics.record_local_registration(
                        round_id,
                        elapsed_wall_ms=elapsed_ms_since(local_skip_started),
                        provider=resolved_strategy.local_provider,
                        method=resolved_strategy.local_method,
                        status='skipped',
                        skip_reason='skipped_low_global_corr',
                        final_corr=float(corr_after_global),
                        backend_metadata=backend_metadata,
                    )
            else:
                local_handler_name = resolved_strategy.local_handler_name
                if local_handler_name is None:
                    raise ValueError(
                        "Round registration strategy enabled local registration without a local handler name"
                    )
                local_started = diagnostics_timer_start()
                local_result = self._dispatch_local_registration(context)
                local_elapsed_ms = elapsed_ms_since(local_started)
                flow_2d = local_result.flow_2d
                flow_3d = local_result.flow_3d
                final_corr = float(local_result.final_corr)
                final_img_qc = local_result.final_img_qc
                backend_metadata = local_result.backend_metadata
                if self._registration_diagnostics is not None and round_id >= 0:
                    local_status = 'completed'
                    local_flow = backend_metadata.get('local_flow') if isinstance(backend_metadata, Mapping) else None
                    if isinstance(local_flow, Mapping):
                        local_status = str(local_flow.get('status', local_status))
                    self._registration_diagnostics.record_local_registration(
                        round_id,
                        elapsed_wall_ms=local_elapsed_ms,
                        provider=resolved_strategy.local_provider,
                        method=resolved_strategy.local_method,
                        status=local_status,
                        final_corr=float(final_corr),
                        backend_metadata=backend_metadata,
                    )
        elif self._registration_diagnostics is not None and round_id >= 0:
            self._registration_diagnostics.record_local_registration(
                round_id,
                elapsed_wall_ms=0.0,
                provider=resolved_strategy.local_provider,
                method=resolved_strategy.local_method,
                status='skipped',
                skip_reason='local_registration_disabled',
                final_corr=float(final_corr),
                backend_metadata=backend_metadata,
            )

        return (
            self._build_round_transform(
                global_shift_3d=global_shift_3d,
                global_corr=global_corr,
                flow_2d=flow_2d,
                flow_3d=flow_3d,
                final_corr=final_corr,
            ),
            final_img_qc,
            backend_metadata,
        )

    def _register_round_native(
        self,
        *,
        ref_scope_3d: FloatArray,
        ref_mip_clean: FloatArray,
        mov_scope_3d: FloatArray,
    ) -> Tuple[Dict[str, Any], FloatArray, Optional[Dict[str, Any]]]:
        """Legacy native wrapper that delegates to the canonical round authority."""
        return self._register_round_orchestrated(
            fov_id=-1,
            round_id=-1,
            ref_round=int(self.reg_cfg.reference_round),
            ref_scope_3d=ref_scope_3d,
            ref_mip_clean=ref_mip_clean,
            mov_scope_3d=mov_scope_3d,
            scope_descriptor=_build_scope_descriptor(ref_scope_3d, 'full_fov'),
            strategy=self._build_round_registration_strategy(
                global_provider='native',
                local_provider='native',
            ),
        )

    def _register_round_matlab_extracted(
        self,
        *,
        fov_id: int,
        round_id: int,
        ref_round: int,
        ref_scope_3d: FloatArray,
        ref_mip_clean: FloatArray,
        mov_scope_3d: FloatArray,
        scope_descriptor: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], FloatArray, Optional[Dict[str, Any]]]:
        """Legacy MATLAB wrapper that delegates to the canonical round authority."""
        return self._register_round_orchestrated(
            fov_id=fov_id,
            round_id=round_id,
            ref_round=ref_round,
            ref_scope_3d=ref_scope_3d,
            ref_mip_clean=ref_mip_clean,
            mov_scope_3d=mov_scope_3d,
            scope_descriptor=scope_descriptor,
            strategy=self._build_round_registration_strategy(
                global_provider='matlab',
                local_provider='matlab',
            ),
        )

    def _register_round(
        self,
        *,
        fov_id: int,
        round_id: int,
        ref_round: int,
        ref_scope_3d: FloatArray,
        ref_mip_clean: FloatArray,
        mov_scope_3d: FloatArray,
        scope_descriptor: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], FloatArray, Optional[Dict[str, Any]]]:
        """Register one moving round against the reference round.

        ``ref_scope_3d`` and ``mov_scope_3d`` are already cropped to the declared
        registration scope. The returned transform dictionary always contains a
        rigid ``global_shift_3d``; local refinement may add either ``flow_2d`` for
        diagnostic 2-D methods or ``flow_3d`` for extraction-ready 3-D image
        warping. ``final_img_qc`` is a MIP for registration QC plots, not the data
        consumed by extraction.
        """
        strategy = self._build_round_registration_strategy()
        return self._register_round_orchestrated(
            fov_id=fov_id,
            round_id=round_id,
            ref_round=ref_round,
            ref_scope_3d=ref_scope_3d,
            ref_mip_clean=ref_mip_clean,
            mov_scope_3d=mov_scope_3d,
            scope_descriptor=scope_descriptor,
            strategy=strategy,
        )

    def register_fov(self, data: xr.DataArray, fov_id: int) -> Dict[int, Dict[str, Any]]:
        """Register all configured rounds for one FOV and reload the manifest.

        ``data`` supplies the round list and FOV context from ``ImageLoader``;
        registration itself reloads canonical clean images from disk so provider
        and preprocessing provenance stay tied to the output contract. The
        manifest returned by this method is already materialized enough for the
        next stage, but the large 3-D flow fields may remain sidecar descriptors
        until extraction requests them.
        """
        base_dir = Path(self.cfg.pipeline.output.directory)
        diagnostics = RegistrationPerformanceRecorder(
            fov_id=int(fov_id),
            providers=build_registration_provider_summary(self.cfg),
        )
        previous_diagnostics = self._registration_diagnostics
        self._registration_diagnostics = diagnostics
        registration_started = diagnostics_timer_start()

        ref_round = self.reg_cfg.reference_round
        all_rounds = sorted(data.coords["round"].values)
        run_started_at = _iso_utc_now()
        round_summary: Dict[int, Dict[str, str]] = {}
        backend_round_metadata: Dict[int, Dict[str, Any]] = {}

        print(f"\n{'='*60}")
        print(f"  [Registration] FOV {fov_id} | Reference: Round {ref_round}")
        print(f"{'='*60}")

        try:
            # --- Phase 1: Preprocess Everything ---
            ref_load_started = diagnostics_timer_start()
            ref_clean_3d = self._load_combined_clean_volume(fov_id, ref_round)
            diagnostics.record_fov_setup_timing(
                "reference_clean_volume_load",
                elapsed_ms_since(ref_load_started),
                details={
                    "round_id": int(ref_round),
                    "volume_shape_zyx": [int(value) for value in ref_clean_3d.shape],
                },
            )

            scope_started = diagnostics_timer_start()
            scope_descriptor = _build_scope_descriptor(ref_clean_3d, self.cfg.pipeline.scope_mode)
            diagnostics.record_fov_setup_timing(
                "scope_descriptor_build",
                elapsed_ms_since(scope_started),
                details={
                    "scope_mode": self.cfg.pipeline.scope_mode,
                    "scope_descriptor": scope_descriptor,
                },
            )

            ref_crop_started = diagnostics_timer_start()
            ref_scope_3d = _crop_volume_to_scope(ref_clean_3d, scope_descriptor)
            diagnostics.record_fov_setup_timing(
                "reference_scope_crop",
                elapsed_ms_since(ref_crop_started),
                details={"scope_shape_zyx": [int(value) for value in ref_scope_3d.shape]},
            )

            ref_mip_started = diagnostics_timer_start()
            ref_mip_clean = ref_scope_3d.max(axis=0)
            diagnostics.record_fov_setup_timing(
                "reference_mip_creation",
                elapsed_ms_since(ref_mip_started),
                details={"mip_shape_yx": [int(value) for value in ref_mip_clean.shape]},
            )
            print(f" [Registration] Scope: {_format_scope_descriptor(scope_descriptor)}")
            if (
                self.reg_cfg.local_provider == 'matlab'
                and self.reg_cfg.enable_local
                and self.reg_cfg.local_method != 'demons_3d'
            ):
                _ensure_supported_matlab_local_method(self.reg_cfg.local_method)

            transforms = {}

            # --- Phase 2: Register Round by Round ---
            for r_id in all_rounds:
                round_id_int = int(r_id)
                round_started_at = _iso_utc_now()
                diagnostics.start_round(
                    round_id_int,
                    is_reference_round=round_id_int == int(ref_round),
                )
                if r_id == ref_round:
                    transforms[r_id] = {
                        'global_shift_3d': np.array([0., 0., 0.]),
                        'global_corr': 1.0,
                        'flow_2d': None,
                        'flow_3d': None,
                        'final_corr': 1.0,
                        'round_id': round_id_int,
                        'is_reference_round': True,
                    }
                    round_summary[round_id_int] = {
                        'status': 'completed',
                        'start_time': round_started_at,
                        'end_time': _iso_utc_now(),
                    }
                    diagnostics.complete_round(round_id_int)
                    continue

                print(f"\n  >> Round {r_id}")

                mov_load_started = diagnostics_timer_start()
                mov_clean_3d = self._load_combined_clean_volume(fov_id, r_id)
                diagnostics.record_round_timing(
                    round_id_int,
                    "moving_clean_volume_load",
                    elapsed_ms_since(mov_load_started),
                    details={"volume_shape_zyx": [int(value) for value in mov_clean_3d.shape]},
                )

                mov_crop_started = diagnostics_timer_start()
                mov_scope_3d = _crop_volume_to_scope(mov_clean_3d, scope_descriptor)
                diagnostics.record_round_timing(
                    round_id_int,
                    "moving_scope_crop",
                    elapsed_ms_since(mov_crop_started),
                    details={"scope_shape_zyx": [int(value) for value in mov_scope_3d.shape]},
                )
                round_transform, final_img_qc, backend_metadata = self._register_round(
                    fov_id=fov_id,
                    round_id=round_id_int,
                    ref_round=int(ref_round),
                    ref_scope_3d=ref_scope_3d,
                    ref_mip_clean=ref_mip_clean,
                    mov_scope_3d=mov_scope_3d,
                    scope_descriptor=scope_descriptor,
                )

                round_transform['round_id'] = round_id_int
                if backend_metadata is not None:
                    round_transform['backend_metadata'] = backend_metadata
                self._spill_round_flow_3d(fov_id, round_id_int, round_transform)
                transforms[r_id] = round_transform
                round_summary[round_id_int] = {
                    'status': 'completed',
                    'start_time': round_started_at,
                    'end_time': _iso_utc_now(),
                }
                diagnostics.complete_round(round_id_int)
                if backend_metadata is not None:
                    backend_round_metadata[round_id_int] = backend_metadata

                if self.cfg.pipeline.qc_images_enabled():
                    paths = get_fov_output_structure(base_dir, fov_id)
                    qc_dir = paths["qc"]
                    qc_started = diagnostics_timer_start()
                    mov_mip_clean = mov_scope_3d.max(axis=0)
                    save_registration_qc(
                        ref_mip_clean, mov_mip_clean, final_img_qc,
                        r_id, simple_correlation(ref_mip_clean, mov_mip_clean), # Raw initial corr
                        float(round_transform['final_corr']), qc_dir, fov_id
                    )
                    diagnostics.record_qc_image_save(round_id_int, elapsed_wall_ms=elapsed_ms_since(qc_started))

            self._annotate_transform_semantics(transforms)
            self._annotate_transform_scope(transforms, scope_descriptor)
            run_ended_at = _iso_utc_now()
            provenance_started = diagnostics_timer_start()
            provenance = self._build_provenance(
                transforms,
                round_summary,
                run_started_at,
                run_ended_at,
                backend_round_metadata=backend_round_metadata or None,
                registration_backend_details=(
                    self._get_matlab_backend().build_provenance_trace(backend_round_metadata)
                    if self.reg_cfg.global_provider == 'matlab' or self.reg_cfg.local_provider == 'matlab'
                    else None
                ),
            )
            diagnostics.record_manifest_timing("provenance_build", elapsed_ms_since(provenance_started))

            save_started = diagnostics_timer_start()
            self._save_transforms(transforms, fov_id, provenance=provenance)
            diagnostics.record_manifest_timing(
                "save_transform_manifest",
                elapsed_ms_since(save_started),
                details={"base_dir": str(base_dir), "fov_id": int(fov_id)},
            )

            load_started = diagnostics_timer_start()
            materialized_manifest = load_transform_manifest(base_dir, fov_id, load_provenance=False)
            diagnostics.record_manifest_timing(
                "load_transform_manifest",
                elapsed_ms_since(load_started),
                details={"load_provenance": False, "hydrate_flow_3d": True},
            )

            source_stage_elapsed_wall_ms = elapsed_ms_since(registration_started)
            diagnostics_path = diagnostics.write(
                base_dir,
                source_stage_elapsed_wall_ms=source_stage_elapsed_wall_ms,
            )
            print(f" [Registration] Performance diagnostics saved: {diagnostics_path}")
            return materialized_manifest
        finally:
            self._registration_diagnostics = previous_diagnostics
