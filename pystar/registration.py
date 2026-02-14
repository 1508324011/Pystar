# pystar/registration.py

import numpy as np
import xarray as xr
import tifffile
import SimpleITK as sitk
from typing import Tuple, Optional, Dict, Any, List
from pathlib import Path
from scipy.ndimage import shift as scipy_shift
from scipy.ndimage import map_coordinates
from skimage.registration import phase_cross_correlation, optical_flow_tvl1
from skimage.transform import warp, resize
from skimage.filters import gaussian
import warnings

from .infrastructure import ExperimentConfig
from .io import ImageLoader
from .io import get_fov_output_structure
from .visualization import save_registration_qc

# ==============================================================================
# SECTION A: Data Extraction Layer
# ==============================================================================

def compute_overlap_roi(shape: Tuple[int, int], shift_2d: np.ndarray) -> np.ndarray:
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
    ref_3d_clean: np.ndarray,
    mov_3d_clean: np.ndarray,
    downsample_factor: int = 1,
    max_shift: int = 200
) -> Tuple[np.ndarray, float]:
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
        normalization=None 
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
        
    return real_shift, correlation

def apply_rigid_shift_3d(img_3d: np.ndarray, shift_3d: np.ndarray) -> np.ndarray:
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
    return scipy_shift(img_3d, shift_3d, order=1, mode='constant', cval=0.0)

# ==============================================================================
# SECTION C: Local Registration (2D Optical Flow with Quality Masking)
# ==============================================================================

def create_quality_mask(
    img_2d: np.ndarray,
    valid_roi_mask: Optional[np.ndarray] = None,
    edge_margin: int = 50,
    threshold: float = 1e-5 
) -> np.ndarray:
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
    ref_2d: np.ndarray,
    mov_2d: np.ndarray,
    mask: Optional[np.ndarray],
    config_obj: Any
) -> Optional[np.ndarray]:
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
    scale_factor = getattr(config_obj, 'coarse_scale', 0.25)
    
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
            attachment=getattr(config_obj, 'attachment', 15.0), # 强力贴合
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
        correction_factor = 1.0 / scale_factor
        
        # Resize Y component
        flow_large[0] = resize(flow_small[0], (h, w), order=1) * correction_factor
        # Resize X component
        flow_large[1] = resize(flow_small[1], (h, w), order=1) * correction_factor
        
        # ---  Final Polish ---
        # 放大插值可能会带来网格效应，最后做一次平滑
        flow_large[0] = gaussian(flow_large[0], sigma=3.0)
        flow_large[1] = gaussian(flow_large[1], sigma=3.0)
        
        return flow_large

    except Exception as e:
        warnings.warn(f"Optical flow crashed: {e}")
        return None

def register_local_bspline(
    ref_2d: np.ndarray, 
    mov_2d: np.ndarray, 
    mask: Optional[np.ndarray],
    config_obj: Any
) -> Optional[np.ndarray]:
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
    ref_3d: np.ndarray,
    mov_3d: np.ndarray,
    config_obj: Any
) -> Optional[np.ndarray]:
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

def apply_warp_field_2d(img_2d: np.ndarray, flow: np.ndarray) -> np.ndarray:
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
    img_2d: np.ndarray,
    shift_2d: np.ndarray,
    flow: Optional[np.ndarray]
) -> np.ndarray:
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
    shifted = scipy_shift(img_2d, shift_2d, order=1, mode='constant', cval=0.0)

    # Step 2: 光流变形
    if flow is not None:
        return apply_warp_field_2d(shifted, flow)

    return shifted

def apply_warp_field_3d(img_3d: np.ndarray, flow_3d: np.ndarray) -> np.ndarray:
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
    
    # 创建网格坐标
    z_coords, y_coords, x_coords = np.meshgrid(
        np.arange(nz), np.arange(ny), np.arange(nx), indexing='ij'
    )
    
    # 应用位移（逆向映射）
    new_z = z_coords + flow_3d[0]
    new_y = y_coords + flow_3d[1]
    new_x = x_coords + flow_3d[2]
    
    # 插值
    warped = map_coordinates(
        img_3d,
        np.array([new_z, new_y, new_x]),
        order=1,
        mode='constant',
        cval=0,
        prefilter=False
    )
    
    return warped.astype(img_3d.dtype)

# ==============================================================================
# SECTION D: Quality Metrics
# ==============================================================================

def simple_correlation(img1: np.ndarray, img2: np.ndarray) -> float:
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

# ==============================================================================
# SECTION E: Registration Engine (Orchestration)
# ==============================================================================

class RegistrationEngine:
    def __init__(self, config: ExperimentConfig):
        self.cfg = config
        self.reg_cfg = config.pipeline.registration

    def _save_transforms(self, transforms: Dict, fov_id: int):
        base_dir = Path(self.cfg.pipeline.output.directory)
        paths = get_fov_output_structure(base_dir, fov_id)
        
        out_file = paths["transforms"] / f"transforms_fov_{fov_id}.npy"
        np.save(out_file, transforms)
        
    def _load_combined_clean_volume(self, fov_id: int, round_id: int) -> np.ndarray:
        """
        加载该轮次所有 Sequence 通道的 Clean Data，
        并计算 Channel-wise Max Projection，生成一个 3D 体积用于配准。
        
        Returns:
            volume_3d: (Z, Y, X) float32
        """
        # 1. 实例化 Loader (如果类初始化时没存 loader，这里实例化一个)
        loader = ImageLoader(self.cfg)
        
        # 2. 找出该轮次的 Seq 通道
        # 我们的逻辑是：用所有经过清洗的测序信号来做配准，这样信号最丰富
        round_structure = self.cfg.dataset.round_structure.get(int(round_id))
        if not round_structure:
            # 兼容 key 是 str 的情况
            round_structure = self.cfg.dataset.round_structure.get(str(round_id))
            
        roles = self.cfg.dataset.channel_roles
        
        # 过滤：既在该轮次存在，又是 seq 类型的通道
        target_channels = [c for c in round_structure if roles.get(c) == 'seq']
        
        if not target_channels:
            raise ValueError(f"Round {round_id} has no SEQ channels for registration!")

        # 3. 加载并合成
        # print(f"   (Loading Clean Data for Round {round_id}: {target_channels})")
        vol_list = []
        for c in target_channels:
            # 这是一个 IO 操作，直接读 clean_data 目录
            vol = loader.load_clean_image(fov_id, round_id, c)
            vol_list.append(vol)
            
        # Stack -> (C, Z, Y, X)
        stack = np.stack(vol_list, axis=0)
        
        # Max Project along Channel Axis -> (Z, Y, X)
        # 这就是这一轮的"总信号图"
        combined_vol = np.max(stack, axis=0)
        
        return combined_vol

    def register_fov(self, data: xr.DataArray, fov_id: int) -> Dict:
        ref_round = self.reg_cfg.reference_round
        all_rounds = sorted(data.coords["round"].values)

        print(f"\n{'='*60}")
        print(f"  [Registration] FOV {fov_id} | Reference: Round {ref_round}")
        print(f"{'='*60}")

        # --- Phase 1: Preprocess Everything ---
        ref_clean_3d = self._load_combined_clean_volume(fov_id, ref_round)
        
        ref_mip_clean = ref_clean_3d.max(axis=0)
        
        transforms = {}

        # --- Phase 2: Register Round by Round ---
        for r_id in all_rounds:
            if r_id == ref_round:
                transforms[r_id] = {
                    'global_shift_3d': np.array([0., 0., 0.]),
                    'global_corr': 1.0, 'flow_2d': None, 'final_corr': 1.0
                }
                continue

            print(f"\n  >> Round {r_id}")
            
            mov_clean_3d = self._load_combined_clean_volume(fov_id, r_id)
            
            # ========== Step A: Global 3D (Clean vs Clean) ==========
            # 注意：参数中不再包含 preprocess 的选项，因为输入已经是 Clean 的了
            global_shift_3d, global_corr = compute_global_shift_3d(
                ref_clean_3d, mov_clean_3d,
                downsample_factor=self.reg_cfg.downsample_factor, # e.g. 4
                max_shift=self.reg_cfg.global_max_shift
            )
            print(f"     Global Shift (3D): {global_shift_3d}, Corr (Est): {global_corr:.4f}")

            # ========== Step B: Prep for Local (Clean MIPs) ==========
            mov_mip_clean = mov_clean_3d.max(axis=0)
            
            # Apply Global Shift to Moving MIP
            shift_2d = global_shift_3d[1:] # [dy, dx]
            mov_mip_shifted = scipy_shift(mov_mip_clean, shift_2d, order=1)

            # Check Correlation
            corr_after_global = simple_correlation(ref_mip_clean, mov_mip_shifted)
            print(f"     After Global (Clean MIP): Corr = {corr_after_global:.4f}")

            # ========== Step C: Local Flow (MIP vs MIP) ==========
            flow_2d = None
            final_corr = corr_after_global
            final_img_qc = mov_mip_shifted

            if self.reg_cfg.enable_local:
                # 1. Masking
                overlap_mask = compute_overlap_roi(ref_mip_clean.shape, shift_2d)
                mask = create_quality_mask(ref_mip_clean, valid_roi_mask=overlap_mask)

                if corr_after_global < 0.2:
                    warnings.warn(f"Low correlation {corr_after_global:.3f}, skipping local.")
                
                elif self.reg_cfg.local_method == "demons_3d":
                    # 使用 3D Demons（推荐）
                    # 注意：这里需要用 3D volume，不是 MIP
                    mov_shifted_3d = apply_rigid_shift_3d(mov_clean_3d, global_shift_3d)
                    
                    flow_3d = register_local_demons_3d(
                        ref_clean_3d, 
                        mov_shifted_3d, 
                        self.cfg.pipeline.registration.demons_3d
                    )
                    
                    if flow_3d is not None:
                        # 验证：对 MIP 做投影检查相关性
                        final_img_qc_clean = apply_warp_field_3d(mov_shifted_3d, flow_3d).max(axis=0)
                        rec_corr = simple_correlation(ref_mip_clean, final_img_qc_clean)
                        
                        if rec_corr < corr_after_global:
                            print(f"  !!! Reverting Local (Worse: {rec_corr:.4f} < {corr_after_global:.4f})")
                            flow_3d = None
                        else:
                            print(f"  After Local 3D: Corr = {rec_corr:.4f} (Δ = {rec_corr - corr_after_global:+.4f})")
                            final_corr = rec_corr
                            final_img_qc = final_img_qc_clean
                elif self.reg_cfg.local_method == "optical_flow":
                    flow_2d = compute_optical_flow_masked(
                        ref_mip_clean, mov_mip_shifted, mask,
                        self.cfg.pipeline.registration.optical_flow
                    )
                elif self.reg_cfg.local_method == "bspline":
                    flow_2d = register_local_bspline(
                        ref_mip_clean, mov_mip_shifted, mask,
                        self.cfg.pipeline.registration.bspline
                    )

                # 2. Validation
                if flow_2d is not None:
                    final_img_qc_clean = composite_transform_2d(mov_mip_clean, shift_2d, flow_2d)
                    rec_corr = simple_correlation(ref_mip_clean, final_img_qc_clean)
                    diff = rec_corr - corr_after_global
                    
                    if rec_corr < corr_after_global:
                        print(f"     !!! Reverting Local (Worse: {rec_corr:.4f} < {corr_after_global:.4f})")
                        flow_2d = None
                        final_corr = corr_after_global
                    else:
                        print(f"     After Local:  Corr = {rec_corr:.4f} (Δ = {diff:+.4f})")
                        final_corr = rec_corr
                        final_img_qc = final_img_qc_clean

            # ========== Step D: Save Results & QC ==========
            transforms[r_id] = {
                'global_shift_3d': global_shift_3d,
                'global_corr': global_corr,
                'flow_2d': flow_2d,
                'final_corr': final_corr
            }

            if self.cfg.pipeline.output.save_qc_images:
                base_dir = Path(self.cfg.pipeline.output.directory)
                paths = get_fov_output_structure(base_dir, fov_id)
                qc_dir = paths["qc"]
                save_registration_qc(
                    ref_mip_clean, mov_mip_clean, final_img_qc,
                    r_id, simple_correlation(ref_mip_clean, mov_mip_clean), # Raw initial corr
                    final_corr, qc_dir, fov_id
                )

        self._save_transforms(transforms, fov_id)
        return transforms