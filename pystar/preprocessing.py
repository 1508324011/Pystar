# pystar/preprocessing.py
import numpy as np
import tifffile
from pathlib import Path
from tqdm import tqdm
import cv2
from skimage import exposure, morphology, img_as_float32, img_as_ubyte
from skimage.transform import resize
from typing import Dict, Any, List, Optional, Tuple
from .infrastructure import ExperimentConfig
from .io import ImageLoader
from .io import get_fov_output_structure

# ==============================================================================
# 1. THE ATOMS
# 所有的输入 img 保证是 Float32 [0.0, 1.0]
# 所有的输出 img 保证是 Float32 [0.0, 1.0]
# ==============================================================================

def op_median_filter(img: np.ndarray, params: Dict, ctx: Any) -> np.ndarray:
    """
    中值滤波。
    OpenCV 的 medianBlur 在某些版本不支持 float32，
    所以这里有一个肮脏但在生产环境必要的类型转换舞步。
    """
    k = params.get('kernel_size', 3)
    # OpenCV 要求 kernel size 必须是大于1的奇数
    if k % 2 == 0: k += 1
    if k < 3: return img

    # Flight check: input is float32 0-1
    # 暂时转回 uint8 域做滤波 (OpenCV 针对 int 优化极好)
    img_u8 = (img * 255).astype(np.uint8)

    if img_u8.ndim == 3:
        # 3D stack: 逐层处理
        res_u8 = np.stack([cv2.medianBlur(s, k) for s in img_u8])
    else:
        res_u8 = cv2.medianBlur(img_u8, k)

    # 转回 float32
    return res_u8.astype(np.float32) / 255.0

def op_gaussian_blur(img: np.ndarray, params: Dict, ctx: Any) -> np.ndarray:
    """
    高斯模糊。
    OpenCV 的 GaussianBlur 是最快的实现。
    Input/Output: Float32 [0.0, 1.0]
    """
    # 获取 sigma，默认 1.0
    sigma = params.get('sigma', 1.0)
    
    # ksize=(0, 0) 告诉 OpenCV 根据 sigma 自动计算卷积核大小
    # 这是最安全的做法
    
    if img.ndim == 3:
        # 3D Stack 必须切片处理，OpenCV 不支持 3D 卷积
        return np.stack([cv2.GaussianBlur(s, (0, 0), sigmaX=sigma, sigmaY=sigma) for s in img])
    else:
        # 2D 图像
        return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)

def op_histogram_match(img: np.ndarray, params: Dict, ctx: Dict) -> np.ndarray:
    """
    直方图匹配。
    依赖 Engine 在 ctx 中注入正确的 'ref_image'。
    """
    scope = params.get('scope', 'none')
    ref_img = None

    # 从上下文中获取参考图
    if scope == 'inter_round':
        ref_img = ctx.get('ref_round_image')
    elif scope == 'intra_round':
        ref_img = ctx.get('ref_channel_image')
    
    if ref_img is None:
        # 如果没有参考图 (比如这是 R1 自身，或者配置写错了)，
        # 什么都不做，原样返回。不要抛错，因为第一张图本来就没有参考对象。
        return img
    
    # skimage 的 match_histograms 支持 float 输入
    matched = exposure.match_histograms(img, ref_img)
    return matched.astype(np.float32)

def op_gamma_correction(img: np.ndarray, params: Dict, ctx: Any) -> np.ndarray:
    """
    非线性亮度调整。
    Gamma < 1.0 提亮暗部 (常用 0.5 - 0.7)。
    Gamma > 1.0 压暗暗部。
    """
    gamma = params.get('gamma', 1.0)
    if gamma == 1.0: return img
    
    # 假设输入已经是 float32 [0, 1]，直接幂运算
    # 为了防止负值导致 NaN (虽然理论上不该有负值)，加个绝对值或 clip
    safe_img = np.maximum(img, 0)
    return np.power(safe_img, gamma)

def op_difference_of_gaussians(img: np.ndarray, params: Dict, ctx: Any) -> np.ndarray:
    """
    DoG 滤波器：带通滤波，增强特定尺寸的斑点。
    Img_DoG = Gaussian(Small_Sigma) - Gaussian(Large_Sigma)
    """
    # 模拟 RNA 点的大小 (像素)
    spot_sigma = params.get('spot_sigma', 1.0) 
    # 模拟背景的大小 (通常是点的 3-5 倍)
    bg_sigma = params.get('bg_sigma', 5.0)
    
    # 复用 op_gaussian_blur 的逻辑 (OpenCV 实现)
    
    def _blur_slice(s, sig):
        return cv2.GaussianBlur(s, (0, 0), sigmaX=sig, sigmaY=sig)
        
    if img.ndim == 3:
        g_small = np.stack([_blur_slice(s, spot_sigma) for s in img])
        g_large = np.stack([_blur_slice(s, bg_sigma) for s in img])
    else:
        g_small = _blur_slice(img, spot_sigma)
        g_large = _blur_slice(img, bg_sigma)
        
    # DoG 结果可能为负 (原来的背景区域)，这里我们将负值截断为 0
    # 因为在荧光图像中，负信号没有物理意义
    diff = g_small - g_large
    return np.maximum(diff, 0)

def op_clip_percentile(img: np.ndarray, params: Dict, ctx: Any) -> np.ndarray:
    """
    鲁棒截断：忽略极值点。
    """
    min_p = params.get('min_percentile', 0.1) # 底部 0.1% 视为 0 (去底噪)
    max_p = params.get('max_percentile', 99.9) # 顶部 0.1% 视为 1 (去热点)
    
    # 计算分位点
    # 注意：对 3D 图像基于 Volume 全局计算更稳定，不容易造成层间闪烁。
    vmin, vmax = np.percentile(img, (min_p, max_p))
    
    # 截断
    return np.clip(img, vmin, vmax)

def op_clahe(img: np.ndarray, params: Dict, ctx: Any) -> np.ndarray:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization)
    """
    clip = params.get('clip_limit', 0.01)
    nbins = params.get('nbins', 256)
    # equalize_adapthist 完美支持 float，且输出也是 float
    return exposure.equalize_adapthist(img, clip_limit=clip, nbins=nbins).astype(np.float32)

def op_morpho_reconstruction_contrast(img: np.ndarray, params: Dict, ctx: Any) -> np.ndarray:
    """
    复杂的背景扣除逻辑：Morphological Reconstruction + TopHat。
    
    Logic:
    1. Marker = Erode(Img)
    2. Background = Reconstruction(Marker, Mask=Img)只在下采样空间计算
    3. W-TopHat-Rec = Img - Background
    4. Enhanced = W-TopHat-Rec + WhiteTopHat(W-TopHat-Rec) - BlackTopHat(W-TopHat-Rec)
    """
    rad = params.get('radius', 10)
    downsample = params.get('downsample_factor', 0.25)
    
    rad_small = max(1, int(rad * downsample))
    selem_full = morphology.disk(rad)     # 大图用的核
    selem_small = morphology.disk(rad_small) # 小图用的核

    def _process_slice_safe(slice_2d):
        h, w = slice_2d.shape
        
        # --- Step A: 快速估算背景 (The Slow Part Optimization) ---
        small_h, small_w = int(h * downsample), int(w * downsample)
        slice_small = resize(slice_2d, (small_h, small_w), order=1, preserve_range=True)
        
        # 在小图上做侵蚀和重建
        marker_s = morphology.erosion(slice_small, selem_small)
        bg_rec_s = morphology.reconstruction(marker_s, slice_small, method='dilation')
        
        # 放大背景
        bg_full = resize(bg_rec_s, (h, w), order=1, preserve_range=True)
        
        # --- Step B: 全分辨率去背景 ---
        # 这一步是快加减法，没压力
        diff = slice_2d - bg_full
        
        # --- Step C: 全分辨率增强 (The Detail Part) ---
        # White/Black Tophat 在 OpenCV/Skimage 里通常优化得不错，比 Reconstruction 快
        # 为了保留 1-2px 的细节，这步还得在原图跑。
        
        # 如果觉得这步还是慢，可以单独给这步开 0.5 的 downsample，而不是 0.25
        w_th = morphology.white_tophat(diff, selem_full)
        b_th = morphology.black_tophat(diff, selem_full)
        
        final = diff + w_th - b_th
        return final

    if img.ndim == 3:
        res = np.stack([_process_slice_safe(s) for s in img])
    else:
        res = _process_slice_safe(img)
        
    return np.clip(res, 0, 1).astype(np.float32)


def op_min_max_normalize(img: np.ndarray, params: Dict, ctx: Any) -> np.ndarray:
    """
    线性拉伸，确保数据占满 [0, 1] 区间。
    """
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-9: # 避免除以 0
        return np.zeros_like(img)
    return (img - mn) / (mx - mn)

# ==============================================================================
# 2. THE REGISTRY (映射表)
# ==============================================================================
PROCESSOR_MAP = {
    "median_filter": op_median_filter,
    "gaussian_blur": op_gaussian_blur, 
    "histogram_match": op_histogram_match,
    "gamma_correction": op_gamma_correction,
    "difference_of_gaussians": op_difference_of_gaussians,
    "clip_percentile": op_clip_percentile,
    "clahe": op_clahe,
    "morpho_reconstruction_contrast": op_morpho_reconstruction_contrast,
    "min_max_normalize": op_min_max_normalize,
    "none": lambda img, p, c: img # Null Object Pattern
}

# ==============================================================================
# 3. THE ENGINE
# ==============================================================================

class DataSanitizer:
    def __init__(self, config):
        self.cfg = config
        self.loader = ImageLoader(config)

    def _split_sequence(self, full_seq: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Phase A: 保持图像特征的步骤 (Denoise, Match) -> 输出用于做 Reference
        Phase B: 改变图像特征/去背景的步骤 (Morpho, Normalize) -> 输出用于存储
        
        策略: 找到第一个名字里带 'morpho' 或 'background' 的步骤，从那里切开。
        """
        split_idx = len(full_seq) # 默认不切分，全在 Phase A
        
        for i, step in enumerate(full_seq):
            name = step.method.lower()
            if 'morpho' in name or 'background' in name:
                split_idx = i
                break
        
        phase_a = full_seq[:split_idx]
        phase_b = full_seq[split_idx:]
        return phase_a, phase_b

    def _run_pipeline(self, img_vol: np.ndarray, pipeline_seq: List[Dict], context: Dict) -> np.ndarray:
        """执行一段流水线"""
        # 1. 确保 Float32
        if img_vol.dtype != np.float32:
            # 假设输入是 uint8/16，归一化到 0-1
            max_val = 255.0 if img_vol.dtype == np.uint8 else 65535.0
            # 简单的防御性检查，有些 TIFF 读进来已经是 float 但数值很大
            if np.issubdtype(img_vol.dtype, np.floating) and img_vol.max() > 1.0:
                current_data = img_vol
            else:
                current_data = img_vol.astype(np.float32) / max_val
        else:
            current_data = img_vol

        # 2. 执行
        for step in pipeline_seq:
            func = PROCESSOR_MAP.get(step.method)
            if func:
                current_data = func(current_data, step.params, context)
            # else: warning handled in upper logic or crash
            
        return current_data

    def sanitize_fov(self, fov_id: int, target_rounds: Optional[List[int]] = None):
        """
        target_rounds: List[int], optional
            如果提供，只处理列表中的轮次 (e.g. [1])。用于快速测试参数。
        """
        print(f"[{'='*20} Sanitizing FOV {fov_id} {'='*20}]")
        
        full_seq = self.cfg.pipeline.preprocessing.sequence
        if not full_seq:
            print("Warning: Pipeline sequence is empty.")
            return

        seq_calibration, seq_extraction = self._split_sequence(full_seq)
        
        # 1. 确定要处理哪些轮次
        all_config_rounds = sorted(self.cfg.dataset.round_structure.keys())
        
        if target_rounds is not None:
            # 取交集，防止用户输入不存在的轮次
            rounds_to_process = sorted([r for r in target_rounds if r in all_config_rounds])
            if not rounds_to_process:
                print(f"Warning: No valid rounds found in target_rounds: {target_rounds}")
                return
            print(f" -> DEBUG: Only processing user-selected rounds: {rounds_to_process}")
        else:
            rounds_to_process = all_config_rounds

        # 2. 调度逻辑：确保 Reference Round (R1) 总是排在队伍最前面
        # 只有当 R1 真的在我们要处理的列表里时，才应用这个排序
        ref_round_id = 1 # 或者从 config 读
        
        final_queue = []
        
        # 如果 R1 在任务清单里，先加它
        if ref_round_id in rounds_to_process:
            final_queue.append(ref_round_id)
            
        # 再加其他的
        for r_id in rounds_to_process:
            if r_id != ref_round_id:
                final_queue.append(r_id)

        # 3. 开始执行
        print(f" -> Pipeline Split: {len(seq_calibration)} Calibration steps + {len(seq_extraction)} Extraction steps")
        
        # 全局缓存 (Inter-round Ref)
        inter_round_ref_cache = {}

        for r_id in final_queue:
            print(f" -> Processing Round {r_id}...")
            
            # 局部缓存 (Intra-round Ref - Reset for each round)
            intra_round_ref_img = None
            
            # 获取该轮的通道
            roles = self.cfg.dataset.channel_roles
            channels_in_round = self.cfg.dataset.round_structure[r_id]
            # 只处理 seq 通道
            seq_channels = sorted([c for c in channels_in_round if roles.get(c) == 'seq'])
            
            for c_id in seq_channels:
                # 1. Load
                path = self.loader._get_path(fov_id, r_id, c_id)
                raw_vol = self.loader._lazy_load_tiff(path).compute()
                
                # 2. Context
                ctx = {
                    'ref_round_image': inter_round_ref_cache.get(c_id),
                    'ref_channel_image': intra_round_ref_img
                }

                # === Phase A: Calibration ===
                img_calibrated = self._run_pipeline(raw_vol, seq_calibration, ctx)
                
                # Update Caches
                if r_id == ref_round_id:
                    inter_round_ref_cache[c_id] = img_calibrated.copy()
                
                if intra_round_ref_img is None:
                    intra_round_ref_img = img_calibrated.copy()

                # === Phase B: Extraction ===
                final_vol = self._run_pipeline(img_calibrated, seq_extraction, ctx)

                # 3. Save
                final_u8 = img_as_ubyte(np.clip(final_vol, 0, 1))
                self._save_clean(final_u8, fov_id, r_id, c_id)
    def _save_clean(self, img, f, r, c):
        base_dir = Path(self.cfg.pipeline.output.directory)
        paths = get_fov_output_structure(base_dir, f)
        out_path = paths['cleaned'] / f"clean_fov_{f}_round_{r}_ch_{c}.tif"
        tifffile.imwrite(out_path, img, compression='zlib')

