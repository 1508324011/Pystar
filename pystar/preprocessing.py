# pystar/preprocessing.py
import numpy as np
import tifffile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import shutil
import os
from datetime import datetime, timezone
from functools import lru_cache
import cv2
from skimage import exposure, morphology
from skimage.transform import resize
from skimage.util import img_as_ubyte
from typing import Any, Callable, Optional, cast
from numpy.typing import NDArray
from .infrastructure import ExperimentConfig, PreprocessingStep
from .io import ImageLoader
from .io import get_fov_output_structure
from .matlab_preprocessing import (
    PREPROCESSING_PROVENANCE_VERSION,
    MATLABPreprocessingBackend,
    write_preprocessing_provenance,
)
from .matlab_engine_bootstrap import MatlabSharedSessionOwner, summarize_matlab_boundary_traces

ImageArray = NDArray[Any]
ProcessorParams = dict[str, Any]
ProcessorContext = dict[str, Any]
ProcessorFunc = Callable[[ImageArray, ProcessorParams, ProcessorContext], ImageArray]
NativeOutputWriter = Callable[[ImageArray, int, int], Path]

NATIVE_PREPROCESSING_TIMING_SCHEMA_NAME = "pystar_native_preprocessing_timing"
NATIVE_PREPROCESSING_TIMING_SCHEMA_VERSION = 1

# ==============================================================================
# 1. THE ATOMS
# 所有的输入 img 保证是 Float32 [0.0, 1.0]
# 所有的输出 img 保证是 Float32 [0.0, 1.0]
# ==============================================================================

def op_median_filter(img: ImageArray, params: ProcessorParams, ctx: ProcessorContext) -> ImageArray:
    """
    中值滤波。
    OpenCV 的 medianBlur 在某些版本不支持 float32，
    所以这里有一个肮脏但在生产环境必要的类型转换。
    """
    k = params.get('kernel_size', 3)
    # OpenCV 要求 kernel size 必须是大于1的奇数
    if k % 2 == 0:
        k += 1
    if k < 3:
        return img

    # Flight check: input is float32 0-1
    # 暂时转回 uint8 域做滤波 (OpenCV 针对 int 优化极好)
    img_u8 = cast(ImageArray, (img * 255).astype(np.uint8))

    if img_u8.ndim == 3:
        # 3D stack: 逐层处理。避免 list + np.stack 产生第二个全体积临时对象。
        if img_u8.shape[0] == 0:
            _raise_empty_stack_like_np_stack()
        res_u8 = cast(ImageArray, np.empty_like(img_u8))
        for z_index in range(img_u8.shape[0]):
            _ = _median_blur_slice_into(
                cast(ImageArray, img_u8[z_index]),
                k,
                cast(ImageArray, res_u8[z_index]),
            )
    else:
        res_u8 = _median_blur_slice(img_u8, k)

    # 转回 float32
    return res_u8.astype(np.float32) / 255.0

def op_gaussian_blur(img: ImageArray, params: ProcessorParams, ctx: ProcessorContext) -> ImageArray:
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
        if img.shape[0] == 0:
            _raise_empty_stack_like_np_stack()
        first_slice = _gaussian_blur_slice(cast(ImageArray, img[0]), sigma)
        res = np.empty((img.shape[0], *first_slice.shape), dtype=first_slice.dtype)
        res[0] = first_slice
        for z_index in range(1, img.shape[0]):
            _ = _gaussian_blur_slice_into(
                cast(ImageArray, img[z_index]),
                sigma,
                cast(ImageArray, res[z_index]),
            )
        return cast(ImageArray, res)
    else:
        # 2D 图像
        return _gaussian_blur_slice(img, sigma)

def op_histogram_match(img: ImageArray, params: ProcessorParams, ctx: ProcessorContext) -> ImageArray:
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
    return matched.astype(np.float32, copy=False)

def op_gamma_correction(img: ImageArray, params: ProcessorParams, ctx: ProcessorContext) -> ImageArray:
    """
    非线性亮度调整。
    Gamma < 1.0 提亮暗部 (常用 0.5 - 0.7)。
    Gamma > 1.0 压暗暗部。
    """
    gamma = params.get('gamma', 1.0)
    if gamma == 1.0:
        return img
    
    # 假设输入已经是 float32 [0, 1]，直接幂运算
    # 为了防止负值导致 NaN (虽然理论上不该有负值)，加个绝对值或 clip
    safe_img = np.maximum(img, 0)
    return np.power(safe_img, gamma)

def op_difference_of_gaussians(img: ImageArray, params: ProcessorParams, ctx: ProcessorContext) -> ImageArray:
    """
    DoG 滤波器：带通滤波，增强特定尺寸的斑点。
    Img_DoG = Gaussian(Small_Sigma) - Gaussian(Large_Sigma)
    """
    # 模拟 RNA 点的大小 (像素)
    spot_sigma = params.get('spot_sigma', 1.0) 
    # 模拟背景的大小 (通常是点的 3-5 倍)
    bg_sigma = params.get('bg_sigma', 5.0)
    
    if img.ndim == 3:
        if img.shape[0] == 0:
            _raise_empty_stack_like_np_stack()
        first_small = _gaussian_blur_slice(cast(ImageArray, img[0]), spot_sigma)
        res = np.empty((img.shape[0], *first_small.shape), dtype=first_small.dtype)
        res[0] = first_small
        scratch = np.empty_like(first_small)
        _ = _gaussian_blur_slice_into(
            cast(ImageArray, img[0]),
            bg_sigma,
            cast(ImageArray, scratch),
        )
        _subtract_and_clip_nonnegative(cast(ImageArray, res[0]), cast(ImageArray, scratch))
        for z_index in range(1, img.shape[0]):
            _ = _gaussian_blur_slice_into(
                cast(ImageArray, img[z_index]),
                spot_sigma,
                cast(ImageArray, res[z_index]),
            )
            _ = _gaussian_blur_slice_into(
                cast(ImageArray, img[z_index]),
                bg_sigma,
                cast(ImageArray, scratch),
            )
            _subtract_and_clip_nonnegative(cast(ImageArray, res[z_index]), cast(ImageArray, scratch))
        return cast(ImageArray, res)
    else:
        g_small = _gaussian_blur_slice(img, spot_sigma)
        g_large = _gaussian_blur_slice(img, bg_sigma)
        
    # DoG 结果可能为负 (原来的背景区域)，这里我们将负值截断为 0
    # 因为在荧光图像中，负信号没有物理意义
    diff = g_small - g_large
    return np.maximum(diff, 0)


def _raise_empty_stack_like_np_stack() -> None:
    raise ValueError("need at least one array to stack")


def _validate_cv2_dst_result(result: Any, dst: ImageArray, *, operation: str) -> ImageArray:
    result_array = np.asarray(result)
    if result_array.shape != dst.shape or result_array.dtype != dst.dtype:
        raise ValueError(
            f"{operation} OpenCV destination drifted; expected {dst.shape}/{dst.dtype}, got "
            f"{result_array.shape}/{result_array.dtype}"
        )
    if result_array is not dst and not np.may_share_memory(result_array, dst):
        dst[...] = result_array
    return dst


def _median_blur_slice(slice_u8: ImageArray, kernel_size: int) -> ImageArray:
    blurred = cv2.medianBlur(cast(Any, np.ascontiguousarray(slice_u8)), kernel_size)
    return cast(ImageArray, blurred)


def _median_blur_slice_into(slice_u8: ImageArray, kernel_size: int, dst: ImageArray) -> ImageArray:
    contiguous_slice = np.ascontiguousarray(slice_u8)
    result = cv2.medianBlur(cast(Any, contiguous_slice), kernel_size, dst=dst)
    return _validate_cv2_dst_result(result, dst, operation="medianBlur")


def _gaussian_blur_slice(slice_2d: ImageArray, sigma: float) -> ImageArray:
    blurred = cv2.GaussianBlur(slice_2d, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return cast(ImageArray, blurred)


def _gaussian_blur_slice_into(slice_2d: ImageArray, sigma: float, dst: ImageArray) -> ImageArray:
    result = cv2.GaussianBlur(slice_2d, (0, 0), sigmaX=sigma, sigmaY=sigma, dst=dst)
    return _validate_cv2_dst_result(result, dst, operation="GaussianBlur")


def _subtract_and_clip_nonnegative(value: ImageArray, baseline: ImageArray) -> None:
    _ = np.subtract(value, baseline, out=value)
    _ = np.maximum(value, 0, out=value)

def op_clip_percentile(img: ImageArray, params: ProcessorParams, ctx: ProcessorContext) -> ImageArray:
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

def op_clahe(img: ImageArray, params: ProcessorParams, ctx: ProcessorContext) -> ImageArray:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization)
    """
    clip = params.get('clip_limit', 0.01)
    nbins = params.get('nbins', 256)
    # equalize_adapthist 完美支持 float，且输出也是 float
    return exposure.equalize_adapthist(img, clip_limit=clip, nbins=nbins).astype(np.float32)


@lru_cache(maxsize=32)
def _morphology_disk(radius: int | float) -> ImageArray:
    return cast(ImageArray, morphology.disk(radius))


def _morpho_reconstruction_contrast_slice(
    slice_2d: ImageArray,
    *,
    small_shape: tuple[int, int],
    selem_full: ImageArray,
    selem_small: ImageArray,
    full_resolution_scratch: tuple[ImageArray, ImageArray] | None = None,
) -> ImageArray:
    h, w = slice_2d.shape

    # --- Step A: 快速估算背景 (The Slow Part Optimization) ---
    slice_small = resize(slice_2d, small_shape, order=1, preserve_range=True)

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
    if full_resolution_scratch is None:
        w_th = np.empty_like(diff)
        b_th = np.empty_like(diff)
    else:
        w_th, b_th = full_resolution_scratch
        expected = (diff.shape, diff.dtype)
        if (w_th.shape, w_th.dtype) != expected or (b_th.shape, b_th.dtype) != expected:
            raise ValueError(
                "morpho_reconstruction_contrast scratch buffers must match the full-resolution "
                + f"diff shape/dtype {diff.shape}/{diff.dtype}; got "
                + f"{w_th.shape}/{w_th.dtype} and {b_th.shape}/{b_th.dtype}"
            )
        if np.may_share_memory(w_th, slice_2d) or np.may_share_memory(b_th, slice_2d):
            raise ValueError("morpho_reconstruction_contrast scratch buffers must not alias input slices")
        if np.may_share_memory(w_th, b_th):
            raise ValueError("morpho_reconstruction_contrast scratch buffers must not alias each other")

    morphology.white_tophat(diff, selem_full, out=w_th)
    morphology.black_tophat(diff, selem_full, out=b_th)

    diff += w_th
    diff -= b_th
    return cast(ImageArray, diff)


def _morpho_reconstruction_contrast_scratch(
    *,
    shape: tuple[int, int],
    dtype: np.dtype[Any],
) -> tuple[ImageArray, ImageArray]:
    """Allocate private full-resolution morphology scratch for one call.

    The buffers are intentionally tiny in ownership scope: callers create them
    for one 3-D volume and pass them slice-by-slice so the top-hat outputs stop
    allocating new full-resolution arrays for every Z plane. They are not shared
    across FOVs, volumes, threads, or calls.
    """

    return (
        cast(ImageArray, np.empty(shape, dtype=dtype)),
        cast(ImageArray, np.empty(shape, dtype=dtype)),
    )


def _morpho_reconstruction_contrast_workers(params: ProcessorParams) -> int:
    raw_workers = params.get("workers", 1)
    if isinstance(raw_workers, (bool, np.bool_)) or not isinstance(raw_workers, (int, np.integer)):
        raise ValueError(
            "morpho_reconstruction_contrast workers must be a positive integer; "
            + f"got {raw_workers!r}"
        )

    workers = int(raw_workers)
    if workers <= 0:
        raise ValueError(
            "morpho_reconstruction_contrast workers must be a positive integer; "
            + f"got {raw_workers!r}"
        )
    return workers


def _morpho_reconstruction_contrast_parallel_slice(
    z_index: int,
    slice_2d: ImageArray,
    *,
    small_shape: tuple[int, int],
    selem_full: ImageArray,
    selem_small: ImageArray,
    scratch_shape: tuple[int, int],
    scratch_dtype: np.dtype[Any],
) -> tuple[int, ImageArray]:
    scratch = _morpho_reconstruction_contrast_scratch(
        shape=scratch_shape,
        dtype=scratch_dtype,
    )
    result = _morpho_reconstruction_contrast_slice(
        slice_2d,
        small_shape=small_shape,
        selem_full=selem_full,
        selem_small=selem_small,
        full_resolution_scratch=scratch,
    )
    return z_index, result

def op_morpho_reconstruction_contrast(img: ImageArray, params: ProcessorParams, ctx: ProcessorContext) -> ImageArray:
    """
    复杂的背景扣除逻辑：Morphological Reconstruction + TopHat。
    
    Logic:
    1. Marker = Erode(Img)
    2. Background = Reconstruction(Marker, Mask=Img)只在下采样空间计算
    3. W-TopHat-Rec = Img - Background
    4. Enhanced = W-TopHat-Rec + WhiteTopHat(W-TopHat-Rec) - BlackTopHat(W-TopHat-Rec)
    """
    rad = float(params.get('radius', 10))
    downsample = float(params.get('downsample_factor', 0.25))

    if downsample <= 0:
        raise ValueError(
            f"morpho_reconstruction_contrast expects downsample_factor > 0; got {downsample!r}"
        )

    if img.ndim not in (2, 3):
        raise ValueError(
            f"morpho_reconstruction_contrast expects a 2D image or 3D stack; got ndim={img.ndim}"
        )

    rad_small = max(1, int(rad * downsample))
    selem_full = _morphology_disk(rad)     # 大图用的核
    selem_small = _morphology_disk(rad_small) # 小图用的核

    h, w = img.shape[-2:]
    small_shape = (max(1, int(h * downsample)), max(1, int(w * downsample)))

    if img.ndim == 3:
        if img.shape[0] == 0:
            raise ValueError("morpho_reconstruction_contrast expects a non-empty 3D stack")
        workers = _morpho_reconstruction_contrast_workers(params)
        first_slice = _morpho_reconstruction_contrast_slice(
            cast(ImageArray, img[0]),
            small_shape=small_shape,
            selem_full=selem_full,
            selem_small=selem_small,
        )
        res = np.empty((img.shape[0], *first_slice.shape), dtype=first_slice.dtype)
        res[0] = first_slice
        scratch_shape = cast(tuple[int, int], first_slice.shape)
        scratch_dtype = np.dtype(first_slice.dtype)

        if workers == 1 or img.shape[0] == 1:
            full_resolution_scratch = _morpho_reconstruction_contrast_scratch(
                shape=scratch_shape,
                dtype=scratch_dtype,
            )
            for z_index in range(1, img.shape[0]):
                res[z_index] = _morpho_reconstruction_contrast_slice(
                    cast(ImageArray, img[z_index]),
                    small_shape=small_shape,
                    selem_full=selem_full,
                    selem_small=selem_small,
                    full_resolution_scratch=full_resolution_scratch,
                )
        else:
            max_workers = min(workers, img.shape[0] - 1, os.cpu_count() or 1)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        _morpho_reconstruction_contrast_parallel_slice,
                        z_index,
                        cast(ImageArray, img[z_index]),
                        small_shape=small_shape,
                        selem_full=selem_full,
                        selem_small=selem_small,
                        scratch_shape=scratch_shape,
                        scratch_dtype=scratch_dtype,
                    )
                    for z_index in range(1, img.shape[0])
                ]
                for future in futures:
                    z_index, slice_result = future.result()
                    res[z_index] = slice_result
    else:
        res = _morpho_reconstruction_contrast_slice(
            img,
            small_shape=small_shape,
            selem_full=selem_full,
            selem_small=selem_small,
        )
        if "workers" in params:
            _ = _morpho_reconstruction_contrast_workers(params)

    return np.clip(res, 0, 1, out=res).astype(np.float32, copy=False)


def op_min_max_normalize(img: ImageArray, params: ProcessorParams, ctx: ProcessorContext) -> ImageArray:
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
def op_noop(img: ImageArray, params: ProcessorParams, ctx: ProcessorContext) -> ImageArray:
    """Return the input image unchanged.

    This null operation is useful when a config needs an explicit placeholder
    step to preserve provider dispatch shape or to document that no operation is
    intended at a given point in the preprocessing sequence.
    """
    return img


PROCESSOR_MAP: dict[str, ProcessorFunc] = {
    "median_filter": op_median_filter,
    "gaussian_blur": op_gaussian_blur, 
    "histogram_match": op_histogram_match,
    "gamma_correction": op_gamma_correction,
    "difference_of_gaussians": op_difference_of_gaussians,
    "clip_percentile": op_clip_percentile,
    "clahe": op_clahe,
    "morpho_reconstruction_contrast": op_morpho_reconstruction_contrast,
    "min_max_normalize": op_min_max_normalize,
    "none": op_noop, # Null Object Pattern
}


def _elapsed_ms_since(start_time: float) -> float:
    return round((time.perf_counter() - start_time) * 1000.0, 3)


def _duration_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "total_duration_ms": 0.0,
            "mean_duration_ms": None,
            "median_duration_ms": None,
            "min_duration_ms": None,
            "max_duration_ms": None,
        }

    sorted_values = sorted(float(value) for value in values)
    count = len(sorted_values)
    total = round(sum(sorted_values), 3)
    midpoint = count // 2
    if count % 2:
        median = sorted_values[midpoint]
    else:
        median = (sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2.0

    return {
        "count": count,
        "total_duration_ms": total,
        "mean_duration_ms": round(total / count, 3),
        "median_duration_ms": round(float(median), 3),
        "min_duration_ms": round(float(sorted_values[0]), 3),
        "max_duration_ms": round(float(sorted_values[-1]), 3),
    }


def _build_native_preprocessing_timing_payload(
    *,
    fov_id: int,
    round_order: list[int],
    volumes: list[dict[str, Any]],
    segment_index: int | None = None,
) -> dict[str, Any]:
    method_durations: dict[str, list[float]] = {}
    phase_durations: dict[str, list[float]] = {
        "load": [],
        "calibration_steps": [],
        "extraction_steps": [],
        "clip_convert": [],
        "write": [],
        "volume_total": [],
    }

    for volume in volumes:
        phase_durations["load"].append(float(volume["load_ms"]))
        phase_durations["clip_convert"].append(float(volume["clip_convert_ms"]))
        phase_durations["write"].append(float(volume["write_ms"]))
        phase_durations["volume_total"].append(float(volume["total_ms"]))

        calibration_total = 0.0
        for step_record in volume["calibration_steps"]:
            duration_ms = float(step_record["duration_ms"])
            calibration_total += duration_ms
            method_durations.setdefault(str(step_record["method"]), []).append(duration_ms)
        phase_durations["calibration_steps"].append(round(calibration_total, 3))

        extraction_total = 0.0
        for step_record in volume["extraction_steps"]:
            duration_ms = float(step_record["duration_ms"])
            extraction_total += duration_ms
            method_durations.setdefault(str(step_record["method"]), []).append(duration_ms)
        phase_durations["extraction_steps"].append(round(extraction_total, 3))

    by_method = {
        method: _duration_summary(durations)
        for method, durations in sorted(method_durations.items())
    }
    by_phase = {
        phase: _duration_summary(durations)
        for phase, durations in phase_durations.items()
    }

    payload = {
        "schema_name": NATIVE_PREPROCESSING_TIMING_SCHEMA_NAME,
        "schema_version": NATIVE_PREPROCESSING_TIMING_SCHEMA_VERSION,
        "fov_id": int(fov_id),
        "round_order": [int(round_id) for round_id in round_order],
        "volume_count": len(volumes),
        "total_volume_ms": by_phase["volume_total"]["total_duration_ms"],
        "volumes": volumes,
        "summary": {
            "by_method": by_method,
            "by_phase": by_phase,
        },
    }
    if segment_index is not None:
        payload["segment_index"] = int(segment_index)
    return payload

# ==============================================================================
# 3. THE ENGINE
# ==============================================================================

class DataSanitizer:
    """Create canonical cleaned image volumes from raw microscope TIFFs.

    The sanitizer is the first stage that writes PyStar-owned artifacts. It
    reads raw files through `ImageLoader`, applies the configured preprocessing
    sequence, and persists one clean 3D TIFF per FOV/round/channel under
    `clean_data/`. Native preprocessing atoms operate on float32 arrays in
    `[0, 1]` and are converted back to uint8 TIFFs for the output contract.

    A sequence may mix `native` and `matlab` providers. In that case the class
    materializes temporary stage directories between provider segments and then
    copies the final stage into the canonical clean-data layout. Provider
    switching is explicit provenance, not fallback behavior.
    """

    def __init__(self, config: ExperimentConfig, matlab_session_owner: Optional[MatlabSharedSessionOwner] = None):
        self.cfg = config
        self.loader = ImageLoader(config)
        self._matlab_session_owner = matlab_session_owner
        self._matlab_backend: Optional[MATLABPreprocessingBackend] = None

    def close(self) -> None:
        if self._matlab_backend is None:
            return
        self._matlab_backend.close()
        self._matlab_backend = None

    def __del__(self):  # pragma: no cover - best-effort cleanup only
        try:
            self.close()
        except Exception:
            pass

    def _base_output_dir(self) -> Path:
        return Path(self.cfg.pipeline.output.directory)

    def _utc_now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _build_native_preprocessing_provenance(
        self,
        *,
        fov_id: int,
        started_at: str,
        finished_at: str,
        duration_ms: float,
        rounds_processed: list[int],
        calibration_steps: list[PreprocessingStep],
        extraction_steps: list[PreprocessingStep],
        output_files: list[str],
        target_rounds: Optional[list[int]],
        preprocessing_timing: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        provenance = {
            "version": PREPROCESSING_PROVENANCE_VERSION,
            "generated_at": finished_at,
            "fov_id": int(fov_id),
            "backend": "native_pystar",
            "provider": "native",
            "duration_ms": duration_ms,
            "started_at": started_at,
            "finished_at": finished_at,
            "input_contract": {
                "raw_data_path": str(self.cfg.dataset.raw_data_path),
                "filename_pattern": self.cfg.dataset.filename_pattern,
                "target_rounds": list(target_rounds) if target_rounds is not None else None,
                "rounds_processed": rounds_processed,
            },
            "pipeline_split": {
                "calibration_steps": [step.method for step in calibration_steps],
                "extraction_steps": [step.method for step in extraction_steps],
            },
            "raw_sequence": [
                {
                    "index": index,
                    "method": step.method,
                    "provider": step.provider,
                    "params": dict(step.params),
                }
                for index, step in enumerate(self.cfg.pipeline.preprocessing.sequence)
            ],
            "output_files": output_files,
        }
        if preprocessing_timing is not None:
            provenance["preprocessing_timing"] = preprocessing_timing
        return provenance

    def _build_provider_dispatch_provenance(
        self,
        *,
        fov_id: int,
        started_at: str,
        finished_at: str,
        duration_ms: float,
        rounds_processed: list[int],
        target_rounds: Optional[list[int]],
        output_files: list[str],
        segment_records: list[dict[str, Any]],
        canonical_copy_ms: float = 0.0,
    ) -> dict[str, Any]:
        providers_used = sorted({record["provider"] for record in segment_records})
        if providers_used == ["native"]:
            backend_label = "native_pystar"
        elif providers_used == ["matlab"]:
            backend_label = "matlab_extracted"
        else:
            backend_label = "provider_dispatch"

        boundary_traces = [
            trace
            for record in segment_records
            if isinstance(record, dict)
            for trace in [record.get("boundary_instrumentation")]
            if isinstance(trace, dict)
        ]
        boundary_summary = summarize_matlab_boundary_traces(boundary_traces) if boundary_traces else None
        if boundary_summary is not None and canonical_copy_ms > 0:
            boundary_summary["provider_dispatch_canonical_copy_ms"] = round(float(canonical_copy_ms), 3)

        provenance = {
            "version": PREPROCESSING_PROVENANCE_VERSION,
            "generated_at": finished_at,
            "fov_id": int(fov_id),
            "backend": backend_label,
            "provider_mode": self.cfg.pipeline.preprocessing_provider_mode(),
            "providers_used": providers_used,
            "duration_ms": duration_ms,
            "started_at": started_at,
            "finished_at": finished_at,
            "input_contract": {
                "raw_data_path": str(self.cfg.dataset.raw_data_path),
                "filename_pattern": self.cfg.dataset.filename_pattern,
                "target_rounds": list(target_rounds) if target_rounds is not None else None,
                "rounds_processed": rounds_processed,
            },
            "raw_sequence": [
                {
                    "index": index,
                    "method": step.method,
                    "provider": step.provider,
                    "params": dict(step.params),
                }
                for index, step in enumerate(self.cfg.pipeline.preprocessing.sequence)
            ],
            "segments": segment_records,
            "output_files": output_files,
        }
        if boundary_summary is not None:
            provenance["boundary_instrumentation_summary"] = boundary_summary
        return provenance

    def _resolve_rounds_to_process(self, target_rounds: Optional[list[int]]) -> list[int]:
        all_config_rounds = sorted(self.cfg.dataset.round_structure.keys())
        if target_rounds is None:
            return all_config_rounds

        rounds_to_process = sorted([r for r in target_rounds if r in all_config_rounds])
        if not rounds_to_process:
            raise ValueError(f"No valid rounds found in target_rounds: {target_rounds}")
        return rounds_to_process

    def _ordered_round_queue(self, rounds_to_process: list[int]) -> list[int]:
        ref_round_id = 1
        final_queue: list[int] = []
        if ref_round_id in rounds_to_process:
            final_queue.append(ref_round_id)
        for round_id in rounds_to_process:
            if round_id != ref_round_id:
                final_queue.append(round_id)
        return final_queue

    def _sequence_segments(self, sequence: list[PreprocessingStep]) -> list[tuple[str, list[PreprocessingStep]]]:
        segments: list[tuple[str, list[PreprocessingStep]]] = []
        if not sequence:
            return segments

        current_provider = sequence[0].provider
        current_steps: list[PreprocessingStep] = []
        for step in sequence:
            if step.provider != current_provider:
                segments.append((current_provider, current_steps))
                current_provider = step.provider
                current_steps = []
            current_steps.append(step)

        if current_steps:
            segments.append((current_provider, current_steps))
        return segments

    def _make_loader(self, input_root: Path, filename_pattern: str) -> ImageLoader:
        temp_dataset = self.cfg.dataset.model_copy(
            update={
                "raw_data_path": Path(input_root),
                "filename_pattern": filename_pattern,
            }
        )
        temp_cfg = self.cfg.model_copy(update={"dataset": temp_dataset})
        return ImageLoader(temp_cfg)

    def _stage_relative_path(self, fov_id: int, round_id: int, channel_id: int) -> Path:
        formatted = self.cfg.dataset.filename_pattern.format(
            round=round_id,
            fov=fov_id,
            ch=f"{channel_id:02d}",
        )
        if "*" in formatted:
            formatted = formatted.replace("*", f"clean_fov_{fov_id}_round_{round_id}")
        return Path(formatted)

    def _save_stage_clean(self, img: ImageArray, stage_root: Path, fov_id: int, round_id: int, channel_id: int) -> Path:
        rel_path = self._stage_relative_path(fov_id, round_id, channel_id)
        output_path = stage_root / rel_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(output_path, img, compression='zlib')
        return output_path

    def _make_stage_output_writer(self, output_root: Path, fov_id: int) -> NativeOutputWriter:
        def write(img: ImageArray, round_id: int, channel_id: int) -> Path:
            return self._save_stage_clean(img, output_root, fov_id, round_id, channel_id)

        return write

    def _make_canonical_output_writer(self, fov_id: int) -> NativeOutputWriter:
        def write(img: ImageArray, round_id: int, channel_id: int) -> Path:
            return self._save_clean(img, fov_id, round_id, channel_id)

        return write

    def _flat_clean_filename(self, fov_id: int, round_id: int, channel_id: int) -> str:
        return f"clean_fov_{fov_id}_round_{round_id}_ch_{channel_id}.tif"

    def _materialize_flat_outputs_to_stage(
        self,
        flat_output_dir: Path,
        stage_root: Path,
        fov_id: int,
        rounds_to_process: list[int],
    ) -> None:
        roles = self.cfg.dataset.channel_roles
        for round_id in rounds_to_process:
            seq_channels = sorted(
                channel_id
                for channel_id in self.cfg.dataset.round_structure[round_id]
                if roles.get(channel_id) == 'seq'
            )
            for channel_id in seq_channels:
                source_path = flat_output_dir / self._flat_clean_filename(fov_id, round_id, channel_id)
                if not source_path.exists():
                    raise FileNotFoundError(
                        f"Expected MATLAB preprocessing output is missing: {source_path}"
                    )
                destination_path = stage_root / self._stage_relative_path(fov_id, round_id, channel_id)
                destination_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, destination_path)

    def _copy_stage_outputs_to_clean_dir(
        self,
        stage_root: Path,
        fov_id: int,
        rounds_to_process: list[int],
    ) -> list[str]:
        loader = self._make_loader(stage_root, self.cfg.dataset.filename_pattern)
        roles = self.cfg.dataset.channel_roles
        output_files: list[str] = []
        for round_id in rounds_to_process:
            seq_channels = sorted(
                channel_id
                for channel_id in self.cfg.dataset.round_structure[round_id]
                if roles.get(channel_id) == 'seq'
            )
            for channel_id in seq_channels:
                stage_path = loader._get_path(fov_id, round_id, channel_id)
                destination_path = self.get_clean_path(fov_id, round_id, channel_id)
                destination_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(stage_path, destination_path)
                output_files.append(str(destination_path))
        return output_files

    def get_clean_path(self, fov_id: int, round_id: int, channel_id: int) -> Path:
        base_dir = self._base_output_dir()
        paths = get_fov_output_structure(base_dir, fov_id)
        return paths['cleaned'] / self._flat_clean_filename(fov_id, round_id, channel_id)

    def _run_pipeline_with_timing(
        self,
        img_vol: ImageArray,
        pipeline_seq: list[PreprocessingStep],
        context: ProcessorContext,
    ) -> tuple[ImageArray, list[dict[str, Any]]]:
        if img_vol.dtype != np.float32:
            max_val = 255.0 if img_vol.dtype == np.uint8 else 65535.0
            if np.issubdtype(img_vol.dtype, np.floating) and img_vol.max() > 1.0:
                current_data = img_vol
            else:
                current_data = img_vol.astype(np.float32) / max_val
        else:
            current_data = img_vol

        step_timings: list[dict[str, Any]] = []
        for step_index, step in enumerate(pipeline_seq):
            func = PROCESSOR_MAP.get(step.method)
            if func:
                step_started = time.perf_counter()
                current_data = func(current_data, step.params, context)
                step_timings.append(
                    {
                        "index": step_index,
                        "method": step.method,
                        "provider": step.provider,
                        "duration_ms": _elapsed_ms_since(step_started),
                    }
                )

        return current_data, step_timings

    def _run_native_preprocessing_kernel(
        self,
        *,
        fov_id: int,
        loader: ImageLoader,
        sequence: list[PreprocessingStep],
        target_rounds: Optional[list[int]],
        output_writer: NativeOutputWriter,
        segment_index: int | None = None,
        print_progress: bool = False,
    ) -> dict[str, Any]:
        if not sequence:
            raise ValueError("Native preprocessing kernel cannot run an empty sequence")

        seq_calibration, seq_extraction = self._split_sequence(sequence)
        rounds_to_process = self._resolve_rounds_to_process(target_rounds)
        if print_progress and target_rounds is not None:
            print(f" -> DEBUG: Only processing user-selected rounds: {rounds_to_process}")
        final_queue = self._ordered_round_queue(rounds_to_process)
        if print_progress:
            print(f" -> Pipeline Split: {len(seq_calibration)} Calibration steps + {len(seq_extraction)} Extraction steps")

        inter_round_ref_cache: dict[int, ImageArray] = {}
        output_files: list[str] = []
        volume_records: list[dict[str, Any]] = []

        for r_id in final_queue:
            if print_progress:
                print(f" -> Processing Round {r_id}...")
            intra_round_ref_img: ImageArray | None = None
            roles = self.cfg.dataset.channel_roles
            channels_in_round = self.cfg.dataset.round_structure[r_id]
            seq_channels = sorted([c for c in channels_in_round if roles.get(c) == 'seq'])

            for c_id in seq_channels:
                volume_started = time.perf_counter()

                load_started = time.perf_counter()
                path = loader._get_path(fov_id, r_id, c_id)
                raw_vol = loader._lazy_load_tiff(path).compute()
                load_ms = _elapsed_ms_since(load_started)

                ctx = {
                    'ref_round_image': inter_round_ref_cache.get(c_id),
                    'ref_channel_image': intra_round_ref_img,
                }

                img_calibrated, calibration_timings = self._run_pipeline_with_timing(raw_vol, seq_calibration, ctx)

                if r_id == 1:
                    inter_round_ref_cache[c_id] = img_calibrated.copy()

                if intra_round_ref_img is None:
                    intra_round_ref_img = img_calibrated.copy()

                final_vol, extraction_timings = self._run_pipeline_with_timing(img_calibrated, seq_extraction, ctx)

                clip_convert_started = time.perf_counter()
                final_u8 = img_as_ubyte(np.clip(final_vol, 0, 1))
                clip_convert_ms = _elapsed_ms_since(clip_convert_started)

                write_started = time.perf_counter()
                output_path = output_writer(final_u8, r_id, c_id)
                write_ms = _elapsed_ms_since(write_started)
                output_files.append(str(output_path))

                volume_records.append(
                    {
                        "round_id": int(r_id),
                        "channel_id": int(c_id),
                        "input_path": str(path),
                        "output_path": str(output_path),
                        "load_ms": load_ms,
                        "calibration_steps": calibration_timings,
                        "extraction_steps": extraction_timings,
                        "clip_convert_ms": clip_convert_ms,
                        "write_ms": write_ms,
                        "total_ms": _elapsed_ms_since(volume_started),
                    }
                )

        return {
            "rounds_to_process": rounds_to_process,
            "round_order": final_queue,
            "calibration_steps": seq_calibration,
            "extraction_steps": seq_extraction,
            "output_files": output_files,
            "preprocessing_timing": _build_native_preprocessing_timing_payload(
                fov_id=fov_id,
                round_order=final_queue,
                volumes=volume_records,
                segment_index=segment_index,
            ),
        }

    def _run_native_sequence_segment(
        self,
        fov_id: int,
        sequence: list[PreprocessingStep],
        *,
        input_root: Path,
        input_filename_pattern: str,
        output_root: Path,
        target_rounds: Optional[list[int]] = None,
        segment_index: int,
    ) -> dict[str, Any]:
        full_seq = sequence
        if not full_seq:
            raise ValueError("Native preprocessing segment cannot be empty")

        loader = self._make_loader(input_root, input_filename_pattern)
        started_at = self._utc_now()
        start_time = time.perf_counter()
        native_result = self._run_native_preprocessing_kernel(
            fov_id=fov_id,
            loader=loader,
            sequence=full_seq,
            target_rounds=target_rounds,
            output_writer=self._make_stage_output_writer(output_root, fov_id),
            segment_index=segment_index,
        )

        finished_at = self._utc_now()
        duration_ms = round((time.perf_counter() - start_time) * 1000.0, 3)
        return {
            "provider": "native",
            "segment_index": segment_index,
            "duration_ms": duration_ms,
            "started_at": started_at,
            "finished_at": finished_at,
            "input_contract": {
                "raw_data_path": str(input_root),
                "filename_pattern": input_filename_pattern,
                "rounds_processed": native_result["round_order"],
                "target_rounds": list(target_rounds) if target_rounds is not None else None,
            },
            "pipeline_split": {
                "calibration_steps": [step.method for step in native_result["calibration_steps"]],
                "extraction_steps": [step.method for step in native_result["extraction_steps"]],
            },
            "raw_sequence": [
                {
                    "index": index,
                    "method": step.method,
                    "provider": step.provider,
                    "params": dict(step.params),
                }
                for index, step in enumerate(full_seq)
            ],
            "output_files": native_result["output_files"],
            "preprocessing_timing": native_result["preprocessing_timing"],
        }

    def _split_sequence(
        self,
        full_seq: list[PreprocessingStep],
    ) -> tuple[list[PreprocessingStep], list[PreprocessingStep]]:
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

    def _run_pipeline(
        self,
        img_vol: ImageArray,
        pipeline_seq: list[PreprocessingStep],
        context: ProcessorContext,
    ) -> ImageArray:
        """Execute one native preprocessing segment on a 2D/3D image array.

        The segment receives either raw TIFF values or the output of an earlier
        preprocessing stage. Non-float inputs are scaled to `[0, 1]` before the
        configured atoms run. The returned array remains float-like so the caller
        can either feed it into additional atoms or convert it to the canonical
        clean TIFF dtype at the persistence boundary.
        """
        current_data, _step_timings = self._run_pipeline_with_timing(img_vol, pipeline_seq, context)
        return current_data

    def _native_sanitize_fov(
        self,
        fov_id: int,
        target_rounds: Optional[list[int]] = None,
    ) -> dict[str, Any]:
        full_seq = self.cfg.pipeline.preprocessing.sequence
        if not full_seq:
            print("Warning: Pipeline sequence is empty.")
            return {
                "version": PREPROCESSING_PROVENANCE_VERSION,
                "generated_at": self._utc_now(),
                "fov_id": int(fov_id),
                "backend": "native_pystar",
                "provider": "native",
                "duration_ms": 0.0,
                "started_at": None,
                "finished_at": None,
                "input_contract": {
                    "raw_data_path": str(self.cfg.dataset.raw_data_path),
                    "filename_pattern": self.cfg.dataset.filename_pattern,
                    "target_rounds": list(target_rounds) if target_rounds is not None else None,
                    "rounds_processed": [],
                },
                "pipeline_split": {
                    "calibration_steps": [],
                    "extraction_steps": [],
                },
                "raw_sequence": [],
                "output_files": [],
            }

        started_at = self._utc_now()
        start_time = time.perf_counter()
        native_result = self._run_native_preprocessing_kernel(
            fov_id=fov_id,
            loader=self.loader,
            sequence=full_seq,
            target_rounds=target_rounds,
            output_writer=self._make_canonical_output_writer(fov_id),
            print_progress=True,
        )

        finished_at = self._utc_now()
        duration_ms = round((time.perf_counter() - start_time) * 1000.0, 3)
        return self._build_native_preprocessing_provenance(
            fov_id=fov_id,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            rounds_processed=native_result["round_order"],
            calibration_steps=native_result["calibration_steps"],
            extraction_steps=native_result["extraction_steps"],
            output_files=native_result["output_files"],
            target_rounds=target_rounds,
            preprocessing_timing=native_result["preprocessing_timing"],
        )

    def _provider_dispatch_sanitize_fov(
        self,
        fov_id: int,
        target_rounds: Optional[list[int]] = None,
    ) -> dict[str, Any]:
        full_seq = self.cfg.pipeline.preprocessing.sequence
        if not full_seq:
            print("Warning: Pipeline sequence is empty.")
            return self._build_provider_dispatch_provenance(
                fov_id=fov_id,
                started_at=self._utc_now(),
                finished_at=self._utc_now(),
                duration_ms=0.0,
                rounds_processed=[],
                target_rounds=target_rounds,
                output_files=[],
                segment_records=[],
            )

        rounds_to_process = self._resolve_rounds_to_process(target_rounds)
        segment_records: list[dict[str, Any]] = []
        started_at = self._utc_now()
        start_time = time.perf_counter()

        with TemporaryDirectory(prefix=f"pystar_preprocessing_fov{fov_id}_") as tmpdir:
            current_input_root = Path(self.cfg.dataset.raw_data_path)
            current_input_pattern = self.cfg.dataset.filename_pattern

            for segment_index, (provider, steps) in enumerate(self._sequence_segments(full_seq)):
                stage_root = Path(tmpdir) / f"segment_{segment_index}_{provider}"
                if provider == "native":
                    segment_record = self._run_native_sequence_segment(
                        fov_id,
                        steps,
                        input_root=current_input_root,
                        input_filename_pattern=current_input_pattern,
                        output_root=stage_root,
                        target_rounds=target_rounds,
                        segment_index=segment_index,
                    )
                elif provider == "matlab":
                    if self._matlab_backend is None:
                        self._matlab_backend = MATLABPreprocessingBackend(
                            self.cfg,
                            matlab_session_owner=self._matlab_session_owner,
                        )
                    matlab_output_root = Path(tmpdir) / f"segment_{segment_index}_{provider}_flat"
                    segment_record = self._matlab_backend.execute_sequence(
                        fov_id,
                        sequence=steps,
                        input_root=current_input_root,
                        input_filename_pattern=current_input_pattern,
                        output_dir=matlab_output_root,
                        target_rounds=target_rounds,
                        segment_label=f"segment_{segment_index}",
                    )
                    materialization_started = time.perf_counter()
                    self._materialize_flat_outputs_to_stage(
                        matlab_output_root,
                        stage_root,
                        fov_id,
                        rounds_to_process,
                    )
                    materialization_ms = round((time.perf_counter() - materialization_started) * 1000.0, 3)
                    boundary_trace = segment_record.get("boundary_instrumentation")
                    if isinstance(boundary_trace, dict):
                        phase_timings = boundary_trace.setdefault("phase_timings_ms", {})
                        phase_details = boundary_trace.setdefault("phase_details", {})
                        seam_costs = boundary_trace.setdefault("seam_costs_ms", {})
                        phase_timings["python_stage_materialization"] = materialization_ms
                        phase_details["python_stage_materialization"] = {
                            "stage_root": str(stage_root),
                            "round_count": len(rounds_to_process),
                        }
                        seam_costs["canonical_persistence_ms"] = round(
                            float(seam_costs.get("canonical_persistence_ms", 0.0) or 0.0) + materialization_ms,
                            3,
                        )
                        boundary_trace["total_duration_ms"] = round(
                            float(boundary_trace.get("total_duration_ms", 0.0) or 0.0) + materialization_ms,
                            3,
                        )
                else:
                    raise ValueError(f"Unsupported preprocessing provider: {provider!r}")

                segment_records.append(segment_record)
                current_input_root = stage_root
                current_input_pattern = self.cfg.dataset.filename_pattern

            canonical_copy_started = time.perf_counter()
            output_files = self._copy_stage_outputs_to_clean_dir(current_input_root, fov_id, rounds_to_process)
            canonical_copy_ms = round((time.perf_counter() - canonical_copy_started) * 1000.0, 3)

        finished_at = self._utc_now()
        duration_ms = round((time.perf_counter() - start_time) * 1000.0, 3)
        return self._build_provider_dispatch_provenance(
            fov_id=fov_id,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            rounds_processed=rounds_to_process,
            target_rounds=target_rounds,
            output_files=output_files,
            segment_records=segment_records,
            canonical_copy_ms=canonical_copy_ms,
        )

    def sanitize_fov(
        self,
        fov_id: int,
        target_rounds: Optional[list[int]] = None,
    ) -> dict[str, Any]:
        """
        Preprocess one FOV and persist clean images plus provenance.

        Parameters
        ----------
        fov_id:
            Position/FOV index from the experiment config.
        target_rounds:
            Optional subset of rounds for parameter testing. Production runs
            normally leave this as `None` so every configured round is cleaned.

        Returns
        -------
        dict
            Provenance payload describing providers, steps, input contract,
            output files, and timing. The same payload is written to disk next to
            the FOV outputs.
        """
        print(f"[{'='*20} Sanitizing FOV {fov_id} {'='*20}]")

        providers_used = set(self.cfg.pipeline.preprocessing_providers_used())
        if providers_used == {"native"}:
            provenance = self._native_sanitize_fov(fov_id, target_rounds=target_rounds)
        else:
            provenance = self._provider_dispatch_sanitize_fov(fov_id, target_rounds=target_rounds)

        write_preprocessing_provenance(self._base_output_dir(), fov_id, provenance)
        return provenance

    def _save_clean(self, img: ImageArray, f: int, r: int, c: int) -> Path:
        out_path = self.get_clean_path(f, r, c)
        tifffile.imwrite(out_path, img, compression='zlib')
        return out_path
