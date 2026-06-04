from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, cast

import cv2
import numpy as np
import pytest
import tifffile
from numpy.typing import NDArray

from pystar import preprocessing as preprocessing_module
from pystar.infrastructure import PreprocessingStep
from pystar.io import get_fov_output_structure
from pystar.preprocessing import (
    DataSanitizer,
    op_difference_of_gaussians,
    op_gaussian_blur,
    op_median_filter,
)


class _FakeDelayedVolume:
    _array: NDArray[Any]

    def __init__(self, array: NDArray[Any]) -> None:
        self._array = array

    def compute(self) -> NDArray[Any]:
        return self._array.copy()


class _FakeLoader:
    volumes: dict[tuple[int, int], NDArray[Any]]
    root: Path

    def __init__(self, volumes: dict[tuple[int, int], NDArray[Any]], root: Path) -> None:
        self.volumes = volumes
        self.root = root

    def _get_path(self, fov: int, round_id: int, channel_id: int) -> Path:
        _ = fov
        return self.root / f"round{round_id}" / f"ch{channel_id}.tif"

    def _lazy_load_tiff(self, path: Path) -> _FakeDelayedVolume:
        round_id = int(path.parent.name.replace("round", ""))
        channel_id = int(path.stem.replace("ch", ""))
        return _FakeDelayedVolume(self.volumes[(round_id, channel_id)])


def _step(method: str, params: dict[str, Any] | None = None) -> PreprocessingStep:
    return PreprocessingStep(method=method, provider="native", params={} if params is None else params)


def _config(tmp_path: Path, sequence: list[PreprocessingStep]) -> Any:
    dataset = SimpleNamespace(
        raw_data_path=tmp_path / "raw",
        filename_pattern="round{round}/Position{fov}/*_ch{ch}.tif",
        round_structure={1: [0, 1], 2: [0, 1]},
        channel_roles={0: "seq", 1: "seq"},
        dimensions={"z": 2, "height": 6, "width": 7},
        io_chunk_size={"z": 2, "y": 6, "x": 7},
    )
    pipeline = SimpleNamespace(
        preprocessing=SimpleNamespace(sequence=sequence),
        output=SimpleNamespace(directory=str(tmp_path / "production_out")),
    )
    pipeline.preprocessing_providers_used = lambda: ["native"]
    pipeline.preprocessing_provider_mode = lambda: "native_only"
    return SimpleNamespace(dataset=dataset, pipeline=pipeline)


def _volumes() -> dict[tuple[int, int], NDArray[Any]]:
    volumes: dict[tuple[int, int], NDArray[Any]] = {}
    for round_id in (1, 2):
        for channel_id in (0, 1):
            base = round_id * 31 + channel_id * 17
            volumes[(round_id, channel_id)] = (
                np.arange(84, dtype=np.uint16).reshape(2, 6, 7) + base
            ).astype(np.uint8)
    return volumes


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _writer_for_root(sanitizer: DataSanitizer, output_root: Path, fov_id: int):
    def write(img: Any, round_id: int, channel_id: int) -> Path:
        paths = get_fov_output_structure(output_root, fov_id)
        path = paths["cleaned"] / sanitizer._flat_clean_filename(fov_id, round_id, channel_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(path, img, compression="zlib")
        return path

    return write


def _baseline_median_filter(img: NDArray[Any], params: dict[str, Any], ctx: dict[str, Any]) -> NDArray[Any]:
    _ = ctx
    k = params.get("kernel_size", 3)
    if k % 2 == 0:
        k += 1
    if k < 3:
        return img

    img_u8 = cast(NDArray[Any], (img * 255).astype(np.uint8))

    def median_blur_slice(slice_u8: NDArray[Any]) -> NDArray[Any]:
        blurred = cv2.medianBlur(cast(Any, np.ascontiguousarray(slice_u8)), k)
        return cast(NDArray[Any], blurred)

    if img_u8.ndim == 3:
        res_u8 = cast(NDArray[Any], np.stack([median_blur_slice(cast(NDArray[Any], s)) for s in img_u8]))
    else:
        res_u8 = median_blur_slice(img_u8)

    return cast(NDArray[Any], res_u8.astype(np.float32) / 255.0)


def _baseline_gaussian_blur(img: NDArray[Any], params: dict[str, Any], ctx: dict[str, Any]) -> NDArray[Any]:
    _ = ctx
    sigma = params.get("sigma", 1.0)
    if img.ndim == 3:
        return cast(
            NDArray[Any],
            np.stack([cv2.GaussianBlur(s, (0, 0), sigmaX=sigma, sigmaY=sigma) for s in img]),
        )
    return cast(NDArray[Any], cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma))


def _baseline_difference_of_gaussians(
    img: NDArray[Any],
    params: dict[str, Any],
    ctx: dict[str, Any],
) -> NDArray[Any]:
    _ = ctx
    spot_sigma = params.get("spot_sigma", 1.0)
    bg_sigma = params.get("bg_sigma", 5.0)

    def blur_slice(slice_2d: NDArray[Any], sigma: float) -> NDArray[Any]:
        return cast(NDArray[Any], cv2.GaussianBlur(slice_2d, (0, 0), sigmaX=sigma, sigmaY=sigma))

    if img.ndim == 3:
        g_small = np.stack([blur_slice(s, spot_sigma) for s in img])
        g_large = np.stack([blur_slice(s, bg_sigma) for s in img])
    else:
        g_small = blur_slice(img, spot_sigma)
        g_large = blur_slice(img, bg_sigma)

    diff = g_small - g_large
    return cast(NDArray[Any], np.maximum(diff, 0))


AtomFunc = Callable[[NDArray[Any], dict[str, Any], dict[str, Any]], NDArray[Any]]


@pytest.mark.parametrize(
    ("atom", "baseline", "params"),
    [
        (op_gaussian_blur, _baseline_gaussian_blur, {"sigma": 0.75}),
        (op_difference_of_gaussians, _baseline_difference_of_gaussians, {"spot_sigma": 0.6, "bg_sigma": 1.4}),
        (op_median_filter, _baseline_median_filter, {"kernel_size": 4}),
    ],
)
def test_stack_kernel_atoms_match_2d_baselines_and_preserve_inputs(
    atom: AtomFunc,
    baseline: AtomFunc,
    params: dict[str, Any],
) -> None:
    image = np.random.default_rng(2701).random((18, 22), dtype=np.float32)
    image_before = image.copy()

    optimized = atom(cast(NDArray[Any], image), params, {})
    expected = baseline(cast(NDArray[Any], image), params, {})

    assert optimized.dtype == expected.dtype
    assert optimized.shape == expected.shape == image.shape
    np.testing.assert_array_equal(optimized, expected)
    np.testing.assert_array_equal(image, image_before)


@pytest.mark.parametrize(
    ("atom", "baseline", "params"),
    [
        (op_gaussian_blur, _baseline_gaussian_blur, {"sigma": 0.75}),
        (op_difference_of_gaussians, _baseline_difference_of_gaussians, {"spot_sigma": 0.6, "bg_sigma": 1.4}),
        (op_median_filter, _baseline_median_filter, {"kernel_size": 5}),
    ],
)
def test_stack_kernel_atoms_match_3d_list_stack_baselines_and_preserve_inputs(
    atom: AtomFunc,
    baseline: AtomFunc,
    params: dict[str, Any],
) -> None:
    image = np.random.default_rng(2702).random((4, 18, 22), dtype=np.float32)
    image_before = image.copy()

    optimized = atom(cast(NDArray[Any], image), params, {})
    expected = baseline(cast(NDArray[Any], image), params, {})

    assert optimized.dtype == expected.dtype
    assert optimized.shape == expected.shape == image.shape
    np.testing.assert_array_equal(optimized, expected)
    np.testing.assert_array_equal(image, image_before)


@pytest.mark.parametrize(
    ("atom", "baseline", "params"),
    [
        (op_gaussian_blur, _baseline_gaussian_blur, {"sigma": 1.1}),
        (op_difference_of_gaussians, _baseline_difference_of_gaussians, {"spot_sigma": 0.5, "bg_sigma": 1.2}),
        (op_median_filter, _baseline_median_filter, {"kernel_size": 3}),
    ],
)
def test_stack_kernel_atoms_handle_single_slice_3d_stacks(
    atom: AtomFunc,
    baseline: AtomFunc,
    params: dict[str, Any],
) -> None:
    image = np.random.default_rng(2703).random((1, 16, 20), dtype=np.float32)

    optimized = atom(cast(NDArray[Any], image), params, {})
    expected = baseline(cast(NDArray[Any], image), params, {})

    assert optimized.shape == (1, 16, 20)
    np.testing.assert_array_equal(optimized, expected)


@pytest.mark.parametrize(
    ("atom", "baseline", "params"),
    [
        (op_gaussian_blur, _baseline_gaussian_blur, {"sigma": 0.9}),
        (op_difference_of_gaussians, _baseline_difference_of_gaussians, {"spot_sigma": 0.7, "bg_sigma": 1.3}),
        (op_median_filter, _baseline_median_filter, {"kernel_size": 3}),
    ],
)
def test_stack_kernel_atoms_match_non_contiguous_3d_view_baselines(
    atom: AtomFunc,
    baseline: AtomFunc,
    params: dict[str, Any],
) -> None:
    base = np.random.default_rng(2704).random((4, 20, 24), dtype=np.float32)
    image = base[:, ::2, 1::2]
    image_before = image.copy()
    assert not image.flags.c_contiguous

    optimized = atom(cast(NDArray[Any], image), params, {})
    expected = baseline(cast(NDArray[Any], image), params, {})

    assert optimized.shape == expected.shape == image.shape
    np.testing.assert_array_equal(optimized, expected)
    np.testing.assert_array_equal(image, image_before)


def test_median_filter_small_kernel_keeps_identity_behavior() -> None:
    image = np.random.default_rng(2705).random((2, 6, 7), dtype=np.float32)

    result = op_median_filter(cast(NDArray[Any], image), {"kernel_size": 1}, {})

    assert result is image


def test_difference_of_gaussians_preserves_nonnegative_output_contract() -> None:
    image = np.random.default_rng(2706).random((3, 18, 22), dtype=np.float32)

    result = op_difference_of_gaussians(
        cast(NDArray[Any], image),
        {"spot_sigma": 0.6, "bg_sigma": 2.0},
        {},
    )

    assert result.dtype == np.float32
    assert result.shape == image.shape
    assert np.isfinite(result).all()
    assert float(result.min()) >= 0.0


def test_native_kernel_clean_tiffs_match_stage27_list_stack_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sequence = [
        _step("median_filter", {"kernel_size": 3}),
        _step("gaussian_blur", {"sigma": 0.75}),
        _step("difference_of_gaussians", {"spot_sigma": 0.6, "bg_sigma": 1.4}),
    ]
    cfg = _config(tmp_path, sequence)
    sanitizer = DataSanitizer(cast(Any, cfg))
    fake_loader = cast(Any, _FakeLoader(_volumes(), tmp_path / "raw"))

    monkeypatch.setitem(preprocessing_module.PROCESSOR_MAP, "median_filter", _baseline_median_filter)
    monkeypatch.setitem(preprocessing_module.PROCESSOR_MAP, "gaussian_blur", _baseline_gaussian_blur)
    monkeypatch.setitem(
        preprocessing_module.PROCESSOR_MAP,
        "difference_of_gaussians",
        _baseline_difference_of_gaussians,
    )
    baseline = sanitizer._run_native_preprocessing_kernel(
        fov_id=1,
        loader=fake_loader,
        sequence=sequence,
        target_rounds=[1, 2],
        output_writer=_writer_for_root(sanitizer, tmp_path / "baseline", 1),
    )

    monkeypatch.setitem(preprocessing_module.PROCESSOR_MAP, "median_filter", op_median_filter)
    monkeypatch.setitem(preprocessing_module.PROCESSOR_MAP, "gaussian_blur", op_gaussian_blur)
    monkeypatch.setitem(preprocessing_module.PROCESSOR_MAP, "difference_of_gaussians", op_difference_of_gaussians)
    candidate = sanitizer._run_native_preprocessing_kernel(
        fov_id=1,
        loader=fake_loader,
        sequence=sequence,
        target_rounds=[1, 2],
        output_writer=_writer_for_root(sanitizer, tmp_path / "candidate", 1),
    )

    baseline_files = {Path(path).name: Path(path) for path in cast(list[str], baseline["output_files"])}
    candidate_files = {Path(path).name: Path(path) for path in cast(list[str], candidate["output_files"])}
    assert set(baseline_files) == set(candidate_files) == {
        "clean_fov_1_round_1_ch_0.tif",
        "clean_fov_1_round_1_ch_1.tif",
        "clean_fov_1_round_2_ch_0.tif",
        "clean_fov_1_round_2_ch_1.tif",
    }

    for name, baseline_path in baseline_files.items():
        candidate_path = candidate_files[name]
        assert _sha256_file(candidate_path) == _sha256_file(baseline_path)
        baseline_array = tifffile.imread(baseline_path)
        candidate_array = tifffile.imread(candidate_path)
        assert baseline_array.dtype == np.uint8
        assert candidate_array.dtype == np.uint8
        np.testing.assert_array_equal(candidate_array, baseline_array)

    assert not Path(cfg.pipeline.output.directory).exists()
