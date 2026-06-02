from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import tifffile
from numpy.typing import NDArray

from pystar import preprocessing as preprocessing_module
from pystar.infrastructure import PreprocessingStep
from pystar.io import get_fov_output_structure
from pystar.preprocessing import (
    DataSanitizer,
    _morpho_reconstruction_contrast_scratch,
    _morpho_reconstruction_contrast_slice,
    _morphology_disk,
    op_morpho_reconstruction_contrast,
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
            base = round_id * 37 + channel_id * 13
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


def _independent_2d_morphology(img: Any, params: dict[str, Any], ctx: dict[str, Any]) -> Any:
    if np.asarray(img).ndim == 3:
        return np.stack(
            [op_morpho_reconstruction_contrast(cast(Any, slice_2d), params, ctx) for slice_2d in img]
        )
    return op_morpho_reconstruction_contrast(img, params, ctx)


def test_morphology_slice_scratch_matches_unscratched_slice_and_keeps_input_immutable() -> None:
    image = np.random.default_rng(26).random((24, 32), dtype=np.float32)
    image_before = image.copy()
    radius = 3.0
    downsample = 0.5
    small_shape = (max(1, int(image.shape[0] * downsample)), max(1, int(image.shape[1] * downsample)))
    selem_full = _morphology_disk(radius)
    selem_small = _morphology_disk(max(1, int(radius * downsample)))

    baseline = _morpho_reconstruction_contrast_slice(
        cast(Any, image),
        small_shape=small_shape,
        selem_full=selem_full,
        selem_small=selem_small,
    )
    scratch = _morpho_reconstruction_contrast_scratch(
        shape=(int(image.shape[0]), int(image.shape[1])),
        dtype=np.dtype(image.dtype),
    )
    optimized = _morpho_reconstruction_contrast_slice(
        cast(Any, image),
        small_shape=small_shape,
        selem_full=selem_full,
        selem_small=selem_small,
        full_resolution_scratch=scratch,
    )

    assert scratch[0].shape == image.shape
    assert scratch[0].dtype == image.dtype
    assert not np.may_share_memory(scratch[0], image)
    assert not np.may_share_memory(scratch[1], image)
    np.testing.assert_array_equal(optimized, baseline)
    np.testing.assert_array_equal(image, image_before)


def test_morphology_scratch_rejects_input_aliasing() -> None:
    image = np.random.default_rng(27).random((12, 14), dtype=np.float32)
    scratch = (cast(Any, image), cast(Any, np.empty_like(image)))

    with pytest.raises(ValueError, match="scratch buffers must not alias input slices"):
        _ = _morpho_reconstruction_contrast_slice(
            cast(Any, image),
            small_shape=(6, 7),
            selem_full=_morphology_disk(2.0),
            selem_small=_morphology_disk(1),
            full_resolution_scratch=scratch,
        )


def test_morphology_scratch_rejects_internal_aliasing() -> None:
    image = np.random.default_rng(270).random((12, 14), dtype=np.float32)
    shared_scratch = np.empty_like(image)
    scratch = (cast(Any, shared_scratch), cast(Any, shared_scratch))

    with pytest.raises(ValueError, match="scratch buffers must not alias each other"):
        _ = _morpho_reconstruction_contrast_slice(
            cast(Any, image),
            small_shape=(6, 7),
            selem_full=_morphology_disk(2.0),
            selem_small=_morphology_disk(1),
            full_resolution_scratch=scratch,
        )


def test_optimized_3d_morphology_equals_independent_2d_stack_and_preserves_contract() -> None:
    image = np.random.default_rng(28).random((4, 24, 32), dtype=np.float32)
    image_before = image.copy()
    params = {"radius": 3, "downsample_factor": 0.5}

    optimized = op_morpho_reconstruction_contrast(cast(Any, image), params, {})
    per_slice = np.stack(
        [op_morpho_reconstruction_contrast(cast(Any, slice_2d), params, {}) for slice_2d in image]
    )

    assert optimized.dtype == np.float32
    assert optimized.shape == image.shape
    assert np.isfinite(optimized).all()
    assert float(optimized.min()) >= 0.0
    assert float(optimized.max()) <= 1.0
    np.testing.assert_array_equal(optimized, per_slice)
    np.testing.assert_array_equal(image, image_before)


def test_optimized_3d_morphology_reuses_one_private_scratch_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = np.random.default_rng(29).random((4, 12, 14), dtype=np.float32)
    original_slice = preprocessing_module._morpho_reconstruction_contrast_slice
    scratch_ids: list[tuple[int, int] | None] = []

    def spy_slice(slice_2d: Any, **kwargs: Any) -> Any:
        scratch = kwargs.get("full_resolution_scratch")
        scratch_ids.append(None if scratch is None else (id(scratch[0]), id(scratch[1])))
        return original_slice(cast(Any, slice_2d), **kwargs)

    monkeypatch.setattr(preprocessing_module, "_morpho_reconstruction_contrast_slice", spy_slice)

    _ = preprocessing_module.op_morpho_reconstruction_contrast(
        cast(Any, image),
        {"radius": 2, "downsample_factor": 0.5},
        {},
    )

    assert scratch_ids[0] is None
    assert len(scratch_ids) == image.shape[0]
    assert scratch_ids[1:] == [scratch_ids[1]] * (image.shape[0] - 1)


def test_morphology_invalid_inputs_still_fail_loudly() -> None:
    with pytest.raises(ValueError, match="expects a non-empty 3D stack"):
        _ = op_morpho_reconstruction_contrast(
            cast(Any, np.ones((0, 8, 8), dtype=np.float32)),
            {"radius": 2, "downsample_factor": 0.5},
            {},
        )

    with pytest.raises(ValueError, match="expects downsample_factor > 0"):
        _ = op_morpho_reconstruction_contrast(
            cast(Any, np.ones((8, 8), dtype=np.float32)),
            {"radius": 2, "downsample_factor": 0.0},
            {},
        )


def test_native_kernel_clean_tiffs_match_independent_2d_morphology_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sequence = [
        _step("min_max_normalize"),
        _step("morpho_reconstruction_contrast", {"radius": 2, "downsample_factor": 0.5}),
    ]
    cfg = _config(tmp_path, sequence)
    sanitizer = DataSanitizer(cast(Any, cfg))
    fake_loader = cast(Any, _FakeLoader(_volumes(), tmp_path / "raw"))

    monkeypatch.setitem(
        preprocessing_module.PROCESSOR_MAP,
        "morpho_reconstruction_contrast",
        _independent_2d_morphology,
    )
    baseline = sanitizer._run_native_preprocessing_kernel(
        fov_id=1,
        loader=fake_loader,
        sequence=sequence,
        target_rounds=[1, 2],
        output_writer=_writer_for_root(sanitizer, tmp_path / "baseline", 1),
    )

    monkeypatch.setitem(
        preprocessing_module.PROCESSOR_MAP,
        "morpho_reconstruction_contrast",
        op_morpho_reconstruction_contrast,
    )
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
