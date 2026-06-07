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
from pystar.preprocessing import DataSanitizer, op_morpho_reconstruction_contrast


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
        dimensions={"z": 3, "height": 6, "width": 7},
        io_chunk_size={"z": 3, "y": 6, "x": 7},
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
            base = round_id * 29 + channel_id * 11
            volumes[(round_id, channel_id)] = (
                np.arange(126, dtype=np.uint16).reshape(3, 6, 7) + base
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


def test_morphology_workers_omitted_and_one_are_the_serial_oracle() -> None:
    image = np.random.default_rng(2801).random((4, 24, 32), dtype=np.float32)
    image_before = image.copy()
    base_params = {"radius": 3, "downsample_factor": 0.5}

    omitted = op_morpho_reconstruction_contrast(cast(Any, image), base_params, {})
    workers_one = op_morpho_reconstruction_contrast(
        cast(Any, image),
        {**base_params, "workers": 1},
        {},
    )

    assert omitted.dtype == np.float32
    assert omitted.shape == image.shape
    np.testing.assert_array_equal(workers_one, omitted)
    np.testing.assert_array_equal(image, image_before)


@pytest.mark.parametrize("workers", [2, 8])
def test_morphology_parallel_workers_match_serial_oracle(workers: int) -> None:
    image = np.random.default_rng(2802).random((5, 24, 32), dtype=np.float32)
    image_before = image.copy()
    base_params = {"radius": 3, "downsample_factor": 0.5}

    serial = op_morpho_reconstruction_contrast(cast(Any, image), base_params, {})
    parallel = op_morpho_reconstruction_contrast(
        cast(Any, image),
        {**base_params, "workers": workers},
        {},
    )

    assert parallel.dtype == serial.dtype == np.float32
    assert parallel.shape == serial.shape == image.shape
    assert np.isfinite(parallel).all()
    assert float(parallel.min()) >= 0.0
    assert float(parallel.max()) <= 1.0
    np.testing.assert_array_equal(parallel, serial)
    np.testing.assert_array_equal(image, image_before)


def test_morphology_parallel_path_handles_non_contiguous_input_without_drift() -> None:
    base = np.random.default_rng(2803).random((5, 32, 36), dtype=np.float32)
    image = base[:, ::2, 1::2]
    image_before = image.copy()
    assert not image.flags.c_contiguous
    params = {"radius": 2, "downsample_factor": 0.5}

    serial = op_morpho_reconstruction_contrast(cast(Any, image), params, {})
    parallel = op_morpho_reconstruction_contrast(
        cast(Any, image),
        {**params, "workers": 3},
        {},
    )

    assert parallel.shape == serial.shape == image.shape
    np.testing.assert_array_equal(parallel, serial)
    np.testing.assert_array_equal(image, image_before)


def test_morphology_single_slice_stack_accepts_parallel_worker_parameter() -> None:
    image = np.random.default_rng(2806).random((1, 24, 32), dtype=np.float32)
    params = {"radius": 3, "downsample_factor": 0.5}

    serial = op_morpho_reconstruction_contrast(cast(Any, image), params, {})
    parallel_requested = op_morpho_reconstruction_contrast(
        cast(Any, image),
        {**params, "workers": 4},
        {},
    )

    assert parallel_requested.shape == serial.shape == image.shape
    np.testing.assert_array_equal(parallel_requested, serial)


def test_morphology_empty_stack_contract_is_unchanged_with_parallel_request() -> None:
    image = np.ones((0, 16, 16), dtype=np.float32)

    with pytest.raises(ValueError, match="expects a non-empty 3D stack"):
        _ = op_morpho_reconstruction_contrast(
            cast(Any, image),
            {"radius": 2, "downsample_factor": 0.5, "workers": 2},
            {},
        )


def test_morphology_2d_behavior_is_unchanged_by_worker_parameter() -> None:
    image = np.random.default_rng(2804).random((24, 32), dtype=np.float32)
    image_before = image.copy()
    params = {"radius": 3, "downsample_factor": 0.5}

    baseline = op_morpho_reconstruction_contrast(cast(Any, image), params, {})
    with_workers = op_morpho_reconstruction_contrast(
        cast(Any, image),
        {**params, "workers": 4},
        {},
    )

    assert baseline.dtype == with_workers.dtype == np.float32
    assert baseline.shape == with_workers.shape == image.shape
    np.testing.assert_array_equal(with_workers, baseline)
    np.testing.assert_array_equal(image, image_before)


@pytest.mark.parametrize("workers", [0, -1, True, 1.5, "2"])
def test_morphology_parallel_workers_fail_loudly_for_invalid_values(workers: Any) -> None:
    image = np.ones((2, 8, 8), dtype=np.float32)

    with pytest.raises(ValueError, match="workers must be a positive integer"):
        _ = op_morpho_reconstruction_contrast(
            cast(Any, image),
            {"radius": 2, "downsample_factor": 0.5, "workers": workers},
            {},
        )


def test_morphology_parallel_path_uses_private_scratch_per_remaining_slice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = np.random.default_rng(2805).random((4, 12, 14), dtype=np.float32)
    original_scratch = preprocessing_module._morpho_reconstruction_contrast_scratch
    retained_scratch: list[tuple[NDArray[Any], NDArray[Any]]] = []

    def spy_scratch(*, shape: tuple[int, int], dtype: np.dtype[Any]):
        scratch = original_scratch(shape=shape, dtype=dtype)
        retained_scratch.append(cast(tuple[NDArray[Any], NDArray[Any]], scratch))
        return scratch

    monkeypatch.setattr(preprocessing_module, "_morpho_reconstruction_contrast_scratch", spy_scratch)

    serial = op_morpho_reconstruction_contrast(
        cast(Any, image),
        {"radius": 2, "downsample_factor": 0.5},
        {},
    )
    retained_scratch.clear()
    parallel = op_morpho_reconstruction_contrast(
        cast(Any, image),
        {"radius": 2, "downsample_factor": 0.5, "workers": 2},
        {},
    )

    assert len(retained_scratch) == image.shape[0] - 1
    scratch_id_pairs = [(id(first), id(second)) for first, second in retained_scratch]
    assert len(set(scratch_id_pairs)) == len(scratch_id_pairs)
    for first, second in retained_scratch:
        assert not np.may_share_memory(first, second)
        assert not np.may_share_memory(first, image)
        assert not np.may_share_memory(second, image)
    np.testing.assert_array_equal(parallel, serial)


def test_native_kernel_clean_tiffs_match_serial_and_parallel_morphology(
    tmp_path: Path,
) -> None:
    serial_sequence = [
        _step("min_max_normalize"),
        _step("morpho_reconstruction_contrast", {"radius": 2, "downsample_factor": 0.5}),
    ]
    parallel_sequence = [
        _step("min_max_normalize"),
        _step(
            "morpho_reconstruction_contrast",
            {"radius": 2, "downsample_factor": 0.5, "workers": 2},
        ),
    ]
    serial_cfg = _config(tmp_path, serial_sequence)
    parallel_cfg = _config(tmp_path, parallel_sequence)
    serial_sanitizer = DataSanitizer(cast(Any, serial_cfg))
    parallel_sanitizer = DataSanitizer(cast(Any, parallel_cfg))
    fake_loader = cast(Any, _FakeLoader(_volumes(), tmp_path / "raw"))

    serial = serial_sanitizer._run_native_preprocessing_kernel(
        fov_id=1,
        loader=fake_loader,
        sequence=serial_sequence,
        target_rounds=[1, 2],
        output_writer=_writer_for_root(serial_sanitizer, tmp_path / "serial", 1),
    )
    parallel = parallel_sanitizer._run_native_preprocessing_kernel(
        fov_id=1,
        loader=fake_loader,
        sequence=parallel_sequence,
        target_rounds=[1, 2],
        output_writer=_writer_for_root(parallel_sanitizer, tmp_path / "parallel", 1),
    )

    serial_files = {Path(path).name: Path(path) for path in cast(list[str], serial["output_files"])}
    parallel_files = {Path(path).name: Path(path) for path in cast(list[str], parallel["output_files"])}
    assert set(serial_files) == set(parallel_files) == {
        "clean_fov_1_round_1_ch_0.tif",
        "clean_fov_1_round_1_ch_1.tif",
        "clean_fov_1_round_2_ch_0.tif",
        "clean_fov_1_round_2_ch_1.tif",
    }

    for name, serial_path in serial_files.items():
        parallel_path = parallel_files[name]
        assert _sha256_file(parallel_path) == _sha256_file(serial_path)
        serial_array = tifffile.imread(serial_path)
        parallel_array = tifffile.imread(parallel_path)
        assert serial_array.dtype == np.uint8
        assert parallel_array.dtype == np.uint8
        np.testing.assert_array_equal(parallel_array, serial_array)

    assert not Path(serial_cfg.pipeline.output.directory).exists()
    assert not Path(parallel_cfg.pipeline.output.directory).exists()
