from __future__ import annotations

import hashlib
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import tifffile
from numpy.typing import NDArray

from pystar import preprocessing as preprocessing_module
from pystar.infrastructure import PreprocessingConfig, PreprocessingStep
from pystar.io import get_fov_output_structure
from pystar.preprocessing import DataSanitizer


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


def _config(
    tmp_path: Path,
    sequence: list[PreprocessingStep],
    *,
    native_volume_workers: int | None = None,
) -> Any:
    preprocessing_attrs: dict[str, Any] = {"sequence": sequence}
    if native_volume_workers is not None:
        preprocessing_attrs["native_volume_workers"] = native_volume_workers
    dataset = SimpleNamespace(
        raw_data_path=tmp_path / "raw",
        filename_pattern="round{round}/Position{fov}/*_ch{ch}.tif",
        round_structure={1: [0, 1, 2], 2: [0, 1, 2]},
        channel_roles={0: "seq", 1: "seq", 2: "seq"},
        dimensions={"z": 1, "height": 5, "width": 6},
        io_chunk_size={"z": 1, "y": 5, "x": 6},
    )
    pipeline = SimpleNamespace(
        preprocessing=SimpleNamespace(**preprocessing_attrs),
        output=SimpleNamespace(directory=str(tmp_path / "production_out")),
    )
    pipeline.preprocessing_providers_used = lambda: ["native"]
    pipeline.preprocessing_provider_mode = lambda: "native_only"
    return SimpleNamespace(dataset=dataset, pipeline=pipeline)


def _histogram_sequence() -> list[PreprocessingStep]:
    return [
        _step("min_max_normalize"),
        _step("histogram_match", {"scope": "inter_round"}),
        _step("histogram_match", {"scope": "intra_round"}),
    ]


def _volumes() -> dict[tuple[int, int], NDArray[Any]]:
    volumes: dict[tuple[int, int], NDArray[Any]] = {}
    for round_id in (1, 2):
        for channel_id in (0, 1, 2):
            base = round_id * 41 + channel_id * 13
            volumes[(round_id, channel_id)] = (
                np.arange(30, dtype=np.uint16).reshape(1, 5, 6) + base
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

    def planned_output_path(round_id: int, channel_id: int) -> Path:
        paths = get_fov_output_structure(output_root, fov_id)
        return paths["cleaned"] / sanitizer._flat_clean_filename(fov_id, round_id, channel_id)

    return preprocessing_module.NativeOutputWriterWithPlanner(
        write=write,
        output_path_for=planned_output_path,
    )


def test_native_volume_workers_default_path_remains_serial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sequence = [_step("none")]
    cfg = _config(tmp_path, sequence)
    sanitizer = DataSanitizer(cast(Any, cfg))

    def fail_if_parallel(*args: Any, **kwargs: Any) -> list[Any]:
        _ = (args, kwargs)
        raise AssertionError("default native preprocessing path must not enter volume-parallel executor")

    monkeypatch.setattr(sanitizer, "_run_native_volume_parallel", fail_if_parallel)

    result = sanitizer._run_native_preprocessing_kernel(
        fov_id=1,
        loader=cast(Any, _FakeLoader(_volumes(), tmp_path / "raw")),
        sequence=sequence,
        target_rounds=[1, 2],
        output_writer=_writer_for_root(sanitizer, tmp_path / "serial_default", 1),
    )

    assert [
        (volume["round_id"], volume["channel_id"])
        for volume in result["preprocessing_timing"]["volumes"]
    ] == [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]


@pytest.mark.parametrize("native_volume_workers", [None, 1])
def test_native_serial_missing_inter_round_reference_preserves_histogram_noop(
    tmp_path: Path,
    native_volume_workers: int | None,
) -> None:
    sequence = [_step("min_max_normalize"), _step("histogram_match", {"scope": "inter_round"})]
    oracle_sequence = [_step("min_max_normalize")]
    fake_loader = cast(Any, _FakeLoader(_volumes(), tmp_path / "raw"))
    cfg = _config(tmp_path, sequence, native_volume_workers=native_volume_workers)
    oracle_cfg = _config(tmp_path, oracle_sequence)
    sanitizer = DataSanitizer(cast(Any, cfg))
    oracle_sanitizer = DataSanitizer(cast(Any, oracle_cfg))

    result = sanitizer._run_native_preprocessing_kernel(
        fov_id=1,
        loader=fake_loader,
        sequence=sequence,
        target_rounds=[2],
        output_writer=_writer_for_root(sanitizer, tmp_path / "serial_missing_inter_reference", 1),
    )
    oracle = oracle_sanitizer._run_native_preprocessing_kernel(
        fov_id=1,
        loader=fake_loader,
        sequence=oracle_sequence,
        target_rounds=[2],
        output_writer=_writer_for_root(oracle_sanitizer, tmp_path / "serial_missing_inter_reference_oracle", 1),
    )

    assert [
        (volume["round_id"], volume["channel_id"])
        for volume in result["preprocessing_timing"]["volumes"]
    ] == [(2, 0), (2, 1), (2, 2)]

    result_files = {Path(path).name: Path(path) for path in cast(list[str], result["output_files"])}
    oracle_files = {Path(path).name: Path(path) for path in cast(list[str], oracle["output_files"])}
    assert list(result_files) == list(oracle_files)
    for name, result_path in result_files.items():
        np.testing.assert_array_equal(tifffile.imread(result_path), tifffile.imread(oracle_files[name]))


def test_native_preprocessing_plan_records_volume_dependency_contracts(tmp_path: Path) -> None:
    sequence = _histogram_sequence()
    cfg = _config(tmp_path, sequence, native_volume_workers=2)
    sanitizer = DataSanitizer(cast(Any, cfg))
    calibration_steps, extraction_steps = sanitizer._split_sequence(sequence)

    plan = sanitizer._build_native_preprocessing_plan(
        fov_id=1,
        loader=cast(Any, _FakeLoader(_volumes(), tmp_path / "raw")),
        calibration_steps=calibration_steps,
        extraction_steps=extraction_steps,
        final_queue=[1, 2],
        output_writer=sanitizer._make_canonical_output_writer(1),
        worker_count=2,
    )

    assert plan.worker_count == 2
    assert plan.round_order == (1, 2)
    assert [(item.key.round_id, item.key.channel_id) for item in plan.items] == [
        (1, 0),
        (1, 1),
        (1, 2),
        (2, 0),
        (2, 1),
        (2, 2),
    ]

    by_key = {item.key: item for item in plan.items}
    round2_channel1 = by_key[preprocessing_module.NativeVolumeKey(round_id=2, channel_id=1)]
    assert round2_channel1.input_path == tmp_path / "raw" / "round2" / "ch1.tif"
    assert round2_channel1.planned_output_path == (
        Path(cfg.pipeline.output.directory)
        / "Position1"
        / "output_pystar"
        / "clean_data"
        / "clean_fov_1_round_2_ch_1.tif"
    )
    assert round2_channel1.required_reference_keys == (
        preprocessing_module.NativeReferenceKey(
            role="inter_round",
            volume_key=preprocessing_module.NativeVolumeKey(round_id=1, channel_id=1),
        ),
        preprocessing_module.NativeReferenceKey(
            role="intra_round",
            volume_key=preprocessing_module.NativeVolumeKey(round_id=2, channel_id=0),
        ),
    )

    round1_channel0 = by_key[preprocessing_module.NativeVolumeKey(round_id=1, channel_id=0)]
    assert round1_channel0.produces_inter_round_reference is True
    assert round1_channel0.produces_intra_round_reference is True


def test_native_volume_parallel_outputs_match_serial_reference_oracle(tmp_path: Path) -> None:
    sequence = _histogram_sequence()
    fake_loader = cast(Any, _FakeLoader(_volumes(), tmp_path / "raw"))
    serial_cfg = _config(tmp_path, sequence)
    parallel_cfg = _config(tmp_path, sequence, native_volume_workers=3)
    serial_sanitizer = DataSanitizer(cast(Any, serial_cfg))
    parallel_sanitizer = DataSanitizer(cast(Any, parallel_cfg))

    serial = serial_sanitizer._run_native_preprocessing_kernel(
        fov_id=1,
        loader=fake_loader,
        sequence=sequence,
        target_rounds=[1, 2],
        output_writer=_writer_for_root(serial_sanitizer, tmp_path / "serial", 1),
    )
    parallel = parallel_sanitizer._run_native_preprocessing_kernel(
        fov_id=1,
        loader=fake_loader,
        sequence=sequence,
        target_rounds=[1, 2],
        output_writer=_writer_for_root(parallel_sanitizer, tmp_path / "parallel", 1),
    )

    serial_files = {Path(path).name: Path(path) for path in cast(list[str], serial["output_files"])}
    parallel_files = {Path(path).name: Path(path) for path in cast(list[str], parallel["output_files"])}
    assert list(serial_files) == list(parallel_files) == [
        "clean_fov_1_round_1_ch_0.tif",
        "clean_fov_1_round_1_ch_1.tif",
        "clean_fov_1_round_1_ch_2.tif",
        "clean_fov_1_round_2_ch_0.tif",
        "clean_fov_1_round_2_ch_1.tif",
        "clean_fov_1_round_2_ch_2.tif",
    ]

    for name, serial_path in serial_files.items():
        parallel_path = parallel_files[name]
        assert _sha256_file(parallel_path) == _sha256_file(serial_path)
        np.testing.assert_array_equal(tifffile.imread(parallel_path), tifffile.imread(serial_path))


def test_native_volume_parallel_result_order_is_deterministic_after_reordered_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sequence = _histogram_sequence()
    cfg = _config(tmp_path, sequence, native_volume_workers=3)
    sanitizer = DataSanitizer(cast(Any, cfg))
    original = sanitizer._load_and_run_native_volume_work_item
    started: list[tuple[int, int]] = []

    def spy_load_and_run(*, item: Any, **kwargs: Any) -> Any:
        started.append((item.key.round_id, item.key.channel_id))
        if item.key == preprocessing_module.NativeVolumeKey(round_id=1, channel_id=1):
            time.sleep(0.05)
        return original(item=item, **kwargs)

    monkeypatch.setattr(sanitizer, "_load_and_run_native_volume_work_item", spy_load_and_run)
    result = sanitizer._run_native_preprocessing_kernel(
        fov_id=1,
        loader=cast(Any, _FakeLoader(_volumes(), tmp_path / "raw")),
        sequence=sequence,
        target_rounds=[1, 2],
        output_writer=_writer_for_root(sanitizer, tmp_path / "parallel_order", 1),
    )

    observed_records = [
        (volume["round_id"], volume["channel_id"])
        for volume in result["preprocessing_timing"]["volumes"]
    ]
    assert observed_records == [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    assert started[0] == (1, 0)
    assert started.index((2, 1)) > started.index((1, 1))
    assert started.index((2, 1)) > started.index((2, 0))
    assert started.index((2, 2)) > started.index((1, 2))
    assert started.index((2, 2)) > started.index((2, 0))


@pytest.mark.parametrize("workers", [0, -1, True, 1.5, "2"])
def test_native_volume_worker_config_rejects_invalid_values(workers: Any) -> None:
    with pytest.raises(ValueError, match="native_volume_workers must be a positive integer"):
        _ = PreprocessingConfig(sequence=[_step("none")], native_volume_workers=workers)


def test_native_reference_plan_fails_loudly_when_required_reference_round_is_absent(tmp_path: Path) -> None:
    sequence = _histogram_sequence()
    cfg = _config(tmp_path, sequence, native_volume_workers=2)
    sanitizer = DataSanitizer(cast(Any, cfg))

    with pytest.raises(ValueError, match="reference dependency plan is impossible"):
        _ = sanitizer._run_native_preprocessing_kernel(
            fov_id=1,
            loader=cast(Any, _FakeLoader(_volumes(), tmp_path / "raw")),
            sequence=sequence,
            target_rounds=[2],
            output_writer=_writer_for_root(sanitizer, tmp_path / "missing_reference", 1),
        )


def test_native_parallel_plan_requires_known_output_paths(tmp_path: Path) -> None:
    sequence = [_step("none")]
    cfg = _config(tmp_path, sequence, native_volume_workers=2)
    sanitizer = DataSanitizer(cast(Any, cfg))

    def write(img: Any, round_id: int, channel_id: int) -> Path:
        _ = (img, round_id, channel_id)
        raise AssertionError("plan validation must fail before writing")

    with pytest.raises(ValueError, match="requires planned output paths"):
        _ = sanitizer._run_native_preprocessing_kernel(
            fov_id=1,
            loader=cast(Any, _FakeLoader(_volumes(), tmp_path / "raw")),
            sequence=sequence,
            target_rounds=[1],
            output_writer=write,
        )


def test_native_parallel_plan_rejects_duplicate_output_paths(tmp_path: Path) -> None:
    sequence = [_step("none")]
    cfg = _config(tmp_path, sequence, native_volume_workers=2)
    sanitizer = DataSanitizer(cast(Any, cfg))

    def write(img: Any, round_id: int, channel_id: int) -> Path:
        _ = (img, round_id, channel_id)
        raise AssertionError("plan validation must fail before writing")

    output_writer = preprocessing_module.NativeOutputWriterWithPlanner(
        write=write,
        output_path_for=lambda _round_id, _channel_id: tmp_path / "same_clean.tif",
    )

    with pytest.raises(ValueError, match="duplicate output paths"):
        _ = sanitizer._run_native_preprocessing_kernel(
            fov_id=1,
            loader=cast(Any, _FakeLoader(_volumes(), tmp_path / "raw")),
            sequence=sequence,
            target_rounds=[1],
            output_writer=output_writer,
        )
