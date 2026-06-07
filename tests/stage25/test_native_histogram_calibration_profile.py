from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import tifffile
from numpy.typing import NDArray
from skimage import exposure

from pystar.infrastructure import PreprocessingStep
from pystar.preprocessing import op_histogram_match

from scripts.profile_native_preprocessing_calibration import (
    CALIBRATION_PROFILE_SCHEMA_NAME,
    CALIBRATION_PROFILE_SCHEMA_VERSION,
    HISTOGRAM_MATCH_PROFILE_SCHEMA_NAME,
    HISTOGRAM_MATCH_PROFILE_SCHEMA_VERSION,
    build_profile_payload,
    write_profile_artifacts,
)


def _step(method: str, params: dict[str, Any] | None = None) -> PreprocessingStep:
    return PreprocessingStep(method=method, provider="native", params={} if params is None else params)


def _config(tmp_path: Path, sequence: list[PreprocessingStep]) -> Any:
    dataset = SimpleNamespace(
        raw_data_path=tmp_path / "raw",
        filename_pattern="round{round}/Position{fov}/image_ch{ch}.tif",
        round_structure={1: [0, 1], 2: [0, 1]},
        channel_roles={0: "seq", 1: "seq"},
        dimensions={"z": 1, "height": 3, "width": 4},
        io_chunk_size={"z": 1, "y": 3, "x": 4},
        parsed_fovs=[1],
    )
    pipeline = SimpleNamespace(
        preprocessing=SimpleNamespace(sequence=sequence),
        output=SimpleNamespace(directory=str(tmp_path / "production_out")),
    )
    pipeline.preprocessing_providers_used = lambda: ["native"]
    pipeline.preprocessing_provider_mode = lambda: "native_only"
    return SimpleNamespace(
        dataset=dataset,
        pipeline=pipeline,
        config_sha256="sha256:stage25-test-config",
        config_source_path=tmp_path / "config.yaml",
    )


def _write_raw_fixture(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    for round_id in (1, 2):
        for channel_id in (0, 1):
            path = raw_root / f"round{round_id}" / "Position1" / f"image_ch{channel_id:02d}.tif"
            path.parent.mkdir(parents=True, exist_ok=True)
            base = round_id * 50 + channel_id * 11
            image = (np.arange(12, dtype=np.uint8).reshape(1, 3, 4) + base).astype(np.uint8)
            tifffile.imwrite(path, image)


def _first_fov(payload: Mapping[str, object]) -> Mapping[str, Any]:
    return cast(Sequence[Mapping[str, Any]], payload["fovs"])[0]


def _histogram_sequence() -> list[PreprocessingStep]:
    return [
        _step("min_max_normalize"),
        _step("histogram_match", {"scope": "inter_round"}),
        _step("histogram_match", {"scope": "intra_round"}),
    ]


def test_histogram_profile_attributes_scope_reference_dtype_shape_and_noops(tmp_path: Path) -> None:
    _write_raw_fixture(tmp_path)
    cfg = _config(tmp_path, _histogram_sequence())

    payload = build_profile_payload(
        config=cast(Any, cfg),
        config_path=tmp_path / "config.yaml",
        fov_ids=(1,),
        target_rounds=(1, 2),
        repeats=1,
        output_dir=tmp_path / "profile_artifacts",
    )

    fov_payload = _first_fov(payload)
    summary = cast(Mapping[str, Any], fov_payload["summary"])
    profile = cast(Mapping[str, Any], summary["histogram_match_profile"])
    assert profile["schema_name"] == HISTOGRAM_MATCH_PROFILE_SCHEMA_NAME
    assert profile["schema_version"] == HISTOGRAM_MATCH_PROFILE_SCHEMA_VERSION
    assert profile["call_count"] == 8
    assert profile["match_call_count"] == 4
    assert profile["no_reference_call_count"] == 4

    by_scope = cast(Mapping[str, Mapping[str, Any]], profile["by_scope"])
    assert set(by_scope) == {"inter_round", "intra_round"}
    for scope in ("inter_round", "intra_round"):
        scope_summary = by_scope[scope]
        assert scope_summary["call_count"] == 4
        assert scope_summary["match_call_count"] == 2
        assert scope_summary["no_reference_call_count"] == 2
        assert scope_summary["input_dtypes"] == ["float32"]
        assert scope_summary["output_dtypes"] == ["float32"]
        assert scope_summary["reference_dtypes"] == ["float32"]
        assert scope_summary["input_shapes"] == [[3, 4]]
        assert scope_summary["output_shapes"] == [[3, 4]]
        assert scope_summary["reference_shapes"] == [[3, 4]]

    repeat = cast(Sequence[Mapping[str, Any]], fov_payload["repeats"])[0]
    calls = cast(Sequence[Mapping[str, Any]], repeat["histogram_match_calls"])
    assert len(calls) == 8
    no_reference_calls = [call for call in calls if not bool(call["has_reference"])]
    matched_calls = [call for call in calls if bool(call["has_reference"])]
    assert {call["operation"] for call in no_reference_calls} == {"no_reference_noop"}
    assert {call["operation"] for call in matched_calls} == {"match_histograms"}
    assert all(call["output_is_input"] is True for call in no_reference_calls)
    assert all(call["output_is_input"] is False for call in matched_calls)
    assert all(call["reference_dtype"] is None for call in no_reference_calls)
    assert all(call["reference_shape"] is None for call in no_reference_calls)


def test_histogram_profile_preserves_byte_equivalent_clean_repeat_outputs(tmp_path: Path) -> None:
    _write_raw_fixture(tmp_path)
    cfg = _config(tmp_path, _histogram_sequence())

    payload = build_profile_payload(
        config=cast(Any, cfg),
        config_path=tmp_path / "config.yaml",
        fov_ids=(1,),
        target_rounds=(1, 2),
        repeats=2,
        output_dir=tmp_path / "profile_artifacts",
    )

    fov_payload = _first_fov(payload)
    equivalence = cast(Mapping[str, Any], fov_payload["clean_output_equivalence"])
    assert equivalence["status"] == "equivalent"
    assert equivalence["file_count"] == 4
    assert equivalence["mismatches"] == []
    assert set(cast(Mapping[str, object], equivalence["baseline_files"])) == {
        "clean_fov_1_round_1_ch_0.tif",
        "clean_fov_1_round_1_ch_1.tif",
        "clean_fov_1_round_2_ch_0.tif",
        "clean_fov_1_round_2_ch_1.tif",
    }

    repeats = cast(Sequence[Mapping[str, Any]], fov_payload["repeats"])
    assert len(repeats) == 2
    for repeat in repeats:
        assert cast(Mapping[str, Any], repeat["histogram_match_profile"])["call_count"] == 8
        output_files = [Path(path) for path in cast(Sequence[str], repeat["output_files"])]
        assert all(path.exists() for path in output_files)
        assert all(path.name.startswith("clean_fov_1_round_") for path in output_files)

    assert not Path(cfg.pipeline.output.directory).exists()


def test_histogram_profile_markdown_reports_top_level_and_histogram_schemas(tmp_path: Path) -> None:
    _write_raw_fixture(tmp_path)
    cfg = _config(tmp_path, _histogram_sequence())
    output_dir = tmp_path / "profile_artifacts"

    payload = build_profile_payload(
        config=cast(Any, cfg),
        config_path=tmp_path / "config.yaml",
        fov_ids=(1,),
        target_rounds=(1, 2),
        repeats=1,
        output_dir=output_dir,
    )

    _json_path, markdown_path = write_profile_artifacts(payload, output_dir)
    markdown = markdown_path.read_text(encoding="utf-8")

    assert (
        f"Calibration profile schema: `{CALIBRATION_PROFILE_SCHEMA_NAME}` "
        f"v`{CALIBRATION_PROFILE_SCHEMA_VERSION}`"
    ) in markdown
    assert (
        f"Histogram profile schema: `{HISTOGRAM_MATCH_PROFILE_SCHEMA_NAME}` "
        f"v`{HISTOGRAM_MATCH_PROFILE_SCHEMA_VERSION}`"
    ) in markdown


def test_histogram_match_copy_false_keeps_float32_result_without_extra_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = np.linspace(0.0, 1.0, 12, dtype=np.float32).reshape(3, 4)
    reference = np.flip(image, axis=1).copy()
    matched = np.full_like(image, 0.25, dtype=np.float32)

    def fake_match_histograms(img: object, ref_img: object) -> NDArray[np.float32]:
        assert img is image
        assert ref_img is reference
        return matched

    monkeypatch.setattr("pystar.preprocessing.exposure.match_histograms", fake_match_histograms)

    result = op_histogram_match(image, {"scope": "inter_round"}, {"ref_round_image": reference})

    assert result is matched
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, matched)


def test_histogram_match_copy_false_is_exact_equivalent_for_real_float32_inputs() -> None:
    image = np.linspace(0.0, 1.0, 12, dtype=np.float32).reshape(3, 4)
    reference = np.flip(image, axis=1).copy()
    image_before = image.copy()
    reference_before = reference.copy()

    raw_expected = exposure.match_histograms(image, reference)
    expected: NDArray[np.float32] = np.asarray(raw_expected, dtype=np.float32)
    result = op_histogram_match(image, {"scope": "inter_round"}, {"ref_round_image": reference})

    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(image, image_before)
    np.testing.assert_array_equal(reference, reference_before)


def test_histogram_match_no_reference_returns_input_without_copy() -> None:
    image = np.linspace(0.0, 1.0, 12, dtype=np.float32).reshape(3, 4)

    result = op_histogram_match(image, {"scope": "inter_round"}, {"ref_round_image": None})

    assert result is image
