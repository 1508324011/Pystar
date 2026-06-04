from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import tifffile
from skimage import exposure

from pystar.infrastructure import PreprocessingStep
from pystar.preprocessing import op_histogram_match

from scripts.profile_native_preprocessing_calibration import (
    HISTOGRAM_REAL_MATCH_ATTRIBUTION_SCHEMA_NAME,
    HISTOGRAM_REAL_MATCH_ATTRIBUTION_SCHEMA_VERSION,
    _summarize_histogram_calls,
    build_profile_payload,
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
        config_sha256="sha256:stage29-test-config",
        config_source_path=tmp_path / "config.yaml",
    )


def _write_raw_fixture(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    for round_id in (1, 2):
        for channel_id in (0, 1):
            path = raw_root / f"round{round_id}" / "Position1" / f"image_ch{channel_id:02d}.tif"
            path.parent.mkdir(parents=True, exist_ok=True)
            base = round_id * 37 + channel_id * 13
            image = (np.arange(12, dtype=np.uint8).reshape(1, 3, 4) + base).astype(np.uint8)
            tifffile.imwrite(path, image)


def _histogram_sequence() -> list[PreprocessingStep]:
    return [
        _step("min_max_normalize"),
        _step("histogram_match", {"scope": "inter_round"}),
        _step("histogram_match", {"scope": "intra_round"}),
    ]


def test_histogram_match_no_reference_is_identity_and_does_not_mutate_input() -> None:
    image = np.linspace(0.0, 1.0, 12, dtype=np.float32).reshape(3, 4)
    before = image.copy()

    result = op_histogram_match(image, {"scope": "inter_round"}, {"ref_round_image": None})

    assert result is image
    np.testing.assert_array_equal(image, before)


def test_histogram_match_real_reference_path_matches_skimage_baseline() -> None:
    image = np.linspace(0.0, 1.0, 12, dtype=np.float32).reshape(3, 4)
    reference = np.flip(image, axis=1).copy()
    image_before = image.copy()
    reference_before = reference.copy()

    expected = np.asarray(exposure.match_histograms(image, reference), dtype=np.float32)
    result = op_histogram_match(image, {"scope": "inter_round"}, {"ref_round_image": reference})

    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(image, image_before)
    np.testing.assert_array_equal(reference, reference_before)


def test_histogram_real_match_attribution_separates_real_work_from_noops() -> None:
    profile = _summarize_histogram_calls(
        [
            {
                "call_index": 0,
                "scope": "inter_round",
                "has_reference": False,
                "operation": "no_reference_noop",
                "duration_ms": 1.0,
                "input_dtype": "float32",
                "input_shape": [3, 4],
                "reference_dtype": None,
                "reference_shape": None,
                "output_dtype": "float32",
                "output_shape": [3, 4],
                "output_is_input": True,
            },
            {
                "call_index": 1,
                "scope": "inter_round",
                "has_reference": True,
                "operation": "match_histograms",
                "duration_ms": 10.0,
                "input_dtype": "float32",
                "input_shape": [3, 4],
                "reference_dtype": "float32",
                "reference_shape": [3, 4],
                "output_dtype": "float32",
                "output_shape": [3, 4],
                "output_is_input": False,
            },
            {
                "call_index": 2,
                "scope": "intra_round",
                "has_reference": False,
                "operation": "no_reference_noop",
                "duration_ms": 2.0,
                "input_dtype": "float32",
                "input_shape": [3, 4],
                "reference_dtype": None,
                "reference_shape": None,
                "output_dtype": "float32",
                "output_shape": [3, 4],
                "output_is_input": True,
            },
            {
                "call_index": 3,
                "scope": "intra_round",
                "has_reference": True,
                "operation": "match_histograms",
                "duration_ms": 30.0,
                "input_dtype": "float32",
                "input_shape": [3, 4],
                "reference_dtype": "float32",
                "reference_shape": [3, 4],
                "output_dtype": "float32",
                "output_shape": [3, 4],
                "output_is_input": False,
            },
        ]
    )

    attribution = cast(Mapping[str, Any], profile["real_match_attribution"])
    assert attribution["schema_name"] == HISTOGRAM_REAL_MATCH_ATTRIBUTION_SCHEMA_NAME
    assert attribution["schema_version"] == HISTOGRAM_REAL_MATCH_ATTRIBUTION_SCHEMA_VERSION
    assert attribution["call_count"] == 4
    assert attribution["real_match_call_count"] == 2
    assert attribution["no_reference_noop_call_count"] == 2
    assert attribution["real_match_total_duration_ms"] == 40.0
    assert attribution["real_match_median_duration_ms"] == 20.0
    assert attribution["no_reference_noop_total_duration_ms"] == 3.0
    assert attribution["no_reference_noop_median_duration_ms"] == 1.5

    by_scope = cast(Mapping[str, Mapping[str, Any]], attribution["by_scope"])
    assert by_scope["inter_round"]["real_match_total_duration_ms"] == 10.0
    assert by_scope["inter_round"]["no_reference_noop_total_duration_ms"] == 1.0
    assert by_scope["intra_round"]["real_match_total_duration_ms"] == 30.0
    assert by_scope["intra_round"]["no_reference_noop_total_duration_ms"] == 2.0
    assert by_scope["inter_round"]["reference_dtypes"] == ["float32"]
    assert by_scope["intra_round"]["reference_shapes"] == [[3, 4]]


def test_stage29_profile_payload_includes_real_match_attribution_without_production_output(
    tmp_path: Path,
) -> None:
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

    fov_payload = cast(Sequence[Mapping[str, Any]], payload["fovs"])[0]
    aggregate_profile = cast(Mapping[str, Any], cast(Mapping[str, Any], fov_payload["summary"])["histogram_match_profile"])
    attribution = cast(Mapping[str, Any], aggregate_profile["real_match_attribution"])

    assert attribution["call_count"] == 8
    assert attribution["real_match_call_count"] == 4
    assert attribution["no_reference_noop_call_count"] == 4
    assert cast(Mapping[str, Any], attribution["real_match_duration_ms"])["count"] == 4
    assert cast(Mapping[str, Any], attribution["no_reference_noop_duration_ms"])["count"] == 4
    by_scope = cast(Mapping[str, Mapping[str, Any]], attribution["by_scope"])
    assert by_scope["inter_round"]["real_match_call_count"] == 2
    assert by_scope["inter_round"]["no_reference_noop_call_count"] == 2
    assert by_scope["intra_round"]["real_match_call_count"] == 2
    assert by_scope["intra_round"]["no_reference_noop_call_count"] == 2

    repeat_profile = cast(
        Mapping[str, Any],
        cast(Sequence[Mapping[str, Any]], fov_payload["repeats"])[0]["histogram_match_profile"],
    )
    assert cast(Mapping[str, Any], repeat_profile["real_match_attribution"])["real_match_call_count"] == 4
    assert not Path(cfg.pipeline.output.directory).exists()
