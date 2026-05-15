from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import yaml

from pystar._performance_telemetry import (
    PERFORMANCE_TELEMETRY_SCHEMA_NAME,
    PERFORMANCE_TELEMETRY_SCHEMA_VERSION,
    record_stage_timing,
    write_performance_telemetry,
)
from pystar.io import get_fov_output_structure
from pystar.serialization import write_backend_metadata


# Tests intentionally use small config-like namespaces instead of full
# ExperimentConfig fixtures so they can isolate the telemetry contract.
# pyright: reportExplicitAny=false, reportAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false


FOV_ID = 3


def _config(output_dir: Path) -> Any:
    pipeline = SimpleNamespace(
        output=SimpleNamespace(directory=str(output_dir)),
        preprocessing=SimpleNamespace(
            sequence=[
                SimpleNamespace(method="none", provider="native", params={}),
                SimpleNamespace(method="min_max_normalize", provider="matlab", params={}),
            ]
        ),
        registration=SimpleNamespace(
            global_provider="matlab",
            local_provider="matlab",
            local_method="demons_3d",
            enable_local=True,
        ),
        spot_finding=SimpleNamespace(provider="matlab", algorithm="peak_local_max"),
        extraction=SimpleNamespace(provider="matlab", method="box_sum", transform_application_mode="image_warp"),
        decoding=SimpleNamespace(gating_mode="pattern_first"),
    )
    pipeline.preprocessing_providers_used = lambda: ["matlab", "native"]
    pipeline.preprocessing_provider_mode = lambda: "mixed"
    pipeline.registration_provider_mode = lambda: "matlab_only"
    return SimpleNamespace(
        config_source_path=Path("/tmp/example_config.yaml"),
        config_sha256="sha256:abc123",
        pipeline=pipeline,
    )


def _write_csv(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _read_json_mapping(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return cast(Mapping[str, Any], payload)


def _stage_timings() -> list[dict[str, object]]:
    return [
        record_stage_timing("preprocessing", 100.0),
        record_stage_timing("registration", 200.0),
        record_stage_timing("spot_finding", 300.0),
        record_stage_timing("signal_extraction", 400.0),
        record_stage_timing("decoding", 500.0),
    ]


def test_record_stage_timing_validates_stage_contract() -> None:
    timing = record_stage_timing("registration", 12.3456)

    assert timing == {
        "stage_id": "registration",
        "order_index": 2,
        "display_label": "Registration",
        "elapsed_wall_ms": 12.346,
        "status": "completed",
    }

    with pytest.raises(ValueError, match="unknown stage ID"):
        _ = record_stage_timing("not_a_stage", 1.0)

    with pytest.raises(ValueError, match="non-negative"):
        _ = record_stage_timing("registration", -1.0)

    with pytest.raises(ValueError, match="finite"):
        _ = record_stage_timing("registration", float("nan"))


def test_write_performance_telemetry_summarizes_artifacts_and_matlab_metadata(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    paths = get_fov_output_structure(tmp_path, FOV_ID)
    boundary_summary = {
        "schema_version": "1.0",
        "call_count": 2,
        "aggregate_seam_costs_ms": {
            "engine_bootstrap_ms": 11.0,
            "runtime_file_validation_ms": 12.0,
            "input_staging_ms": 13.0,
            "matlab_call_ms": 14.0,
            "result_validation_ms": 15.0,
            "canonical_persistence_ms": 16.0,
        },
        "session_lifecycle_summary": {
            "schema_version": "1.0",
            "session_count": 1,
            "sessions_with_reuse": 1,
        },
    }

    preprocessing_payload = {"provider_mode": "mixed", "boundary_instrumentation_summary": boundary_summary}
    _ = (paths["qc"] / "preprocessing_provenance.yaml").write_text(
        yaml.safe_dump(preprocessing_payload, sort_keys=False),
        encoding="utf-8",
    )
    np.save(
        paths["transforms"] / f"transforms_fov_{FOV_ID}.npy",
        cast(Any, {
            1: {"global_shift_3d": np.zeros(3)},
            "_provenance": {
                "runtime_context": {
                    "registration_backend_details": {
                        "boundary_instrumentation_summary": boundary_summary,
                    }
                }
            },
        }),
    )
    np.save(paths["transforms"] / f"transforms_fov_{FOV_ID}_round_2_flow_3d.npy", np.zeros((3, 2, 4, 5)))

    _write_csv(paths["spots"] / f"spots_fov_{FOV_ID}.csv", ["z,y,x,intensity", "1,2,3,4", "5,6,7,8"])
    np.save(paths["extraction"] / f"intensity_matrix_fov_{FOV_ID}.npy", np.zeros((2, 3, 4), dtype=np.float32))
    write_backend_metadata(
        paths["extraction"] / f"intensity_matrix_fov_{FOV_ID}_metadata.json",
        {"schema_name": "test", "matrix_shape": [2, 3, 4]},
    )
    _write_csv(paths["decoded"] / f"decoded_fov_{FOV_ID}.csv", ["z,y,x,barcode,quality,intensity,gene", "1,2,3,01,0.1,4,GeneA"])
    _write_csv(paths["decoded"] / f"decoded_fov_{FOV_ID}_goodreads.csv", ["z,y,x,barcode,quality,intensity,gene"])
    _write_csv(paths["decoded"] / f"decoded_fov_{FOV_ID}_pre_pattern_check.csv", ["z,y,x,barcode,quality,intensity,gene", "1,2,3,01,0.1,4,GeneA"])
    write_backend_metadata(
        paths["qc"] / f"spot_finding_backend_fov_{FOV_ID}.json",
        {"boundary_instrumentation_summary": boundary_summary},
    )
    write_backend_metadata(
        paths["qc"] / f"extraction_backend_fov_{FOV_ID}.json",
        {"boundary_instrumentation_summary": boundary_summary},
    )

    telemetry_path = write_performance_telemetry(
        config=cfg,
        fov_id=FOV_ID,
        stage_timings=_stage_timings(),
        run_started_at_utc="2026-05-15T00:00:00+00:00",
        run_finished_at_utc="2026-05-15T00:00:10+00:00",
        total_elapsed_ms=1500.0,
    )

    assert telemetry_path == paths["qc"] / f"performance_fov_{FOV_ID}.json"
    payload = _read_json_mapping(telemetry_path)
    assert payload["schema_name"] == PERFORMANCE_TELEMETRY_SCHEMA_NAME
    assert payload["schema_version"] == PERFORMANCE_TELEMETRY_SCHEMA_VERSION
    assert payload["stage_order"] == [
        "preprocessing",
        "registration",
        "spot_finding",
        "signal_extraction",
        "decoding",
    ]
    assert [stage["elapsed_wall_ms"] for stage in payload["stages"]] == [100.0, 200.0, 300.0, 400.0, 500.0]
    assert payload["providers"]["registration"]["global_provider"] == "matlab"
    assert payload["artifacts"]["spot_table"]["row_count"] == 2
    assert payload["artifacts"]["intensity_matrix"]["shape"] == [2, 3, 4]
    assert payload["artifacts"]["flow_3d_sidecars"][0]["shape"] == [3, 2, 4, 5]
    assert payload["artifacts"]["decoded_outputs"]["active"]["row_count"] == 1
    assert payload["artifacts"]["decoded_outputs"]["goodreads"]["row_count"] == 0
    assert payload["stages"][0]["matlab"]["boundary_instrumentation_summary"]["call_count"] == 2
    assert payload["stages"][3]["matlab"]["session_lifecycle_summary"]["session_count"] == 1


def test_performance_telemetry_records_absent_optional_artifacts(tmp_path: Path) -> None:
    cfg = _config(tmp_path)

    telemetry_path = write_performance_telemetry(
        config=cfg,
        fov_id=FOV_ID,
        stage_timings=_stage_timings(),
    )

    payload = _read_json_mapping(telemetry_path)
    assert payload["artifacts"]["transform_manifest"]["exists"] is False
    assert payload["artifacts"]["spot_table"]["read_status"] == "absent"
    assert payload["artifacts"]["intensity_matrix"]["read_status"] == "absent"
    assert payload["stages"][2]["matlab"]["metadata_sources"][0]["read_status"] == "absent"


def test_performance_telemetry_rejects_stage_order_drift(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    drifted = [
        record_stage_timing("preprocessing", 100.0),
        record_stage_timing("spot_finding", 300.0),
        record_stage_timing("registration", 200.0),
        record_stage_timing("signal_extraction", 400.0),
        record_stage_timing("decoding", 500.0),
    ]

    with pytest.raises(ValueError, match="order drift detected"):
        _ = write_performance_telemetry(config=cfg, fov_id=FOV_ID, stage_timings=drifted)


def test_performance_telemetry_rejects_missing_stage_timing(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    incomplete = _stage_timings()[:-1]

    with pytest.raises(ValueError, match="order drift detected"):
        _ = write_performance_telemetry(config=cfg, fov_id=FOV_ID, stage_timings=incomplete)


def test_performance_telemetry_requires_completed_outputs_when_enabled(tmp_path: Path) -> None:
    cfg = _config(tmp_path)

    with pytest.raises(FileNotFoundError, match="required completed artifact transform_manifest"):
        _ = write_performance_telemetry(
            config=cfg,
            fov_id=FOV_ID,
            stage_timings=_stage_timings(),
            require_completed_outputs=True,
        )


def test_performance_telemetry_fails_loud_for_missing_matlab_metadata(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    paths = get_fov_output_structure(tmp_path, FOV_ID)
    boundary_summary = {
        "schema_version": "1.0",
        "call_count": 1,
        "aggregate_seam_costs_ms": {"matlab_call_ms": 1.0},
    }
    _ = (paths["qc"] / "preprocessing_provenance.yaml").write_text(
        yaml.safe_dump({"boundary_instrumentation_summary": boundary_summary}, sort_keys=False),
        encoding="utf-8",
    )
    np.save(
        paths["transforms"] / f"transforms_fov_{FOV_ID}.npy",
        cast(Any, {
            1: {"global_shift_3d": np.zeros(3)},
            "_provenance": {
                "runtime_context": {
                    "registration_backend_details": {
                        "boundary_instrumentation_summary": boundary_summary,
                    }
                }
            },
        }),
    )
    _write_csv(paths["spots"] / f"spots_fov_{FOV_ID}.csv", ["z,y,x,intensity"])
    np.save(paths["extraction"] / f"intensity_matrix_fov_{FOV_ID}.npy", np.zeros((0, 3, 4), dtype=np.float32))
    _write_csv(paths["decoded"] / f"decoded_fov_{FOV_ID}.csv", ["z,y,x,barcode,quality,intensity,gene"])
    _write_csv(paths["decoded"] / f"decoded_fov_{FOV_ID}_goodreads.csv", ["z,y,x,barcode,quality,intensity,gene"])
    _write_csv(paths["decoded"] / f"decoded_fov_{FOV_ID}_pre_pattern_check.csv", ["z,y,x,barcode,quality,intensity,gene"])

    with pytest.raises(FileNotFoundError, match="MATLAB spot_finding"):
        _ = write_performance_telemetry(
            config=cfg,
            fov_id=FOV_ID,
            stage_timings=_stage_timings(),
            require_completed_outputs=True,
        )
