from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest

from pystar._performance_summary import (
    PERFORMANCE_SUMMARY_SCHEMA_NAME,
    PERFORMANCE_SUMMARY_SCHEMA_VERSION,
    build_performance_summary_payload,
    parse_fov_ids,
    write_performance_summary,
)
from pystar._performance_telemetry import (
    PERFORMANCE_TELEMETRY_SCHEMA_NAME,
    PERFORMANCE_TELEMETRY_SCHEMA_VERSION,
)
from pystar.io import get_fov_output_structure
from pystar.serialization import write_backend_metadata


def _stage_entry(stage_id: str, order_index: int, elapsed_ms: float, *, matlab: dict[str, object] | None = None) -> dict[str, object]:
    labels = {
        "preprocessing": "Preprocessing (Sanitization)",
        "registration": "Registration",
        "spot_finding": "Spot Finding",
        "signal_extraction": "Signal Extraction",
        "decoding": "Decoding",
    }
    return {
        "stage_id": stage_id,
        "order_index": order_index,
        "display_label": labels[stage_id],
        "elapsed_wall_ms": elapsed_ms,
        "status": "completed",
        "runner_calls": [],
        "matlab": matlab
        if matlab is not None
        else {
            "metadata_sources": [],
            "boundary_instrumentation_summary": None,
            "session_lifecycle_summary": None,
        },
    }


def _matlab_summary(*, path: Path) -> dict[str, object]:
    return {
        "metadata_sources": [
            {
                "path": str(path),
                "exists": True,
                "size_bytes": 100,
                "read_status": "read",
                "kind": "synthetic_backend_metadata",
            }
        ],
        "boundary_instrumentation_summary": {
            "schema_version": "1.0",
            "call_count": 2,
            "engine_reused_calls": 1,
            "aggregate_seam_costs_ms": {
                "engine_bootstrap_ms": 10.0,
                "runtime_file_validation_ms": 2.0,
                "input_staging_ms": 3.0,
                "matlab_call_ms": 40.0,
                "result_validation_ms": 5.0,
                "canonical_persistence_ms": 6.0,
            },
        },
        "session_lifecycle_summary": {
            "schema_version": "1.0",
            "session_count": 1,
            "aggregate_counts": {
                "engine_bootstrap_count": 1,
                "engine_reuse_count": 1,
            },
            "aggregate_timing_ms": {
                "engine_bootstrap_ms": 10.0,
                "runtime_file_validation_ms": 2.0,
            },
        },
    }


def _telemetry_payload(
    tmp_path: Path,
    *,
    fov_id: int,
    stage_elapsed: dict[str, float],
    matlab_stage_ids: set[str] | None = None,
) -> dict[str, object]:
    paths = get_fov_output_structure(tmp_path, fov_id)
    matlab_stage_ids = matlab_stage_ids or set()
    stage_ids = ["preprocessing", "registration", "spot_finding", "signal_extraction", "decoding"]
    stages = [
        _stage_entry(
            stage_id,
            index,
            stage_elapsed[stage_id],
            matlab=(
                _matlab_summary(path=paths["qc"] / f"{stage_id}_backend.json")
                if stage_id in matlab_stage_ids
                else None
            ),
        )
        for index, stage_id in enumerate(stage_ids, start=1)
    ]
    return {
        "schema_name": PERFORMANCE_TELEMETRY_SCHEMA_NAME,
        "schema_version": PERFORMANCE_TELEMETRY_SCHEMA_VERSION,
        "generated_at_utc": "2026-05-15T00:00:00+00:00",
        "fov_id": fov_id,
        "config": {"output_directory": str(tmp_path)},
        "providers": {
            "preprocessing": {"providers": ["native"], "provider_mode": "native_only"},
            "registration": {"provider_mode": "native_only", "global_provider": "native", "local_provider": "native"},
            "spot_finding": {"provider": "native", "algorithm": "peak_local_max"},
            "signal_extraction": {"provider": "native", "method": "box_sum", "transform_application_mode": "image_warp"},
            "decoding": {"gating_mode": "pattern_first"},
        },
        "run": {
            "started_at_utc": "2026-05-15T00:00:00+00:00",
            "finished_at_utc": "2026-05-15T00:00:01+00:00",
            "total_elapsed_ms": sum(stage_elapsed.values()) + 10.0,
        },
        "stage_order": stage_ids,
        "stages": stages,
        "artifacts": {
            "output_paths": {key: str(value) for key, value in paths.items()},
            "transform_manifest": {"path": str(paths["transforms"] / f"transforms_fov_{fov_id}.npy"), "exists": True, "size_bytes": 1000},
            "flow_3d_sidecars": [
                {
                    "path": str(paths["transforms"] / f"transforms_fov_{fov_id}_round_2_flow_3d.npy"),
                    "exists": True,
                    "size_bytes": 2000,
                    "read_status": "read",
                    "shape": [3, 2, 4, 5],
                    "dtype": "float64",
                }
            ],
            "spot_table": {
                "path": str(paths["spots"] / f"spots_fov_{fov_id}.csv"),
                "exists": True,
                "size_bytes": 300,
                "read_status": "read",
                "row_count": fov_id,
            },
            "intensity_matrix": {
                "path": str(paths["extraction"] / f"intensity_matrix_fov_{fov_id}.npy"),
                "exists": True,
                "size_bytes": 400,
                "read_status": "read",
                "shape": [fov_id, 3, 4],
                "dtype": "float32",
            },
            "intensity_matrix_metadata": {
                "path": str(paths["extraction"] / f"intensity_matrix_fov_{fov_id}_metadata.json"),
                "exists": True,
                "size_bytes": 50,
            },
            "decoded_outputs": {
                "active": {
                    "path": str(paths["decoded"] / f"decoded_fov_{fov_id}.csv"),
                    "exists": True,
                    "size_bytes": 500,
                    "read_status": "read",
                    "row_count": fov_id + 1,
                },
                "goodreads": {
                    "path": str(paths["decoded"] / f"decoded_fov_{fov_id}_goodreads.csv"),
                    "exists": True,
                    "size_bytes": 250,
                    "read_status": "read",
                    "row_count": fov_id - 1,
                },
                "pre_pattern_check": {
                    "path": str(paths["decoded"] / f"decoded_fov_{fov_id}_pre_pattern_check.csv"),
                    "exists": True,
                    "size_bytes": 550,
                    "read_status": "read",
                    "row_count": fov_id + 2,
                },
            },
            "backend_metadata_sidecars": {
                "preprocessing_provenance": {"path": str(paths["qc"] / "preprocessing_provenance.yaml"), "exists": False, "size_bytes": None},
                "spot_finding_backend": {"path": str(paths["qc"] / f"spot_finding_backend_fov_{fov_id}.json"), "exists": False, "size_bytes": None},
                "extraction_backend": {"path": str(paths["qc"] / f"extraction_backend_fov_{fov_id}.json"), "exists": False, "size_bytes": None},
            },
        },
    }


def _write_telemetry(tmp_path: Path, payload: dict[str, object]) -> Path:
    fov_id = cast(int, payload["fov_id"])
    path = get_fov_output_structure(tmp_path, fov_id)["qc"] / f"performance_fov_{fov_id}.json"
    write_backend_metadata(path, payload)
    return path


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return cast(dict[str, Any], payload)


def test_parse_fov_ids_accepts_ranges_and_rejects_bad_input() -> None:
    assert parse_fov_ids("1-3,7,3") == (1, 2, 3, 7)
    assert parse_fov_ids(None) == ()

    with pytest.raises(ValueError, match="Invalid FOV range"):
        _ = parse_fov_ids("3-1")


def test_performance_summary_aggregates_native_style_telemetry(tmp_path: Path) -> None:
    _write_telemetry(
        tmp_path,
        _telemetry_payload(
            tmp_path,
            fov_id=1,
            stage_elapsed={
                "preprocessing": 10.0,
                "registration": 50.0,
                "spot_finding": 20.0,
                "signal_extraction": 40.0,
                "decoding": 30.0,
            },
        ),
    )
    _write_telemetry(
        tmp_path,
        _telemetry_payload(
            tmp_path,
            fov_id=2,
            stage_elapsed={
                "preprocessing": 20.0,
                "registration": 70.0,
                "spot_finding": 25.0,
                "signal_extraction": 35.0,
                "decoding": 45.0,
            },
        ),
    )

    json_path, markdown_path = write_performance_summary(base_dir=tmp_path)

    payload = _read_json(json_path)
    assert markdown_path.read_text(encoding="utf-8").startswith("# PyStar Performance Telemetry Summary")
    assert payload["schema_name"] == PERFORMANCE_SUMMARY_SCHEMA_NAME
    assert payload["schema_version"] == PERFORMANCE_SUMMARY_SCHEMA_VERSION
    assert payload["inputs"]["read_fov_ids"] == [1, 2]
    assert payload["inputs"]["absent_fov_ids"] == []
    registration = payload["stage_aggregates"]["registration"]
    assert registration["count"] == 2
    assert registration["total_elapsed_wall_ms"] == 120.0
    assert registration["mean_elapsed_wall_ms"] == 60.0
    assert registration["median_elapsed_wall_ms"] == 60.0
    assert payload["stage_rankings"][0]["stage_id"] == "registration"
    assert payload["slow_fov_rankings"][0]["fov_id"] == 2
    assert payload["artifact_summary"]["spot_table"]["row_count"]["total"] == 3.0
    assert payload["artifact_summary"]["flow_3d_sidecars"]["total_sidecar_count"] == 2


def test_performance_summary_preserves_matlab_boundary_and_session_data(tmp_path: Path) -> None:
    _write_telemetry(
        tmp_path,
        _telemetry_payload(
            tmp_path,
            fov_id=4,
            stage_elapsed={
                "preprocessing": 10.0,
                "registration": 500.0,
                "spot_finding": 20.0,
                "signal_extraction": 200.0,
                "decoding": 30.0,
            },
            matlab_stage_ids={"registration", "signal_extraction"},
        ),
    )

    payload = build_performance_summary_payload(base_dir=tmp_path)

    registration = payload["matlab_summary"]["registration"]
    assert registration["boundary_summary_count"] == 1
    assert registration["aggregate_boundary_call_count"] == 2
    assert registration["aggregate_seam_costs_ms"]["matlab_call_ms"] == 40.0
    assert registration["session_summary_count"] == 1
    assert registration["aggregate_session_count"] == 1
    assert registration["aggregate_session_counts"]["engine_reuse_count"] == 1.0
    assert payload["matlab_summary"]["spot_finding"]["boundary_summary_absent_count"] == 1


def test_performance_summary_reports_missing_requested_fov_telemetry(tmp_path: Path) -> None:
    _write_telemetry(
        tmp_path,
        _telemetry_payload(
            tmp_path,
            fov_id=1,
            stage_elapsed={
                "preprocessing": 1.0,
                "registration": 2.0,
                "spot_finding": 3.0,
                "signal_extraction": 4.0,
                "decoding": 5.0,
            },
        ),
    )

    payload = build_performance_summary_payload(base_dir=tmp_path, fov_ids=(1, 2))

    assert payload["inputs"]["read_fov_ids"] == [1]
    assert payload["inputs"]["absent_fov_ids"] == [2]
    missing = next(record for record in payload["fovs"] if record["fov_id"] == 2)
    assert missing["telemetry_status"] == "absent"
    assert missing["telemetry_path"].endswith("Position2/output_pystar/qc_reports/performance_fov_2.json")


def test_performance_summary_fails_loud_for_malformed_telemetry(tmp_path: Path) -> None:
    path = get_fov_output_structure(tmp_path, 9)["qc"] / "performance_fov_9.json"
    _ = path.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(ValueError, match=r"Malformed performance telemetry at .*performance_fov_9\.json"):
        _ = build_performance_summary_payload(base_dir=tmp_path)


def test_summary_cli_writes_requested_outputs(tmp_path: Path) -> None:
    _write_telemetry(
        tmp_path,
        _telemetry_payload(
            tmp_path,
            fov_id=5,
            stage_elapsed={
                "preprocessing": 1.0,
                "registration": 2.0,
                "spot_finding": 3.0,
                "signal_extraction": 4.0,
                "decoding": 5.0,
            },
        ),
    )
    script = Path(__file__).resolve().parents[2] / "scripts" / "summarize_performance_telemetry.py"
    output_json = tmp_path / "qc_reports" / "summary.json"
    output_md = tmp_path / "qc_reports" / "summary.md"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--base-dir",
            str(tmp_path),
            "--fovs",
            "5-6",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Performance telemetry JSON summary" in result.stdout
    payload = _read_json(output_json)
    assert payload["inputs"]["read_fov_ids"] == [5]
    assert payload["inputs"]["absent_fov_ids"] == [6]
    assert "Slowest FOVs" in output_md.read_text(encoding="utf-8")
