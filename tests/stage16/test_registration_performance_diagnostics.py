from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from pystar._registration_performance import (
    REGISTRATION_PERFORMANCE_SCHEMA_NAME,
    REGISTRATION_PERFORMANCE_SCHEMA_VERSION,
    RegistrationPerformanceRecorder,
    get_registration_performance_path,
    load_registration_performance_diagnostics,
    timing_record,
    validate_registration_performance_payload,
)
from pystar.io import get_fov_output_structure


FOV_ID = 5


def _boundary_trace(
    *,
    stage_name: str,
    matlab_call_ms: float,
    engine_reused_this_call: bool = True,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "stage_name": stage_name,
        "runtime_path": "/tmp/pystar-runtime",
        "entrypoint": f"{stage_name}_entrypoint",
        "call_scope": {"fov_id": FOV_ID, "round_id": 2},
        "started_at": "2026-05-15T00:00:00+00:00",
        "finished_at": "2026-05-15T00:00:01+00:00",
        "total_duration_ms": round(matlab_call_ms + 6.0, 3),
        "engine_reused_this_call": engine_reused_this_call,
        "session_lifecycle_before": {},
        "session_lifecycle_after": {},
        "phase_timings_ms": {"matlab_call": matlab_call_ms},
        "phase_details": {},
        "seam_costs_ms": {
            "engine_bootstrap_ms": 1.0,
            "runtime_file_validation_ms": 2.0,
            "input_staging_ms": 3.0,
            "matlab_call_ms": matlab_call_ms,
            "result_validation_ms": 4.0,
            "canonical_persistence_ms": 5.0,
            "teardown_ms": 0.0,
        },
    }


def _session_lifecycle() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "session_id": "session-1",
        "consumer": "registration",
        "engine_bootstrap_count": 1,
        "engine_reuse_count": 2,
        "aggregate_timing_ms": {"engine_bootstrap_ms": 12.0},
    }


def _session_lifecycle_summary() -> dict[str, Any]:
    return {
        "session_count": 1,
        "aggregate_counts": {"engine_reuse_count": 2},
        "aggregate_timing_ms": {"engine_bootstrap_ms": 12.0},
    }


def _matlab_internal_metadata(
    *,
    total_duration_ms: float = 100.0,
    steps: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "round_id": 2,
        "reference_round": 1,
        "total_duration_ms": total_duration_ms,
        "steps": steps
        if steps is not None
        else [
            {"name": "new_LoadMultipageTiff", "duration_ms": 10.0, "details": {"input_count": 2}},
            {"name": "imregdemons", "duration_ms": 80.0, "details": {"iterations": 50}},
            {"name": "save_local_flow_mat", "duration_ms": 5.0, "details": {"format": "mat_v7"}},
        ],
    }


def _worker_lifecycle(*, tile_index: int, worker_process_pid: int, total_tile_wall_ms: float) -> dict[str, Any]:
    return {
        "worker_process_pid": worker_process_pid,
        "tile_index": tile_index,
        "status": "completed",
        "backend_construct_ms": 2.0,
        "matlab_session_start_or_attach_ms": 1.0,
        "runtime_validation_ms": 0.5,
        "matlab_addpath_or_bootstrap_ms": 0.25,
        "input_staging_ms": 3.0,
        "matlab_call_ms": 20.0,
        "mat_output_load_ms": 4.0,
        "result_validation_ms": 1.0,
        "backend_close_ms": 0.75,
        "total_tile_wall_ms": total_tile_wall_ms,
    }


def _providers(provider_mode: str = "native_only") -> dict[str, Any]:
    return {
        "stage_id": "registration",
        "provider_mode": provider_mode,
        "global_provider": "matlab" if provider_mode == "matlab_only" else "native",
        "global_method": "phase_corr_3d",
        "local_provider": "matlab" if provider_mode == "matlab_only" else "native",
        "local_method": "demons_3d",
        "enable_local": True,
        "reference_round": 1,
    }


def test_native_style_registration_diagnostics_round_trip_and_manifest_timings(tmp_path: Path) -> None:
    recorder = RegistrationPerformanceRecorder(fov_id=FOV_ID, providers=_providers())
    recorder.record_fov_setup_timing("reference_clean_volume_load", 1.0, details={"round_id": 1})
    recorder.record_fov_setup_timing("reference_scope_crop", 2.0)
    recorder.record_fov_setup_timing("reference_mip_creation", 3.0)

    recorder.start_round(1, is_reference_round=True)
    recorder.complete_round(1)
    recorder.start_round(2, is_reference_round=False)
    recorder.record_round_timing(2, "moving_clean_volume_load", 4.0)
    recorder.record_round_timing(2, "moving_scope_crop", 5.0)
    recorder.record_global_registration(
        2,
        elapsed_wall_ms=6.0,
        provider="native",
        method="phase_corr_3d",
        global_shift_3d=np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
        global_corr=0.91,
    )
    recorder.record_post_global_qc(2, elapsed_wall_ms=7.0, corr_after_global=0.88)
    recorder.record_local_registration(
        2,
        elapsed_wall_ms=8.0,
        provider="native",
        method="demons_3d",
        status="accepted",
        final_corr=0.93,
    )
    recorder.record_final_qc(2, elapsed_wall_ms=9.0, final_corr=0.93)
    recorder.record_manifest_timing("provenance_build", 10.0)
    recorder.record_manifest_timing("save_transform_manifest", 11.0)
    recorder.record_manifest_timing("load_transform_manifest", 12.0)

    output_path = recorder.write(tmp_path, source_stage_elapsed_wall_ms=100.0)

    assert output_path == get_registration_performance_path(tmp_path, FOV_ID)
    payload = load_registration_performance_diagnostics(output_path, expected_fov_id=FOV_ID)
    assert payload["schema_name"] == REGISTRATION_PERFORMANCE_SCHEMA_NAME
    assert payload["schema_version"] == REGISTRATION_PERFORMANCE_SCHEMA_VERSION
    assert payload["stage_id"] == "registration"
    assert payload["source_stage_elapsed_wall_ms"] == 100.0
    assert payload["summary"]["moving_round_count"] == 1
    assert payload["summary"]["matlab_boundary_call_count"] == 0
    assert payload["manifest"]["save_transform_manifest"]["elapsed_wall_ms"] == 11.0
    assert payload["manifest"]["load_transform_manifest"]["elapsed_wall_ms"] == 12.0
    assert "save_transform_manifest" not in payload["rounds"][1]


def test_matlab_boundary_instrumentation_is_aggregated_without_matlab() -> None:
    recorder = RegistrationPerformanceRecorder(fov_id=FOV_ID, providers=_providers("matlab_only"))
    recorder.start_round(2, is_reference_round=False)
    recorder.record_global_registration(
        2,
        elapsed_wall_ms=20.0,
        provider="matlab",
        method="phase_corr_3d",
        global_shift_3d=[1.0, 2.0, 3.0],
        global_corr=0.9,
        backend_metadata={"boundary_instrumentation": _boundary_trace(stage_name="matlab_registration_global", matlab_call_ms=40.0)},
    )
    recorder.record_local_registration(
        2,
        elapsed_wall_ms=30.0,
        provider="matlab",
        method="demons_3d",
        status="accepted",
        final_corr=0.92,
        backend_metadata={
            "local_flow": {
                "boundary_instrumentation": _boundary_trace(
                    stage_name="matlab_registration_local",
                    matlab_call_ms=60.0,
                    engine_reused_this_call=False,
                )
            },
            "session_lifecycle": _session_lifecycle(),
            "session_lifecycle_summary": _session_lifecycle_summary(),
        },
    )

    payload = recorder.build_payload()
    summary = payload["summary"]

    assert summary["matlab_global_boundary_call_count"] == 1
    assert summary["matlab_local_boundary_call_count"] == 1
    assert summary["matlab_boundary_call_count"] == 2
    assert summary["matlab_boundary_seam_cost_totals_ms"]["matlab_call_ms"] == 100.0
    assert summary["matlab_boundary_seam_cost_totals_ms"]["runtime_file_validation_ms"] == 4.0
    assert summary["matlab_boundary_summary"]["call_count"] == 2
    assert summary["matlab_boundary_summary"]["engine_reused_calls"] == 1
    local_record = payload["rounds"][0]["local_registration"]
    assert local_record["session_lifecycle"]["session_id"] == "session-1"
    assert local_record["session_lifecycle_summary"]["session_count"] == 1


def test_global_matlab_steps_do_not_create_local_internal_timing() -> None:
    recorder = RegistrationPerformanceRecorder(fov_id=FOV_ID, providers=_providers("matlab_only"))
    recorder.start_round(2, is_reference_round=False)
    recorder.record_local_registration(
        2,
        elapsed_wall_ms=30.0,
        provider="matlab",
        method="demons_3d",
        status="accepted",
        final_corr=0.92,
        backend_metadata={
            "matlab_metadata": {
                "round_id": 2,
                "reference_round": 1,
                "total_duration_ms": 30.0,
                "steps": [{"name": "pystar_phasecorr_global_shift", "duration_ms": 25.0}],
            },
            "local_flow": {
                "boundary_instrumentation": _boundary_trace(
                    stage_name="matlab_registration_local",
                    matlab_call_ms=60.0,
                ),
            },
        },
    )

    payload = recorder.build_payload()

    assert "matlab_internal_timing" not in payload["rounds"][0]["local_registration"]
    assert payload["summary"]["matlab_internal_timing_status"] == "absent"


def test_tiled_matlab_local_diagnostics_preserve_tile_identity_and_boundary_mapping() -> None:
    recorder = RegistrationPerformanceRecorder(fov_id=FOV_ID, providers=_providers("matlab_only"))
    recorder.start_round(2, is_reference_round=False)
    recorder.record_tiled_local_tile(
        2,
        tile_identity={
            "tile_index": 3,
            "grid_position_yx": [0, 2],
            "grid_shape_yx": [2, 2],
            "region_origin_zyx": [0, 10, 20],
            "region_shape_zyx": [4, 30, 40],
            "write_origin_zyx": [0, 12, 22],
            "write_shape_zyx": [4, 26, 36],
            "write_offset_zyx": [0, 2, 2],
            "full_volume_shape_zyx": [4, 60, 80],
        },
        total_elapsed_wall_ms=50.0,
        extraction_elapsed_wall_ms=5.0,
        backend_call_elapsed_wall_ms=40.0,
        flow_validation_elapsed_wall_ms=5.0,
        boundary_instrumentation=_boundary_trace(stage_name="matlab_registration_local", matlab_call_ms=35.0),
        session_lifecycle=_session_lifecycle(),
        session_lifecycle_summary=_session_lifecycle_summary(),
        normalized_result={"flow_3d_shape": [3, 4, 30, 40], "flow_3d_dtype": "float32"},
    )
    recorder.record_tiled_local_summary(
        2,
        layout_summary={"enabled": True, "grid_shape_yx": [2, 2], "tile_count": 4},
        stitch_elapsed_wall_ms=6.0,
    )

    payload = recorder.build_payload()
    tile = payload["rounds"][0]["tiled_local"]["tiles"][0]

    assert tile["tile_index"] == 3
    assert tile["grid_position_yx"] == [0, 2]
    assert tile["region_origin_zyx"] == [0, 10, 20]
    assert tile["timings"]["backend_call"]["elapsed_wall_ms"] == 40.0
    assert tile["boundary_instrumentation"]["call_scope"]["round_id"] == 2
    assert tile["session_lifecycle"]["session_id"] == "session-1"
    assert tile["session_lifecycle_summary"]["aggregate_counts"]["engine_reuse_count"] == 2
    assert payload["summary"]["tile_count"] == 1
    assert payload["summary"]["matlab_local_boundary_call_count"] == 1
    assert payload["summary"]["slowest_tiles"][0]["tile_index"] == 3


def test_matlab_local_internal_steps_normalize_and_aggregate_without_matlab() -> None:
    recorder = RegistrationPerformanceRecorder(fov_id=FOV_ID, providers=_providers("matlab_only"))
    recorder.start_round(2, is_reference_round=False)
    boundary = _boundary_trace(stage_name="matlab_registration_local", matlab_call_ms=106.0)
    recorder.record_local_registration(
        2,
        elapsed_wall_ms=120.0,
        provider="matlab",
        method="demons_3d",
        status="accepted",
        final_corr=0.92,
        backend_metadata={
            "local_flow": {
                "matlab_metadata": _matlab_internal_metadata(),
                "boundary_instrumentation": boundary,
            }
        },
    )

    payload = recorder.build_payload()
    local_record = payload["rounds"][0]["local_registration"]
    internal = local_record["matlab_internal_timing"]
    summary = payload["summary"]

    assert internal["source"] == "matlab_metadata.steps"
    assert internal["total_duration_ms"] == 100.0
    assert internal["step_total_duration_ms"] == 95.0
    assert internal["unaccounted_duration_ms"] == 5.0
    assert internal["boundary_matlab_call_ms"] == 106.0
    assert internal["boundary_minus_matlab_total_ms"] == 6.0
    assert internal["dominant_step"] == {"name": "imregdemons", "duration_ms": 80.0}
    assert summary["matlab_internal_timing_status"] == "present"
    assert summary["matlab_internal_call_count"] == 1
    assert summary["matlab_internal_total_duration_ms"] == 100.0
    assert summary["matlab_internal_step_totals_ms"] == {
        "imregdemons": 80.0,
        "new_LoadMultipageTiff": 10.0,
        "save_local_flow_mat": 5.0,
    }
    assert summary["matlab_internal_unaccounted_total_ms"] == 5.0
    assert summary["matlab_boundary_minus_internal_total_ms"] == 6.0
    assert summary["matlab_internal_dominant_step_counts"] == {"imregdemons": 1}
    assert summary["slowest_rounds"][0]["matlab_internal_total_duration_ms"] == 100.0


def test_tiled_matlab_internal_steps_preserve_dominant_step_and_aggregate() -> None:
    recorder = RegistrationPerformanceRecorder(fov_id=FOV_ID, providers=_providers("matlab_only"))
    recorder.start_round(2, is_reference_round=False)

    tile_identity = {
        "tile_index": 3,
        "grid_position_yx": [0, 2],
        "grid_shape_yx": [2, 2],
        "region_origin_zyx": [0, 10, 20],
        "region_shape_zyx": [4, 30, 40],
        "write_origin_zyx": [0, 12, 22],
        "write_shape_zyx": [4, 26, 36],
        "write_offset_zyx": [0, 2, 2],
        "full_volume_shape_zyx": [4, 60, 80],
    }
    recorder.record_tiled_local_tile(
        2,
        tile_identity=tile_identity,
        total_elapsed_wall_ms=150.0,
        extraction_elapsed_wall_ms=5.0,
        backend_call_elapsed_wall_ms=140.0,
        flow_validation_elapsed_wall_ms=5.0,
        boundary_instrumentation=_boundary_trace(stage_name="matlab_registration_local", matlab_call_ms=121.0),
        normalized_result={"flow_3d_shape": [3, 4, 30, 40], "flow_3d_dtype": "float32"},
        matlab_metadata=_matlab_internal_metadata(total_duration_ms=118.0),
    )
    recorder.record_tiled_local_summary(
        2,
        layout_summary={"enabled": True, "grid_shape_yx": [2, 2], "tile_count": 4},
        stitch_elapsed_wall_ms=6.0,
    )

    payload = recorder.build_payload()
    tile = payload["rounds"][0]["tiled_local"]["tiles"][0]
    summary = payload["summary"]
    slow_tile = summary["slowest_tiles"][0]

    assert tile["matlab_internal_timing"]["total_duration_ms"] == 118.0
    assert tile["matlab_internal_timing"]["unaccounted_duration_ms"] == 23.0
    assert tile["matlab_internal_timing"]["boundary_minus_matlab_total_ms"] == 3.0
    assert tile["matlab_internal_timing"]["dominant_step"]["name"] == "imregdemons"
    assert summary["matlab_internal_call_count"] == 1
    assert summary["matlab_internal_step_totals_ms"]["imregdemons"] == 80.0
    assert summary["slowest_rounds"][0]["matlab_internal_total_duration_ms"] == 118.0
    assert slow_tile["tile_index"] == 3
    assert slow_tile["matlab_internal_total_duration_ms"] == 118.0
    assert slow_tile["matlab_internal_dominant_step"] == {"name": "imregdemons", "duration_ms": 80.0}


def test_tiled_process_parallel_worker_lifecycle_persists_and_aggregates_without_matlab() -> None:
    recorder = RegistrationPerformanceRecorder(fov_id=FOV_ID, providers=_providers("matlab_only"))
    recorder.start_round(2, is_reference_round=False)

    for tile_index, worker_pid, total_ms in ((1, 4101, 40.0), (2, 4102, 60.0)):
        recorder.record_tiled_local_tile(
            2,
            tile_identity={
                "tile_index": tile_index,
                "grid_position_yx": [0, tile_index - 1],
                "grid_shape_yx": [1, 2],
                "region_origin_zyx": [0, 0, 20 * (tile_index - 1)],
                "region_shape_zyx": [4, 30, 40],
                "write_origin_zyx": [0, 0, 20 * (tile_index - 1)],
                "write_shape_zyx": [4, 30, 40],
                "write_offset_zyx": [0, 0, 0],
                "full_volume_shape_zyx": [4, 30, 80],
            },
            total_elapsed_wall_ms=total_ms,
            extraction_elapsed_wall_ms=5.0,
            backend_call_elapsed_wall_ms=total_ms - 10.0,
            flow_validation_elapsed_wall_ms=5.0,
            boundary_instrumentation=_boundary_trace(stage_name="matlab_registration_local", matlab_call_ms=20.0),
            normalized_result={"flow_3d_shape": [3, 4, 30, 40], "flow_3d_dtype": "float32"},
            worker_lifecycle=_worker_lifecycle(
                tile_index=tile_index,
                worker_process_pid=worker_pid,
                total_tile_wall_ms=total_ms,
            ),
        )
    recorder.record_tiled_local_summary(
        2,
        layout_summary={"enabled": True, "grid_shape_yx": [1, 2], "tile_count": 2},
        stitch_elapsed_wall_ms=6.0,
        execution_report={
            "requested_mode": "process_parallel",
            "effective_mode": "process_parallel",
            "worker_count": 2,
            "tile_count": 2,
            "strict_equivalence_audit": True,
            "tile_indices": [1, 2],
            "failures": [],
            "worker_lifecycle": {
                "status": "present",
                "worker_process_count": 2,
                "worker_process_ids": [4101, 4102],
                "worker_tile_counts": {"4101": 1, "4102": 1},
                "worker_overhead_totals_ms": {
                    "backend_construct_ms": 4.0,
                    "matlab_session_start_or_attach_ms": 2.0,
                    "runtime_validation_ms": 1.0,
                    "matlab_addpath_or_bootstrap_ms": 0.5,
                    "input_staging_ms": 6.0,
                    "matlab_call_ms": 40.0,
                    "mat_output_load_ms": 8.0,
                    "result_validation_ms": 2.0,
                    "backend_close_ms": 1.5,
                    "total_tile_wall_ms": 100.0,
                },
                "worker_overhead_percentages": {
                    "backend_construct_ms": 4.0,
                    "matlab_session_start_or_attach_ms": 2.0,
                    "runtime_validation_ms": 1.0,
                    "matlab_addpath_or_bootstrap_ms": 0.5,
                    "input_staging_ms": 6.0,
                    "matlab_call_ms": 40.0,
                    "mat_output_load_ms": 8.0,
                    "result_validation_ms": 2.0,
                    "backend_close_ms": 1.5,
                    "total_tile_wall_ms": 100.0,
                },
                "slowest_workers": [
                    {
                        "worker_process_pid": 4102,
                        "tile_count": 1,
                        "total_tile_wall_ms": 60.0,
                        "mean_tile_wall_ms": 60.0,
                        "max_tile_wall_ms": 60.0,
                    }
                ],
            },
        },
    )

    payload = recorder.build_payload()
    tiled = payload["rounds"][0]["tiled_local"]
    first_tile = tiled["tiles"][0]
    summary = payload["summary"]

    assert tiled["execution"]["requested_mode"] == "process_parallel"
    assert tiled["execution"]["worker_lifecycle"]["worker_process_count"] == 2
    assert first_tile["worker_lifecycle"]["worker_process_pid"] == 4101
    assert first_tile["worker_lifecycle"]["backend_construct_ms"] == 2.0
    assert summary["tiled_local_execution_status"] == "present"
    assert summary["tiled_local_worker_lifecycle_status"] == "present"
    assert summary["tiled_local_worker_process_count"] == 2
    assert summary["tiled_local_worker_tile_counts"] == {"4101": 1, "4102": 1}
    assert summary["tiled_local_worker_overhead_totals_ms"]["total_tile_wall_ms"] == 100.0
    assert summary["tiled_local_worker_overhead_totals_ms"]["matlab_call_ms"] == 40.0
    assert summary["tiled_local_worker_overhead_percentages"]["matlab_call_ms"] == 40.0
    assert summary["tiled_local_slowest_workers"][0]["worker_process_pid"] == 4102
    assert summary["tiled_local_slowest_worker_tiles"][0]["tile_index"] == 2


def test_missing_matlab_internal_steps_remain_valid_and_absent() -> None:
    recorder = RegistrationPerformanceRecorder(fov_id=FOV_ID, providers=_providers("matlab_only"))
    recorder.start_round(2, is_reference_round=False)
    recorder.record_local_registration(
        2,
        elapsed_wall_ms=30.0,
        provider="matlab",
        method="demons_3d",
        status="accepted",
        final_corr=0.92,
        backend_metadata={
            "local_flow": {
                "matlab_metadata": {"round_id": 2, "reference_round": 1, "total_duration_ms": 100.0},
                "boundary_instrumentation": _boundary_trace(stage_name="matlab_registration_local", matlab_call_ms=60.0),
            }
        },
    )

    payload = recorder.build_payload()

    assert "matlab_internal_timing" not in payload["rounds"][0]["local_registration"]
    assert payload["summary"]["matlab_internal_timing_status"] == "absent"
    assert payload["summary"]["matlab_internal_call_count"] == 0
    assert payload["summary"]["matlab_internal_step_totals_ms"] == {}


def test_internal_timing_without_boundary_reports_null_closure_delta() -> None:
    recorder = RegistrationPerformanceRecorder(fov_id=FOV_ID, providers=_providers("matlab_only"))
    recorder.start_round(2, is_reference_round=False)
    recorder.record_local_registration(
        2,
        elapsed_wall_ms=120.0,
        provider="matlab",
        method="demons_3d",
        status="accepted",
        final_corr=0.92,
        backend_metadata={"local_flow": {"matlab_metadata": _matlab_internal_metadata()}},
    )

    payload = recorder.build_payload()
    internal = payload["rounds"][0]["local_registration"]["matlab_internal_timing"]

    assert internal["boundary_matlab_call_ms"] is None
    assert internal["boundary_minus_matlab_total_ms"] is None
    assert payload["summary"]["matlab_boundary_minus_internal_total_ms"] is None


def test_step_durations_exceeding_total_fail_loudly() -> None:
    recorder = RegistrationPerformanceRecorder(fov_id=FOV_ID, providers=_providers("matlab_only"))
    recorder.start_round(2, is_reference_round=False)

    with pytest.raises(ValueError, match="unaccounted_duration_ms.*step durations exceed"):
        recorder.record_local_registration(
            2,
            elapsed_wall_ms=30.0,
            provider="matlab",
            method="demons_3d",
            status="accepted",
            final_corr=0.92,
            backend_metadata={
                "local_flow": {
                    "matlab_metadata": _matlab_internal_metadata(
                        total_duration_ms=10.0,
                        steps=[{"name": "imregdemons", "duration_ms": 11.0}],
                    ),
                }
            },
        )


@pytest.mark.parametrize(
    ("steps", "match"),
    [
        ([{"duration_ms": 1.0}], "name"),
        ([{"name": "imregdemons"}], "duration_ms"),
        ([{"name": "imregdemons", "duration_ms": -1.0}], "finite and non-negative|non-negative elapsed_wall_ms"),
        ([{"name": "imregdemons", "duration_ms": float("nan")}], "finite and non-negative"),
        ([{"name": "imregdemons", "duration_ms": float("inf")}], "finite and non-negative"),
        ([{"name": "imregdemons", "duration_ms": 1.0, "details": "bad"}], "details"),
    ],
)
def test_malformed_matlab_internal_steps_fail_loudly(steps: list[dict[str, Any]], match: str) -> None:
    recorder = RegistrationPerformanceRecorder(fov_id=FOV_ID, providers=_providers("matlab_only"))
    recorder.start_round(2, is_reference_round=False)

    with pytest.raises(ValueError, match=match):
        recorder.record_local_registration(
            2,
            elapsed_wall_ms=30.0,
            provider="matlab",
            method="demons_3d",
            status="accepted",
            final_corr=0.92,
            backend_metadata={
                "local_flow": {
                    "matlab_metadata": _matlab_internal_metadata(steps=steps),
                    "boundary_instrumentation": _boundary_trace(stage_name="matlab_registration_local", matlab_call_ms=60.0),
                }
            },
        )


def test_present_malformed_matlab_internal_block_fails_payload_validation() -> None:
    recorder = RegistrationPerformanceRecorder(fov_id=FOV_ID, providers=_providers("matlab_only"))
    payload = recorder.build_payload()
    malformed = cast(dict[str, Any], dict(payload))
    malformed["rounds"] = [
        {
            "round_id": 2,
            "is_reference_round": False,
            "status": "completed",
            "local_registration": {
                "phase_id": "local_registration",
                "elapsed_wall_ms": 1.0,
                "status": "completed",
                "matlab_internal_timing": {
                    "source": "matlab_metadata.steps",
                    "status": "present",
                    "total_duration_ms": 1.0,
                    "step_total_duration_ms": 1.0,
                    "unaccounted_duration_ms": 0.0,
                    "boundary_matlab_call_ms": None,
                    "boundary_minus_matlab_total_ms": None,
                    "steps": [{"name": "imregdemons", "duration_ms": True}],
                    "dominant_step": {"name": "imregdemons", "duration_ms": 1.0},
                },
            },
        }
    ]

    with pytest.raises(ValueError, match="matlab_internal_timing.*duration_ms"):
        validate_registration_performance_payload(malformed, expected_fov_id=FOV_ID)


def test_present_malformed_worker_lifecycle_block_fails_payload_validation() -> None:
    recorder = RegistrationPerformanceRecorder(fov_id=FOV_ID, providers=_providers("matlab_only"))
    recorder.start_round(2, is_reference_round=False)
    recorder.record_tiled_local_tile(
        2,
        tile_identity={
            "tile_index": 3,
            "grid_position_yx": [0, 2],
            "grid_shape_yx": [2, 2],
            "region_origin_zyx": [0, 10, 20],
            "region_shape_zyx": [4, 30, 40],
            "write_origin_zyx": [0, 12, 22],
            "write_shape_zyx": [4, 26, 36],
            "write_offset_zyx": [0, 2, 2],
            "full_volume_shape_zyx": [4, 60, 80],
        },
        total_elapsed_wall_ms=50.0,
        extraction_elapsed_wall_ms=5.0,
        backend_call_elapsed_wall_ms=40.0,
        flow_validation_elapsed_wall_ms=5.0,
        worker_lifecycle=_worker_lifecycle(tile_index=3, worker_process_pid=4103, total_tile_wall_ms=50.0),
    )
    payload = recorder.build_payload()
    malformed = cast(dict[str, Any], dict(payload))
    rounds = cast(list[dict[str, Any]], malformed["rounds"])
    tiled = cast(dict[str, Any], rounds[0]["tiled_local"])
    tile = cast(dict[str, Any], tiled["tiles"][0])
    lifecycle = cast(dict[str, Any], dict(tile["worker_lifecycle"]))
    lifecycle["backend_construct_ms"] = -1.0
    tile["worker_lifecycle"] = lifecycle

    with pytest.raises(ValueError, match="worker_lifecycle.*backend_construct_ms"):
        validate_registration_performance_payload(malformed, expected_fov_id=FOV_ID)


def test_tiled_local_tile_identity_drift_fails_payload_validation() -> None:
    recorder = RegistrationPerformanceRecorder(fov_id=FOV_ID, providers=_providers("matlab_only"))
    recorder.start_round(2, is_reference_round=False)
    recorder.record_tiled_local_tile(
        2,
        tile_identity={
            "tile_index": 3,
            "grid_position_yx": [0, 2],
            "grid_shape_yx": [2, 2],
            "region_origin_zyx": [0, 10, 20],
            "region_shape_zyx": [4, 30, 40],
            "write_origin_zyx": [0, 12, 22],
            "write_shape_zyx": [4, 26, 36],
            "write_offset_zyx": [0, 2, 2],
            "full_volume_shape_zyx": [4, 60, 80],
        },
        total_elapsed_wall_ms=50.0,
        extraction_elapsed_wall_ms=5.0,
        backend_call_elapsed_wall_ms=40.0,
        flow_validation_elapsed_wall_ms=5.0,
    )
    payload = recorder.build_payload()
    malformed = cast(dict[str, Any], dict(payload))
    rounds = cast(list[dict[str, Any]], malformed["rounds"])
    tiled = cast(dict[str, Any], rounds[0]["tiled_local"])
    tile = cast(dict[str, Any], tiled["tiles"][0])
    del tile["write_offset_zyx"]

    with pytest.raises(ValueError, match="write_offset_zyx"):
        validate_registration_performance_payload(malformed, expected_fov_id=FOV_ID)


def test_tiled_local_execution_failures_fail_payload_validation() -> None:
    recorder = RegistrationPerformanceRecorder(fov_id=FOV_ID, providers=_providers("matlab_only"))
    recorder.start_round(2, is_reference_round=False)
    recorder.record_tiled_local_tile(
        2,
        tile_identity={
            "tile_index": 1,
            "grid_position_yx": [0, 0],
            "grid_shape_yx": [1, 1],
            "region_origin_zyx": [0, 0, 0],
            "region_shape_zyx": [4, 30, 40],
            "write_origin_zyx": [0, 0, 0],
            "write_shape_zyx": [4, 30, 40],
            "write_offset_zyx": [0, 0, 0],
            "full_volume_shape_zyx": [4, 30, 40],
        },
        total_elapsed_wall_ms=50.0,
        extraction_elapsed_wall_ms=5.0,
        backend_call_elapsed_wall_ms=40.0,
        flow_validation_elapsed_wall_ms=5.0,
        worker_lifecycle=_worker_lifecycle(tile_index=1, worker_process_pid=4101, total_tile_wall_ms=50.0),
    )
    recorder.record_tiled_local_summary(
        2,
        layout_summary={"enabled": True, "grid_shape_yx": [1, 1], "tile_count": 1},
        stitch_elapsed_wall_ms=6.0,
        execution_report={
            "requested_mode": "process_parallel",
            "effective_mode": "process_parallel",
            "worker_count": 1,
            "tile_count": 1,
            "strict_equivalence_audit": True,
            "tile_indices": [1],
            "failures": [],
            "worker_lifecycle": {
                "status": "present",
                "worker_process_count": 1,
                "worker_process_ids": [4101],
                "worker_tile_counts": {"4101": 1},
                "worker_overhead_totals_ms": {
                    "backend_construct_ms": 2.0,
                    "matlab_session_start_or_attach_ms": 1.0,
                    "runtime_validation_ms": 0.5,
                    "matlab_addpath_or_bootstrap_ms": 0.25,
                    "input_staging_ms": 3.0,
                    "matlab_call_ms": 20.0,
                    "mat_output_load_ms": 4.0,
                    "result_validation_ms": 1.0,
                    "backend_close_ms": 0.75,
                    "total_tile_wall_ms": 50.0,
                },
                "worker_overhead_percentages": {
                    "backend_construct_ms": 4.0,
                    "matlab_session_start_or_attach_ms": 2.0,
                    "runtime_validation_ms": 1.0,
                    "matlab_addpath_or_bootstrap_ms": 0.5,
                    "input_staging_ms": 6.0,
                    "matlab_call_ms": 40.0,
                    "mat_output_load_ms": 8.0,
                    "result_validation_ms": 2.0,
                    "backend_close_ms": 1.5,
                    "total_tile_wall_ms": 100.0,
                },
                "slowest_workers": [],
            },
        },
    )
    payload = recorder.build_payload()
    malformed = cast(dict[str, Any], dict(payload))
    rounds = cast(list[dict[str, Any]], malformed["rounds"])
    tiled = cast(dict[str, Any], rounds[0]["tiled_local"])
    execution = cast(dict[str, Any], dict(tiled["execution"]))
    execution["failures"] = [{"tile_index": 1, "error": "synthetic worker failure"}]
    tiled["execution"] = execution

    with pytest.raises(ValueError, match="failures must be empty"):
        validate_registration_performance_payload(malformed, expected_fov_id=FOV_ID)


def test_elapsed_timing_validation_rejects_negative_nan_and_infinite_values() -> None:
    for bad_value in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and non-negative|non-negative elapsed_wall_ms"):
            _ = timing_record("bad_phase", bad_value)


def test_flow_sidecar_persistence_summary_records_descriptor_path_shape_dtype_and_size(tmp_path: Path) -> None:
    paths = get_fov_output_structure(tmp_path, FOV_ID)
    sidecar_path = paths["transforms"] / f"transforms_fov_{FOV_ID}_round_2_flow_3d.npy"
    flow = np.zeros((3, 2, 4, 5), dtype=np.float32)
    np.save(sidecar_path, flow, allow_pickle=False)
    descriptor = {
        "storage": "round_level_sidecar_npy",
        "path": sidecar_path.name,
        "shape": [3, 2, 4, 5],
        "dtype": "float32",
    }
    recorder = RegistrationPerformanceRecorder(fov_id=FOV_ID, providers=_providers())
    recorder.start_round(2, is_reference_round=False)
    recorder.record_flow_sidecar_persistence(
        2,
        elapsed_wall_ms=13.0,
        descriptor=descriptor,
        sidecar_path=sidecar_path,
    )

    payload = recorder.build_payload()
    flow_summary = payload["summary"]["flow_sidecars"][0]
    round_record = payload["rounds"][0]["flow_sidecar_persistence"]

    assert payload["summary"]["flow_sidecar_count"] == 1
    assert payload["summary"]["flow_sidecar_total_bytes"] == sidecar_path.stat().st_size
    assert flow_summary["path"] == str(sidecar_path)
    assert flow_summary["descriptor"] == descriptor
    assert flow_summary["exists"] is True
    assert flow_summary["size_bytes"] == sidecar_path.stat().st_size
    assert round_record["details"]["descriptor"]["shape"] == [3, 2, 4, 5]
    assert round_record["details"]["descriptor"]["dtype"] == "float32"


def test_load_diagnostics_rejects_malformed_schema_and_fov_mismatch(tmp_path: Path) -> None:
    path = get_registration_performance_path(tmp_path, FOV_ID)
    payload = {
        "schema_name": REGISTRATION_PERFORMANCE_SCHEMA_NAME,
        "schema_version": REGISTRATION_PERFORMANCE_SCHEMA_VERSION,
        "generated_at_utc": "2026-05-15T00:00:00+00:00",
        "fov_id": FOV_ID,
        "stage_id": "registration",
        "source_stage_elapsed_wall_ms": None,
        "providers": _providers(),
        "registration_method": _providers(),
        "summary": {},
        "fov_setup": {},
        "rounds": [],
        "manifest": {},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="fov/path mismatch"):
        _ = load_registration_performance_diagnostics(path, expected_fov_id=FOV_ID + 1)

    malformed = cast(dict[str, Any], dict(payload))
    malformed["stage_id"] = "spot_finding"
    with pytest.raises(ValueError, match="stage_id"):
        validate_registration_performance_payload(malformed, expected_fov_id=FOV_ID)


def test_validate_diagnostics_rejects_missing_required_schema_fields() -> None:
    recorder = RegistrationPerformanceRecorder(fov_id=FOV_ID, providers=_providers())
    payload = recorder.build_payload()

    missing_generated_at = cast(dict[str, Any], dict(payload))
    del missing_generated_at["generated_at_utc"]
    with pytest.raises(ValueError, match="generated_at_utc"):
        validate_registration_performance_payload(missing_generated_at, expected_fov_id=FOV_ID)

    missing_summary_field = cast(dict[str, Any], dict(payload))
    summary = cast(dict[str, Any], dict(cast(dict[str, Any], missing_summary_field["summary"])))
    del summary["phase_totals"]
    missing_summary_field["summary"] = summary
    with pytest.raises(ValueError, match="summary is missing required fields"):
        validate_registration_performance_payload(missing_summary_field, expected_fov_id=FOV_ID)
