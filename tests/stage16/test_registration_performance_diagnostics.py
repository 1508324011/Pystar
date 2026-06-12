from __future__ import annotations

import json
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import xarray as xr
from numpy.typing import NDArray

from pystar._registration_performance import (
    REGISTRATION_PERFORMANCE_SCHEMA_NAME,
    REGISTRATION_PERFORMANCE_SCHEMA_VERSION,
    RegistrationPerformanceRecorder,
    array_diagnostics_descriptor,
    array_shape_diagnostics_descriptor,
    get_registration_performance_path,
    load_registration_performance_diagnostics,
    memory_delta_diagnostics,
    timing_record,
    validate_registration_performance_payload,
)
from pystar.io import get_flow_3d_sidecar_filename, get_fov_output_structure, load_transform_manifest
from pystar.registration import RegistrationEngine, _run_tiled_native_demons_registration
from pystar.tiling import build_yx_tile_layout, extract_tile, stitch_tiles


FOV_ID = 5
FloatArray = NDArray[np.float32]


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
    ref_volume = np.zeros((2, 4, 5), dtype=np.float32)
    mov_volume = np.ones((2, 4, 5), dtype=np.float32)
    recorder.record_registration_work_plan(
        2,
        {
            "round_id": 2,
            "reference_round": 1,
            "global_provider": "native",
            "global_method": "phase_corr_3d",
            "local_provider": "native",
            "local_method": "demons_3d",
            "local_enabled": True,
            "local_execution_mode": "full_volume_native_demons_3d",
            "selection_reason": "synthetic test plan",
            "scope_descriptor": {
                "coverage_mode": "full_fov",
                "region_origin_zyx": [0, 0, 0],
                "region_shape_zyx": [2, 4, 5],
                "full_volume_shape_zyx": [2, 4, 5],
            },
            "scope_coverage_mode": "full_fov",
            "scope_region_origin_zyx": [0, 0, 0],
            "scope_region_shape_zyx": [2, 4, 5],
            "reference_volume": array_diagnostics_descriptor(ref_volume, role="reference_scope_3d"),
            "moving_volume": array_diagnostics_descriptor(mov_volume, role="moving_scope_3d"),
            "expected_local_flow": array_shape_diagnostics_descriptor((3, 2, 4, 5), role="expected_flow_3d"),
            "expected_flow_3d": array_shape_diagnostics_descriptor((3, 2, 4, 5), role="expected_flow_3d"),
            "flow_3d_persisted": False,
            "flow_3d_sidecar_path": None,
            "flow_3d_sidecar_size_bytes": None,
        },
    )
    recorder.record_memory_telemetry(
        2,
        memory_delta_diagnostics(
            "native_demons_registration_compute",
            {"source": "procfs", "rss_bytes": 1000, "available_memory_bytes": 8000},
            {"source": "procfs", "rss_bytes": 1256, "available_memory_bytes": 7600},
            details={"local_execution_mode": "full_volume_native_demons_3d"},
        ),
    )
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
    recorder.record_manifest_timing(
        "load_transform_manifest",
        12.0,
        details={"load_provenance": False, "hydrate_flow_3d": False},
    )

    output_path = recorder.write(tmp_path, source_stage_elapsed_wall_ms=100.0)

    assert output_path == get_registration_performance_path(tmp_path, FOV_ID)
    payload = load_registration_performance_diagnostics(output_path, expected_fov_id=FOV_ID)
    assert payload["schema_name"] == REGISTRATION_PERFORMANCE_SCHEMA_NAME
    assert payload["schema_version"] == REGISTRATION_PERFORMANCE_SCHEMA_VERSION
    assert payload["stage_id"] == "registration"
    assert payload["source_stage_elapsed_wall_ms"] == 100.0
    assert payload["summary"]["moving_round_count"] == 1
    assert payload["summary"]["registration_work_plan_status"] == "present"
    assert payload["summary"]["registration_work_plan_count"] == 1
    assert payload["summary"]["registration_local_execution_mode_counts"] == {"full_volume_native_demons_3d": 1}
    assert payload["summary"]["registration_reference_volume_total_nbytes"] == ref_volume.nbytes
    assert payload["summary"]["registration_moving_volume_total_nbytes"] == mov_volume.nbytes
    assert payload["summary"]["registration_expected_flow_total_nbytes"] == 3 * ref_volume.nbytes
    assert payload["summary"]["registration_memory_telemetry_status"] == "present"
    assert payload["summary"]["registration_memory_telemetry_sample_count"] == 1
    assert payload["summary"]["registration_rss_delta_max_bytes"] == 256
    assert payload["rounds"][1]["work_plan"]["reference_volume"]["nbytes"] == ref_volume.nbytes
    assert payload["rounds"][1]["memory_telemetry"][0]["rss_delta_bytes"] == 256
    assert payload["summary"]["matlab_boundary_call_count"] == 0
    assert payload["manifest"]["save_transform_manifest"]["elapsed_wall_ms"] == 11.0
    assert payload["manifest"]["load_transform_manifest"]["elapsed_wall_ms"] == 12.0
    assert payload["manifest"]["load_transform_manifest"]["details"] == {
        "load_provenance": False,
        "hydrate_flow_3d": False,
    }
    assert "save_transform_manifest" not in payload["rounds"][1]


def test_register_fov_returns_lazy_sidecar_manifest_and_records_lazy_reload(tmp_path: Path) -> None:
    field_semantics = SimpleNamespace(
        as_dict=lambda: {
            "representation": "residual",
            "composition": "sequential_global_then_local",
            "status": "settled",
        }
    )
    registration = SimpleNamespace(
        reference_round=1,
        field_semantics=field_semantics,
        global_provider="native",
        local_provider="native",
        local_method="demons_3d",
        enable_local=True,
        global_stage=SimpleNamespace(method="phase_corr_3d"),
    )
    pipeline = SimpleNamespace(
        output=SimpleNamespace(directory=str(tmp_path)),
        registration=registration,
        scope_mode="full_fov",
        registration_provider_mode=lambda: "native_only",
        qc_images_enabled=lambda: False,
    )
    cfg = cast(Any, SimpleNamespace(pipeline=pipeline))
    engine = RegistrationEngine(cfg)

    flow_3d = np.arange(3 * 2 * 4 * 4, dtype=np.float32).reshape(3, 2, 4, 4)

    def fake_load_volume(self: RegistrationEngine, fov_id: int, round_id: int) -> FloatArray:
        return np.full((2, 4, 4), float(round_id), dtype=np.float32)

    def fake_register_round(self: RegistrationEngine, **kwargs: Any) -> tuple[dict[str, Any], FloatArray, dict[str, Any]]:
        return (
            {
                "global_shift_3d": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
                "global_corr": 0.95,
                "flow_2d": None,
                "flow_3d": flow_3d,
                "final_corr": 0.96,
                "is_reference_round": False,
            },
            np.zeros((4, 4), dtype=np.float32),
            {"backend": "synthetic"},
        )

    def no_provenance(self: RegistrationEngine, *args: Any, **kwargs: Any) -> None:
        return None

    engine._load_combined_clean_volume = MethodType(fake_load_volume, engine)  # type: ignore[method-assign]
    engine._register_round = MethodType(fake_register_round, engine)  # type: ignore[method-assign]
    engine._build_provenance = MethodType(no_provenance, engine)  # type: ignore[method-assign]

    data = xr.DataArray(
        np.zeros((2,), dtype=np.float32),
        dims=("round",),
        coords={"round": [1, 2]},
    )

    returned_manifest = engine.register_fov(data, FOV_ID)
    round_two = cast(dict[str, Any], returned_manifest[2])
    returned_flow = cast(dict[str, Any], round_two["flow_3d"])

    assert returned_flow == {
        "storage": "round_level_sidecar_npy",
        "path": get_flow_3d_sidecar_filename(FOV_ID, 2),
        "shape": [3, 2, 4, 4],
        "dtype": "float32",
    }

    default_loaded = load_transform_manifest(tmp_path, FOV_ID)
    default_flow = cast(dict[int, dict[str, Any]], default_loaded)[2]["flow_3d"]
    assert isinstance(default_flow, np.memmap)
    assert default_flow.mode == "r"
    assert default_flow.flags.writeable is False
    np.testing.assert_array_equal(cast(FloatArray, default_flow), flow_3d)

    diagnostics_path = get_registration_performance_path(tmp_path, FOV_ID)
    payload = load_registration_performance_diagnostics(diagnostics_path, expected_fov_id=FOV_ID)
    assert payload["manifest"]["load_transform_manifest"]["details"] == {
        "load_provenance": False,
        "hydrate_flow_3d": False,
    }


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


def test_tiled_native_local_diagnostics_preserve_array_descriptors_without_matlab_boundary() -> None:
    recorder = RegistrationPerformanceRecorder(fov_id=FOV_ID, providers=_providers())
    recorder.start_round(2, is_reference_round=False)
    recorder.record_tiled_local_tile(
        2,
        tile_identity={
            "tile_index": 1,
            "grid_position_yx": [0, 0],
            "grid_shape_yx": [1, 2],
            "region_origin_zyx": [0, 0, 0],
            "region_shape_zyx": [2, 4, 5],
            "write_origin_zyx": [0, 0, 0],
            "write_shape_zyx": [2, 4, 5],
            "write_offset_zyx": [0, 0, 0],
            "full_volume_shape_zyx": [2, 4, 10],
        },
        total_elapsed_wall_ms=30.0,
        extraction_elapsed_wall_ms=4.0,
        backend_call_elapsed_wall_ms=21.0,
        flow_validation_elapsed_wall_ms=5.0,
        normalized_result={
            "provider": "native",
            "method": "demons_3d",
            "flow_3d_shape": [3, 2, 4, 5],
            "flow_3d_dtype": "float32",
            "flow_3d_nbytes": 3 * 2 * 4 * 5 * 4,
            "flow_3d_descriptor": array_shape_diagnostics_descriptor(
                (3, 2, 4, 5),
                dtype=np.float32,
                role="native_tiled_flow_3d",
            ),
            "mean_abs_displacement": 0.25,
        },
    )
    recorder.record_tiled_local_summary(
        2,
        layout_summary={"enabled": True, "grid_shape_yx": [1, 2], "tile_count": 2},
        stitch_elapsed_wall_ms=3.0,
    )

    payload = recorder.build_payload()
    tile = payload["rounds"][0]["tiled_local"]["tiles"][0]

    assert tile["normalized_result"]["provider"] == "native"
    assert tile["normalized_result"]["method"] == "demons_3d"
    assert tile["normalized_result"]["flow_3d_shape"] == [3, 2, 4, 5]
    assert tile["normalized_result"]["flow_3d_dtype"] == "float32"
    assert tile["normalized_result"]["flow_3d_nbytes"] == 480
    assert tile["normalized_result"]["flow_3d_descriptor"]["nbytes"] == 480
    assert tile["normalized_result"]["mean_abs_displacement"] == 0.25
    assert tile["timings"]["tile_extraction"]["elapsed_wall_ms"] == 4.0
    assert tile["timings"]["backend_call"]["elapsed_wall_ms"] == 21.0
    assert tile["timings"]["flow_validation"]["elapsed_wall_ms"] == 5.0
    assert payload["summary"]["tile_count"] == 1
    assert payload["summary"]["matlab_local_boundary_call_count"] == 0
    assert payload["summary"]["nested_phase_totals_ms"]["backend_call"] == 21.0
    assert payload["summary"]["slowest_tiles"][0]["tile_index"] == 1


def test_native_tiled_demons_runtime_records_per_tile_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:
    import pystar.registration as registration_module

    ref = np.arange(2 * 4 * 4, dtype=np.float32).reshape((2, 4, 4))
    mov = ref + np.float32(1.0)
    layout = build_yx_tile_layout(
        (2, 4, 4),
        grid_shape_yx=(2, 2),
        overlap_yx=(0, 0),
        grid_source="stage16_native_fixture",
    )
    recorder = RegistrationPerformanceRecorder(fov_id=FOV_ID, providers=_providers())
    recorder.start_round(2, is_reference_round=False)

    def fake_native_demons(ref_tile: FloatArray, mov_tile: FloatArray, config_obj: object) -> FloatArray:
        del mov_tile, config_obj
        return np.full((3, *ref_tile.shape), float(ref_tile.mean()), dtype=np.float32)

    monkeypatch.setattr(registration_module, "register_local_demons_3d", fake_native_demons)

    stitched, summary = _run_tiled_native_demons_registration(
        ref,
        mov,
        object(),
        layout,
        round_id=2,
        diagnostics=recorder,
    )
    assert stitched is not None

    expected_tiles = []
    for tile in layout.tiles:
        expected_tile_value = float(np.asarray(extract_tile(ref, tile), dtype=np.float32).mean())
        expected_tiles.append(
            (
                tile,
                np.full((3, *tile.region_shape_zyx), expected_tile_value, dtype=np.float32),
            )
        )
    expected = stitch_tiles(expected_tiles, full_shape_zyx=(2, 4, 4))
    np.testing.assert_array_equal(stitched, expected)
    assert summary["tiles"][0]["provider"] == "native"

    payload = recorder.build_payload()
    tiled = payload["rounds"][0]["tiled_local"]
    assert tiled["layout"]["tile_count"] == 4
    assert len(tiled["tiles"]) == 4
    first_tile = tiled["tiles"][0]
    assert first_tile["timings"]["tile_extraction"]["elapsed_wall_ms"] >= 0.0
    assert first_tile["timings"]["backend_call"]["elapsed_wall_ms"] >= 0.0
    assert first_tile["timings"]["flow_validation"]["elapsed_wall_ms"] >= 0.0
    assert first_tile["normalized_result"]["provider"] == "native"
    assert first_tile["normalized_result"]["method"] == "demons_3d"
    assert first_tile["normalized_result"]["flow_3d_shape"] == [3, 2, 2, 2]
    assert first_tile["normalized_result"]["flow_3d_dtype"] == "float32"
    assert first_tile["normalized_result"]["flow_3d_descriptor"]["role"] == "native_tiled_flow_3d"
    assert first_tile["normalized_result"]["flow_3d_nbytes"] == 3 * 2 * 2 * 2 * 4
    expected_first_mean = float(np.asarray(extract_tile(ref, layout.tiles[0]), dtype=np.float32).mean())
    assert first_tile["normalized_result"]["mean_abs_displacement"] == pytest.approx(expected_first_mean)
    assert payload["summary"]["tile_count"] == 4
    assert payload["summary"]["nested_phase_totals_ms"]["tiled_local_stitch"] >= 0.0
    assert payload["summary"]["matlab_local_boundary_call_count"] == 0
    assert payload["summary"]["slowest_tiles"][0]["tile_index"] in {1, 2, 3, 4}


def test_native_demons_work_plan_records_shifted_volume_descriptor(monkeypatch: pytest.MonkeyPatch) -> None:
    import pystar.registration as registration_module

    registration_cfg = SimpleNamespace(
        demons_3d=SimpleNamespace(use_tiling=False),
        guards=SimpleNamespace(reject_if_correlation_worse=True),
    )
    cfg = cast(Any, SimpleNamespace(pipeline=SimpleNamespace(registration=registration_cfg)))
    engine = RegistrationEngine(cfg)
    recorder = RegistrationPerformanceRecorder(fov_id=FOV_ID, providers=_providers())
    engine._registration_diagnostics = recorder
    recorder.start_round(2, is_reference_round=False)

    ref_volume = np.arange(2 * 4 * 4, dtype=np.float32).reshape((2, 4, 4))
    mov_volume = np.array(ref_volume, copy=True)
    ref_mip = np.asarray(ref_volume.max(axis=0), dtype=np.float32)
    mov_mip = np.asarray(mov_volume.max(axis=0), dtype=np.float32)
    global_shift = np.zeros(3, dtype=np.float32)
    shift_2d = np.zeros(2, dtype=np.float32)
    scope_descriptor = {
        "coverage_mode": "full_fov",
        "region_origin_zyx": [0, 0, 0],
        "region_shape_zyx": [2, 4, 4],
        "full_volume_shape_zyx": [2, 4, 4],
    }
    recorder.record_registration_work_plan(
        2,
        {
            "round_id": 2,
            "reference_round": 1,
            "global_provider": "native",
            "global_method": "phase_corr_3d",
            "local_provider": "native",
            "local_method": "demons_3d",
            "local_enabled": True,
            "local_execution_mode": "native_demons_3d",
            "selection_reason": "synthetic shifted-volume work-plan test",
            "scope_descriptor": scope_descriptor,
            "scope_coverage_mode": "full_fov",
            "scope_region_origin_zyx": [0, 0, 0],
            "scope_region_shape_zyx": [2, 4, 4],
            "reference_volume": array_diagnostics_descriptor(ref_volume, role="reference_scope_3d"),
            "moving_volume": array_diagnostics_descriptor(mov_volume, role="moving_scope_3d"),
            "expected_local_flow": array_shape_diagnostics_descriptor((3, 2, 4, 4), role="expected_flow_3d"),
            "expected_flow_3d": array_shape_diagnostics_descriptor((3, 2, 4, 4), role="expected_flow_3d"),
            "flow_3d_persisted": False,
            "flow_3d_sidecar_path": None,
            "flow_3d_sidecar_size_bytes": None,
        },
    )

    def fake_native_demons(ref_tile: FloatArray, mov_tile: FloatArray, config_obj: object) -> FloatArray:
        del ref_tile, mov_tile, config_obj
        return np.zeros((3, 2, 4, 4), dtype=np.float32)

    monkeypatch.setattr(registration_module, "register_local_demons_3d", fake_native_demons)
    context = engine._build_local_registration_context(
        fov_id=FOV_ID,
        round_id=2,
        ref_round=1,
        ref_scope_3d=ref_volume,
        ref_mip_clean=ref_mip,
        mov_scope_3d=mov_volume,
        mov_mip_clean=mov_mip,
        mov_mip_shifted=mov_mip,
        shift_2d=shift_2d,
        global_shift_3d=global_shift,
        corr_after_global=0.9,
        scope_descriptor=scope_descriptor,
        backend_metadata=None,
        local_provider="native",
        local_method="demons_3d",
        local_handler_name="_run_local_native_demons_3d",
    )

    outcome = engine._run_local_native_demons_3d(context)
    assert outcome.flow_3d is not None

    payload = recorder.build_payload()
    shifted_descriptor = payload["rounds"][0]["work_plan"]["moving_shifted_volume"]

    assert shifted_descriptor == {
        "shape": [2, 4, 4],
        "dtype": "float32",
        "nbytes": int(mov_volume.nbytes),
        "ndim": 3,
        "role": "moving_shifted_scope_3d",
    }


def test_native_tiled_demons_bad_tile_flow_shape_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    import pystar.registration as registration_module

    ref = np.zeros((2, 4, 4), dtype=np.float32)
    mov = np.ones((2, 4, 4), dtype=np.float32)
    layout = build_yx_tile_layout(
        (2, 4, 4),
        grid_shape_yx=(2, 2),
        overlap_yx=(0, 0),
        grid_source="stage16_native_fixture",
    )

    def fake_bad_native_demons(ref_tile: FloatArray, mov_tile: FloatArray, config_obj: object) -> FloatArray:
        del mov_tile, config_obj
        return np.zeros((3, ref_tile.shape[0], ref_tile.shape[1], ref_tile.shape[2] + 1), dtype=np.float32)

    monkeypatch.setattr(registration_module, "register_local_demons_3d", fake_bad_native_demons)

    with pytest.raises(
        ValueError,
        match="Native tiled demons_3d returned flow_3d with incompatible tile shape",
    ):
        _ = _run_tiled_native_demons_registration(
            ref,
            mov,
            object(),
            layout,
        )


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

    hot_path = summary["matlab_local_hot_path_profile"]
    assert hot_path["status"] == "present"
    assert hot_path["source"] == "matlab_metadata.steps + boundary_instrumentation.seam_costs_ms.matlab_call_ms"
    assert hot_path["scope"] == "matlab_local_registration"
    assert hot_path["call_count"] == 1
    assert hot_path["boundary_closure_count"] == 1
    assert hot_path["boundary_closure_complete"] is True
    assert hot_path["boundary_matlab_call_total_ms"] == 106.0
    assert hot_path["matlab_internal_total_duration_ms"] == 100.0
    assert hot_path["boundary_minus_internal_total_ms"] == 6.0
    assert hot_path["matlab_internal_unaccounted_total_ms"] == 5.0
    assert hot_path["step_totals_ms"] == summary["matlab_internal_step_totals_ms"]
    assert hot_path["step_call_counts"] == {
        "imregdemons": 1,
        "new_LoadMultipageTiff": 1,
        "save_local_flow_mat": 1,
    }
    assert hot_path["step_percent_of_matlab_internal_total"]["imregdemons"] == 80.0
    assert hot_path["step_percent_of_boundary_matlab_call"]["imregdemons"] == 75.472
    assert hot_path["dominant_internal_step"] == {
        "name": "imregdemons",
        "total_duration_ms": 80.0,
        "call_count": 1,
        "percent_of_matlab_internal_total": 80.0,
        "percent_of_boundary_matlab_call": 75.472,
        "owner": "matlab_runtime/pystar_registration/pystar_register_local_demons_entry.m",
    }
    assert hot_path["hot_path_rankings"][0]["component_type"] == "matlab_step"
    assert hot_path["hot_path_rankings"][0]["name"] == "imregdemons"
    assert hot_path["hot_path_rankings"][0]["total_duration_ms"] == 80.0


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
    assert payload["summary"]["matlab_local_hot_path_profile"]["status"] == "absent"
    assert payload["summary"]["matlab_local_hot_path_profile"]["call_count"] == 0
    assert payload["summary"]["matlab_local_hot_path_profile"]["hot_path_rankings"] == []


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
    assert payload["summary"]["matlab_local_hot_path_profile"]["boundary_closure_complete"] is False
    assert payload["summary"]["matlab_local_hot_path_profile"]["boundary_matlab_call_total_ms"] is None
    assert payload["summary"]["matlab_local_hot_path_profile"]["boundary_minus_internal_total_ms"] is None


def test_stage19_hot_path_profile_is_optional_for_older_diagnostics() -> None:
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
    legacy_payload = dict(payload)
    legacy_summary = dict(legacy_payload["summary"])
    legacy_summary.pop("matlab_local_hot_path_profile")
    legacy_payload["summary"] = legacy_summary

    validate_registration_performance_payload(legacy_payload, expected_fov_id=FOV_ID)


def test_malformed_stage19_hot_path_profile_fails_payload_validation() -> None:
    recorder = RegistrationPerformanceRecorder(fov_id=FOV_ID, providers=_providers("matlab_only"))
    payload = recorder.build_payload()
    malformed = dict(payload)
    malformed_summary = dict(malformed["summary"])
    malformed_summary["matlab_local_hot_path_profile"] = {
        "schema_version": "1.0",
        "status": "present",
        "source": "matlab_metadata.steps + boundary_instrumentation.seam_costs_ms.matlab_call_ms",
        "scope": "matlab_local_registration",
        "call_count": 1,
        "boundary_closure_count": 1,
        "boundary_closure_complete": True,
        "boundary_matlab_call_total_ms": 10.0,
        "matlab_internal_total_duration_ms": 9.0,
        "boundary_minus_internal_total_ms": 1.0,
        "matlab_internal_unaccounted_total_ms": 0.0,
        "step_totals_ms": {"imregdemons": -1.0},
        "step_call_counts": {"imregdemons": 1},
        "step_percent_of_matlab_internal_total": {"imregdemons": 100.0},
        "step_percent_of_boundary_matlab_call": {"imregdemons": 90.0},
        "dominant_internal_step": None,
        "hot_path_rankings": [],
    }
    malformed["summary"] = malformed_summary

    with pytest.raises(ValueError, match="matlab_local_hot_path_profile.*step_totals_ms"):
        validate_registration_performance_payload(malformed, expected_fov_id=FOV_ID)


def test_partial_stage19_hot_path_boundary_closure_fails_payload_validation() -> None:
    recorder = RegistrationPerformanceRecorder(fov_id=FOV_ID, providers=_providers("matlab_only"))
    payload = recorder.build_payload()
    malformed = dict(payload)
    malformed_summary = dict(malformed["summary"])
    malformed_summary["matlab_local_hot_path_profile"] = {
        "schema_version": "1.0",
        "status": "present",
        "source": "matlab_metadata.steps + boundary_instrumentation.seam_costs_ms.matlab_call_ms",
        "scope": "matlab_local_registration",
        "call_count": 2,
        "boundary_closure_count": 1,
        "boundary_closure_complete": False,
        "boundary_matlab_call_total_ms": 10.0,
        "matlab_internal_total_duration_ms": 9.0,
        "boundary_minus_internal_total_ms": 1.0,
        "matlab_internal_unaccounted_total_ms": 0.0,
        "step_totals_ms": {"imregdemons": 9.0},
        "step_call_counts": {"imregdemons": 1},
        "step_percent_of_matlab_internal_total": {"imregdemons": 100.0},
        "step_percent_of_boundary_matlab_call": {},
        "dominant_internal_step": {
            "name": "imregdemons",
            "total_duration_ms": 9.0,
            "call_count": 1,
            "percent_of_matlab_internal_total": 100.0,
            "percent_of_boundary_matlab_call": None,
            "owner": "matlab_runtime/pystar_registration/pystar_register_local_demons_entry.m",
        },
        "hot_path_rankings": [
            {
                "component_type": "matlab_step",
                "name": "imregdemons",
                "owner": "matlab_runtime/pystar_registration/pystar_register_local_demons_entry.m",
                "total_duration_ms": 9.0,
                "percent_of_matlab_internal_total": 100.0,
                "percent_of_boundary_matlab_call": None,
                "call_count": 1,
            }
        ],
    }
    malformed["summary"] = malformed_summary

    with pytest.raises(ValueError, match="boundary_matlab_call_total_ms.*closure is incomplete"):
        validate_registration_performance_payload(malformed, expected_fov_id=FOV_ID)


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


def test_malformed_work_plan_and_memory_telemetry_fail_payload_validation() -> None:
    recorder = RegistrationPerformanceRecorder(fov_id=FOV_ID, providers=_providers())
    recorder.start_round(2, is_reference_round=False)
    recorder.record_registration_work_plan(
        2,
        {
            "round_id": 2,
            "reference_round": 1,
            "local_execution_mode": "full_volume_native_demons_3d",
            "reference_volume": array_shape_diagnostics_descriptor((2, 4, 5), role="reference_scope_3d"),
            "moving_volume": array_shape_diagnostics_descriptor((2, 4, 5), role="moving_scope_3d"),
            "expected_local_flow": array_shape_diagnostics_descriptor((3, 2, 4, 5), role="expected_flow_3d"),
        },
    )
    recorder.record_memory_telemetry(
        2,
        memory_delta_diagnostics(
            "local_registration_dispatch",
            {"source": "procfs", "rss_bytes": 1000, "available_memory_bytes": None},
            {"source": "procfs", "rss_bytes": 900, "available_memory_bytes": None},
        ),
    )
    payload = recorder.build_payload()

    malformed_work_plan = cast(dict[str, Any], dict(payload))
    malformed_work_plan["rounds"] = [dict(round_entry) for round_entry in cast(list[dict[str, Any]], payload["rounds"])]
    bad_plan = dict(cast(dict[str, Any], malformed_work_plan["rounds"][0]["work_plan"]))
    bad_reference_volume = dict(cast(dict[str, Any], bad_plan["reference_volume"]))
    bad_reference_volume["nbytes"] = bad_reference_volume["nbytes"] + 1
    bad_plan["reference_volume"] = bad_reference_volume
    malformed_work_plan["rounds"][0]["work_plan"] = bad_plan
    with pytest.raises(ValueError, match="work_plan.*reference_volume.*nbytes"):
        validate_registration_performance_payload(malformed_work_plan, expected_fov_id=FOV_ID)

    malformed_sidecar_path = cast(dict[str, Any], dict(payload))
    malformed_sidecar_path["rounds"] = [
        dict(round_entry) for round_entry in cast(list[dict[str, Any]], payload["rounds"])
    ]
    bad_sidecar_plan = dict(cast(dict[str, Any], malformed_sidecar_path["rounds"][0]["work_plan"]))
    bad_sidecar_plan["flow_3d_sidecar_path"] = 123
    malformed_sidecar_path["rounds"][0]["work_plan"] = bad_sidecar_plan
    with pytest.raises(ValueError, match="work_plan.*flow_3d_sidecar_path"):
        validate_registration_performance_payload(malformed_sidecar_path, expected_fov_id=FOV_ID)

    malformed_memory = cast(dict[str, Any], dict(payload))
    malformed_memory["rounds"] = [dict(round_entry) for round_entry in cast(list[dict[str, Any]], payload["rounds"])]
    memory_records = [dict(record) for record in cast(list[dict[str, Any]], malformed_memory["rounds"][0]["memory_telemetry"])]
    memory_records[0]["rss_delta_bytes"] = 12345
    malformed_memory["rounds"][0]["memory_telemetry"] = memory_records
    with pytest.raises(ValueError, match="memory_telemetry.*rss_delta_bytes"):
        validate_registration_performance_payload(malformed_memory, expected_fov_id=FOV_ID)


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
    recorder.record_registration_work_plan(
        2,
        {
            "round_id": 2,
            "reference_round": 1,
            "global_provider": "native",
            "global_method": "phase_corr_3d",
            "local_provider": "native",
            "local_method": "demons_3d",
            "local_execution_mode": "full_volume_native_demons_3d",
            "selection_reason": "synthetic sidecar test",
            "reference_volume": array_shape_diagnostics_descriptor((2, 4, 5), role="reference_scope_3d"),
            "moving_volume": array_shape_diagnostics_descriptor((2, 4, 5), role="moving_scope_3d"),
            "expected_local_flow": array_shape_diagnostics_descriptor(flow.shape, role="expected_flow_3d"),
            "expected_flow_3d": array_shape_diagnostics_descriptor(flow.shape, role="expected_flow_3d"),
            "flow_3d_persisted": False,
            "flow_3d_sidecar_path": None,
            "flow_3d_sidecar_size_bytes": None,
        },
    )
    recorder.record_flow_sidecar_persistence(
        2,
        elapsed_wall_ms=13.0,
        descriptor=descriptor,
        sidecar_path=sidecar_path,
        in_memory_flow=flow,
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
    assert round_record["details"]["in_memory_flow"]["nbytes"] == flow.nbytes
    assert payload["rounds"][0]["work_plan"]["flow_3d_persisted"] is True
    assert payload["rounds"][0]["work_plan"]["flow_3d_sidecar_path"] == str(sidecar_path)
    assert payload["rounds"][0]["work_plan"]["flow_3d_sidecar_size_bytes"] == sidecar_path.stat().st_size
    assert payload["rounds"][0]["work_plan"]["produced_flow_3d"]["nbytes"] == flow.nbytes


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
