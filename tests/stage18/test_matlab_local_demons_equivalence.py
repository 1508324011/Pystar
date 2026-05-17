from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pystar._io_paths import (
    get_flow_3d_sidecar_filename,
    get_fov_output_structure,
    get_transform_manifest_path,
)
from pystar._registration_equivalence import (
    build_registration_equivalence_report,
    compare_local_demons_requests,
    compare_registration_diagnostics_schema,
    compare_tiled_flow_outputs,
    compare_transform_artifacts,
    write_equivalence_report,
)
from pystar._registration_performance import RegistrationPerformanceRecorder
from pystar.tiling import TileSpec, build_yx_tile_layout


FOV_ID = 7
ROUND_ID = 2
REFERENCE_ROUND = 1
FULL_SHAPE_ZYX = (2, 4, 4)


def _tiles() -> tuple[TileSpec, ...]:
    return build_yx_tile_layout(
        FULL_SHAPE_ZYX,
        grid_shape_yx=(2, 2),
        overlap_yx=(0, 0),
        grid_source="stage18_fixture",
    ).tiles


def _flow_tile(tile: TileSpec) -> NDArray[np.float32]:
    flow = np.zeros((3, *tile.region_shape_zyx), dtype=np.float32)
    flow[0].fill(float(tile.tile_index))
    flow[1].fill(float(tile.grid_position_yx[0]))
    flow[2].fill(float(tile.grid_position_yx[1]))
    return flow


def _request(tile: TileSpec, *, volatile_suffix: str) -> dict[str, Any]:
    return {
        "fov_id": FOV_ID,
        "round_id": ROUND_ID,
        "reference_round": REFERENCE_ROUND,
        "provider": "matlab",
        "method": "demons_3d",
        "coverage_mode": "tile_local",
        "global_shift_already_applied": True,
        "compute_tile": tile.as_dict(),
        "runtime": {
            "entrypoint": "pystar_register_local_demons_entry",
            "manifest_sha256": "sha256:fixture-runtime-manifest",
        },
        "reference_volume_shape_zyx": list(FULL_SHAPE_ZYX),
        "moving_volume_shape_zyx": list(FULL_SHAPE_ZYX),
        "reference_volume_path": f"/tmp/baseline-{volatile_suffix}.tif",
        "moving_volume_path": f"/tmp/candidate-{volatile_suffix}.tif",
        "tmpdir": f"/tmp/stage18-{volatile_suffix}",
        "session_id": f"session-{volatile_suffix}",
        "elapsed_wall_ms": 123.0,
    }


def _flat_matlab_request(tile: TileSpec, *, volatile_suffix: str) -> dict[str, Any]:
    tile_payload = tile.as_dict()
    return {
        "fov_id": FOV_ID,
        "round_id": ROUND_ID,
        "reference_round": REFERENCE_ROUND,
        "provider": "matlab",
        "method": "demons_3d",
        "coverage_mode": "tile_local",
        "global_shift_already_applied": True,
        "compute_tile_index": tile_payload["tile_index"],
        "compute_tile_grid_position_yx": tile_payload["grid_position_yx"],
        "compute_tile_grid_shape_yx": tile_payload["grid_shape_yx"],
        "compute_tile_origin_zyx": tile_payload["region_origin_zyx"],
        "compute_tile_shape_zyx": tile_payload["region_shape_zyx"],
        "compute_tile_write_origin_zyx": tile_payload["write_origin_zyx"],
        "compute_tile_write_shape_zyx": tile_payload["write_shape_zyx"],
        "compute_tile_write_offset_zyx": tile_payload["write_offset_zyx"],
        "full_volume_shape_zyx": tile_payload["full_volume_shape_zyx"],
        "runtime": {
            "entrypoint": "pystar_register_local_demons_entry",
            "runtime_manifest": "/repo/matlab_runtime/pystar_registration/runtime_manifest.json",
        },
        "volume_shape_zyx": list(tile.region_shape_zyx),
        "reference_volume_shape_zyx": list(tile.region_shape_zyx),
        "moving_volume_shape_zyx": list(tile.region_shape_zyx),
        "flow_output_path": f"/tmp/local-flow-{volatile_suffix}.mat",
        "session_name": f"session-{volatile_suffix}",
        "duration_ms": 999.0,
    }


def _tile_results(*, candidate: bool = False) -> list[dict[str, Any]]:
    suffix = "candidate" if candidate else "baseline"
    return [
        {"tile": tile, "flow_tile": _flow_tile(tile), "request": _request(tile, volatile_suffix=suffix)}
        for tile in _tiles()
    ]


def test_identical_tile_flows_pass_and_candidate_order_is_normalized() -> None:
    baseline = _tile_results(candidate=False)
    candidate = list(reversed(_tile_results(candidate=True)))

    report = compare_tiled_flow_outputs(baseline, candidate, full_shape_zyx=FULL_SHAPE_ZYX)

    assert report.passed is True
    assert report.tile_count == 4
    assert report.candidate_order_normalized is True
    assert report.stitched_equal is True
    assert report.stitched_max_abs_diff == 0.0
    assert [record.tile_index for record in report.tile_records] == [1, 2, 3, 4]
    assert all(record.layout_equal for record in report.tile_records)
    assert all(record.request_equal is True for record in report.tile_records)
    assert all(record.flow_equal for record in report.tile_records)


def test_tile_layout_mismatch_fails_loudly() -> None:
    baseline = _tile_results()
    candidate = _tile_results(candidate=True)
    changed_tile = replace(candidate[1]["tile"], write_shape_zyx=(2, 1, 2))
    candidate[1] = {**candidate[1], "tile": changed_tile}

    report = compare_tiled_flow_outputs(baseline, candidate, full_shape_zyx=FULL_SHAPE_ZYX)

    assert report.passed is False
    assert report.tile_records[1].layout_equal is False
    assert any("write_shape_zyx" in diff for diff in report.tile_records[1].differences)
    assert any("tile field 'write_shape_zyx' mismatch" in diff for diff in report.differences)


def test_per_tile_flow_value_mismatch_fails_tile_and_stitched_comparison() -> None:
    baseline = _tile_results()
    candidate = _tile_results(candidate=True)
    changed_flow = np.array(candidate[2]["flow_tile"], copy=True)
    changed_flow[0, 0, 0, 0] += np.float32(1.0)
    candidate[2] = {**candidate[2], "flow_tile": changed_flow}

    report = compare_tiled_flow_outputs(baseline, candidate, full_shape_zyx=FULL_SHAPE_ZYX)

    assert report.passed is False
    assert report.tile_records[2].flow_equal is False
    assert report.tile_records[2].max_abs_diff == 1.0
    assert report.stitched_equal is False
    assert report.stitched_max_abs_diff == 1.0
    assert any("tile 3 flow_tile max_abs_diff" in diff for diff in report.differences)
    assert any("stitched flow_3d max_abs_diff" in diff for diff in report.differences)


def test_request_fingerprint_ignores_volatile_paths_sessions_and_timings() -> None:
    tile = _tiles()[0]
    baseline = _request(tile, volatile_suffix="baseline")
    candidate = _request(tile, volatile_suffix="candidate")

    equivalent = compare_local_demons_requests(baseline, candidate)

    assert equivalent.passed is True
    assert equivalent.baseline_hash == equivalent.candidate_hash

    drifted = dict(candidate)
    drifted["method"] = "native_demons"
    report = compare_local_demons_requests(baseline, drifted)

    assert report.passed is False
    assert any("method" in diff for diff in report.differences)


def test_request_comparison_accepts_actual_flat_matlab_plan_tile_fields() -> None:
    tile = _tiles()[0]
    baseline = _flat_matlab_request(tile, volatile_suffix="baseline")
    candidate = _flat_matlab_request(tile, volatile_suffix="candidate")

    equivalent = compare_local_demons_requests(baseline, candidate)

    assert equivalent.passed is True
    assert equivalent.baseline_hash == equivalent.candidate_hash

    drifted = _flat_matlab_request(tile, volatile_suffix="candidate")
    drifted["compute_tile_write_origin_zyx"] = [0, 1, 0]
    report = compare_local_demons_requests(baseline, drifted)

    assert report.passed is False
    assert any("compute_tile" in diff for diff in report.differences)


def test_request_comparison_requires_runtime_tile_and_volume_identity() -> None:
    tile = _tiles()[0]
    baseline = _request(tile, volatile_suffix="baseline")

    missing_entrypoint = _request(tile, volatile_suffix="candidate")
    missing_entrypoint["runtime"] = {"manifest_sha256": "sha256:fixture-runtime-manifest"}
    entrypoint_report = compare_local_demons_requests(baseline, missing_entrypoint)

    assert entrypoint_report.passed is False
    assert any("runtime entrypoint identity" in diff for diff in entrypoint_report.differences)

    missing_shape = _request(tile, volatile_suffix="candidate")
    del missing_shape["moving_volume_shape_zyx"]
    shape_report = compare_local_demons_requests(baseline, missing_shape)

    assert shape_report.passed is False
    assert any("moving shape absent" in diff for diff in shape_report.differences)

    invalid_coverage = _request(tile, volatile_suffix="candidate")
    invalid_coverage["coverage_mode"] = "partial_debug"
    coverage_report = compare_local_demons_requests(baseline, invalid_coverage)

    assert coverage_report.passed is False
    assert any("coverage_mode" in diff and "tile_local" in diff for diff in coverage_report.differences)


def _providers() -> dict[str, Any]:
    return {
        "stage_id": "registration",
        "provider_mode": "matlab_only",
        "global_provider": "matlab",
        "global_method": "phase_corr_3d",
        "local_provider": "matlab",
        "local_method": "demons_3d",
        "enable_local": True,
        "reference_round": REFERENCE_ROUND,
    }


def _boundary_trace(*, matlab_call_ms: float) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "stage_name": "matlab_registration_local",
        "runtime_path": "/tmp/pystar-runtime",
        "entrypoint": "pystar_register_local_demons_entry",
        "call_scope": {"fov_id": FOV_ID, "round_id": ROUND_ID},
        "started_at": "2026-05-16T00:00:00+00:00",
        "finished_at": "2026-05-16T00:00:01+00:00",
        "total_duration_ms": matlab_call_ms + 10.0,
        "engine_reused_this_call": True,
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


def _matlab_internal_metadata(*, total_duration_ms: float) -> dict[str, Any]:
    return {
        "round_id": ROUND_ID,
        "reference_round": REFERENCE_ROUND,
        "total_duration_ms": total_duration_ms,
        "steps": [
            {"name": "new_LoadMultipageTiff", "duration_ms": 1.0, "details": {}},
            {"name": "imregdemons", "duration_ms": total_duration_ms - 2.0, "details": {"iterations": 50}},
            {"name": "save_local_flow_mat", "duration_ms": 1.0, "details": {}},
        ],
    }


def _write_diagnostics(base_dir: Path, *, local_elapsed_ms: float, internal_total_ms: float) -> Path:
    recorder = RegistrationPerformanceRecorder(fov_id=FOV_ID, providers=_providers())
    recorder.start_round(ROUND_ID, is_reference_round=False)
    recorder.record_local_registration(
        ROUND_ID,
        elapsed_wall_ms=local_elapsed_ms,
        provider="matlab",
        method="demons_3d",
        status="accepted",
        final_corr=0.92,
        backend_metadata={
            "local_flow": {
                "matlab_metadata": _matlab_internal_metadata(total_duration_ms=internal_total_ms),
                "boundary_instrumentation": _boundary_trace(matlab_call_ms=internal_total_ms + 5.0),
            }
        },
    )
    recorder.complete_round(ROUND_ID)
    return recorder.write(base_dir, source_stage_elapsed_wall_ms=local_elapsed_ms + 100.0)


def test_diagnostics_timing_differences_are_ignored_when_schema_is_compatible(tmp_path: Path) -> None:
    baseline_path = _write_diagnostics(tmp_path / "baseline", local_elapsed_ms=100.0, internal_total_ms=80.0)
    candidate_path = _write_diagnostics(tmp_path / "candidate", local_elapsed_ms=200.0, internal_total_ms=160.0)

    report = compare_registration_diagnostics_schema(
        baseline_path,
        candidate_path,
        expected_fov_id=FOV_ID,
    )

    assert report.passed is True
    assert report.schema_compatible is True
    assert report.timing_fields_ignored is True
    assert report.differences == ()


def test_diagnostics_missing_stage17_summary_field_fails(tmp_path: Path) -> None:
    baseline_path = _write_diagnostics(tmp_path / "baseline", local_elapsed_ms=100.0, internal_total_ms=80.0)
    candidate_path = _write_diagnostics(tmp_path / "candidate", local_elapsed_ms=200.0, internal_total_ms=160.0)
    payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    del payload["summary"]["matlab_internal_step_totals_ms"]
    candidate_path.write_text(json.dumps(payload), encoding="utf-8")

    report = compare_registration_diagnostics_schema(
        baseline_path,
        candidate_path,
        expected_fov_id=FOV_ID,
    )

    assert report.passed is False
    assert any("matlab_internal_step_totals_ms" in diff for diff in report.differences)


def _release_contract() -> dict[str, Any]:
    return {
        "requested_scope_mode": "full_fov",
        "delivered_coverage": "full_fov",
        "scope_valid": True,
        "scope_status": "valid",
        "requested_intent": {"registration": "image_warp"},
        "delivered_capability": {"registration": "image_warp"},
        "field_semantics_contract": {
            "flow_3d": {
                "representation": "total",
                "composition": "sequential_global_then_local",
                "status": "settled",
            }
        },
        "release_gate": {"status": "valid"},
    }


def _round_entry(descriptor: dict[str, Any], *, semantics_status: str) -> dict[str, Any]:
    return {
        "global_shift_3d": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        "global_corr": 0.9,
        "flow_2d": None,
        "flow_3d": descriptor,
        "final_corr": 0.91,
        "round_id": ROUND_ID,
        "_scope": {
            "coverage_mode": "full_fov",
            "region_origin_zyx": [0, 0, 0],
            "region_shape_zyx": list(FULL_SHAPE_ZYX),
            "full_volume_shape_zyx": list(FULL_SHAPE_ZYX),
        },
        "_semantics": {
            "representation": "total",
            "composition": "sequential_global_then_local",
            "status": semantics_status,
        },
        "backend_metadata": {"elapsed_wall_ms": 123.0},
    }


def _write_transform_bundle(base_dir: Path, *, flow: NDArray[np.float32], semantics_status: str = "settled") -> Path:
    _ = get_fov_output_structure(base_dir, FOV_ID)
    manifest_path = get_transform_manifest_path(base_dir, FOV_ID)
    sidecar_name = get_flow_3d_sidecar_filename(FOV_ID, ROUND_ID)
    sidecar_path = manifest_path.parent / sidecar_name
    np.save(sidecar_path, flow, allow_pickle=False)
    descriptor = {
        "storage": "round_level_sidecar_npy",
        "path": sidecar_name,
        "shape": list(flow.shape),
        "dtype": str(flow.dtype),
    }
    contract = _release_contract()
    payload = {
        ROUND_ID: _round_entry(descriptor, semantics_status=semantics_status),
        "_contract": contract,
        "_provenance": {
            "release_contract": contract,
            "runtime_context": {"duration_ms": 999.0, "tmpdir": "/tmp/ignored"},
        },
    }
    np.save(manifest_path, np.asarray(payload, dtype=object))
    return manifest_path


def test_transform_artifact_comparison_detects_sidecar_and_manifest_semantic_drift(tmp_path: Path) -> None:
    flow = np.zeros((3, *FULL_SHAPE_ZYX), dtype=np.float32)
    _write_transform_bundle(tmp_path / "baseline", flow=flow)
    _write_transform_bundle(tmp_path / "candidate", flow=flow)

    equivalent = compare_transform_artifacts(
        tmp_path / "baseline",
        tmp_path / "candidate",
        fov_id=FOV_ID,
        round_id=ROUND_ID,
    )

    assert equivalent.passed is True
    assert equivalent.sidecar_contract_equal is True
    assert equivalent.manifest_semantics_equal is True

    changed_flow = np.array(flow, copy=True)
    changed_flow[0, 0, 0, 0] = np.float32(1.0)
    _write_transform_bundle(tmp_path / "candidate", flow=changed_flow)
    sidecar_drifted = compare_transform_artifacts(
        tmp_path / "baseline",
        tmp_path / "candidate",
        fov_id=FOV_ID,
        round_id=ROUND_ID,
    )

    assert sidecar_drifted.passed is False
    assert sidecar_drifted.sidecar_contract_equal is False
    assert sidecar_drifted.manifest_semantics_equal is True
    assert any("flow_3d sidecar raw byte hash mismatch" in diff for diff in sidecar_drifted.differences)

    _write_transform_bundle(tmp_path / "candidate", flow=flow, semantics_status="provisional")
    drifted = compare_transform_artifacts(
        tmp_path / "baseline",
        tmp_path / "candidate",
        fov_id=FOV_ID,
        round_id=ROUND_ID,
    )

    assert drifted.passed is False
    assert drifted.sidecar_contract_equal is True
    assert drifted.manifest_semantics_equal is False
    assert any("round entry semantic payload drifted" in diff for diff in drifted.differences)


def test_equivalence_report_writer_uses_json_safe_report_shape(tmp_path: Path) -> None:
    tiled_report = compare_tiled_flow_outputs(_tile_results(), _tile_results(candidate=True), full_shape_zyx=FULL_SHAPE_ZYX)
    combined = build_registration_equivalence_report(tiled_flow=tiled_report)

    path = write_equivalence_report(tmp_path / "stage18_equivalence.json", combined)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["passed"] is True
    assert payload["tiled_flow"]["tile_count"] == 4
    assert payload["differences"] == []
