from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from pystar._registration_equivalence import compare_local_demons_requests, compare_tiled_flow_outputs
from pystar._registration_tile_executor import (
    MatlabLocalTileResult,
    build_matlab_local_tile_execution_plan,
    compare_tile_result_sequence,
    normalize_matlab_local_tile_results,
)
from pystar.infrastructure import MatlabLocalParallelConfig, RegistrationConfig
from pystar.matlab_registration import build_matlab_local_registration_plan
from pystar.registration import _run_tiled_matlab_demons_registration
from pystar.tiling import TileLayout, TileSpec, build_yx_tile_layout, stitch_tiles


FOV_ID = 7
ROUND_ID = 2
REFERENCE_ROUND = 1
FULL_SHAPE_ZYX = (2, 4, 4)


def _layout() -> TileLayout:
    return build_yx_tile_layout(
        FULL_SHAPE_ZYX,
        grid_shape_yx=(2, 2),
        overlap_yx=(0, 0),
        grid_source="stage18b_fixture",
    )


def _scope_descriptor() -> dict[str, Any]:
    return {
        "coverage_mode": "full_fov",
        "region_origin_zyx": [0, 0, 0],
        "region_shape_zyx": list(FULL_SHAPE_ZYX),
        "full_volume_shape_zyx": list(FULL_SHAPE_ZYX),
    }


def _flow_tile(tile: TileSpec) -> np.ndarray:
    flow = np.zeros((3, *tile.region_shape_zyx), dtype=np.float32)
    flow[0].fill(float(tile.tile_index))
    flow[1].fill(float(tile.grid_position_yx[0]))
    flow[2].fill(float(tile.grid_position_yx[1]))
    return flow


def _request(tile: TileSpec) -> dict[str, Any]:
    return {
        "fov_id": FOV_ID,
        "round_id": ROUND_ID,
        "reference_round": REFERENCE_ROUND,
        "provider": "matlab",
        "method": "demons_3d",
        "coverage_mode": "full_fov",
        "global_shift_already_applied": True,
        "compute_tile": tile.as_dict(),
        "runtime": {
            "entrypoint": "pystar_register_local_demons_entry",
            "manifest_sha256": "sha256:fixture-runtime-manifest",
        },
        "reference_volume_shape_zyx": list(tile.region_shape_zyx),
        "moving_volume_shape_zyx": list(tile.region_shape_zyx),
    }


def _backend_metadata(tile: TileSpec, flow: np.ndarray) -> dict[str, Any]:
    return {
        "request": _request(tile),
        "normalized_result": {
            "flow_3d_shape": [int(value) for value in flow.shape],
            "flow_3d_dtype": str(flow.dtype),
            "mean_abs_displacement": float(np.abs(flow).mean()),
        },
    }


def _result(tile: TileSpec, *, flow: np.ndarray | None = None, status: str = "completed") -> MatlabLocalTileResult:
    flow_tile = _flow_tile(tile) if flow is None and status == "completed" else flow
    return MatlabLocalTileResult(
        tile=tile,
        flow_tile=flow_tile,
        backend_metadata={} if flow_tile is None else _backend_metadata(tile, flow_tile),
        total_elapsed_wall_ms=10.0 + float(tile.tile_index),
        extraction_elapsed_wall_ms=1.0,
        backend_call_elapsed_wall_ms=8.0,
        flow_validation_elapsed_wall_ms=1.0,
        status=status,
        error=None if status == "completed" else "synthetic failure",
    )


def _config(*, parallel: bool) -> Any:
    return SimpleNamespace(
        pipeline=SimpleNamespace(
            registration=SimpleNamespace(
                matlab_local_parallel=SimpleNamespace(
                    enabled=parallel,
                    workers=2,
                    strict_equivalence_audit=True,
                )
            )
        )
    )


def _matlab_plan_config() -> Any:
    return SimpleNamespace(
        providers=SimpleNamespace(
            matlab=SimpleNamespace(
                registration=SimpleNamespace(
                    volume_transfer_mode="temporary_tiff",
                    input_volume_dtype="uint8",
                    use_gpu=False,
                )
            )
        ),
        pipeline=SimpleNamespace(
            registration=SimpleNamespace(
                downsample_factor=4,
                global_max_shift=200,
                local_method="demons_3d",
                demons_3d=SimpleNamespace(
                    num_iter=50,
                    smoothing_sigma=1.0,
                    pyramid_levels=None,
                ),
            )
        ),
    )


class _SerialBackend:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def compute_local_flow(
        self,
        ref_tile: np.ndarray,
        mov_tile: np.ndarray,
        *,
        fov_id: int,
        round_id: int,
        reference_round: int,
        scope_descriptor: dict[str, Any],
        compute_tile: dict[str, Any],
    ) -> dict[str, Any]:
        del ref_tile, mov_tile, scope_descriptor
        tile = TileSpec(
            tile_index=int(compute_tile["tile_index"]),
            grid_position_yx=tuple(int(value) for value in compute_tile["grid_position_yx"]),
            grid_shape_yx=tuple(int(value) for value in compute_tile["grid_shape_yx"]),
            region_origin_zyx=tuple(int(value) for value in compute_tile["region_origin_zyx"]),
            region_shape_zyx=tuple(int(value) for value in compute_tile["region_shape_zyx"]),
            write_origin_zyx=tuple(int(value) for value in compute_tile["write_origin_zyx"]),
            write_shape_zyx=tuple(int(value) for value in compute_tile["write_shape_zyx"]),
            write_offset_zyx=tuple(int(value) for value in compute_tile["write_offset_zyx"]),
            full_volume_shape_zyx=tuple(int(value) for value in compute_tile["full_volume_shape_zyx"]),
        )
        self.calls.append(
            {
                "fov_id": fov_id,
                "round_id": round_id,
                "reference_round": reference_round,
                "compute_tile": compute_tile,
            }
        )
        flow = _flow_tile(tile)
        return {"flow_3d": flow, "backend_metadata": _backend_metadata(tile, flow)}


def test_parallel_config_defaults_to_serial_off() -> None:
    cfg = MatlabLocalParallelConfig()

    assert cfg.enabled is False
    assert cfg.workers == 2
    assert cfg.strict_equivalence_audit is True


def test_parallel_config_requires_positive_workers_when_enabled() -> None:
    with pytest.raises(ValueError, match="workers must be a positive integer"):
        _ = MatlabLocalParallelConfig(enabled=True, workers=0)

    with pytest.raises(ValueError, match="workers must be a positive integer"):
        _ = MatlabLocalParallelConfig(enabled=True, workers=-1)

    with pytest.raises(ValueError, match="workers must be an integer, not a boolean"):
        _ = MatlabLocalParallelConfig(enabled=True, workers=True)


def test_registration_config_rejects_parallel_without_matlab_tiled_local() -> None:
    source = {"method": "mip_all_channels", "mip_channels": [0, 1, 2]}

    with pytest.raises(ValueError, match="requires registration.local.enabled=true"):
        _ = RegistrationConfig(
            reference_round=1,
            source=source,
            matlab_local_parallel={"enabled": True, "workers": 2},
        )

    with pytest.raises(ValueError, match="provider='matlab'.*method='demons_3d'"):
        _ = RegistrationConfig(
            reference_round=1,
            source=source,
            local={"enabled": True, "method": "demons_3d", "provider": "native"},
            matlab_local_parallel={"enabled": True, "workers": 2},
        )

    with pytest.raises(ValueError, match="use_tiling=true"):
        _ = RegistrationConfig(
            reference_round=1,
            source=source,
            local={"enabled": True, "method": "demons_3d", "provider": "matlab"},
            matlab_local_parallel={"enabled": True, "workers": 2},
        )


def test_default_off_path_keeps_serial_backend_loop_and_legacy_summary_shape() -> None:
    layout = _layout()
    backend = _SerialBackend()
    ref = np.arange(np.prod(FULL_SHAPE_ZYX), dtype=np.float32).reshape(FULL_SHAPE_ZYX)
    mov = ref + np.float32(1.0)

    flow, summary = _run_tiled_matlab_demons_registration(
        backend=cast(Any, backend),
        config=_config(parallel=False),
        ref_volume_zyx=ref,
        mov_volume_zyx=mov,
        fov_id=FOV_ID,
        round_id=ROUND_ID,
        reference_round=REFERENCE_ROUND,
        scope_descriptor=_scope_descriptor(),
        layout=layout,
        diagnostics=None,
    )

    expected = stitch_tiles([(tile, _flow_tile(tile)) for tile in layout.tiles], full_shape_zyx=FULL_SHAPE_ZYX)
    np.testing.assert_array_equal(flow, expected)
    assert [call["compute_tile"] for call in backend.calls] == [tile.as_dict() for tile in layout.tiles]
    assert "execution" not in summary
    assert [tile_summary["tile_index"] for tile_summary in summary["tiles"]] == [1, 2, 3, 4]


def test_tile_execution_plan_preserves_tile_identity_and_extracts_inputs() -> None:
    layout = _layout()
    ref = np.arange(np.prod(FULL_SHAPE_ZYX), dtype=np.float32).reshape(FULL_SHAPE_ZYX)
    mov = ref + np.float32(10.0)

    plan = build_matlab_local_tile_execution_plan(
        config=_config(parallel=True),
        ref_volume_zyx=ref,
        mov_volume_zyx=mov,
        fov_id=FOV_ID,
        round_id=ROUND_ID,
        reference_round=REFERENCE_ROUND,
        scope_descriptor=_scope_descriptor(),
        layout=layout,
        worker_count=2,
        execution_mode="process_parallel",
        strict_equivalence_audit=True,
    )

    assert plan.execution_mode == "process_parallel"
    assert plan.worker_count == 2
    assert [job.tile.as_dict() for job in plan.jobs] == [tile.as_dict() for tile in layout.tiles]
    assert [job.tile_index for job in plan.jobs] == [1, 2, 3, 4]
    for job in plan.jobs:
        assert tuple(job.ref_tile.shape) == job.tile.region_shape_zyx
        assert tuple(job.mov_tile.shape) == job.tile.region_shape_zyx


def test_actual_matlab_local_plan_carries_stage18a_request_identity() -> None:
    tile = _layout().tiles[0]
    request = build_matlab_local_registration_plan(
        _matlab_plan_config(),
        fov_id=FOV_ID,
        round_id=ROUND_ID,
        reference_round=REFERENCE_ROUND,
        scope_descriptor=_scope_descriptor(),
        volume_shape_zyx=tile.region_shape_zyx,
        compute_tile=tile.as_dict(),
    )
    request["runtime"] = {
        "entrypoint": "pystar_register_local_demons_entry",
        "runtime_manifest_sha256": "sha256:fixture-runtime-manifest",
    }
    request["reference_volume_shape_zyx"] = list(tile.region_shape_zyx)
    request["moving_volume_shape_zyx"] = list(tile.region_shape_zyx)

    candidate = dict(request)
    candidate["flow_output_path"] = "/tmp/stage18b-candidate.mat"
    report = compare_local_demons_requests(request, candidate)

    assert request["provider"] == "matlab"
    assert request["method"] == "demons_3d"
    assert request["coverage_mode"] == "full_fov"
    assert request["global_shift_already_applied"] is True
    assert request["compute_tile_index"] == tile.tile_index
    assert report.passed is True


def test_parallel_results_are_restored_by_tile_index_before_stitching() -> None:
    layout = _layout()
    out_of_order = tuple(_result(tile) for tile in reversed(layout.tiles))

    ordered, report = normalize_matlab_local_tile_results(
        out_of_order,
        layout=layout,
        requested_mode="process_parallel",
        effective_mode="process_parallel",
        worker_count=2,
        strict_equivalence_audit=True,
    )

    assert [result.tile_index for result in ordered] == [1, 2, 3, 4]
    assert report.to_dict()["tile_indices"] == [1, 2, 3, 4]
    stitched = stitch_tiles([(result.tile, cast(np.ndarray, result.flow_tile)) for result in ordered], full_shape_zyx=FULL_SHAPE_ZYX)
    expected = stitch_tiles([(tile, _flow_tile(tile)) for tile in layout.tiles], full_shape_zyx=FULL_SHAPE_ZYX)
    np.testing.assert_array_equal(stitched, expected)


def test_parallel_worker_failure_fails_loudly_without_serial_fallback() -> None:
    layout = _layout()
    failed_results = [_result(layout.tiles[0], status="failed")]
    failed_results.extend(_result(tile) for tile in layout.tiles[1:])

    with pytest.raises(RuntimeError, match="MATLAB local tile executor failed"):
        _ = normalize_matlab_local_tile_results(
            failed_results,
            layout=layout,
            requested_mode="process_parallel",
            effective_mode="process_parallel",
            worker_count=2,
            strict_equivalence_audit=True,
        )


def test_parallel_result_with_same_index_but_drifted_geometry_fails_loudly() -> None:
    layout = _layout()
    drifted_tile = replace(layout.tiles[0], write_shape_zyx=(2, 1, 2))
    drifted_results = [_result(drifted_tile)]
    drifted_results.extend(_result(tile) for tile in layout.tiles[1:])

    with pytest.raises(RuntimeError, match="tile geometry mismatch"):
        _ = normalize_matlab_local_tile_results(
            drifted_results,
            layout=layout,
            requested_mode="process_parallel",
            effective_mode="process_parallel",
            worker_count=2,
            strict_equivalence_audit=True,
        )


def test_synthetic_parallel_path_uses_executor_and_stage18a_equivalence() -> None:
    layout = _layout()
    ref = np.arange(np.prod(FULL_SHAPE_ZYX), dtype=np.float32).reshape(FULL_SHAPE_ZYX)
    mov = ref + np.float32(1.0)

    def fake_executor(plan: Any) -> tuple[MatlabLocalTileResult, ...]:
        assert plan.execution_mode == "process_parallel"
        assert plan.worker_count == 2
        return tuple(
            replace(
                _result(job.tile),
                extracted_ref_tile=job.ref_tile,
                extracted_mov_tile=job.mov_tile,
            )
            for job in reversed(plan.jobs)
        )

    flow, summary = _run_tiled_matlab_demons_registration(
        backend=cast(Any, _SerialBackend()),
        config=_config(parallel=True),
        ref_volume_zyx=ref,
        mov_volume_zyx=mov,
        fov_id=FOV_ID,
        round_id=ROUND_ID,
        reference_round=REFERENCE_ROUND,
        scope_descriptor=_scope_descriptor(),
        layout=layout,
        diagnostics=None,
        executor=fake_executor,
    )

    expected = stitch_tiles([(tile, _flow_tile(tile)) for tile in layout.tiles], full_shape_zyx=FULL_SHAPE_ZYX)
    np.testing.assert_array_equal(flow, expected)
    assert summary["execution"]["requested_mode"] == "process_parallel"
    assert [tile_summary["tile_index"] for tile_summary in summary["tiles"]] == [1, 2, 3, 4]

    serial_rows = [
        {"tile": tile, "flow_tile": _flow_tile(tile), "request": _request(tile)}
        for tile in layout.tiles
    ]
    candidate_rows = [
        {"tile": tile, "flow_tile": _flow_tile(tile), "request": _request(tile)}
        for tile in reversed(layout.tiles)
    ]
    report = compare_tiled_flow_outputs(serial_rows, candidate_rows, full_shape_zyx=FULL_SHAPE_ZYX)
    assert report.passed is True
    assert report.candidate_order_normalized is True


def test_executor_sequence_equivalence_helper_delegates_to_stage18a() -> None:
    layout = _layout()
    serial = tuple(_result(tile) for tile in layout.tiles)
    candidate = tuple(_result(tile) for tile in reversed(layout.tiles))

    report = compare_tile_result_sequence(serial, candidate, full_shape_zyx=FULL_SHAPE_ZYX)

    assert report.passed is True
    assert report.candidate_order_normalized is True
