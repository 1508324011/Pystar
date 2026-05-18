"""Private Stage18B executor seam for MATLAB local demons tile jobs.

The executor in this module is intentionally narrow: it schedules the existing
``MATLABRegistrationBackend.compute_local_flow(..., compute_tile=tile.as_dict())``
semantic unit for independent tiles, restores results by tile identity, and
hands the ordered outputs back to the existing ``stitch_tiles`` authority.

It does not change MATLAB parameters, tile geometry, artifact persistence,
provider routing, or diagnostics schemas.
"""

from __future__ import annotations

import multiprocessing as mp
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence, cast

import numpy as np
from numpy.typing import NDArray

from .matlab_registration import MATLABRegistrationBackend
from .tiling import TileLayout, TileSpec, extract_tile


FloatArray = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class MatlabLocalTileJob:
    """One existing MATLAB local demons tile call serialized for execution."""

    tile: TileSpec
    ref_tile: FloatArray
    mov_tile: FloatArray
    fov_id: int
    round_id: int
    reference_round: int
    scope_descriptor: dict[str, Any]
    config: Any

    @property
    def tile_index(self) -> int:
        return int(self.tile.tile_index)


@dataclass(frozen=True, slots=True)
class MatlabLocalTileResult:
    """Result for one MATLAB local demons tile call."""

    tile: TileSpec
    flow_tile: FloatArray | None
    backend_metadata: dict[str, Any]
    total_elapsed_wall_ms: float
    extraction_elapsed_wall_ms: float
    backend_call_elapsed_wall_ms: float
    flow_validation_elapsed_wall_ms: float
    extracted_ref_tile: FloatArray | None = None
    extracted_mov_tile: FloatArray | None = None
    status: str = "completed"
    error: str | None = None
    traceback: str | None = None

    @property
    def tile_index(self) -> int:
        return int(self.tile.tile_index)


@dataclass(frozen=True, slots=True)
class TileExecutionPlan:
    """Concrete execution plan for one tiled MATLAB local demons round."""

    layout: TileLayout
    jobs: tuple[MatlabLocalTileJob, ...]
    execution_mode: str
    worker_count: int
    strict_equivalence_audit: bool


@dataclass(frozen=True, slots=True)
class TileExecutionReport:
    """Non-persisted report describing the private tile executor run."""

    requested_mode: str
    effective_mode: str
    worker_count: int
    tile_count: int
    strict_equivalence_audit: bool
    tile_indices: tuple[int, ...]
    failures: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_mode": self.requested_mode,
            "effective_mode": self.effective_mode,
            "worker_count": int(self.worker_count),
            "tile_count": int(self.tile_count),
            "strict_equivalence_audit": bool(self.strict_equivalence_audit),
            "tile_indices": [int(value) for value in self.tile_indices],
            "failures": [dict(item) for item in self.failures],
        }


ExecutorCallable = Callable[[TileExecutionPlan], Sequence[MatlabLocalTileResult]]


def _elapsed_ms_since(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 3)


def build_matlab_local_tile_execution_plan(
    *,
    config: Any,
    ref_volume_zyx: NDArray[Any],
    mov_volume_zyx: NDArray[Any],
    fov_id: int,
    round_id: int,
    reference_round: int,
    scope_descriptor: Mapping[str, Any],
    layout: TileLayout,
    worker_count: int,
    execution_mode: str,
    strict_equivalence_audit: bool,
) -> TileExecutionPlan:
    """Extract tile inputs and build a private execution plan.

    Tile extraction stays in the parent process so the process worker owns only
    the MATLAB backend/session boundary and the existing per-tile backend call.
    """

    if int(worker_count) <= 0:
        raise ValueError("MATLAB local tile executor worker_count must be positive")
    if execution_mode not in {"serial", "process_parallel"}:
        raise ValueError(f"Unsupported MATLAB local tile execution_mode={execution_mode!r}")

    ref_shape = tuple(int(value) for value in np.asarray(ref_volume_zyx).shape)
    mov_shape = tuple(int(value) for value in np.asarray(mov_volume_zyx).shape)
    if ref_shape != tuple(layout.full_volume_shape_zyx):
        raise ValueError(
            "MATLAB local tile execution plan reference volume shape mismatch: "
            f"expected {layout.full_volume_shape_zyx}, got {ref_shape}"
        )
    if mov_shape != tuple(layout.full_volume_shape_zyx):
        raise ValueError(
            "MATLAB local tile execution plan moving volume shape mismatch: "
            f"expected {layout.full_volume_shape_zyx}, got {mov_shape}"
        )

    jobs: list[MatlabLocalTileJob] = []
    for tile in layout.tiles:
        jobs.append(
            MatlabLocalTileJob(
                tile=tile,
                ref_tile=np.asarray(extract_tile(ref_volume_zyx, tile), dtype=np.float32),
                mov_tile=np.asarray(extract_tile(mov_volume_zyx, tile), dtype=np.float32),
                fov_id=int(fov_id),
                round_id=int(round_id),
                reference_round=int(reference_round),
                scope_descriptor=dict(scope_descriptor),
                config=config,
            )
        )

    return TileExecutionPlan(
        layout=layout,
        jobs=tuple(jobs),
        execution_mode=execution_mode,
        worker_count=int(worker_count),
        strict_equivalence_audit=bool(strict_equivalence_audit),
    )


def _run_matlab_local_tile_job(job: MatlabLocalTileJob) -> MatlabLocalTileResult:
    """Worker function: create a backend/session boundary and compute one tile."""

    total_started = time.perf_counter()
    backend_call_started = time.perf_counter()
    backend: MATLABRegistrationBackend | None = None
    try:
        backend = MATLABRegistrationBackend(job.config)
        local_result = backend.compute_local_flow(
            job.ref_tile,
            job.mov_tile,
            fov_id=int(job.fov_id),
            round_id=int(job.round_id),
            reference_round=int(job.reference_round),
            scope_descriptor=job.scope_descriptor,
            compute_tile=job.tile.as_dict(),
        )
        backend_call_ms = _elapsed_ms_since(backend_call_started)

        validation_started = time.perf_counter()
        flow_tile = np.asarray(local_result["flow_3d"], dtype=np.float32)
        expected_shape = (3, *job.tile.region_shape_zyx)
        if flow_tile.shape != expected_shape:
            raise ValueError(
                "MATLAB tiled local registration returned flow_3d with incompatible tile shape: "
                f"expected {expected_shape}, got {flow_tile.shape} for tile {job.tile_index}"
            )
        flow_validation_ms = _elapsed_ms_since(validation_started)
        backend_metadata = cast(dict[str, Any], dict(local_result.get("backend_metadata", {})))
        return MatlabLocalTileResult(
            tile=job.tile,
            flow_tile=flow_tile,
            backend_metadata=backend_metadata,
            total_elapsed_wall_ms=_elapsed_ms_since(total_started),
            extraction_elapsed_wall_ms=0.0,
            backend_call_elapsed_wall_ms=backend_call_ms,
            flow_validation_elapsed_wall_ms=flow_validation_ms,
        )
    except Exception as exc:  # pragma: no cover - exact MATLAB failures depend on local install
        return MatlabLocalTileResult(
            tile=job.tile,
            flow_tile=None,
            backend_metadata={},
            total_elapsed_wall_ms=_elapsed_ms_since(total_started),
            extraction_elapsed_wall_ms=0.0,
            backend_call_elapsed_wall_ms=_elapsed_ms_since(backend_call_started),
            flow_validation_elapsed_wall_ms=0.0,
            status="failed",
            error=f"{exc.__class__.__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
    finally:
        if backend is not None:
            backend.close()


def run_matlab_local_tile_process_parallel(plan: TileExecutionPlan) -> tuple[MatlabLocalTileResult, ...]:
    """Execute tile jobs in worker processes and return raw completion order."""

    if plan.execution_mode != "process_parallel":
        raise ValueError("process-parallel tile executor requires execution_mode='process_parallel'")
    if int(plan.worker_count) <= 0:
        raise ValueError("process-parallel tile executor worker_count must be positive")
    if not plan.jobs:
        return ()

    context = mp.get_context("spawn")
    max_workers = min(int(plan.worker_count), len(plan.jobs))
    results: list[MatlabLocalTileResult] = []
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=context) as executor:
        future_to_tile = {executor.submit(_run_matlab_local_tile_job, job): job.tile_index for job in plan.jobs}
        for future in as_completed(future_to_tile):
            tile_index = int(future_to_tile[future])
            try:
                results.append(future.result())
            except Exception as exc:  # pragma: no cover - executor infrastructure failure path
                tile = next(job.tile for job in plan.jobs if job.tile_index == tile_index)
                results.append(
                    MatlabLocalTileResult(
                        tile=tile,
                        flow_tile=None,
                        backend_metadata={},
                        total_elapsed_wall_ms=0.0,
                        extraction_elapsed_wall_ms=0.0,
                        backend_call_elapsed_wall_ms=0.0,
                        flow_validation_elapsed_wall_ms=0.0,
                        status="failed",
                        error=f"{exc.__class__.__name__}: {exc}",
                        traceback=traceback.format_exc(),
                    )
                )
    return tuple(results)


def run_matlab_local_tile_serial(plan: TileExecutionPlan, backend: MATLABRegistrationBackend) -> tuple[MatlabLocalTileResult, ...]:
    """Execute the same per-tile semantic unit serially with an existing backend."""

    results: list[MatlabLocalTileResult] = []
    for job in plan.jobs:
        total_started = time.perf_counter()
        backend_call_started = time.perf_counter()
        local_result = backend.compute_local_flow(
            job.ref_tile,
            job.mov_tile,
            fov_id=int(job.fov_id),
            round_id=int(job.round_id),
            reference_round=int(job.reference_round),
            scope_descriptor=job.scope_descriptor,
            compute_tile=job.tile.as_dict(),
        )
        backend_call_ms = _elapsed_ms_since(backend_call_started)
        validation_started = time.perf_counter()
        flow_tile = np.asarray(local_result["flow_3d"], dtype=np.float32)
        expected_shape = (3, *job.tile.region_shape_zyx)
        if flow_tile.shape != expected_shape:
            raise ValueError(
                "MATLAB tiled local registration returned flow_3d with incompatible tile shape: "
                f"expected {expected_shape}, got {flow_tile.shape} for tile {job.tile_index}"
            )
        results.append(
            MatlabLocalTileResult(
                tile=job.tile,
                flow_tile=flow_tile,
                backend_metadata=cast(dict[str, Any], dict(local_result.get("backend_metadata", {}))),
                total_elapsed_wall_ms=_elapsed_ms_since(total_started),
                extraction_elapsed_wall_ms=0.0,
                backend_call_elapsed_wall_ms=backend_call_ms,
                flow_validation_elapsed_wall_ms=_elapsed_ms_since(validation_started),
            )
        )
    return tuple(results)


def compare_tile_result_sequence(
    baseline: Sequence[MatlabLocalTileResult],
    candidate: Sequence[MatlabLocalTileResult],
    *,
    full_shape_zyx: Sequence[int],
) -> Any:
    """Run the Stage18A tiled-flow comparator on executor result sequences.

    This helper is deliberately small so Stage18B reuses the Stage18A comparison
    truth instead of growing a second equivalence implementation in the executor.
    """

    from ._registration_equivalence import compare_tiled_flow_outputs

    def minimal_request(result: MatlabLocalTileResult) -> dict[str, Any]:
        tile_payload = result.tile.as_dict()
        return {
            "fov_id": 0,
            "round_id": 0,
            "reference_round": 0,
            "provider": "matlab",
            "method": "demons_3d",
            "coverage_mode": "full_fov",
            "global_shift_already_applied": True,
            "compute_tile": tile_payload,
            "runtime": {
                "entrypoint": "pystar_register_local_demons_entry",
                "manifest_sha256": "sha256:stage18b-executor-synthetic-runtime",
            },
            "reference_volume_shape_zyx": [int(value) for value in result.tile.region_shape_zyx],
            "moving_volume_shape_zyx": [int(value) for value in result.tile.region_shape_zyx],
        }

    def as_equivalence_rows(results: Sequence[MatlabLocalTileResult]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for result in results:
            request = result.backend_metadata.get("request")
            rows.append(
                {
                    "tile": result.tile,
                    "flow_tile": result.flow_tile,
                    "request": request if isinstance(request, Mapping) else minimal_request(result),
                }
            )
        return rows

    return compare_tiled_flow_outputs(
        as_equivalence_rows(baseline),
        as_equivalence_rows(candidate),
        full_shape_zyx=tuple(int(value) for value in full_shape_zyx),
    )


def normalize_matlab_local_tile_results(
    results: Sequence[MatlabLocalTileResult],
    *,
    layout: TileLayout,
    requested_mode: str,
    effective_mode: str,
    worker_count: int,
    strict_equivalence_audit: bool,
) -> tuple[tuple[MatlabLocalTileResult, ...], TileExecutionReport]:
    """Validate tile results, fail loudly on errors, and sort by tile identity."""

    expected_indices = tuple(int(tile.tile_index) for tile in layout.tiles)
    expected_tiles_by_index = {int(tile.tile_index): tile for tile in layout.tiles}
    by_index: dict[int, MatlabLocalTileResult] = {}
    failures: list[dict[str, Any]] = []

    for result in results:
        tile_index = int(result.tile_index)
        expected_tile = expected_tiles_by_index.get(tile_index)
        if expected_tile is None:
            failures.append(
                {
                    "tile_index": tile_index,
                    "error": "unexpected tile result",
                    "expected_tile_indices": [int(value) for value in expected_indices],
                }
            )
            continue
        if tile_index in by_index:
            failures.append({"tile_index": tile_index, "error": "duplicate tile result"})
            continue
        by_index[tile_index] = result
        if result.tile.as_dict() != expected_tile.as_dict():
            failures.append(
                {
                    "tile_index": tile_index,
                    "error": "tile geometry mismatch",
                    "expected_tile": expected_tile.as_dict(),
                    "actual_tile": result.tile.as_dict(),
                }
            )
        if result.status != "completed" or result.error is not None:
            failures.append(
                {
                    "tile_index": tile_index,
                    "status": result.status,
                    "error": result.error,
                    "traceback": result.traceback,
                }
            )

    for tile_index in expected_indices:
        if tile_index not in by_index:
            failures.append({"tile_index": tile_index, "error": "missing tile result"})

    report = TileExecutionReport(
        requested_mode=requested_mode,
        effective_mode=effective_mode,
        worker_count=int(worker_count),
        tile_count=int(layout.tile_count),
        strict_equivalence_audit=bool(strict_equivalence_audit),
        tile_indices=expected_indices,
        failures=tuple(failures),
    )
    if failures:
        raise RuntimeError(f"MATLAB local tile executor failed: {report.to_dict()}")

    ordered = tuple(by_index[tile_index] for tile_index in expected_indices)
    for result in ordered:
        expected_shape = (3, *result.tile.region_shape_zyx)
        if result.flow_tile is None:
            raise RuntimeError(f"MATLAB local tile executor produced no flow for tile {result.tile_index}")
        if tuple(result.flow_tile.shape) != expected_shape:
            raise ValueError(
                "MATLAB local tile executor result shape mismatch after ordering: "
                f"expected {expected_shape}, got {result.flow_tile.shape} for tile {result.tile_index}"
            )
    return ordered, report
