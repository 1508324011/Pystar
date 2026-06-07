"""Private Stage18B executor seam for MATLAB local demons tile jobs.

The executor in this module is intentionally narrow: it schedules the existing
``MATLABRegistrationBackend.compute_local_flow(..., compute_tile=tile.as_dict())``
semantic unit for independent tiles, restores results by tile identity, and
hands the ordered outputs back to the existing ``stitch_tiles`` authority.

It does not change MATLAB parameters, tile geometry, artifact persistence,
provider routing, or diagnostics schemas.
"""

from __future__ import annotations

import atexit
import math
import multiprocessing as mp
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence, cast

import numpy as np
from numpy.typing import NDArray

from .matlab_registration import MATLABRegistrationBackend
from .tiling import TileLayout, TileSpec, extract_tile
from ._registration_worker_lifecycle import (
    WORKER_LIFECYCLE_MS_FIELDS,
    WORKER_LIFECYCLE_OPTIONAL_MS_FIELDS,
    WORKER_LIFECYCLE_REQUIRED_FIELDS,
)


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
    worker_lifecycle: dict[str, Any] | None = None
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
    worker_lifecycle: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_mode": self.requested_mode,
            "effective_mode": self.effective_mode,
            "worker_count": int(self.worker_count),
            "tile_count": int(self.tile_count),
            "strict_equivalence_audit": bool(self.strict_equivalence_audit),
            "tile_indices": [int(value) for value in self.tile_indices],
            "failures": [dict(item) for item in self.failures],
            "worker_lifecycle": dict(self.worker_lifecycle or {"status": "absent"}),
        }


ExecutorCallable = Callable[[TileExecutionPlan], Sequence[MatlabLocalTileResult]]


@dataclass(slots=True)
class _WorkerBackendState:
    """Worker-process-local MATLAB backend/session state."""

    backend: MATLABRegistrationBackend
    config_fingerprint: tuple[str, ...]
    initialized_at: float
    tiles_completed: int = 0


_worker_backend_state: _WorkerBackendState | None = None
_worker_backend_atexit_registered = False


def _elapsed_ms_since(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 3)


def _coerce_non_negative_ms(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    number = float(value)
    if not math.isfinite(number) or number < 0:
        return 0.0
    return round(number, 3)


def _mapping_value(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _nested_attr(value: Any, *names: str) -> Any:
    current = value
    for name in names:
        current = getattr(current, name, None)
        if current is None:
            return None
    return current


def _worker_backend_config_fingerprint(config: Any) -> tuple[str, ...]:
    """Return a conservative identity for one worker-local backend instance."""

    registration = _nested_attr(config, "pipeline", "registration")
    demons = getattr(registration, "demons_3d", None) if registration is not None else None
    parallel = getattr(registration, "matlab_local_parallel", None) if registration is not None else None
    matlab_registration = _nested_attr(config, "providers", "matlab", "registration")
    local_entrypoints = getattr(matlab_registration, "local_entrypoints", None)
    local_demons_entrypoint = (
        local_entrypoints.get("demons_3d")
        if isinstance(local_entrypoints, Mapping)
        else None
    )

    return (
        type(config).__module__,
        type(config).__qualname__,
        str(getattr(config, "config_sha256", None) or ""),
        str(getattr(config, "config_source_path", None) or ""),
        str(getattr(matlab_registration, "runtime_path", None) or ""),
        str(getattr(matlab_registration, "entrypoint", None) or ""),
        str(local_demons_entrypoint or ""),
        str(getattr(matlab_registration, "input_volume_dtype", None) or ""),
        str(getattr(matlab_registration, "volume_transfer_mode", None) or ""),
        str(getattr(matlab_registration, "use_gpu", None) or ""),
        str(getattr(registration, "local_method", None) or ""),
        repr(demons),
        repr(parallel),
    )


def _ensure_worker_backend_atexit_registered() -> None:
    global _worker_backend_atexit_registered

    if _worker_backend_atexit_registered:
        return
    _ = atexit.register(_close_worker_backend)
    _worker_backend_atexit_registered = True


def _close_worker_backend() -> float:
    """Close and clear the worker-local backend/session, returning teardown ms."""

    global _worker_backend_state

    state = _worker_backend_state
    if state is None:
        return 0.0

    teardown_started = time.perf_counter()
    try:
        state.backend.close()
    finally:
        _worker_backend_state = None
    return _elapsed_ms_since(teardown_started)


def _worker_session_lifetime_ms() -> float:
    state = _worker_backend_state
    if state is None:
        return 0.0
    return _elapsed_ms_since(state.initialized_at)


def _get_or_create_worker_backend(job: MatlabLocalTileJob) -> tuple[MATLABRegistrationBackend, float, bool, int]:
    """Return this process's reusable MATLAB backend for a tile job."""

    global _worker_backend_state

    config_fingerprint = _worker_backend_config_fingerprint(job.config)
    state = _worker_backend_state
    if state is not None:
        if state.config_fingerprint != config_fingerprint:
            raise RuntimeError(
                "MATLAB local process-parallel worker received a different config identity while "
                "holding a worker-local backend; refusing cross-config backend reuse"
            )
        return state.backend, 0.0, True, int(state.tiles_completed) + 1

    initialized_at = time.perf_counter()
    construct_started = time.perf_counter()
    backend = MATLABRegistrationBackend(job.config)
    backend_construct_ms = _elapsed_ms_since(construct_started)
    _worker_backend_state = _WorkerBackendState(
        backend=backend,
        config_fingerprint=config_fingerprint,
        initialized_at=initialized_at,
    )
    _ensure_worker_backend_atexit_registered()
    return backend, backend_construct_ms, False, 1


def _record_worker_tile_completed() -> int:
    state = _worker_backend_state
    if state is None:
        return 0
    state.tiles_completed = int(state.tiles_completed) + 1
    return int(state.tiles_completed)


def _boundary_instrumentation(metadata: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(metadata, Mapping):
        return {}
    boundary = metadata.get("boundary_instrumentation")
    return _mapping_value(boundary)


def _boundary_seam_costs(boundary: Mapping[str, Any]) -> Mapping[str, Any]:
    seam_costs = boundary.get("seam_costs_ms")
    if isinstance(seam_costs, Mapping):
        return seam_costs
    aggregate = boundary.get("aggregate_seam_costs_ms")
    if isinstance(aggregate, Mapping):
        return aggregate
    return {}


def _boundary_phase_timings(boundary: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping_value(boundary.get("phase_timings_ms"))


def _boundary_engine_bootstrap_details(boundary: Mapping[str, Any]) -> Mapping[str, Any]:
    phase_details = _mapping_value(boundary.get("phase_details"))
    return _mapping_value(phase_details.get("engine_bootstrap"))


def _session_start_or_attach_ms(engine_bootstrap: Mapping[str, Any]) -> float:
    return round(
        sum(
            _coerce_non_negative_ms(engine_bootstrap.get(key))
            for key in ("start_matlab_ms", "connect_matlab_ms")
        ),
        3,
    )


def _matlab_addpath_or_bootstrap_ms(engine_bootstrap: Mapping[str, Any], seam_costs: Mapping[str, Any]) -> float:
    addpath_ms = _coerce_non_negative_ms(engine_bootstrap.get("addpath_ms"))
    if addpath_ms > 0.0:
        return addpath_ms
    return _coerce_non_negative_ms(seam_costs.get("engine_bootstrap_ms"))


def _build_worker_lifecycle(
    *,
    worker_process_pid: int,
    tile_index: int,
    backend_construct_ms: float,
    backend_metadata: Mapping[str, Any] | None,
    backend_close_ms: float,
    total_tile_wall_ms: float,
    status: str,
    worker_backend_reused: bool | None = None,
    worker_backend_reuse_index: int | None = None,
    worker_initializer_ms: float | None = None,
    worker_teardown_ms: float | None = None,
    worker_session_lifetime_ms: float | None = None,
    worker_tiles_completed: int | None = None,
) -> dict[str, Any]:
    boundary = _boundary_instrumentation(backend_metadata)
    seam_costs = _boundary_seam_costs(boundary)
    phase_timings = _boundary_phase_timings(boundary)
    engine_bootstrap = _boundary_engine_bootstrap_details(boundary)
    lifecycle = {
        "worker_process_pid": int(worker_process_pid),
        "tile_index": int(tile_index),
        "status": str(status),
        "backend_construct_ms": _coerce_non_negative_ms(backend_construct_ms),
        "matlab_session_start_or_attach_ms": _session_start_or_attach_ms(engine_bootstrap),
        "runtime_validation_ms": _coerce_non_negative_ms(seam_costs.get("runtime_file_validation_ms")),
        "matlab_addpath_or_bootstrap_ms": _matlab_addpath_or_bootstrap_ms(engine_bootstrap, seam_costs),
        "input_staging_ms": _coerce_non_negative_ms(seam_costs.get("input_staging_ms")),
        "matlab_call_ms": _coerce_non_negative_ms(seam_costs.get("matlab_call_ms")),
        "mat_output_load_ms": _coerce_non_negative_ms(phase_timings.get("mat_output_load")),
        "result_validation_ms": _coerce_non_negative_ms(seam_costs.get("result_validation_ms")),
        "backend_close_ms": _coerce_non_negative_ms(backend_close_ms),
        "total_tile_wall_ms": _coerce_non_negative_ms(total_tile_wall_ms),
    }
    if worker_backend_reused is not None:
        lifecycle["worker_backend_reused"] = bool(worker_backend_reused)
    if worker_backend_reuse_index is not None:
        lifecycle["worker_backend_reuse_index"] = int(worker_backend_reuse_index)
    if worker_initializer_ms is not None:
        lifecycle["worker_initializer_ms"] = _coerce_non_negative_ms(worker_initializer_ms)
    if worker_teardown_ms is not None:
        lifecycle["worker_teardown_ms"] = _coerce_non_negative_ms(worker_teardown_ms)
    if worker_session_lifetime_ms is not None:
        lifecycle["worker_session_lifetime_ms"] = _coerce_non_negative_ms(worker_session_lifetime_ms)
    if worker_tiles_completed is not None:
        lifecycle["worker_tiles_completed"] = int(worker_tiles_completed)
    return lifecycle


def _validate_worker_lifecycle(lifecycle: Mapping[str, Any], *, tile_index: int) -> None:
    missing = [key for key in WORKER_LIFECYCLE_REQUIRED_FIELDS if key not in lifecycle]
    if missing:
        raise ValueError(
            f"MATLAB local tile worker lifecycle telemetry is missing fields for tile {tile_index}: "
            + ", ".join(missing)
        )
    lifecycle_tile_index = lifecycle.get("tile_index")
    if isinstance(lifecycle_tile_index, bool) or not isinstance(lifecycle_tile_index, int):
        raise ValueError("MATLAB local tile worker lifecycle tile_index must be an integer")
    if int(lifecycle_tile_index) != int(tile_index):
        raise ValueError(
            "MATLAB local tile worker lifecycle tile_index mismatch: "
            f"expected {tile_index}, got {lifecycle_tile_index}"
        )
    worker_process_pid = lifecycle.get("worker_process_pid")
    if isinstance(worker_process_pid, bool) or not isinstance(worker_process_pid, int) or worker_process_pid <= 0:
        raise ValueError("MATLAB local tile worker lifecycle worker_process_pid must be a positive integer")
    for key in WORKER_LIFECYCLE_MS_FIELDS:
        value = lifecycle.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"MATLAB local tile worker lifecycle {key} must be finite and non-negative")
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise ValueError(f"MATLAB local tile worker lifecycle {key} must be finite and non-negative")
    if "worker_backend_reused" in lifecycle and not isinstance(lifecycle.get("worker_backend_reused"), bool):
        raise ValueError("MATLAB local tile worker lifecycle worker_backend_reused must be a boolean")
    reuse_index = lifecycle.get("worker_backend_reuse_index")
    if reuse_index is not None and (
        isinstance(reuse_index, bool) or not isinstance(reuse_index, int) or int(reuse_index) <= 0
    ):
        raise ValueError("MATLAB local tile worker lifecycle worker_backend_reuse_index must be a positive integer")
    tiles_completed = lifecycle.get("worker_tiles_completed")
    if tiles_completed is not None and (
        isinstance(tiles_completed, bool) or not isinstance(tiles_completed, int) or int(tiles_completed) < 0
    ):
        raise ValueError("MATLAB local tile worker lifecycle worker_tiles_completed must be a non-negative integer")
    for key in WORKER_LIFECYCLE_OPTIONAL_MS_FIELDS:
        if key not in lifecycle:
            continue
        value = lifecycle.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"MATLAB local tile worker lifecycle {key} must be finite and non-negative")
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise ValueError(f"MATLAB local tile worker lifecycle {key} must be finite and non-negative")


def _summarize_worker_lifecycle(results: Sequence[MatlabLocalTileResult]) -> dict[str, Any]:
    lifecycles: list[Mapping[str, Any]] = []
    for result in results:
        lifecycle = result.worker_lifecycle
        if isinstance(lifecycle, Mapping):
            _validate_worker_lifecycle(lifecycle, tile_index=result.tile_index)
            lifecycles.append(lifecycle)
    if not lifecycles:
        return {
            "status": "absent",
            "worker_process_count": 0,
            "worker_process_ids": [],
            "worker_tile_counts": {},
            "worker_overhead_totals_ms": {key: 0.0 for key in WORKER_LIFECYCLE_MS_FIELDS},
            "worker_overhead_percentages": {key: 0.0 for key in WORKER_LIFECYCLE_MS_FIELDS},
            "slowest_workers": [],
        }

    totals = {key: 0.0 for key in WORKER_LIFECYCLE_MS_FIELDS}
    by_pid: dict[int, dict[str, Any]] = {}
    first_use_tile_count = 0
    reused_tile_count = 0
    for lifecycle in lifecycles:
        pid = int(lifecycle["worker_process_pid"])
        worker = by_pid.setdefault(
            pid,
            {
                "worker_process_pid": pid,
                "tile_count": 0,
                "first_use_tile_count": 0,
                "reused_tile_count": 0,
                "total_tile_wall_ms": 0.0,
                "max_tile_wall_ms": 0.0,
            },
        )
        worker["tile_count"] = int(worker["tile_count"]) + 1
        worker_backend_reused = lifecycle.get("worker_backend_reused")
        if worker_backend_reused is True:
            reused_tile_count += 1
            worker["reused_tile_count"] = int(worker["reused_tile_count"]) + 1
        elif worker_backend_reused is False:
            first_use_tile_count += 1
            worker["first_use_tile_count"] = int(worker["first_use_tile_count"]) + 1
        tile_wall = _coerce_non_negative_ms(lifecycle.get("total_tile_wall_ms"))
        worker["total_tile_wall_ms"] = round(float(worker["total_tile_wall_ms"]) + tile_wall, 3)
        worker["max_tile_wall_ms"] = round(max(float(worker["max_tile_wall_ms"]), tile_wall), 3)
        for key in WORKER_LIFECYCLE_MS_FIELDS:
            totals[key] = round(totals[key] + _coerce_non_negative_ms(lifecycle.get(key)), 3)

    total_tile_wall = totals["total_tile_wall_ms"]
    percentages = {
        key: round((value / total_tile_wall * 100.0), 3) if total_tile_wall > 0.0 else 0.0
        for key, value in totals.items()
    }
    slowest_workers = []
    for worker in by_pid.values():
        tile_count = int(worker["tile_count"])
        total_wall = _coerce_non_negative_ms(worker["total_tile_wall_ms"])
        slowest_workers.append(
            {
                "worker_process_pid": int(worker["worker_process_pid"]),
                "tile_count": tile_count,
                "first_use_tile_count": int(worker.get("first_use_tile_count", 0)),
                "reused_tile_count": int(worker.get("reused_tile_count", 0)),
                "total_tile_wall_ms": total_wall,
                "mean_tile_wall_ms": round(total_wall / tile_count, 3) if tile_count else 0.0,
                "max_tile_wall_ms": _coerce_non_negative_ms(worker["max_tile_wall_ms"]),
            }
        )

    return {
        "status": "present",
        "worker_process_count": len(by_pid),
        "worker_process_ids": sorted(by_pid),
        "worker_tile_counts": {str(pid): int(row["tile_count"]) for pid, row in sorted(by_pid.items())},
        "worker_overhead_totals_ms": {key: round(value, 3) for key, value in totals.items()},
        "worker_overhead_percentages": percentages,
        "worker_backend_reuse": {
            "first_use_tile_count": int(first_use_tile_count),
            "reused_tile_count": int(reused_tile_count),
            "observed_tile_count": int(first_use_tile_count + reused_tile_count),
            "reused_tile_fraction": round(
                reused_tile_count / (first_use_tile_count + reused_tile_count),
                6,
            )
            if first_use_tile_count + reused_tile_count
            else 0.0,
        },
        "slowest_workers": sorted(
            slowest_workers,
            key=lambda item: (-float(item["total_tile_wall_ms"]), int(item["worker_process_pid"])),
        ),
    }


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
    """Worker function: reuse this process's backend/session for one tile."""

    total_started = time.perf_counter()
    backend_construct_ms = 0.0
    backend_call_ms = 0.0
    backend_close_ms = 0.0
    backend_metadata: dict[str, Any] = {}
    worker_pid = int(os.getpid())
    backend_reused = False
    backend_reuse_index = 1
    worker_initializer_ms = 0.0

    try:
        backend, backend_construct_ms, backend_reused, backend_reuse_index = _get_or_create_worker_backend(job)
        worker_initializer_ms = backend_construct_ms if not backend_reused else 0.0
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
        backend_metadata = cast(dict[str, Any], dict(local_result.get("backend_metadata", {})))

        validation_started = time.perf_counter()
        flow_tile = np.asarray(local_result["flow_3d"], dtype=np.float32)
        expected_shape = (3, *job.tile.region_shape_zyx)
        if flow_tile.shape != expected_shape:
            raise ValueError(
                "MATLAB tiled local registration returned flow_3d with incompatible tile shape: "
                f"expected {expected_shape}, got {flow_tile.shape} for tile {job.tile_index}"
            )
        flow_validation_ms = _elapsed_ms_since(validation_started)
        worker_tiles_completed = _record_worker_tile_completed()
        total_elapsed_ms = _elapsed_ms_since(total_started)
        worker_lifecycle = _build_worker_lifecycle(
            worker_process_pid=worker_pid,
            tile_index=job.tile_index,
            backend_construct_ms=backend_construct_ms,
            backend_metadata=backend_metadata,
            backend_close_ms=0.0,
            total_tile_wall_ms=total_elapsed_ms,
            status="completed",
            worker_backend_reused=backend_reused,
            worker_backend_reuse_index=backend_reuse_index,
            worker_initializer_ms=worker_initializer_ms,
            worker_session_lifetime_ms=_worker_session_lifetime_ms(),
            worker_tiles_completed=worker_tiles_completed,
        )
        return MatlabLocalTileResult(
            tile=job.tile,
            flow_tile=flow_tile,
            backend_metadata=backend_metadata,
            total_elapsed_wall_ms=total_elapsed_ms,
            extraction_elapsed_wall_ms=0.0,
            backend_call_elapsed_wall_ms=backend_call_ms,
            flow_validation_elapsed_wall_ms=flow_validation_ms,
            worker_lifecycle=worker_lifecycle,
        )
    except Exception as exc:  # pragma: no cover - exact MATLAB failures depend on local install
        session_lifetime_ms = _worker_session_lifetime_ms()
        try:
            backend_close_ms = _close_worker_backend()
        except Exception as close_exc:  # pragma: no cover - close failure depends on MATLAB Engine state
            backend_close_ms = 0.0
            error = f"{exc.__class__.__name__}: {exc}; backend close failed with {close_exc.__class__.__name__}: {close_exc}"
            trace = traceback.format_exc()
        else:
            error = f"{exc.__class__.__name__}: {exc}"
            trace = traceback.format_exc()
        total_elapsed_ms = _elapsed_ms_since(total_started)
        worker_lifecycle = _build_worker_lifecycle(
            worker_process_pid=worker_pid,
            tile_index=job.tile_index,
            backend_construct_ms=backend_construct_ms,
            backend_metadata=backend_metadata,
            backend_close_ms=backend_close_ms,
            total_tile_wall_ms=total_elapsed_ms,
            status="failed",
            worker_backend_reused=backend_reused,
            worker_backend_reuse_index=backend_reuse_index,
            worker_initializer_ms=worker_initializer_ms,
            worker_teardown_ms=backend_close_ms,
            worker_session_lifetime_ms=session_lifetime_ms,
        )
        return MatlabLocalTileResult(
            tile=job.tile,
            flow_tile=None,
            backend_metadata=backend_metadata,
            total_elapsed_wall_ms=total_elapsed_ms,
            extraction_elapsed_wall_ms=0.0,
            backend_call_elapsed_wall_ms=backend_call_ms,
            flow_validation_elapsed_wall_ms=0.0,
            worker_lifecycle=worker_lifecycle,
            status="failed",
            error=error,
            traceback=trace,
        )


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
                raise RuntimeError(
                    "MATLAB local tile process_parallel infrastructure failure for "
                    f"tile {tile_index}; refusing serial/native fallback"
                ) from exc
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
                worker_lifecycle=None,
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
        if effective_mode == "process_parallel":
            if not isinstance(result.worker_lifecycle, Mapping):
                failures.append(
                    {
                        "tile_index": tile_index,
                        "error": "missing worker lifecycle telemetry",
                    }
                )
            else:
                try:
                    _validate_worker_lifecycle(result.worker_lifecycle, tile_index=tile_index)
                except ValueError as exc:
                    failures.append({"tile_index": tile_index, "error": str(exc)})

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
        worker_lifecycle=_summarize_worker_lifecycle(tuple(by_index[tile_index] for tile_index in expected_indices if tile_index in by_index)),
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
