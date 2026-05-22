"""Registration-internal performance diagnostics for Stage16.

This private module owns the versioned diagnostics sidecar written by the
registration stage.  The sidecar is measurement-only: it records timings and
already-produced MATLAB boundary metadata without changing provider dispatch,
registration algorithms, transform semantics, or canonical artifact paths.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np

from ._io_paths import get_fov_output_structure
from ._registration_worker_lifecycle import (
    WORKER_LIFECYCLE_MS_FIELDS,
    WORKER_LIFECYCLE_OPTIONAL_MS_FIELDS,
    WORKER_LIFECYCLE_REQUIRED_FIELDS,
)
from ._stage_contracts import get_stage_spec
from .matlab_engine_bootstrap import MATLAB_BOUNDARY_SEAM_COST_KEYS, summarize_matlab_boundary_traces
from .serialization import write_backend_metadata


REGISTRATION_PERFORMANCE_SCHEMA_NAME = "pystar_registration_performance_diagnostics"
REGISTRATION_PERFORMANCE_SCHEMA_VERSION = 1
REGISTRATION_PERFORMANCE_STAGE_ID = "registration"

def diagnostics_timer_start() -> float:
    """Return a high-resolution timer token for registration diagnostics."""

    return time.perf_counter()


def elapsed_ms_since(start_time: float) -> float:
    """Return elapsed wall time in milliseconds from a timer token."""

    return round((time.perf_counter() - start_time) * 1000.0, 3)


def _validate_elapsed_ms(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite non-negative elapsed_wall_ms value")
    elapsed = float(value)
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return round(elapsed, 3)


def _validate_finite_number(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite numeric value")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    return round(number, 3)


def _validate_status(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty status string")
    return value.strip()


def timing_record(
    phase_id: str,
    elapsed_wall_ms: float,
    *,
    status: str = "completed",
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one validated timing record with explicit millisecond units."""

    if not isinstance(phase_id, str) or not phase_id.strip():
        raise ValueError("registration diagnostics phase_id must be a non-empty string")
    record: dict[str, Any] = {
        "phase_id": phase_id.strip(),
        "elapsed_wall_ms": _validate_elapsed_ms(
            elapsed_wall_ms,
            field_name=f"registration diagnostics phase {phase_id!r}",
        ),
        "status": _validate_status(status, field_name=f"registration diagnostics phase {phase_id!r}.status"),
    }
    if details is not None:
        record["details"] = dict(details)
    return record


def get_registration_performance_path(base_dir: Path, fov_id: int) -> Path:
    """Return the canonical Stage16 registration diagnostics sidecar path."""

    paths = get_fov_output_structure(Path(base_dir), int(fov_id))
    return paths["qc"] / f"registration_performance_fov_{int(fov_id)}.json"


def _to_float_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=np.float64)
    except Exception:
        return None
    if array.ndim == 0:
        return [float(array)]
    return [float(item) for item in cast(list[float], array.reshape(-1).tolist())]


def _extract_boundary_instrumentation(
    metadata: Mapping[str, Any] | None,
    *,
    prefer_local_flow: bool = False,
) -> dict[str, Any] | None:
    if not isinstance(metadata, Mapping):
        return None
    local_flow = metadata.get("local_flow")
    if prefer_local_flow and isinstance(local_flow, Mapping):
        nested = local_flow.get("boundary_instrumentation")
        if isinstance(nested, Mapping):
            return dict(nested)
    direct = metadata.get("boundary_instrumentation")
    if isinstance(direct, Mapping):
        return dict(direct)
    if isinstance(local_flow, Mapping):
        nested = local_flow.get("boundary_instrumentation")
        if isinstance(nested, Mapping):
            return dict(nested)
    return None


def _extract_session_lifecycle_summary(
    metadata: Mapping[str, Any] | None,
    *,
    prefer_local_flow: bool = False,
) -> dict[str, Any] | None:
    if not isinstance(metadata, Mapping):
        return None
    local_flow = metadata.get("local_flow")
    if prefer_local_flow and isinstance(local_flow, Mapping):
        nested = local_flow.get("session_lifecycle_summary")
        if isinstance(nested, Mapping):
            return dict(nested)
    direct = metadata.get("session_lifecycle_summary")
    if isinstance(direct, Mapping):
        return dict(direct)
    if isinstance(local_flow, Mapping):
        nested = local_flow.get("session_lifecycle_summary")
        if isinstance(nested, Mapping):
            return dict(nested)
    return None


def _extract_session_lifecycle(
    metadata: Mapping[str, Any] | None,
    *,
    prefer_local_flow: bool = False,
) -> dict[str, Any] | None:
    if not isinstance(metadata, Mapping):
        return None
    local_flow = metadata.get("local_flow")
    if prefer_local_flow and isinstance(local_flow, Mapping):
        nested = local_flow.get("session_lifecycle")
        if isinstance(nested, Mapping):
            return dict(nested)
    direct = metadata.get("session_lifecycle")
    if isinstance(direct, Mapping):
        return dict(direct)
    if isinstance(local_flow, Mapping):
        nested = local_flow.get("session_lifecycle")
        if isinstance(nested, Mapping):
            return dict(nested)
    return None


def _is_local_matlab_metadata(metadata: Mapping[str, Any]) -> bool:
    return any(
        key in metadata
        for key in (
            "flow_storage_format",
            "flow_variable",
            "flow_layout",
            "flow_shape_yxz_component",
        )
    )


def _extract_matlab_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Return MATLAB local-call metadata without mistaking global metadata for local timing."""

    if not isinstance(metadata, Mapping):
        return None
    local_flow = metadata.get("local_flow")
    if isinstance(local_flow, Mapping):
        nested = local_flow.get("matlab_metadata")
        if isinstance(nested, Mapping):
            return dict(nested)
    direct = metadata.get("matlab_metadata")
    if isinstance(direct, Mapping) and _is_local_matlab_metadata(direct):
        return dict(direct)
    return None


def _boundary_seam_costs(boundary: Mapping[str, Any]) -> Mapping[str, Any]:
    seam_costs = boundary.get("seam_costs_ms")
    if isinstance(seam_costs, Mapping):
        return seam_costs
    aggregate = boundary.get("aggregate_seam_costs_ms")
    if isinstance(aggregate, Mapping):
        return aggregate
    return {}


def _validate_boundary_instrumentation(boundary: Mapping[str, Any], *, field_name: str) -> None:
    seam_costs = _boundary_seam_costs(boundary)
    for key, value in seam_costs.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{field_name}.seam_costs_ms keys must be non-empty strings")
        _ = _validate_elapsed_ms(value, field_name=f"{field_name}.seam_costs_ms.{key}")

    total_duration = boundary.get("total_duration_ms")
    if total_duration is not None:
        _ = _validate_elapsed_ms(total_duration, field_name=f"{field_name}.total_duration_ms")


def _validate_worker_lifecycle(value: Any, *, field_name: str, tile_index: int | None = None) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    missing = [key for key in WORKER_LIFECYCLE_REQUIRED_FIELDS if key not in value]
    if missing:
        raise ValueError(f"{field_name} is missing required fields: " + ", ".join(missing))
    lifecycle_tile_index = value.get("tile_index")
    if isinstance(lifecycle_tile_index, bool) or not isinstance(lifecycle_tile_index, int):
        raise ValueError(f"{field_name}.tile_index must be an integer")
    if tile_index is not None and int(lifecycle_tile_index) != int(tile_index):
        raise ValueError(
            f"{field_name}.tile_index mismatch: expected {int(tile_index)}, got {lifecycle_tile_index}"
        )
    worker_process_pid = value.get("worker_process_pid")
    if isinstance(worker_process_pid, bool) or not isinstance(worker_process_pid, int) or worker_process_pid <= 0:
        raise ValueError(f"{field_name}.worker_process_pid must be a positive integer")
    normalized: dict[str, Any] = {
        "worker_process_pid": int(worker_process_pid),
        "tile_index": int(lifecycle_tile_index),
    }
    if "status" in value:
        normalized["status"] = _validate_status(value.get("status"), field_name=f"{field_name}.status")
    for key in WORKER_LIFECYCLE_MS_FIELDS:
        normalized[key] = _validate_elapsed_ms(value.get(key), field_name=f"{field_name}.{key}")
    if "worker_backend_reused" in value:
        worker_backend_reused = value.get("worker_backend_reused")
        if not isinstance(worker_backend_reused, bool):
            raise ValueError(f"{field_name}.worker_backend_reused must be a boolean")
        normalized["worker_backend_reused"] = bool(worker_backend_reused)
    if "worker_backend_reuse_index" in value:
        reuse_index = value.get("worker_backend_reuse_index")
        if isinstance(reuse_index, bool) or not isinstance(reuse_index, int) or int(reuse_index) <= 0:
            raise ValueError(f"{field_name}.worker_backend_reuse_index must be a positive integer")
        normalized["worker_backend_reuse_index"] = int(reuse_index)
    for key in WORKER_LIFECYCLE_OPTIONAL_MS_FIELDS:
        if key in value:
            normalized[key] = _validate_elapsed_ms(value.get(key), field_name=f"{field_name}.{key}")
    if "worker_tiles_completed" in value:
        tiles_completed = value.get("worker_tiles_completed")
        if isinstance(tiles_completed, bool) or not isinstance(tiles_completed, int) or int(tiles_completed) < 0:
            raise ValueError(f"{field_name}.worker_tiles_completed must be a non-negative integer")
        normalized["worker_tiles_completed"] = int(tiles_completed)
    return normalized


def _validate_int_field(value: Any, *, field_name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{field_name} must be an integer")
    number = int(value)
    if minimum is not None and number < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}")
    return number


def _validate_int_sequence(
    value: Any,
    *,
    field_name: str,
    length: int,
    minimum: int | None = None,
) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field_name} must be a length-{length} integer sequence")
    if len(value) != length:
        raise ValueError(f"{field_name} must be a length-{length} integer sequence")
    return [
        _validate_int_field(item, field_name=f"{field_name}[{index}]", minimum=minimum)
        for index, item in enumerate(value)
    ]


def _validate_tiled_local_tile_identity(value: Mapping[str, Any], *, field_name: str) -> int:
    tile_index = _validate_int_field(value.get("tile_index"), field_name=f"{field_name}.tile_index", minimum=1)
    _validate_int_sequence(value.get("grid_position_yx"), field_name=f"{field_name}.grid_position_yx", length=2, minimum=0)
    _validate_int_sequence(value.get("grid_shape_yx"), field_name=f"{field_name}.grid_shape_yx", length=2, minimum=1)
    _validate_int_sequence(value.get("region_origin_zyx"), field_name=f"{field_name}.region_origin_zyx", length=3, minimum=0)
    _validate_int_sequence(value.get("region_shape_zyx"), field_name=f"{field_name}.region_shape_zyx", length=3, minimum=1)
    _validate_int_sequence(value.get("write_origin_zyx"), field_name=f"{field_name}.write_origin_zyx", length=3, minimum=0)
    _validate_int_sequence(value.get("write_shape_zyx"), field_name=f"{field_name}.write_shape_zyx", length=3, minimum=1)
    _validate_int_sequence(value.get("write_offset_zyx"), field_name=f"{field_name}.write_offset_zyx", length=3, minimum=0)
    _validate_int_sequence(value.get("full_volume_shape_zyx"), field_name=f"{field_name}.full_volume_shape_zyx", length=3, minimum=1)
    return tile_index


def _validate_execution_worker_lifecycle_summary(value: Any, *, field_name: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    status = value.get("status")
    if status not in {"present", "absent"}:
        raise ValueError(f"{field_name}.status must be 'present' or 'absent'")
    worker_process_count = value.get("worker_process_count")
    if isinstance(worker_process_count, bool) or not isinstance(worker_process_count, int) or worker_process_count < 0:
        raise ValueError(f"{field_name}.worker_process_count must be a non-negative integer")
    worker_process_ids = value.get("worker_process_ids")
    if not isinstance(worker_process_ids, list):
        raise ValueError(f"{field_name}.worker_process_ids must be a list")
    for index, pid in enumerate(worker_process_ids):
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            raise ValueError(f"{field_name}.worker_process_ids[{index}] must be a positive integer")
    worker_tile_counts = value.get("worker_tile_counts")
    if not isinstance(worker_tile_counts, Mapping):
        raise ValueError(f"{field_name}.worker_tile_counts must be a mapping")
    for pid, count in worker_tile_counts.items():
        if not isinstance(pid, str) or not pid:
            raise ValueError(f"{field_name}.worker_tile_counts keys must be non-empty strings")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"{field_name}.worker_tile_counts.{pid} must be a non-negative integer")
    for mapping_name in ("worker_overhead_totals_ms", "worker_overhead_percentages"):
        mapping = value.get(mapping_name)
        if not isinstance(mapping, Mapping):
            raise ValueError(f"{field_name}.{mapping_name} must be a mapping")
        for key in WORKER_LIFECYCLE_MS_FIELDS:
            if key not in mapping:
                raise ValueError(f"{field_name}.{mapping_name} is missing {key}")
            if mapping_name == "worker_overhead_totals_ms":
                _ = _validate_elapsed_ms(mapping.get(key), field_name=f"{field_name}.{mapping_name}.{key}")
            else:
                percentage = _validate_finite_number(mapping.get(key), field_name=f"{field_name}.{mapping_name}.{key}")
                if percentage < 0:
                    raise ValueError(f"{field_name}.{mapping_name}.{key} must be non-negative")
    slowest_workers = value.get("slowest_workers")
    if not isinstance(slowest_workers, list):
        raise ValueError(f"{field_name}.slowest_workers must be a list")
    for index, worker in enumerate(slowest_workers):
        if not isinstance(worker, Mapping):
            raise ValueError(f"{field_name}.slowest_workers[{index}] must be a mapping")
        pid = worker.get("worker_process_pid")
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            raise ValueError(f"{field_name}.slowest_workers[{index}].worker_process_pid must be a positive integer")
        count = worker.get("tile_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"{field_name}.slowest_workers[{index}].tile_count must be a non-negative integer")
        for key in ("total_tile_wall_ms", "mean_tile_wall_ms", "max_tile_wall_ms"):
            if key in worker:
                _ = _validate_elapsed_ms(worker.get(key), field_name=f"{field_name}.slowest_workers[{index}].{key}")


def _validate_tiled_local_execution_report(
    value: Any,
    *,
    field_name: str,
    tile_indices: Sequence[int],
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    requested_mode = value.get("requested_mode")
    if requested_mode not in {"serial", "process_parallel"}:
        raise ValueError(f"{field_name}.requested_mode must be 'serial' or 'process_parallel'")
    effective_mode = value.get("effective_mode")
    if effective_mode not in {"serial", "process_parallel"}:
        raise ValueError(f"{field_name}.effective_mode must be 'serial' or 'process_parallel'")
    worker_count = value.get("worker_count")
    if isinstance(worker_count, bool) or not isinstance(worker_count, int) or worker_count <= 0:
        raise ValueError(f"{field_name}.worker_count must be a positive integer")
    tile_count = value.get("tile_count")
    if isinstance(tile_count, bool) or not isinstance(tile_count, int) or tile_count < 0:
        raise ValueError(f"{field_name}.tile_count must be a non-negative integer")
    if int(tile_count) != len(tile_indices):
        raise ValueError(
            f"{field_name}.tile_count mismatch: expected {len(tile_indices)} recorded tiles, got {tile_count}"
        )
    if not isinstance(value.get("strict_equivalence_audit"), bool):
        raise ValueError(f"{field_name}.strict_equivalence_audit must be a boolean")
    raw_tile_indices = value.get("tile_indices")
    if not isinstance(raw_tile_indices, list):
        raise ValueError(f"{field_name}.tile_indices must be a list")
    normalized_indices: list[int] = []
    for index, raw_index in enumerate(raw_tile_indices):
        normalized_indices.append(
            _validate_int_field(raw_index, field_name=f"{field_name}.tile_indices[{index}]", minimum=1)
        )
    if len(normalized_indices) != len(set(normalized_indices)):
        raise ValueError(f"{field_name}.tile_indices must not contain duplicate tile identities")
    if tuple(normalized_indices) != tuple(tile_indices):
        raise ValueError(
            f"{field_name}.tile_indices must match recorded tile order: expected {list(tile_indices)}, got {normalized_indices}"
        )
    failures = value.get("failures")
    if not isinstance(failures, list):
        raise ValueError(f"{field_name}.failures must be a list")
    if failures:
        raise ValueError(f"{field_name}.failures must be empty for persisted successful diagnostics")
    lifecycle_summary = value.get("worker_lifecycle")
    if not isinstance(lifecycle_summary, Mapping):
        raise ValueError(f"{field_name}.worker_lifecycle must be a mapping")
    _validate_execution_worker_lifecycle_summary(
        lifecycle_summary,
        field_name=f"{field_name}.worker_lifecycle",
    )
    if effective_mode == "process_parallel" and lifecycle_summary.get("status") != "present":
        raise ValueError(f"{field_name}.worker_lifecycle.status must be 'present' for process_parallel execution")


def _boundary_matlab_call_ms(boundary: Mapping[str, Any] | None, *, field_name: str) -> float | None:
    if not isinstance(boundary, Mapping):
        return None
    seam_costs = _boundary_seam_costs(boundary)
    if "matlab_call_ms" not in seam_costs:
        return None
    return _validate_elapsed_ms(
        seam_costs.get("matlab_call_ms"),
        field_name=f"{field_name}.seam_costs_ms.matlab_call_ms",
    )


def normalize_matlab_internal_timing(
    matlab_metadata: Mapping[str, Any] | None,
    *,
    boundary_instrumentation: Mapping[str, Any] | None = None,
    field_name: str = "matlab_metadata",
) -> dict[str, Any] | None:
    """Normalize MATLAB local demons ``metadata.steps`` into diagnostics shape.

    Missing ``steps`` means older/native metadata did not claim internal timing and
    remains compatible.  Present-but-malformed timing fails loudly with the
    offending field path.
    """

    if not isinstance(matlab_metadata, Mapping):
        return None
    if "steps" not in matlab_metadata or matlab_metadata.get("steps") is None:
        return None

    steps_value = matlab_metadata.get("steps")
    if not isinstance(steps_value, Sequence) or isinstance(steps_value, (str, bytes)) or not steps_value:
        raise ValueError(f"{field_name}.steps must be a non-empty sequence of timing mappings")

    normalized_steps: list[dict[str, Any]] = []
    for index, raw_step in enumerate(steps_value):
        step_field = f"{field_name}.steps[{index}]"
        if not isinstance(raw_step, Mapping):
            raise ValueError(f"{step_field} must be a mapping")
        name = raw_step.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{step_field}.name must be a non-empty string")
        duration_ms = _validate_elapsed_ms(
            raw_step.get("duration_ms"),
            field_name=f"{step_field}.duration_ms",
        )
        step: dict[str, Any] = {
            "name": name.strip(),
            "duration_ms": duration_ms,
        }
        if "details" in raw_step:
            details = raw_step.get("details")
            if not isinstance(details, Mapping):
                raise ValueError(f"{step_field}.details must be a mapping when present")
            step["details"] = dict(details)
        normalized_steps.append(step)

    total_duration_ms = _validate_elapsed_ms(
        matlab_metadata.get("total_duration_ms"),
        field_name=f"{field_name}.total_duration_ms",
    )
    step_total_duration_ms = round(sum(float(step["duration_ms"]) for step in normalized_steps), 3)
    unaccounted_duration_ms = round(total_duration_ms - step_total_duration_ms, 3)
    if unaccounted_duration_ms < 0:
        raise ValueError(
            f"{field_name}.unaccounted_duration_ms must be finite and non-negative; "
            "step durations exceed total_duration_ms"
        )
    boundary_call_ms = _boundary_matlab_call_ms(
        boundary_instrumentation,
        field_name=f"{field_name}.boundary_instrumentation",
    )
    boundary_delta = None if boundary_call_ms is None else round(boundary_call_ms - total_duration_ms, 3)
    dominant_step = max(normalized_steps, key=lambda step: float(step["duration_ms"]))

    return {
        "source": "matlab_metadata.steps",
        "status": "present",
        "total_duration_ms": total_duration_ms,
        "step_total_duration_ms": step_total_duration_ms,
        "unaccounted_duration_ms": unaccounted_duration_ms,
        "boundary_matlab_call_ms": boundary_call_ms,
        "boundary_minus_matlab_total_ms": boundary_delta,
        "steps": normalized_steps,
        "dominant_step": {
            "name": dominant_step["name"],
            "duration_ms": dominant_step["duration_ms"],
        },
    }


def _validate_matlab_internal_timing_block(value: Any, *, field_name: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    source = value.get("source")
    if not isinstance(source, str) or not source.strip():
        raise ValueError(f"{field_name}.source must be a non-empty string")
    if "status" in value:
        _ = _validate_status(value.get("status"), field_name=f"{field_name}.status")
    _ = _validate_elapsed_ms(value.get("total_duration_ms"), field_name=f"{field_name}.total_duration_ms")
    _ = _validate_elapsed_ms(value.get("step_total_duration_ms"), field_name=f"{field_name}.step_total_duration_ms")
    _ = _validate_elapsed_ms(value.get("unaccounted_duration_ms"), field_name=f"{field_name}.unaccounted_duration_ms")
    boundary_call = value.get("boundary_matlab_call_ms")
    if boundary_call is not None:
        _ = _validate_elapsed_ms(boundary_call, field_name=f"{field_name}.boundary_matlab_call_ms")
    boundary_delta = value.get("boundary_minus_matlab_total_ms")
    if boundary_delta is not None:
        _ = _validate_finite_number(boundary_delta, field_name=f"{field_name}.boundary_minus_matlab_total_ms")

    steps = value.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"{field_name}.steps must be a non-empty list")
    normalized_steps: list[dict[str, Any]] = []
    for index, raw_step in enumerate(steps):
        step_field = f"{field_name}.steps[{index}]"
        if not isinstance(raw_step, Mapping):
            raise ValueError(f"{step_field} must be a mapping")
        name = raw_step.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{step_field}.name must be a non-empty string")
        duration_ms = _validate_elapsed_ms(raw_step.get("duration_ms"), field_name=f"{step_field}.duration_ms")
        if "details" in raw_step and not isinstance(raw_step.get("details"), Mapping):
            raise ValueError(f"{step_field}.details must be a mapping when present")
        normalized_steps.append({"name": name.strip(), "duration_ms": duration_ms})

    expected_step_total = round(sum(float(step["duration_ms"]) for step in normalized_steps), 3)
    actual_step_total = _validate_elapsed_ms(
        value.get("step_total_duration_ms"),
        field_name=f"{field_name}.step_total_duration_ms",
    )
    if actual_step_total != expected_step_total:
        raise ValueError(
            f"{field_name}.step_total_duration_ms must equal the sum of step durations "
            f"({expected_step_total:.3f}); got {actual_step_total:.3f}"
        )

    expected_unaccounted = round(
        _validate_elapsed_ms(value.get("total_duration_ms"), field_name=f"{field_name}.total_duration_ms")
        - expected_step_total,
        3,
    )
    actual_unaccounted = _validate_elapsed_ms(
        value.get("unaccounted_duration_ms"),
        field_name=f"{field_name}.unaccounted_duration_ms",
    )
    if actual_unaccounted != expected_unaccounted:
        raise ValueError(
            f"{field_name}.unaccounted_duration_ms must equal total_duration_ms - step_total_duration_ms "
            f"({expected_unaccounted:.3f}); got {actual_unaccounted:.3f}"
        )

    dominant_step = value.get("dominant_step")
    if not isinstance(dominant_step, Mapping):
        raise ValueError(f"{field_name}.dominant_step must be a mapping")
    dominant_name = dominant_step.get("name")
    if not isinstance(dominant_name, str) or not dominant_name.strip():
        raise ValueError(f"{field_name}.dominant_step.name must be a non-empty string")
    dominant_duration_ms = _validate_elapsed_ms(
        dominant_step.get("duration_ms"),
        field_name=f"{field_name}.dominant_step.duration_ms",
    )
    expected_dominant_step = max(normalized_steps, key=lambda step: float(step["duration_ms"]))
    if dominant_name.strip() != expected_dominant_step["name"] or dominant_duration_ms != expected_dominant_step["duration_ms"]:
        raise ValueError(
            f"{field_name}.dominant_step must match the longest step {expected_dominant_step['name']!r} "
            f"at {expected_dominant_step['duration_ms']:.3f} ms"
        )


def _validate_optional_percentage(value: Any, *, field_name: str) -> None:
    if value is None:
        return
    percentage = _validate_finite_number(value, field_name=field_name)
    if percentage < 0:
        raise ValueError(f"{field_name} must be non-negative")


def _validate_hot_path_percentage_mapping(value: Any, *, field_name: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    for key, percentage in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{field_name} keys must be non-empty strings")
        _validate_optional_percentage(percentage, field_name=f"{field_name}.{key}")


def _validate_matlab_local_hot_path_profile(value: Any, *, field_name: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    if value.get("schema_version") != "1.0":
        raise ValueError(f"{field_name}.schema_version must be '1.0'")
    status = value.get("status")
    if status not in {"present", "absent"}:
        raise ValueError(f"{field_name}.status must be 'present' or 'absent'")
    source = value.get("source")
    if not isinstance(source, str) or not source.strip():
        raise ValueError(f"{field_name}.source must be a non-empty string")
    scope = value.get("scope")
    if not isinstance(scope, str) or not scope.strip():
        raise ValueError(f"{field_name}.scope must be a non-empty string")
    counts: dict[str, int] = {}
    for count_field in ("call_count", "boundary_closure_count"):
        count_value = value.get(count_field)
        if isinstance(count_value, bool) or not isinstance(count_value, int) or count_value < 0:
            raise ValueError(f"{field_name}.{count_field} must be a non-negative integer")
        counts[count_field] = count_value
    if not isinstance(value.get("boundary_closure_complete"), bool):
        raise ValueError(f"{field_name}.boundary_closure_complete must be a boolean")
    call_count = counts["call_count"]
    boundary_closure_count = counts["boundary_closure_count"]
    if status == "absent" and call_count != 0:
        raise ValueError(f"{field_name}.call_count must be 0 when status is 'absent'")
    if boundary_closure_count > call_count:
        raise ValueError(f"{field_name}.boundary_closure_count must be <= call_count")
    expected_closure_complete = call_count > 0 and boundary_closure_count == call_count
    if bool(value.get("boundary_closure_complete")) is not expected_closure_complete:
        raise ValueError(
            f"{field_name}.boundary_closure_complete must reflect whether every counted call has boundary closure data"
        )

    boundary_total = value.get("boundary_matlab_call_total_ms")
    if boundary_total is not None:
        _ = _validate_elapsed_ms(boundary_total, field_name=f"{field_name}.boundary_matlab_call_total_ms")
    _ = _validate_elapsed_ms(
        value.get("matlab_internal_total_duration_ms"),
        field_name=f"{field_name}.matlab_internal_total_duration_ms",
    )
    boundary_delta = value.get("boundary_minus_internal_total_ms")
    if boundary_delta is not None:
        _ = _validate_finite_number(boundary_delta, field_name=f"{field_name}.boundary_minus_internal_total_ms")
    if not bool(value.get("boundary_closure_complete")):
        if boundary_total is not None:
            raise ValueError(f"{field_name}.boundary_matlab_call_total_ms must be null when boundary closure is incomplete")
        if boundary_delta is not None:
            raise ValueError(f"{field_name}.boundary_minus_internal_total_ms must be null when boundary closure is incomplete")
    _ = _validate_elapsed_ms(
        value.get("matlab_internal_unaccounted_total_ms"),
        field_name=f"{field_name}.matlab_internal_unaccounted_total_ms",
    )

    step_totals = value.get("step_totals_ms")
    if not isinstance(step_totals, Mapping):
        raise ValueError(f"{field_name}.step_totals_ms must be a mapping")
    for step_name, duration_ms in step_totals.items():
        if not isinstance(step_name, str) or not step_name.strip():
            raise ValueError(f"{field_name}.step_totals_ms keys must be non-empty strings")
        _ = _validate_elapsed_ms(duration_ms, field_name=f"{field_name}.step_totals_ms.{step_name}")

    step_counts = value.get("step_call_counts")
    if not isinstance(step_counts, Mapping):
        raise ValueError(f"{field_name}.step_call_counts must be a mapping")
    for step_name, count in step_counts.items():
        if not isinstance(step_name, str) or not step_name.strip():
            raise ValueError(f"{field_name}.step_call_counts keys must be non-empty strings")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"{field_name}.step_call_counts.{step_name} must be a non-negative integer")

    _validate_hot_path_percentage_mapping(
        value.get("step_percent_of_matlab_internal_total"),
        field_name=f"{field_name}.step_percent_of_matlab_internal_total",
    )
    _validate_hot_path_percentage_mapping(
        value.get("step_percent_of_boundary_matlab_call"),
        field_name=f"{field_name}.step_percent_of_boundary_matlab_call",
    )

    dominant = value.get("dominant_internal_step")
    if dominant is not None:
        if not isinstance(dominant, Mapping):
            raise ValueError(f"{field_name}.dominant_internal_step must be null or a mapping")
        dominant_name = dominant.get("name")
        if not isinstance(dominant_name, str) or not dominant_name.strip():
            raise ValueError(f"{field_name}.dominant_internal_step.name must be a non-empty string")
        _ = _validate_elapsed_ms(
            dominant.get("total_duration_ms"),
            field_name=f"{field_name}.dominant_internal_step.total_duration_ms",
        )
        dominant_count = dominant.get("call_count")
        if isinstance(dominant_count, bool) or not isinstance(dominant_count, int) or dominant_count < 0:
            raise ValueError(f"{field_name}.dominant_internal_step.call_count must be a non-negative integer")
        _validate_optional_percentage(
            dominant.get("percent_of_matlab_internal_total"),
            field_name=f"{field_name}.dominant_internal_step.percent_of_matlab_internal_total",
        )
        _validate_optional_percentage(
            dominant.get("percent_of_boundary_matlab_call"),
            field_name=f"{field_name}.dominant_internal_step.percent_of_boundary_matlab_call",
        )
        owner = dominant.get("owner")
        if not isinstance(owner, str) or not owner.strip():
            raise ValueError(f"{field_name}.dominant_internal_step.owner must be a non-empty string")

    rankings = value.get("hot_path_rankings")
    if not isinstance(rankings, list):
        raise ValueError(f"{field_name}.hot_path_rankings must be a list")
    for index, row in enumerate(rankings):
        row_field = f"{field_name}.hot_path_rankings[{index}]"
        if not isinstance(row, Mapping):
            raise ValueError(f"{row_field} must be a mapping")
        for text_field in ("component_type", "name", "owner"):
            text_value = row.get(text_field)
            if not isinstance(text_value, str) or not text_value.strip():
                raise ValueError(f"{row_field}.{text_field} must be a non-empty string")
        _ = _validate_elapsed_ms(row.get("total_duration_ms"), field_name=f"{row_field}.total_duration_ms")
        _validate_optional_percentage(
            row.get("percent_of_matlab_internal_total"),
            field_name=f"{row_field}.percent_of_matlab_internal_total",
        )
        _validate_optional_percentage(
            row.get("percent_of_boundary_matlab_call"),
            field_name=f"{row_field}.percent_of_boundary_matlab_call",
        )
        if "call_count" in row:
            row_count = row.get("call_count")
            if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count < 0:
                raise ValueError(f"{row_field}.call_count must be a non-negative integer")


def _sum_numeric_mapping(target: dict[str, float], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        if not isinstance(key, str):
            continue
        number = _validate_elapsed_ms(value, field_name=f"MATLAB boundary seam cost {key}")
        target[key] = round(target.get(key, 0.0) + number, 3)


def _direct_round_timing_keys() -> tuple[str, ...]:
    return (
        "moving_clean_volume_load",
        "moving_scope_crop",
        "global_registration",
        "post_global_qc",
        "local_registration",
        "final_qc",
        "flow_sidecar_persistence",
        "qc_image_save",
    )


def _timing_elapsed(record: Any) -> float | None:
    if isinstance(record, Mapping) and "elapsed_wall_ms" in record:
        return _validate_elapsed_ms(record.get("elapsed_wall_ms"), field_name="registration timing record")
    return None


def _phase_total_row(phase_id: str, elapsed_ms: float, total_ms: float) -> dict[str, Any]:
    return {
        "phase_id": phase_id,
        "total_elapsed_wall_ms": round(float(elapsed_ms), 3),
        "percent_of_total_measured_registration_diagnostic_time": round((elapsed_ms / total_ms * 100.0), 3)
        if total_ms > 0
        else 0.0,
    }


def build_registration_provider_summary(config: Any) -> dict[str, Any]:
    """Build a config-derived registration route summary without dispatching."""

    pipeline = getattr(config, "pipeline", None)
    registration = getattr(pipeline, "registration", None)
    global_stage = getattr(registration, "global_stage", None)
    registration_provider_mode = getattr(pipeline, "registration_provider_mode", None)
    provider_mode = registration_provider_mode() if callable(registration_provider_mode) else None
    return {
        "stage_id": REGISTRATION_PERFORMANCE_STAGE_ID,
        "provider_mode": provider_mode,
        "global_provider": getattr(registration, "global_provider", None),
        "global_method": getattr(global_stage, "method", None),
        "local_provider": getattr(registration, "local_provider", None),
        "local_method": getattr(registration, "local_method", None),
        "enable_local": getattr(registration, "enable_local", None),
        "reference_round": getattr(registration, "reference_round", None),
    }


@dataclass
class RegistrationPerformanceRecorder:
    """Mutable collector for one FOV's registration diagnostics.

    Registration code records coarse ownership-seam facts through these methods;
    this class owns the persisted schema shape, validation, and aggregation.
    """

    fov_id: int
    providers: Mapping[str, Any]
    registration_method: Mapping[str, Any] | None = None
    fov_setup: dict[str, dict[str, Any]] = field(default_factory=dict)
    rounds_by_id: dict[int, dict[str, Any]] = field(default_factory=dict)
    manifest: dict[str, dict[str, Any]] = field(default_factory=dict)

    def record_fov_setup_timing(
        self,
        phase_id: str,
        elapsed_wall_ms: float,
        *,
        status: str = "completed",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.fov_setup[phase_id] = timing_record(phase_id, elapsed_wall_ms, status=status, details=details)

    def start_round(self, round_id: int, *, is_reference_round: bool, status: str = "in_progress") -> None:
        self.rounds_by_id[int(round_id)] = {
            "round_id": int(round_id),
            "is_reference_round": bool(is_reference_round),
            "status": _validate_status(status, field_name=f"round {round_id}.status"),
        }

    def _round(self, round_id: int) -> dict[str, Any]:
        round_id = int(round_id)
        if round_id not in self.rounds_by_id:
            self.start_round(round_id, is_reference_round=False)
        return self.rounds_by_id[round_id]

    def complete_round(self, round_id: int, *, status: str = "completed") -> None:
        self._round(round_id)["status"] = _validate_status(status, field_name=f"round {round_id}.status")

    def record_round_timing(
        self,
        round_id: int,
        phase_id: str,
        elapsed_wall_ms: float,
        *,
        status: str = "completed",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self._round(round_id)[phase_id] = timing_record(
            phase_id,
            elapsed_wall_ms,
            status=status,
            details=details,
        )

    def record_global_registration(
        self,
        round_id: int,
        *,
        elapsed_wall_ms: float,
        provider: str | None,
        method: str | None,
        global_shift_3d: Any,
        global_corr: float,
        backend_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        metadata = dict(backend_metadata) if isinstance(backend_metadata, Mapping) else None
        record = timing_record("global_registration", elapsed_wall_ms)
        record.update(
            {
                "provider": provider,
                "method": method,
                "global_shift_3d": _to_float_list(global_shift_3d),
                "global_corr": float(global_corr),
            }
        )
        boundary = _extract_boundary_instrumentation(metadata)
        if boundary is not None:
            record["boundary_instrumentation"] = boundary
        session_lifecycle = _extract_session_lifecycle(metadata)
        if session_lifecycle is not None:
            record["session_lifecycle"] = session_lifecycle
        session_summary = _extract_session_lifecycle_summary(metadata)
        if session_summary is not None:
            record["session_lifecycle_summary"] = session_summary
        self._round(round_id)["global_registration"] = record

    def record_post_global_qc(
        self,
        round_id: int,
        *,
        elapsed_wall_ms: float,
        corr_after_global: float,
    ) -> None:
        self._round(round_id)["post_global_qc"] = timing_record(
            "post_global_qc",
            elapsed_wall_ms,
            details={"corr_after_global": float(corr_after_global)},
        )

    def record_local_registration(
        self,
        round_id: int,
        *,
        elapsed_wall_ms: float,
        provider: str | None,
        method: str | None,
        status: str,
        skip_reason: str | None = None,
        final_corr: float | None = None,
        backend_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        metadata = dict(backend_metadata) if isinstance(backend_metadata, Mapping) else None
        record = timing_record("local_registration", elapsed_wall_ms, status=status)
        record.update({"provider": provider, "method": method})
        if skip_reason is not None:
            record["skip_reason"] = skip_reason
        if final_corr is not None:
            record["final_corr"] = float(final_corr)
        boundary = _extract_boundary_instrumentation(metadata, prefer_local_flow=True)
        if boundary is not None:
            record["boundary_instrumentation"] = boundary
        matlab_internal_timing = normalize_matlab_internal_timing(
            _extract_matlab_metadata(metadata),
            boundary_instrumentation=boundary,
            field_name=f"round {round_id}.local_registration.matlab_metadata",
        )
        if matlab_internal_timing is not None:
            record["matlab_internal_timing"] = matlab_internal_timing
        session_lifecycle = _extract_session_lifecycle(metadata, prefer_local_flow=True)
        if session_lifecycle is not None:
            record["session_lifecycle"] = session_lifecycle
        session_summary = _extract_session_lifecycle_summary(metadata, prefer_local_flow=True)
        if session_summary is not None:
            record["session_lifecycle_summary"] = session_summary
        round_entry = self._round(round_id)
        existing = round_entry.get("local_registration")
        if isinstance(existing, Mapping):
            merged = dict(existing)
            merged.update(record)
            round_entry["local_registration"] = merged
        else:
            round_entry["local_registration"] = record

    def record_local_internal_phase(
        self,
        round_id: int,
        phase_id: str,
        elapsed_wall_ms: float,
        *,
        status: str = "completed",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        round_entry = self._round(round_id)
        local_internal = round_entry.setdefault("local_internal_timings", {})
        if not isinstance(local_internal, dict):
            local_internal = {}
            round_entry["local_internal_timings"] = local_internal
        local_internal[phase_id] = timing_record(phase_id, elapsed_wall_ms, status=status, details=details)

    def record_tiled_local_summary(
        self,
        round_id: int,
        *,
        layout_summary: Mapping[str, Any],
        stitch_elapsed_wall_ms: float,
        execution_report: Mapping[str, Any] | None = None,
    ) -> None:
        round_entry = self._round(round_id)
        tiled: dict[str, Any] | None = cast(dict[str, Any] | None, round_entry.get("tiled_local"))
        if not isinstance(tiled, dict):
            tiled = {"tiles": []}
            round_entry["tiled_local"] = tiled
        tiled["layout"] = dict(layout_summary)
        tiled["stitch"] = timing_record("tiled_local_stitch", stitch_elapsed_wall_ms)
        if execution_report is not None:
            tiled["execution"] = dict(execution_report)

    def record_tiled_local_tile(
        self,
        round_id: int,
        *,
        tile_identity: Mapping[str, Any],
        total_elapsed_wall_ms: float,
        extraction_elapsed_wall_ms: float,
        backend_call_elapsed_wall_ms: float,
        flow_validation_elapsed_wall_ms: float,
        boundary_instrumentation: Mapping[str, Any] | None = None,
        session_lifecycle: Mapping[str, Any] | None = None,
        session_lifecycle_summary: Mapping[str, Any] | None = None,
        normalized_result: Mapping[str, Any] | None = None,
        matlab_metadata: Mapping[str, Any] | None = None,
        worker_lifecycle: Mapping[str, Any] | None = None,
    ) -> None:
        round_entry = self._round(round_id)
        tiled: dict[str, Any] | None = cast(dict[str, Any] | None, round_entry.get("tiled_local"))
        if not isinstance(tiled, dict):
            tiled = {"tiles": []}
            round_entry["tiled_local"] = tiled
        tiles: list[dict[str, Any]] | None = cast(list[dict[str, Any]] | None, tiled.get("tiles"))
        if not isinstance(tiles, list):
            tiles = []
            tiled["tiles"] = tiles
        tile_record: dict[str, Any] = {
            **dict(tile_identity),
            "timings": {
                "tile_total": timing_record("tile_total", total_elapsed_wall_ms),
                "tile_extraction": timing_record("tile_extraction", extraction_elapsed_wall_ms),
                "backend_call": timing_record("tile_backend_call", backend_call_elapsed_wall_ms),
                "flow_validation": timing_record("tile_flow_validation", flow_validation_elapsed_wall_ms),
            },
        }
        if boundary_instrumentation is not None:
            tile_record["boundary_instrumentation"] = dict(boundary_instrumentation)
        matlab_internal_timing = normalize_matlab_internal_timing(
            matlab_metadata,
            boundary_instrumentation=boundary_instrumentation,
            field_name=f"round {round_id}.tiled_local.tile {tile_identity.get('tile_index', '?')}.matlab_metadata",
        )
        if matlab_internal_timing is not None:
            tile_record["matlab_internal_timing"] = matlab_internal_timing
        if session_lifecycle is not None:
            tile_record["session_lifecycle"] = dict(session_lifecycle)
        if session_lifecycle_summary is not None:
            tile_record["session_lifecycle_summary"] = dict(session_lifecycle_summary)
        if normalized_result is not None:
            tile_record["normalized_result"] = dict(normalized_result)
        if worker_lifecycle is not None:
            tile_record["worker_lifecycle"] = _validate_worker_lifecycle(
                worker_lifecycle,
                field_name=f"round {round_id}.tiled_local.tile {tile_identity.get('tile_index', '?')}.worker_lifecycle",
                tile_index=int(tile_identity["tile_index"]) if isinstance(tile_identity.get("tile_index"), int) else None,
            )
        tiles.append(tile_record)

    def record_final_qc(
        self,
        round_id: int,
        *,
        elapsed_wall_ms: float,
        final_corr: float,
    ) -> None:
        self._round(round_id)["final_qc"] = timing_record(
            "final_qc",
            elapsed_wall_ms,
            details={"final_corr": float(final_corr)},
        )

    def record_flow_sidecar_persistence(
        self,
        round_id: int,
        *,
        elapsed_wall_ms: float,
        descriptor: Mapping[str, Any] | None,
        sidecar_path: Path | None,
    ) -> None:
        details: dict[str, Any] = {
            "descriptor": None if descriptor is None else dict(descriptor),
            "path": None if sidecar_path is None else str(sidecar_path),
            "exists": None,
            "size_bytes": None,
        }
        if sidecar_path is not None:
            try:
                exists = sidecar_path.exists()
                details["exists"] = bool(exists)
                if exists:
                    details["size_bytes"] = int(sidecar_path.stat().st_size)
            except OSError as exc:
                details["stat_error"] = {"type": exc.__class__.__name__, "message": str(exc)}
        self._round(round_id)["flow_sidecar_persistence"] = timing_record(
            "flow_sidecar_persistence",
            elapsed_wall_ms,
            details=details,
        )

    def record_qc_image_save(self, round_id: int, *, elapsed_wall_ms: float, status: str = "completed") -> None:
        self._round(round_id)["qc_image_save"] = timing_record("qc_image_save", elapsed_wall_ms, status=status)

    def record_manifest_timing(
        self,
        phase_id: str,
        elapsed_wall_ms: float,
        *,
        status: str = "completed",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.manifest[phase_id] = timing_record(phase_id, elapsed_wall_ms, status=status, details=details)

    def build_payload(self, *, source_stage_elapsed_wall_ms: float | None = None) -> dict[str, Any]:
        payload = build_registration_performance_payload(
            fov_id=self.fov_id,
            providers=dict(self.providers),
            registration_method=dict(self.registration_method or self.providers),
            fov_setup=self.fov_setup,
            rounds=[self.rounds_by_id[key] for key in sorted(self.rounds_by_id)],
            manifest=self.manifest,
            source_stage_elapsed_wall_ms=source_stage_elapsed_wall_ms,
        )
        validate_registration_performance_payload(payload, expected_fov_id=self.fov_id)
        return payload

    def write(self, base_dir: Path, *, source_stage_elapsed_wall_ms: float | None = None) -> Path:
        payload = self.build_payload(source_stage_elapsed_wall_ms=source_stage_elapsed_wall_ms)
        return write_registration_performance_diagnostics(
            base_dir=base_dir,
            fov_id=self.fov_id,
            payload=payload,
        )


def _phase_totals_from_payload(
    *,
    fov_setup: Mapping[str, Any],
    rounds: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> dict[str, float]:
    totals: dict[str, float] = {}

    for phase_id, record in fov_setup.items():
        elapsed = _timing_elapsed(record)
        if elapsed is not None:
            totals[str(phase_id)] = round(totals.get(str(phase_id), 0.0) + elapsed, 3)

    for round_entry in rounds:
        for phase_id in _direct_round_timing_keys():
            elapsed = _timing_elapsed(round_entry.get(phase_id))
            if elapsed is not None:
                totals[phase_id] = round(totals.get(phase_id, 0.0) + elapsed, 3)

    for phase_id, record in manifest.items():
        elapsed = _timing_elapsed(record)
        if elapsed is not None:
            totals[str(phase_id)] = round(totals.get(str(phase_id), 0.0) + elapsed, 3)

    return totals


def _nested_phase_totals_from_rounds(rounds: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for round_entry in rounds:
        local_internal = round_entry.get("local_internal_timings")
        if isinstance(local_internal, Mapping):
            for phase_id, record in local_internal.items():
                elapsed = _timing_elapsed(record)
                if elapsed is not None:
                    totals[str(phase_id)] = round(totals.get(str(phase_id), 0.0) + elapsed, 3)

        tiled = round_entry.get("tiled_local")
        if isinstance(tiled, Mapping):
            stitch = _timing_elapsed(tiled.get("stitch"))
            if stitch is not None:
                totals["tiled_local_stitch"] = round(totals.get("tiled_local_stitch", 0.0) + stitch, 3)
            tiles = tiled.get("tiles")
            if isinstance(tiles, Sequence) and not isinstance(tiles, (str, bytes)):
                for tile in tiles:
                    if not isinstance(tile, Mapping):
                        continue
                    timings = tile.get("timings")
                    if not isinstance(timings, Mapping):
                        continue
                    for phase_id, record in timings.items():
                        elapsed = _timing_elapsed(record)
                        if elapsed is not None:
                            totals[str(phase_id)] = round(totals.get(str(phase_id), 0.0) + elapsed, 3)
    return totals


def _round_total_elapsed(round_entry: Mapping[str, Any]) -> float:
    total = 0.0
    for phase_id in _direct_round_timing_keys():
        elapsed = _timing_elapsed(round_entry.get(phase_id))
        if elapsed is not None:
            total += elapsed
    return round(total, 3)


def _collect_boundary_traces(rounds: Sequence[Mapping[str, Any]]) -> tuple[list[Mapping[str, Any]], int, int]:
    traces: list[Mapping[str, Any]] = []
    global_count = 0
    local_count = 0
    for round_entry in rounds:
        global_registration = round_entry.get("global_registration")
        if isinstance(global_registration, Mapping):
            boundary = global_registration.get("boundary_instrumentation")
            if isinstance(boundary, Mapping):
                _validate_boundary_instrumentation(boundary, field_name="round.global_registration.boundary_instrumentation")
                traces.append(boundary)
                global_count += 1

        local_registration = round_entry.get("local_registration")
        if isinstance(local_registration, Mapping):
            boundary = local_registration.get("boundary_instrumentation")
            if isinstance(boundary, Mapping):
                _validate_boundary_instrumentation(boundary, field_name="round.local_registration.boundary_instrumentation")
                traces.append(boundary)
                local_count += 1

        tiled = round_entry.get("tiled_local")
        if isinstance(tiled, Mapping):
            tiles = tiled.get("tiles")
            if isinstance(tiles, Sequence) and not isinstance(tiles, (str, bytes)):
                for tile in tiles:
                    if not isinstance(tile, Mapping):
                        continue
                    boundary = tile.get("boundary_instrumentation")
                    if isinstance(boundary, Mapping):
                        _validate_boundary_instrumentation(boundary, field_name="round.tiled_local.tiles[].boundary_instrumentation")
                        traces.append(boundary)
                        local_count += 1
    return traces, global_count, local_count


def _flow_sidecar_summary(rounds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = 0
    total_size = 0
    entries: list[dict[str, Any]] = []
    for round_entry in rounds:
        record = round_entry.get("flow_sidecar_persistence")
        if not isinstance(record, Mapping):
            continue
        details = record.get("details")
        if not isinstance(details, Mapping):
            continue
        descriptor = details.get("descriptor")
        if not isinstance(descriptor, Mapping):
            continue
        size_bytes = details.get("size_bytes")
        count += 1
        if isinstance(size_bytes, (int, float)) and not isinstance(size_bytes, bool):
            total_size += int(size_bytes)
        entries.append(
            {
                "round_id": int(round_entry.get("round_id", -1)),
                "path": details.get("path"),
                "descriptor": dict(descriptor),
                "exists": details.get("exists"),
                "size_bytes": size_bytes,
                "elapsed_wall_ms": record.get("elapsed_wall_ms"),
            }
        )
    return {
        "flow_sidecar_count": count,
        "flow_sidecar_total_bytes": total_size,
        "flow_sidecars": entries,
    }


def _iter_matlab_internal_timing_blocks(
    rounds: Sequence[Mapping[str, Any]],
) -> list[tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any] | None]]:
    blocks: list[tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any] | None]] = []
    for round_entry in rounds:
        local_registration = round_entry.get("local_registration")
        if isinstance(local_registration, Mapping):
            internal = local_registration.get("matlab_internal_timing")
            if isinstance(internal, Mapping):
                blocks.append((round_entry, internal, None))

        tiled = round_entry.get("tiled_local")
        if not isinstance(tiled, Mapping):
            continue
        tiles = tiled.get("tiles")
        if not isinstance(tiles, Sequence) or isinstance(tiles, (str, bytes)):
            continue
        for tile in tiles:
            if not isinstance(tile, Mapping):
                continue
            internal = tile.get("matlab_internal_timing")
            if isinstance(internal, Mapping):
                blocks.append((round_entry, internal, tile))
    return blocks


def _matlab_internal_timing_summary(rounds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    call_count = 0
    total_duration_ms = 0.0
    step_totals_ms: dict[str, float] = {}
    unaccounted_total_ms = 0.0
    boundary_minus_total_ms = 0.0
    boundary_delta_count = 0
    dominant_counts: dict[str, int] = {}
    dominant_totals_ms: dict[str, float] = {}

    for _round_entry, internal, _tile in _iter_matlab_internal_timing_blocks(rounds):
        _validate_matlab_internal_timing_block(internal, field_name="matlab_internal_timing")
        call_count += 1
        total_duration_ms = round(total_duration_ms + float(internal["total_duration_ms"]), 3)
        unaccounted_total_ms = round(unaccounted_total_ms + float(internal["unaccounted_duration_ms"]), 3)
        boundary_delta = internal.get("boundary_minus_matlab_total_ms")
        if boundary_delta is not None:
            boundary_minus_total_ms = round(boundary_minus_total_ms + float(boundary_delta), 3)
            boundary_delta_count += 1
        steps = internal.get("steps")
        if isinstance(steps, Sequence) and not isinstance(steps, (str, bytes)):
            for raw_step in steps:
                if not isinstance(raw_step, Mapping):
                    continue
                name = raw_step.get("name")
                if not isinstance(name, str) or not name.strip():
                    continue
                duration_ms = _validate_elapsed_ms(
                    raw_step.get("duration_ms"),
                    field_name=f"matlab_internal_timing.steps.{name}.duration_ms",
                )
                step_totals_ms[name] = round(step_totals_ms.get(name, 0.0) + duration_ms, 3)
        dominant = internal.get("dominant_step")
        if isinstance(dominant, Mapping):
            name = dominant.get("name")
            if isinstance(name, str) and name.strip():
                duration_ms = _validate_elapsed_ms(
                    dominant.get("duration_ms"),
                    field_name=f"matlab_internal_timing.dominant_step.{name}.duration_ms",
                )
                dominant_counts[name] = dominant_counts.get(name, 0) + 1
                dominant_totals_ms[name] = round(dominant_totals_ms.get(name, 0.0) + duration_ms, 3)

    boundary_minus_internal_total_ms = (
        round(boundary_minus_total_ms, 3)
        if call_count == 0 or boundary_delta_count == call_count
        else None
    )
    return {
        "matlab_internal_timing_status": "present" if call_count else "absent",
        "matlab_internal_call_count": int(call_count),
        "matlab_internal_total_duration_ms": round(total_duration_ms, 3),
        "matlab_internal_step_totals_ms": {key: round(value, 3) for key, value in sorted(step_totals_ms.items())},
        "matlab_internal_unaccounted_total_ms": round(unaccounted_total_ms, 3),
        "matlab_boundary_minus_internal_total_ms": boundary_minus_internal_total_ms,
        "matlab_internal_dominant_step_counts": dict(sorted(dominant_counts.items())),
        "matlab_internal_dominant_step_totals_ms": {
            key: round(value, 3) for key, value in sorted(dominant_totals_ms.items())
        },
    }


def _percentage(value: float, total: float) -> float:
    if total <= 0:
        return 0.0
    return round(float(value) / float(total) * 100.0, 3)


def _percentage_mapping(values: Mapping[str, float], total: float) -> dict[str, float]:
    return {key: _percentage(value, total) for key, value in sorted(values.items())}


def _profile_hot_path_row(
    *,
    component_type: str,
    name: str,
    owner: str,
    total_duration_ms: float,
    internal_total_ms: float,
    boundary_call_total_ms: float | None,
    call_count: int | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "component_type": component_type,
        "name": name,
        "owner": owner,
        "total_duration_ms": round(float(total_duration_ms), 3),
        "percent_of_matlab_internal_total": _percentage(float(total_duration_ms), internal_total_ms),
        "percent_of_boundary_matlab_call": None
        if boundary_call_total_ms is None
        else _percentage(float(total_duration_ms), boundary_call_total_ms),
    }
    if call_count is not None:
        row["call_count"] = int(call_count)
    return row


def _matlab_local_hot_path_profile(rounds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build an additive Stage19 local MATLAB hot-path profile.

    The profile derives only from already-recorded Stage17 internal timing and
    outer MATLAB boundary data.  It does not introduce a new timing owner or
    change MATLAB runtime behavior; older diagnostics without this optional
    block remain valid.
    """

    call_count = 0
    boundary_closure_count = 0
    boundary_call_total_ms = 0.0
    boundary_minus_internal_total_ms = 0.0
    internal_total_ms = 0.0
    internal_unaccounted_total_ms = 0.0
    step_totals_ms: dict[str, float] = {}
    step_call_counts: dict[str, int] = {}

    for _round_entry, internal, _tile in _iter_matlab_internal_timing_blocks(rounds):
        _validate_matlab_internal_timing_block(internal, field_name="matlab_local_hot_path_profile.matlab_internal_timing")
        call_count += 1
        internal_total_ms = round(internal_total_ms + float(internal["total_duration_ms"]), 3)
        internal_unaccounted_total_ms = round(
            internal_unaccounted_total_ms + float(internal["unaccounted_duration_ms"]),
            3,
        )

        boundary_call = internal.get("boundary_matlab_call_ms")
        boundary_delta = internal.get("boundary_minus_matlab_total_ms")
        if boundary_call is not None and boundary_delta is not None:
            boundary_call_total_ms = round(boundary_call_total_ms + float(boundary_call), 3)
            boundary_minus_internal_total_ms = round(boundary_minus_internal_total_ms + float(boundary_delta), 3)
            boundary_closure_count += 1

        steps = internal.get("steps")
        if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
            continue
        for raw_step in steps:
            if not isinstance(raw_step, Mapping):
                continue
            name = raw_step.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            duration_ms = _validate_elapsed_ms(
                raw_step.get("duration_ms"),
                field_name=f"matlab_local_hot_path_profile.steps.{name}.duration_ms",
            )
            step_name = name.strip()
            step_totals_ms[step_name] = round(step_totals_ms.get(step_name, 0.0) + duration_ms, 3)
            step_call_counts[step_name] = step_call_counts.get(step_name, 0) + 1

    boundary_closure_complete = call_count > 0 and boundary_closure_count == call_count
    closed_boundary_call_total_ms = round(boundary_call_total_ms, 3) if boundary_closure_complete else None
    closed_boundary_delta_total_ms = (
        round(boundary_minus_internal_total_ms, 3)
        if boundary_closure_complete
        else None
    )
    sorted_step_totals = {key: round(value, 3) for key, value in sorted(step_totals_ms.items())}
    step_percent_internal = _percentage_mapping(sorted_step_totals, internal_total_ms)
    step_percent_boundary = (
        _percentage_mapping(sorted_step_totals, closed_boundary_call_total_ms)
        if closed_boundary_call_total_ms is not None
        else {}
    )
    dominant_step_name: str | None = None
    dominant_step_total_ms = 0.0
    if sorted_step_totals:
        dominant_step_name, dominant_step_total_ms = max(
            sorted_step_totals.items(),
            key=lambda item: (float(item[1]), item[0]),
        )

    profile: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "present" if call_count else "absent",
        "source": "matlab_metadata.steps + boundary_instrumentation.seam_costs_ms.matlab_call_ms",
        "scope": "matlab_local_registration",
        "call_count": int(call_count),
        "boundary_closure_count": int(boundary_closure_count),
        "boundary_closure_complete": bool(boundary_closure_complete),
        "boundary_matlab_call_total_ms": closed_boundary_call_total_ms,
        "matlab_internal_total_duration_ms": round(internal_total_ms, 3),
        "boundary_minus_internal_total_ms": closed_boundary_delta_total_ms,
        "matlab_internal_unaccounted_total_ms": round(internal_unaccounted_total_ms, 3),
        "step_totals_ms": sorted_step_totals,
        "step_call_counts": dict(sorted(step_call_counts.items())),
        "step_percent_of_matlab_internal_total": step_percent_internal,
        "step_percent_of_boundary_matlab_call": step_percent_boundary,
        "dominant_internal_step": None,
        "hot_path_rankings": [],
    }

    rankings: list[dict[str, Any]] = []
    for step_name, duration_ms in sorted(
        sorted_step_totals.items(),
        key=lambda item: (-float(item[1]), item[0]),
    ):
        rankings.append(
            _profile_hot_path_row(
                component_type="matlab_step",
                name=step_name,
                owner="matlab_runtime/pystar_registration/pystar_register_local_demons_entry.m",
                total_duration_ms=duration_ms,
                internal_total_ms=internal_total_ms,
                boundary_call_total_ms=closed_boundary_call_total_ms,
                call_count=step_call_counts.get(step_name, 0),
            )
        )

    if closed_boundary_delta_total_ms is not None and closed_boundary_delta_total_ms >= 0:
        rankings.append(
            _profile_hot_path_row(
                component_type="python_matlab_boundary_closure",
                name="boundary_minus_internal",
                owner="pystar/matlab_registration.py + MATLAB Engine call boundary",
                total_duration_ms=closed_boundary_delta_total_ms,
                internal_total_ms=internal_total_ms,
                boundary_call_total_ms=closed_boundary_call_total_ms,
                call_count=boundary_closure_count,
            )
        )

    if internal_unaccounted_total_ms > 0:
        rankings.append(
            _profile_hot_path_row(
                component_type="matlab_entrypoint_unaccounted",
                name="metadata_unaccounted",
                owner="matlab_runtime/pystar_registration/pystar_register_local_demons_entry.m",
                total_duration_ms=internal_unaccounted_total_ms,
                internal_total_ms=internal_total_ms,
                boundary_call_total_ms=closed_boundary_call_total_ms,
                call_count=call_count,
            )
        )

    profile["hot_path_rankings"] = sorted(
        rankings,
        key=lambda item: (-float(item["total_duration_ms"]), str(item["component_type"]), str(item["name"])),
    )
    if dominant_step_name is not None:
        profile["dominant_internal_step"] = {
            "name": dominant_step_name,
            "total_duration_ms": round(dominant_step_total_ms, 3),
            "call_count": int(step_call_counts.get(dominant_step_name, 0)),
            "percent_of_matlab_internal_total": step_percent_internal.get(dominant_step_name, 0.0),
            "percent_of_boundary_matlab_call": step_percent_boundary.get(dominant_step_name),
            "owner": "matlab_runtime/pystar_registration/pystar_register_local_demons_entry.m",
        }

    return profile


def _iter_tiled_local_executions(rounds: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    executions: list[Mapping[str, Any]] = []
    for round_entry in rounds:
        tiled = round_entry.get("tiled_local")
        if not isinstance(tiled, Mapping):
            continue
        execution = tiled.get("execution")
        if isinstance(execution, Mapping):
            executions.append(execution)
    return executions


def _iter_worker_lifecycles(rounds: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    lifecycles: list[Mapping[str, Any]] = []
    for round_entry in rounds:
        tiled = round_entry.get("tiled_local")
        if not isinstance(tiled, Mapping):
            continue
        tiles = tiled.get("tiles")
        if not isinstance(tiles, Sequence) or isinstance(tiles, (str, bytes)):
            continue
        for tile in tiles:
            if not isinstance(tile, Mapping):
                continue
            lifecycle = tile.get("worker_lifecycle")
            if isinstance(lifecycle, Mapping):
                lifecycles.append(lifecycle)
    return lifecycles


def _worker_lifecycle_summary(rounds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    executions = _iter_tiled_local_executions(rounds)
    lifecycles = _iter_worker_lifecycles(rounds)
    execution_modes = sorted(
        {
            str(execution.get("effective_mode"))
            for execution in executions
            if isinstance(execution.get("effective_mode"), str) and str(execution.get("effective_mode")).strip()
        }
    )
    requested_modes = sorted(
        {
            str(execution.get("requested_mode"))
            for execution in executions
            if isinstance(execution.get("requested_mode"), str) and str(execution.get("requested_mode")).strip()
        }
    )
    worker_count_set: set[int] = set()
    for execution in executions:
        raw_worker_count = execution.get("worker_count")
        if isinstance(raw_worker_count, int) and not isinstance(raw_worker_count, bool):
            worker_count_set.add(int(raw_worker_count))
    worker_counts = sorted(worker_count_set)
    strict_flags = sorted(
        {
            bool(execution.get("strict_equivalence_audit"))
            for execution in executions
            if isinstance(execution.get("strict_equivalence_audit"), bool)
        }
    )
    failures: list[Any] = []
    tile_indices: list[int] = []
    for execution in executions:
        execution_failures = execution.get("failures")
        if isinstance(execution_failures, Sequence) and not isinstance(execution_failures, (str, bytes)):
            failures.extend(list(execution_failures))
        raw_indices = execution.get("tile_indices")
        if isinstance(raw_indices, Sequence) and not isinstance(raw_indices, (str, bytes)):
            for raw_index in raw_indices:
                if isinstance(raw_index, int) and not isinstance(raw_index, bool):
                    tile_indices.append(int(raw_index))

    totals = {key: 0.0 for key in WORKER_LIFECYCLE_MS_FIELDS}
    by_pid: dict[int, dict[str, Any]] = {}
    slowest_tiles: list[dict[str, Any]] = []
    first_use_tile_count = 0
    reused_tile_count = 0
    for lifecycle in lifecycles:
        normalized = _validate_worker_lifecycle(lifecycle, field_name="worker_lifecycle_summary.worker_lifecycle")
        pid = int(normalized["worker_process_pid"])
        tile_index = int(normalized["tile_index"])
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
        tile_wall = float(normalized["total_tile_wall_ms"])
        worker["total_tile_wall_ms"] = round(float(worker["total_tile_wall_ms"]) + tile_wall, 3)
        worker["max_tile_wall_ms"] = round(max(float(worker["max_tile_wall_ms"]), tile_wall), 3)
        worker_backend_reused = normalized.get("worker_backend_reused")
        if worker_backend_reused is True:
            reused_tile_count += 1
            worker["reused_tile_count"] = int(worker["reused_tile_count"]) + 1
        elif worker_backend_reused is False:
            first_use_tile_count += 1
            worker["first_use_tile_count"] = int(worker["first_use_tile_count"]) + 1
        slowest_tile = {
            "worker_process_pid": pid,
            "tile_index": tile_index,
            "total_tile_wall_ms": tile_wall,
            "backend_construct_ms": normalized["backend_construct_ms"],
            "matlab_call_ms": normalized["matlab_call_ms"],
            "backend_close_ms": normalized["backend_close_ms"],
        }
        if "worker_backend_reused" in normalized:
            slowest_tile["worker_backend_reused"] = normalized["worker_backend_reused"]
        if "worker_backend_reuse_index" in normalized:
            slowest_tile["worker_backend_reuse_index"] = normalized["worker_backend_reuse_index"]
        slowest_tiles.append(slowest_tile)
        for key in WORKER_LIFECYCLE_MS_FIELDS:
            totals[key] = round(totals[key] + float(normalized[key]), 3)

    total_tile_wall = totals["total_tile_wall_ms"]
    percentages = {
        key: round((value / total_tile_wall * 100.0), 3) if total_tile_wall > 0 else 0.0
        for key, value in totals.items()
    }
    slowest_workers = []
    for pid, row in sorted(by_pid.items()):
        tile_count = int(row["tile_count"])
        total_wall = round(float(row["total_tile_wall_ms"]), 3)
        slowest_workers.append(
            {
                "worker_process_pid": int(pid),
                "tile_count": tile_count,
                "first_use_tile_count": int(row.get("first_use_tile_count", 0)),
                "reused_tile_count": int(row.get("reused_tile_count", 0)),
                "total_tile_wall_ms": total_wall,
                "mean_tile_wall_ms": round(total_wall / tile_count, 3) if tile_count else 0.0,
                "max_tile_wall_ms": round(float(row["max_tile_wall_ms"]), 3),
            }
        )

    return {
        "tiled_local_execution_status": "present" if executions else "absent",
        "tiled_local_execution_modes": execution_modes,
        "tiled_local_requested_modes": requested_modes,
        "tiled_local_worker_counts": worker_counts,
        "tiled_local_strict_equivalence_audit_values": strict_flags,
        "tiled_local_execution_tile_indices": tile_indices,
        "tiled_local_execution_failure_count": len(failures),
        "tiled_local_worker_lifecycle_status": "present" if lifecycles else "absent",
        "tiled_local_worker_process_count": len(by_pid),
        "tiled_local_worker_process_ids": sorted(by_pid),
        "tiled_local_worker_tile_counts": {str(pid): int(row["tile_count"]) for pid, row in sorted(by_pid.items())},
        "tiled_local_worker_backend_reuse": {
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
        "tiled_local_worker_overhead_totals_ms": {key: round(value, 3) for key, value in totals.items()},
        "tiled_local_worker_overhead_percentages": percentages,
        "tiled_local_slowest_workers": sorted(
            slowest_workers,
            key=lambda item: (-float(item["total_tile_wall_ms"]), int(item["worker_process_pid"])),
        ),
        "tiled_local_slowest_worker_tiles": sorted(
            slowest_tiles,
            key=lambda item: (-float(item["total_tile_wall_ms"]), int(item["tile_index"])),
        )[:10],
    }


def _round_matlab_internal_total(round_entry: Mapping[str, Any]) -> float:
    total = 0.0
    for _candidate_round, internal, _tile in _iter_matlab_internal_timing_blocks([round_entry]):
        _validate_matlab_internal_timing_block(internal, field_name="round.matlab_internal_timing")
        total = round(total + float(internal["total_duration_ms"]), 3)
    return total


def _slowest_tiles(rounds: Sequence[Mapping[str, Any]], *, limit: int = 10) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for round_entry in rounds:
        round_id = int(round_entry.get("round_id", -1))
        tiled = round_entry.get("tiled_local")
        if not isinstance(tiled, Mapping):
            continue
        tiles = tiled.get("tiles")
        if not isinstance(tiles, Sequence) or isinstance(tiles, (str, bytes)):
            continue
        for tile in tiles:
            if not isinstance(tile, Mapping):
                continue
            timings = tile.get("timings")
            if not isinstance(timings, Mapping):
                continue
            total_record = timings.get("tile_total")
            elapsed = _timing_elapsed(total_record)
            if elapsed is None:
                continue
            row = {
                "round_id": round_id,
                "tile_index": tile.get("tile_index"),
                "grid_position_yx": tile.get("grid_position_yx"),
                "region_origin_zyx": tile.get("region_origin_zyx"),
                "region_shape_zyx": tile.get("region_shape_zyx"),
                "elapsed_wall_ms": elapsed,
            }
            internal = tile.get("matlab_internal_timing")
            if isinstance(internal, Mapping):
                _validate_matlab_internal_timing_block(internal, field_name="slowest_tiles[].matlab_internal_timing")
                row["matlab_internal_total_duration_ms"] = internal.get("total_duration_ms")
                row["matlab_internal_dominant_step"] = internal.get("dominant_step")
            rows.append(row)
    return sorted(rows, key=lambda item: (-float(item["elapsed_wall_ms"]), int(item.get("round_id") or 0)))[:limit]


def _round_sort_key(item: Mapping[str, Any]) -> tuple[float, int]:
    elapsed = _validate_elapsed_ms(item.get("total_elapsed_wall_ms"), field_name="summary.slowest_rounds.total_elapsed_wall_ms")
    round_id = item.get("round_id")
    if isinstance(round_id, bool) or not isinstance(round_id, int):
        raise ValueError("summary.slowest_rounds.round_id must be an integer")
    return (-elapsed, int(round_id))


def _build_summary(
    *,
    fov_setup: Mapping[str, Any],
    rounds: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    phase_totals = _phase_totals_from_payload(fov_setup=fov_setup, rounds=rounds, manifest=manifest)
    total_measured = round(sum(phase_totals.values()), 3)
    nested_phase_totals = _nested_phase_totals_from_rounds(rounds)
    boundary_traces, global_boundary_count, local_boundary_count = _collect_boundary_traces(rounds)
    seam_cost_totals: dict[str, float] = {key: 0.0 for key in MATLAB_BOUNDARY_SEAM_COST_KEYS}
    for boundary in boundary_traces:
        _sum_numeric_mapping(seam_cost_totals, _boundary_seam_costs(boundary))

    boundary_summary = summarize_matlab_boundary_traces(boundary_traces) if boundary_traces else None
    flow_summary = _flow_sidecar_summary(rounds)
    matlab_internal_summary = _matlab_internal_timing_summary(rounds)
    matlab_hot_path_profile = _matlab_local_hot_path_profile(rounds)
    worker_summary = _worker_lifecycle_summary(rounds)
    round_totals = [
        {
            "round_id": int(round_entry.get("round_id", -1)),
            "is_reference_round": bool(round_entry.get("is_reference_round", False)),
            "status": round_entry.get("status"),
            "total_elapsed_wall_ms": _round_total_elapsed(round_entry),
            "matlab_internal_total_duration_ms": _round_matlab_internal_total(round_entry),
        }
        for round_entry in rounds
    ]
    slowest_rounds = sorted(round_totals, key=_round_sort_key)[:10]
    moving_round_count = sum(1 for item in rounds if not bool(item.get("is_reference_round", False)))
    tile_count = 0
    for round_entry in rounds:
        tiled = round_entry.get("tiled_local")
        if not isinstance(tiled, Mapping):
            continue
        tiles = tiled.get("tiles")
        if isinstance(tiles, Sequence) and not isinstance(tiles, (str, bytes)):
            tile_count += len(tiles)
    return {
        "total_measured_registration_diagnostic_time_ms": total_measured,
        "phase_totals": {
            phase_id: _phase_total_row(phase_id, elapsed, total_measured)
            for phase_id, elapsed in sorted(phase_totals.items())
        },
        "nested_phase_totals_ms": {key: round(value, 3) for key, value in sorted(nested_phase_totals.items())},
        "moving_round_count": moving_round_count,
        "tile_count": int(tile_count),
        "matlab_global_boundary_call_count": global_boundary_count,
        "matlab_local_boundary_call_count": local_boundary_count,
        "matlab_boundary_call_count": len(boundary_traces),
        "matlab_boundary_seam_cost_totals_ms": {
            key: round(value, 3) for key, value in sorted(seam_cost_totals.items())
        },
        "matlab_boundary_summary": boundary_summary,
        **matlab_internal_summary,
        "matlab_local_hot_path_profile": matlab_hot_path_profile,
        **worker_summary,
        "flow_sidecar_count": flow_summary["flow_sidecar_count"],
        "flow_sidecar_total_bytes": flow_summary["flow_sidecar_total_bytes"],
        "flow_sidecars": flow_summary["flow_sidecars"],
        "slowest_rounds": slowest_rounds,
        "slowest_tiles": _slowest_tiles(rounds),
    }


def build_registration_performance_payload(
    *,
    fov_id: int,
    providers: Mapping[str, Any],
    registration_method: Mapping[str, Any] | None = None,
    fov_setup: Mapping[str, Any] | None = None,
    rounds: Sequence[Mapping[str, Any]] | None = None,
    manifest: Mapping[str, Any] | None = None,
    source_stage_elapsed_wall_ms: float | None = None,
) -> dict[str, Any]:
    """Build a versioned registration diagnostics payload."""

    stage_spec = get_stage_spec(REGISTRATION_PERFORMANCE_STAGE_ID)
    normalized_fov_setup = {str(key): dict(value) for key, value in (fov_setup or {}).items() if isinstance(value, Mapping)}
    normalized_rounds = [dict(round_entry) for round_entry in (rounds or [])]
    normalized_manifest = {str(key): dict(value) for key, value in (manifest or {}).items() if isinstance(value, Mapping)}
    payload: dict[str, Any] = {
        "schema_name": REGISTRATION_PERFORMANCE_SCHEMA_NAME,
        "schema_version": REGISTRATION_PERFORMANCE_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "fov_id": int(fov_id),
        "stage_id": stage_spec.stage_id,
        "source_stage_elapsed_wall_ms": None
        if source_stage_elapsed_wall_ms is None
        else _validate_elapsed_ms(source_stage_elapsed_wall_ms, field_name="source_stage_elapsed_wall_ms"),
        "providers": dict(providers),
        "registration_method": dict(registration_method or providers),
        "summary": _build_summary(
            fov_setup=normalized_fov_setup,
            rounds=normalized_rounds,
            manifest=normalized_manifest,
        ),
        "fov_setup": normalized_fov_setup,
        "rounds": normalized_rounds,
        "manifest": normalized_manifest,
    }
    validate_registration_performance_payload(payload, expected_fov_id=int(fov_id))
    return payload


def _validate_timing_record(value: Any, *, field_name: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a timing mapping")
    phase_id = value.get("phase_id")
    if not isinstance(phase_id, str) or not phase_id.strip():
        raise ValueError(f"{field_name}.phase_id must be a non-empty string")
    _ = _validate_elapsed_ms(value.get("elapsed_wall_ms"), field_name=f"{field_name}.elapsed_wall_ms")
    _ = _validate_status(value.get("status"), field_name=f"{field_name}.status")


def _validate_nested_timings(round_entry: Mapping[str, Any], *, field_name: str) -> None:
    for phase_id in _direct_round_timing_keys():
        if phase_id in round_entry:
            _validate_timing_record(round_entry[phase_id], field_name=f"{field_name}.{phase_id}")
    local_internal = round_entry.get("local_internal_timings")
    if isinstance(local_internal, Mapping):
        for phase_id, record in local_internal.items():
            _validate_timing_record(record, field_name=f"{field_name}.local_internal_timings.{phase_id}")
    tiled = round_entry.get("tiled_local")
    if isinstance(tiled, Mapping):
        stitch = tiled.get("stitch")
        if stitch is not None:
            _validate_timing_record(stitch, field_name=f"{field_name}.tiled_local.stitch")
        execution = tiled.get("execution")
        execution_effective_mode = execution.get("effective_mode") if isinstance(execution, Mapping) else None
        tiles = tiled.get("tiles")
        if isinstance(tiles, Sequence) and not isinstance(tiles, (str, bytes)):
            tile_indices: list[int] = []
            for index, tile in enumerate(tiles):
                if not isinstance(tile, Mapping):
                    raise ValueError(f"{field_name}.tiled_local.tiles[{index}] must be a mapping")
                tile_index = _validate_tiled_local_tile_identity(
                    tile,
                    field_name=f"{field_name}.tiled_local.tiles[{index}]",
                )
                if tile_index in tile_indices:
                    raise ValueError(
                        f"{field_name}.tiled_local.tiles contains duplicate tile identity {tile_index}"
                    )
                tile_indices.append(tile_index)
                boundary = tile.get("boundary_instrumentation")
                if isinstance(boundary, Mapping):
                    _validate_boundary_instrumentation(
                        boundary,
                        field_name=f"{field_name}.tiled_local.tiles[{index}].boundary_instrumentation",
                    )
                internal = tile.get("matlab_internal_timing")
                if internal is not None:
                    _validate_matlab_internal_timing_block(
                        internal,
                        field_name=f"{field_name}.tiled_local.tiles[{index}].matlab_internal_timing",
                    )
                lifecycle = tile.get("worker_lifecycle")
                if lifecycle is not None:
                    _validate_worker_lifecycle(
                        lifecycle,
                        field_name=f"{field_name}.tiled_local.tiles[{index}].worker_lifecycle",
                        tile_index=tile_index,
                    )
                elif execution_effective_mode == "process_parallel":
                    raise ValueError(
                        f"{field_name}.tiled_local.tiles[{index}].worker_lifecycle is required for process_parallel execution"
                    )
                timings = tile.get("timings")
                if not isinstance(timings, Mapping):
                    raise ValueError(f"{field_name}.tiled_local.tiles[{index}].timings must be a mapping")
                for timing_name, record in timings.items():
                    _validate_timing_record(record, field_name=f"{field_name}.tiled_local.tiles[{index}].timings.{timing_name}")
            if isinstance(execution, Mapping):
                _validate_tiled_local_execution_report(
                    execution,
                    field_name=f"{field_name}.tiled_local.execution",
                    tile_indices=tile_indices,
                )


def validate_registration_performance_payload(
    payload: Mapping[str, Any],
    *,
    expected_fov_id: int | None = None,
) -> None:
    """Fail loudly for malformed Stage16 diagnostics payloads."""

    if not isinstance(payload, Mapping):
        raise ValueError("Registration diagnostics payload must be a JSON object")
    if payload.get("schema_name") != REGISTRATION_PERFORMANCE_SCHEMA_NAME:
        raise ValueError(
            f"Registration diagnostics schema_name must be {REGISTRATION_PERFORMANCE_SCHEMA_NAME!r}, got {payload.get('schema_name')!r}"
        )
    if payload.get("schema_version") != REGISTRATION_PERFORMANCE_SCHEMA_VERSION:
        raise ValueError(
            "Registration diagnostics schema_version must be "
            f"{REGISTRATION_PERFORMANCE_SCHEMA_VERSION}, got {payload.get('schema_version')!r}"
        )
    generated_at = payload.get("generated_at_utc")
    if not isinstance(generated_at, str) or not generated_at.strip():
        raise ValueError("Registration diagnostics generated_at_utc must be a non-empty string")
    if payload.get("stage_id") != REGISTRATION_PERFORMANCE_STAGE_ID:
        raise ValueError("Registration diagnostics stage_id must be 'registration'")
    fov_id = payload.get("fov_id")
    if isinstance(fov_id, bool) or not isinstance(fov_id, int):
        raise ValueError("Registration diagnostics fov_id must be an integer")
    if expected_fov_id is not None and int(fov_id) != int(expected_fov_id):
        raise ValueError(
            f"Registration diagnostics fov/path mismatch: payload fov_id={fov_id}, expected {expected_fov_id}"
        )
    source_elapsed = payload.get("source_stage_elapsed_wall_ms")
    if source_elapsed is not None:
        _ = _validate_elapsed_ms(source_elapsed, field_name="source_stage_elapsed_wall_ms")
    if not isinstance(payload.get("providers"), Mapping):
        raise ValueError("Registration diagnostics providers must be a mapping")
    if not isinstance(payload.get("registration_method"), Mapping):
        raise ValueError("Registration diagnostics registration_method must be a mapping")

    fov_setup = payload.get("fov_setup")
    if not isinstance(fov_setup, Mapping):
        raise ValueError("Registration diagnostics fov_setup must be a mapping")
    for phase_id, record in fov_setup.items():
        _validate_timing_record(record, field_name=f"fov_setup.{phase_id}")

    rounds = payload.get("rounds")
    if not isinstance(rounds, list):
        raise ValueError("Registration diagnostics rounds must be a list")
    for index, round_entry in enumerate(rounds):
        if not isinstance(round_entry, Mapping):
            raise ValueError(f"Registration diagnostics rounds[{index}] must be a mapping")
        round_id = round_entry.get("round_id")
        if isinstance(round_id, bool) or not isinstance(round_id, int):
            raise ValueError(f"Registration diagnostics rounds[{index}].round_id must be an integer")
        if not isinstance(round_entry.get("is_reference_round"), bool):
            raise ValueError(f"Registration diagnostics rounds[{index}].is_reference_round must be a boolean")
        _ = _validate_status(round_entry.get("status"), field_name=f"rounds[{index}].status")
        for boundary_key in ("global_registration", "local_registration"):
            maybe_record = round_entry.get(boundary_key)
            if isinstance(maybe_record, Mapping):
                boundary = maybe_record.get("boundary_instrumentation")
                if isinstance(boundary, Mapping):
                    _validate_boundary_instrumentation(
                        boundary,
                        field_name=f"rounds[{index}].{boundary_key}.boundary_instrumentation",
                    )
                internal = maybe_record.get("matlab_internal_timing")
                if internal is not None:
                    _validate_matlab_internal_timing_block(
                        internal,
                        field_name=f"rounds[{index}].{boundary_key}.matlab_internal_timing",
                    )
        _validate_nested_timings(round_entry, field_name=f"rounds[{index}]")

    manifest = payload.get("manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("Registration diagnostics manifest must be a mapping")
    for phase_id, record in manifest.items():
        _validate_timing_record(record, field_name=f"manifest.{phase_id}")

    summary = payload.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError("Registration diagnostics summary must be a mapping")
    required_summary_fields = (
        "total_measured_registration_diagnostic_time_ms",
        "phase_totals",
        "nested_phase_totals_ms",
        "moving_round_count",
        "tile_count",
        "matlab_global_boundary_call_count",
        "matlab_local_boundary_call_count",
        "matlab_boundary_call_count",
        "matlab_boundary_seam_cost_totals_ms",
        "matlab_boundary_summary",
        "tiled_local_execution_status",
        "tiled_local_worker_lifecycle_status",
        "tiled_local_worker_process_count",
        "tiled_local_worker_tile_counts",
        "tiled_local_worker_overhead_totals_ms",
        "tiled_local_worker_overhead_percentages",
        "tiled_local_slowest_workers",
        "tiled_local_slowest_worker_tiles",
        "flow_sidecar_count",
        "flow_sidecar_total_bytes",
        "flow_sidecars",
        "slowest_rounds",
        "slowest_tiles",
    )
    missing_summary_fields = [field for field in required_summary_fields if field not in summary]
    if missing_summary_fields:
        raise ValueError(
            "Registration diagnostics summary is missing required fields: "
            + ", ".join(missing_summary_fields)
        )
    _ = _validate_elapsed_ms(
        summary.get("total_measured_registration_diagnostic_time_ms"),
        field_name="summary.total_measured_registration_diagnostic_time_ms",
    )
    for count_field in (
        "moving_round_count",
        "tile_count",
        "matlab_global_boundary_call_count",
        "matlab_local_boundary_call_count",
        "matlab_boundary_call_count",
        "tiled_local_worker_process_count",
        "flow_sidecar_count",
        "flow_sidecar_total_bytes",
    ):
        value = summary.get(count_field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Registration diagnostics summary.{count_field} must be a non-negative integer")
    if not isinstance(summary.get("phase_totals"), Mapping):
        raise ValueError("Registration diagnostics summary.phase_totals must be a mapping")
    if not isinstance(summary.get("nested_phase_totals_ms"), Mapping):
        raise ValueError("Registration diagnostics summary.nested_phase_totals_ms must be a mapping")
    if not isinstance(summary.get("matlab_boundary_seam_cost_totals_ms"), Mapping):
        raise ValueError("Registration diagnostics summary.matlab_boundary_seam_cost_totals_ms must be a mapping")
    if summary.get("tiled_local_execution_status") not in {"present", "absent"}:
        raise ValueError("Registration diagnostics summary.tiled_local_execution_status must be 'present' or 'absent'")
    if summary.get("tiled_local_worker_lifecycle_status") not in {"present", "absent"}:
        raise ValueError("Registration diagnostics summary.tiled_local_worker_lifecycle_status must be 'present' or 'absent'")
    if not isinstance(summary.get("tiled_local_worker_tile_counts"), Mapping):
        raise ValueError("Registration diagnostics summary.tiled_local_worker_tile_counts must be a mapping")
    for pid, count in cast(Mapping[str, Any], summary.get("tiled_local_worker_tile_counts")).items():
        if not isinstance(pid, str) or not pid.strip():
            raise ValueError("Registration diagnostics summary.tiled_local_worker_tile_counts keys must be non-empty strings")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"Registration diagnostics summary.tiled_local_worker_tile_counts.{pid} must be a non-negative integer")
    for mapping_name in ("tiled_local_worker_overhead_totals_ms", "tiled_local_worker_overhead_percentages"):
        mapping = summary.get(mapping_name)
        if not isinstance(mapping, Mapping):
            raise ValueError(f"Registration diagnostics summary.{mapping_name} must be a mapping")
        for key in WORKER_LIFECYCLE_MS_FIELDS:
            if key not in mapping:
                raise ValueError(f"Registration diagnostics summary.{mapping_name} is missing {key}")
            if mapping_name == "tiled_local_worker_overhead_totals_ms":
                _ = _validate_elapsed_ms(mapping.get(key), field_name=f"summary.{mapping_name}.{key}")
            else:
                percentage = _validate_finite_number(mapping.get(key), field_name=f"summary.{mapping_name}.{key}")
                if percentage < 0:
                    raise ValueError(f"Registration diagnostics summary.{mapping_name}.{key} must be non-negative")
    if not isinstance(summary.get("tiled_local_slowest_workers"), list):
        raise ValueError("Registration diagnostics summary.tiled_local_slowest_workers must be a list")
    if not isinstance(summary.get("tiled_local_slowest_worker_tiles"), list):
        raise ValueError("Registration diagnostics summary.tiled_local_slowest_worker_tiles must be a list")
    stage17_summary_fields = (
        "matlab_internal_timing_status",
        "matlab_internal_call_count",
        "matlab_internal_total_duration_ms",
        "matlab_internal_step_totals_ms",
        "matlab_internal_unaccounted_total_ms",
        "matlab_boundary_minus_internal_total_ms",
        "matlab_internal_dominant_step_counts",
        "matlab_internal_dominant_step_totals_ms",
    )
    present_stage17_fields = [field for field in stage17_summary_fields if field in summary]
    if present_stage17_fields:
        missing_stage17_fields = [field for field in stage17_summary_fields if field not in summary]
        if missing_stage17_fields:
            raise ValueError(
                "Registration diagnostics summary has partial MATLAB internal timing fields; missing: "
                + ", ".join(missing_stage17_fields)
            )
        internal_count = summary.get("matlab_internal_call_count")
        if isinstance(internal_count, bool) or not isinstance(internal_count, int) or internal_count < 0:
            raise ValueError("Registration diagnostics summary.matlab_internal_call_count must be a non-negative integer")
        internal_status = summary.get("matlab_internal_timing_status")
        if internal_status not in {"present", "absent"}:
            raise ValueError("Registration diagnostics summary.matlab_internal_timing_status must be 'present' or 'absent'")
        _ = _validate_elapsed_ms(
            summary.get("matlab_internal_total_duration_ms"),
            field_name="summary.matlab_internal_total_duration_ms",
        )
        _ = _validate_elapsed_ms(
            summary.get("matlab_internal_unaccounted_total_ms"),
            field_name="summary.matlab_internal_unaccounted_total_ms",
        )
        summary_boundary_delta = summary.get("matlab_boundary_minus_internal_total_ms")
        if summary_boundary_delta is not None:
            _ = _validate_finite_number(
                summary_boundary_delta,
                field_name="summary.matlab_boundary_minus_internal_total_ms",
            )
        if not isinstance(summary.get("matlab_internal_step_totals_ms"), Mapping):
            raise ValueError("Registration diagnostics summary.matlab_internal_step_totals_ms must be a mapping")
        for step_name, duration_ms in cast(Mapping[str, Any], summary.get("matlab_internal_step_totals_ms")).items():
            if not isinstance(step_name, str) or not step_name.strip():
                raise ValueError("Registration diagnostics summary.matlab_internal_step_totals_ms keys must be non-empty strings")
            _ = _validate_elapsed_ms(
                duration_ms,
                field_name=f"summary.matlab_internal_step_totals_ms.{step_name}",
            )
        if not isinstance(summary.get("matlab_internal_dominant_step_counts"), Mapping):
            raise ValueError("Registration diagnostics summary.matlab_internal_dominant_step_counts must be a mapping")
        for step_name, count in cast(Mapping[str, Any], summary.get("matlab_internal_dominant_step_counts")).items():
            if not isinstance(step_name, str) or not step_name.strip():
                raise ValueError("Registration diagnostics summary.matlab_internal_dominant_step_counts keys must be non-empty strings")
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError(f"Registration diagnostics summary.matlab_internal_dominant_step_counts.{step_name} must be a non-negative integer")
        if not isinstance(summary.get("matlab_internal_dominant_step_totals_ms"), Mapping):
            raise ValueError("Registration diagnostics summary.matlab_internal_dominant_step_totals_ms must be a mapping")
        for step_name, duration_ms in cast(Mapping[str, Any], summary.get("matlab_internal_dominant_step_totals_ms")).items():
            if not isinstance(step_name, str) or not step_name.strip():
                raise ValueError("Registration diagnostics summary.matlab_internal_dominant_step_totals_ms keys must be non-empty strings")
            _ = _validate_elapsed_ms(
                duration_ms,
                field_name=f"summary.matlab_internal_dominant_step_totals_ms.{step_name}",
            )
    hot_path_profile = summary.get("matlab_local_hot_path_profile")
    if hot_path_profile is not None:
        _validate_matlab_local_hot_path_profile(
            hot_path_profile,
            field_name="summary.matlab_local_hot_path_profile",
        )
    if not isinstance(summary.get("flow_sidecars"), list):
        raise ValueError("Registration diagnostics summary.flow_sidecars must be a list")
    if not isinstance(summary.get("slowest_rounds"), list):
        raise ValueError("Registration diagnostics summary.slowest_rounds must be a list")
    if not isinstance(summary.get("slowest_tiles"), list):
        raise ValueError("Registration diagnostics summary.slowest_tiles must be a list")


def write_registration_performance_diagnostics(
    *,
    base_dir: Path,
    fov_id: int,
    payload: Mapping[str, Any],
) -> Path:
    """Persist the Stage16 diagnostics JSON sidecar through shared metadata I/O."""

    validate_registration_performance_payload(payload, expected_fov_id=int(fov_id))
    output_path = get_registration_performance_path(base_dir, int(fov_id))
    write_backend_metadata(output_path, cast(dict[str, object], dict(payload)))
    return output_path


def load_registration_performance_diagnostics(
    path: Path,
    *,
    expected_fov_id: int | None = None,
) -> dict[str, Any]:
    """Load and validate a Stage16 registration diagnostics sidecar."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Malformed registration diagnostics at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Malformed registration diagnostics at {path}: expected JSON object")
    validate_registration_performance_payload(payload, expected_fov_id=expected_fov_id)
    return cast(dict[str, Any], payload)
