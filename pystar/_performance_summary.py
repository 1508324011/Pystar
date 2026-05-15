# pyright: reportPrivateUsage=false, reportExplicitAny=false, reportAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false
"""Run-level summaries for Stage14 per-FOV performance telemetry.

This module is intentionally read-only with respect to scientific artifacts.  It
loads the existing ``qc_reports/performance_fov_<fov>.json`` sidecars, aggregates
their already-recorded stage timings / artifact facts, and writes a small JSON +
Markdown summary for commit-to-commit comparisons.  It does not inspect raw
images, rerun stages, or change batch runner semantics.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, cast

from ._io_paths import _build_fov_output_paths
from ._performance_telemetry import (
    PERFORMANCE_TELEMETRY_SCHEMA_NAME,
    PERFORMANCE_TELEMETRY_SCHEMA_VERSION,
    utc_now_iso,
)
from ._stage_contracts import get_ordered_stage_ids, get_stage_spec, validate_stage_ids
from .serialization import write_backend_metadata


PERFORMANCE_SUMMARY_SCHEMA_NAME = "pystar_performance_summary"
PERFORMANCE_SUMMARY_SCHEMA_VERSION = 1
PERFORMANCE_TELEMETRY_GLOB = "Position*/output_pystar/qc_reports/performance_fov_*.json"

_POSITION_DIR_RE = re.compile(r"^Position(?P<fov_id>\d+)$")
_TELEMETRY_FILE_RE = re.compile(r"^performance_fov_(?P<fov_id>\d+)\.json$")


@dataclass(frozen=True)
class PerformanceTelemetrySource:
    """One expected or discovered Stage14 telemetry sidecar."""

    fov_id: int
    path: Path


def parse_fov_ids(raw_value: str | None) -> tuple[int, ...]:
    """Parse comma/range FOV syntax such as ``"1-3,7"``.

    The helper is shared by the CLI and tests so missing-FOV behavior is
    explicit rather than inferred from whichever telemetry files happen to be
    present.
    """

    if raw_value is None or not raw_value.strip():
        return ()

    fov_ids: list[int] = []
    seen: set[int] = set()
    for raw_part in raw_value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            bounds = [item.strip() for item in part.split("-", 1)]
            if len(bounds) != 2 or not bounds[0] or not bounds[1]:
                raise ValueError(f"Invalid FOV range {part!r}; expected syntax like '1-3,7'")
            try:
                start = int(bounds[0])
                stop = int(bounds[1])
            except ValueError as exc:
                raise ValueError(f"Invalid FOV range {part!r}; range bounds must be integers") from exc
            if start > stop:
                raise ValueError(f"Invalid FOV range {part!r}; start must be <= stop")
            values = range(start, stop + 1)
        else:
            try:
                values = (int(part),)
            except ValueError as exc:
                raise ValueError(f"Invalid FOV id {part!r}; expected integer or range") from exc

        for fov_id in values:
            if fov_id < 0:
                raise ValueError(f"Invalid FOV id {fov_id}; expected non-negative integer")
            if fov_id not in seen:
                fov_ids.append(fov_id)
                seen.add(fov_id)

    return tuple(fov_ids)


def _canonical_telemetry_path(base_dir: Path, fov_id: int) -> Path:
    paths = _build_fov_output_paths(base_dir, fov_id)
    return paths["qc"] / f"performance_fov_{fov_id}.json"


def _dedupe_ints(values: Sequence[int]) -> tuple[int, ...]:
    ordered: list[int] = []
    seen: set[int] = set()
    for value in values:
        coerced = int(value)
        if coerced < 0:
            raise ValueError(f"Invalid FOV id {coerced}; expected non-negative integer")
        if coerced not in seen:
            ordered.append(coerced)
            seen.add(coerced)
    return tuple(ordered)


def discover_performance_telemetry_sources(
    base_dir: Path,
    *,
    fov_ids: Sequence[int] | None = None,
) -> tuple[PerformanceTelemetrySource, ...]:
    """Discover canonical per-FOV telemetry sidecars under ``base_dir``.

    If ``fov_ids`` is provided, every requested FOV is returned even when its
    sidecar is absent.  Without explicit FOVs, the summary includes all
    ``Position<id>`` directories and any matching telemetry sidecars discovered
    under the canonical PyStar output tree.
    """

    base_dir = Path(base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"Performance telemetry base directory does not exist: {base_dir}")
    if not base_dir.is_dir():
        raise ValueError(f"Performance telemetry base path is not a directory: {base_dir}")

    if fov_ids is not None and len(fov_ids) > 0:
        expected_fov_ids = _dedupe_ints([int(value) for value in fov_ids])
    else:
        discovered: set[int] = set()
        for child in base_dir.iterdir():
            if not child.is_dir():
                continue
            match = _POSITION_DIR_RE.match(child.name)
            if match is not None:
                discovered.add(int(match.group("fov_id")))
        for path in base_dir.glob(PERFORMANCE_TELEMETRY_GLOB):
            match = _TELEMETRY_FILE_RE.match(path.name)
            if match is not None:
                discovered.add(int(match.group("fov_id")))
        expected_fov_ids = tuple(sorted(discovered))

    return tuple(
        PerformanceTelemetrySource(
            fov_id=fov_id,
            path=_canonical_telemetry_path(base_dir, fov_id),
        )
        for fov_id in expected_fov_ids
    )


def _schema_error(path: Path, detail: str) -> ValueError:
    return ValueError(
        f"Malformed performance telemetry at {path}: {detail}. Expected Stage14 JSON object "
        + f"with schema_name={PERFORMANCE_TELEMETRY_SCHEMA_NAME!r}, schema_version="
        + f"{PERFORMANCE_TELEMETRY_SCHEMA_VERSION}, canonical stage_order, stages[], artifacts, and run mappings."
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise _schema_error(path, f"unable to parse JSON ({exc.__class__.__name__}: {exc})") from exc
    if not isinstance(payload, dict):
        raise _schema_error(path, f"root payload must be a JSON object, got {type(payload)!r}")
    return cast(dict[str, Any], payload)


def _require_mapping(payload: Mapping[str, Any], key: str, path: Path) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise _schema_error(path, f"field {key!r} must be a JSON object")
    return dict(value)


def _require_stage_mappings(payload: Mapping[str, Any], path: Path) -> list[dict[str, Any]]:
    stages = payload.get("stages")
    if not isinstance(stages, list):
        raise _schema_error(path, "field 'stages' must be a JSON list")
    normalized: list[dict[str, Any]] = []
    stage_ids: list[str] = []
    for index, stage in enumerate(stages):
        if not isinstance(stage, Mapping):
            raise _schema_error(path, f"stages[{index}] must be a JSON object")
        stage_id = stage.get("stage_id")
        if not isinstance(stage_id, str):
            raise _schema_error(path, f"stages[{index}].stage_id must be a string")
        elapsed = stage.get("elapsed_wall_ms")
        if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)):
            raise _schema_error(path, f"stages[{index}].elapsed_wall_ms must be numeric")
        elapsed_value = float(elapsed)
        if not math.isfinite(elapsed_value) or elapsed_value < 0:
            raise _schema_error(path, f"stages[{index}].elapsed_wall_ms must be finite and non-negative")
        status = stage.get("status")
        if not isinstance(status, str) or not status.strip():
            raise _schema_error(path, f"stages[{index}].status must be a non-empty string")
        stage_ids.append(stage_id)
        normalized.append(dict(stage))

    try:
        _ = validate_stage_ids(tuple(stage_ids))
    except ValueError as exc:
        raise _schema_error(path, str(exc)) from exc
    return normalized


def _validate_telemetry_payload(
    payload: Mapping[str, Any],
    *,
    path: Path,
    expected_fov_id: int,
) -> dict[str, Any]:
    schema_name = payload.get("schema_name")
    if schema_name != PERFORMANCE_TELEMETRY_SCHEMA_NAME:
        raise _schema_error(path, f"schema_name is {schema_name!r}")
    schema_version = payload.get("schema_version")
    if schema_version != PERFORMANCE_TELEMETRY_SCHEMA_VERSION:
        raise _schema_error(path, f"schema_version is {schema_version!r}")
    fov_id = payload.get("fov_id")
    if isinstance(fov_id, bool) or not isinstance(fov_id, int):
        raise _schema_error(path, "field 'fov_id' must be an integer")
    if int(fov_id) != int(expected_fov_id):
        raise _schema_error(path, f"field 'fov_id' is {fov_id}, expected {expected_fov_id} from canonical path")
    stage_order = payload.get("stage_order")
    if not isinstance(stage_order, list) or not all(isinstance(item, str) for item in stage_order):
        raise _schema_error(path, "field 'stage_order' must be a string list")
    try:
        _ = validate_stage_ids(tuple(stage_order))
    except ValueError as exc:
        raise _schema_error(path, str(exc)) from exc

    stages = _require_stage_mappings(payload, path)
    artifacts = _require_mapping(payload, "artifacts", path)
    run = _require_mapping(payload, "run", path)
    providers = _require_mapping(payload, "providers", path)
    return {
        "fov_id": int(fov_id),
        "stage_order": list(stage_order),
        "stages": stages,
        "artifacts": artifacts,
        "run": run,
        "providers": providers,
    }


def _round_float(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 3)


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    coerced = float(value)
    return coerced if math.isfinite(coerced) else None


def _stage_elapsed_by_id(stages: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    return {str(stage["stage_id"]): float(stage["elapsed_wall_ms"]) for stage in stages}


def _stage_payload_by_id(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    stages = payload.get("stages")
    if not isinstance(stages, Sequence) or isinstance(stages, (str, bytes)):
        return {}
    return {
        str(stage.get("stage_id")): dict(stage)
        for stage in stages
        if isinstance(stage, Mapping) and isinstance(stage.get("stage_id"), str)
    }


def _build_read_fov_record(path: Path, telemetry: Mapping[str, Any]) -> dict[str, Any]:
    stages = cast(list[dict[str, Any]], telemetry["stages"])
    elapsed_by_stage = _stage_elapsed_by_id(stages)
    total_stage_elapsed = sum(elapsed_by_stage.values())
    slowest_stage_id = max(elapsed_by_stage.items(), key=lambda item: item[1])[0] if elapsed_by_stage else None
    run_total = _finite_number(cast(Mapping[str, Any], telemetry["run"]).get("total_elapsed_ms"))
    return {
        "fov_id": int(telemetry["fov_id"]),
        "telemetry_path": str(path),
        "telemetry_status": "read",
        "total_stage_elapsed_wall_ms": _round_float(total_stage_elapsed),
        "run_total_elapsed_ms": _round_float(run_total),
        "slowest_stage": None
        if slowest_stage_id is None
        else {
            "stage_id": slowest_stage_id,
            "elapsed_wall_ms": _round_float(elapsed_by_stage[slowest_stage_id]),
        },
        "stages": [
            {
                "stage_id": str(stage["stage_id"]),
                "elapsed_wall_ms": _round_float(float(stage["elapsed_wall_ms"])),
                "status": str(stage["status"]),
            }
            for stage in stages
        ],
        "artifact_highlights": _artifact_highlights(cast(Mapping[str, Any], telemetry["artifacts"])),
    }


def _build_absent_fov_record(source: PerformanceTelemetrySource) -> dict[str, Any]:
    return {
        "fov_id": int(source.fov_id),
        "telemetry_path": str(source.path),
        "telemetry_status": "absent",
        "total_stage_elapsed_wall_ms": None,
        "run_total_elapsed_ms": None,
        "slowest_stage": None,
        "stages": [],
        "artifact_highlights": {},
    }


def _artifact_highlights(artifacts: Mapping[str, Any]) -> dict[str, Any]:
    spot_table = _mapping_or_empty(artifacts.get("spot_table"))
    intensity_matrix = _mapping_or_empty(artifacts.get("intensity_matrix"))
    decoded_outputs = _mapping_or_empty(artifacts.get("decoded_outputs"))
    active_decoded = _mapping_or_empty(decoded_outputs.get("active"))
    goodreads = _mapping_or_empty(decoded_outputs.get("goodreads"))
    return {
        "spot_rows": _finite_number(spot_table.get("row_count")),
        "intensity_matrix_shape": intensity_matrix.get("shape"),
        "intensity_matrix_size_bytes": _finite_number(intensity_matrix.get("size_bytes")),
        "decoded_active_rows": _finite_number(active_decoded.get("row_count")),
        "decoded_goodreads_rows": _finite_number(goodreads.get("row_count")),
    }


def _elapsed_stats(values: Sequence[float], *, total_all_stages_ms: float) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "total_elapsed_wall_ms": 0.0,
            "mean_elapsed_wall_ms": None,
            "median_elapsed_wall_ms": None,
            "min_elapsed_wall_ms": None,
            "max_elapsed_wall_ms": None,
            "percent_of_total_measured_stage_time": 0.0,
        }
    total = float(sum(values))
    percent = (total / total_all_stages_ms * 100.0) if total_all_stages_ms > 0 else 0.0
    return {
        "count": len(values),
        "total_elapsed_wall_ms": _round_float(total),
        "mean_elapsed_wall_ms": _round_float(total / len(values)),
        "median_elapsed_wall_ms": _round_float(float(median(values))),
        "min_elapsed_wall_ms": _round_float(min(values)),
        "max_elapsed_wall_ms": _round_float(max(values)),
        "percent_of_total_measured_stage_time": round(percent, 3),
    }


def _build_stage_aggregates(read_payloads: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    values_by_stage: dict[str, list[float]] = {stage_id: [] for stage_id in get_ordered_stage_ids()}
    for payload in read_payloads:
        for stage in cast(Sequence[Mapping[str, Any]], payload["stages"]):
            values_by_stage[str(stage["stage_id"])].append(float(stage["elapsed_wall_ms"]))

    total_all_stages_ms = sum(sum(values) for values in values_by_stage.values())
    return {
        stage_id: {
            "stage_id": stage_id,
            "order_index": get_stage_spec(stage_id).order_index,
            "display_label": get_stage_spec(stage_id).display_label,
            **_elapsed_stats(values_by_stage[stage_id], total_all_stages_ms=total_all_stages_ms),
        }
        for stage_id in get_ordered_stage_ids()
    }


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return cast(Mapping[str, Any], value) if isinstance(value, Mapping) else {}


def _read_status(info: Mapping[str, Any]) -> str:
    status = info.get("read_status")
    if isinstance(status, str) and status:
        return status
    if info.get("exists") is True:
        return "present"
    if info.get("exists") is False:
        return "absent"
    return "unknown"


def _summarize_numeric_values(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "total": 0.0, "mean": None, "min": None, "max": None}
    total = sum(values)
    return {
        "count": len(values),
        "total": _round_float(total),
        "mean": _round_float(total / len(values)),
        "min": _round_float(min(values)),
        "max": _round_float(max(values)),
    }


def _summarize_file_artifact(
    read_payloads: Sequence[Mapping[str, Any]],
    accessor: Callable[[Mapping[str, Any]], Any],
) -> dict[str, Any]:
    present_count = 0
    absent_count = 0
    error_count = 0
    status_counts: dict[str, int] = {}
    size_values: list[float] = []
    row_values: list[float] = []
    shapes: set[tuple[int, ...]] = set()

    for payload in read_payloads:
        artifacts = _mapping_or_empty(payload.get("artifacts"))
        info = _mapping_or_empty(accessor(artifacts))
        status = _read_status(info)
        status_counts[status] = status_counts.get(status, 0) + 1
        if info.get("exists") is True:
            present_count += 1
        else:
            absent_count += 1
        if status == "error" or "stat_error" in info or "read_error" in info:
            error_count += 1
        size_value = _finite_number(info.get("size_bytes"))
        if size_value is not None:
            size_values.append(size_value)
        row_value = _finite_number(info.get("row_count"))
        if row_value is not None:
            row_values.append(row_value)
        shape = info.get("shape")
        if isinstance(shape, list) and all(isinstance(item, int) and not isinstance(item, bool) for item in shape):
            shapes.add(tuple(int(item) for item in shape))

    return {
        "present_count": present_count,
        "absent_count": absent_count,
        "error_count": error_count,
        "read_status_counts": status_counts,
        "size_bytes": _summarize_numeric_values(size_values),
        "row_count": _summarize_numeric_values(row_values),
        "shape_observations": [list(shape) for shape in sorted(shapes)],
    }


def _build_flow_sidecar_summary(read_payloads: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total_count = 0
    total_size_bytes = 0.0
    fovs_with_sidecars = 0
    shapes: set[tuple[int, ...]] = set()
    by_fov: list[dict[str, Any]] = []
    for payload in read_payloads:
        artifacts = _mapping_or_empty(payload.get("artifacts"))
        raw_sidecars = artifacts.get("flow_3d_sidecars")
        sidecars = raw_sidecars if isinstance(raw_sidecars, list) else []
        sidecar_size = 0.0
        for sidecar in sidecars:
            info = _mapping_or_empty(sidecar)
            size_value = _finite_number(info.get("size_bytes"))
            if size_value is not None:
                sidecar_size += size_value
            shape = info.get("shape")
            if isinstance(shape, list) and all(isinstance(item, int) and not isinstance(item, bool) for item in shape):
                shapes.add(tuple(int(item) for item in shape))
        count = len(sidecars)
        total_count += count
        total_size_bytes += sidecar_size
        if count > 0:
            fovs_with_sidecars += 1
        by_fov.append(
            {
                "fov_id": int(payload["fov_id"]),
                "sidecar_count": count,
                "size_bytes": _round_float(sidecar_size),
            }
        )
    return {
        "total_sidecar_count": total_count,
        "fovs_with_sidecars_count": fovs_with_sidecars,
        "total_size_bytes": _round_float(total_size_bytes),
        "shape_observations": [list(shape) for shape in sorted(shapes)],
        "by_fov": by_fov,
    }


def _build_artifact_summary(read_payloads: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "source_telemetry_count": len(read_payloads),
        "transform_manifest": _summarize_file_artifact(read_payloads, lambda artifacts: artifacts.get("transform_manifest")),
        "flow_3d_sidecars": _build_flow_sidecar_summary(read_payloads),
        "spot_table": _summarize_file_artifact(read_payloads, lambda artifacts: artifacts.get("spot_table")),
        "intensity_matrix": _summarize_file_artifact(read_payloads, lambda artifacts: artifacts.get("intensity_matrix")),
        "intensity_matrix_metadata": _summarize_file_artifact(read_payloads, lambda artifacts: artifacts.get("intensity_matrix_metadata")),
        "decoded_outputs": {
            "active": _summarize_file_artifact(
                read_payloads,
                lambda artifacts: _mapping_or_empty(artifacts.get("decoded_outputs")).get("active"),
            ),
            "goodreads": _summarize_file_artifact(
                read_payloads,
                lambda artifacts: _mapping_or_empty(artifacts.get("decoded_outputs")).get("goodreads"),
            ),
            "pre_pattern_check": _summarize_file_artifact(
                read_payloads,
                lambda artifacts: _mapping_or_empty(artifacts.get("decoded_outputs")).get("pre_pattern_check"),
            ),
        },
        "backend_metadata_sidecars": {
            "preprocessing_provenance": _summarize_file_artifact(
                read_payloads,
                lambda artifacts: _mapping_or_empty(artifacts.get("backend_metadata_sidecars")).get("preprocessing_provenance"),
            ),
            "spot_finding_backend": _summarize_file_artifact(
                read_payloads,
                lambda artifacts: _mapping_or_empty(artifacts.get("backend_metadata_sidecars")).get("spot_finding_backend"),
            ),
            "extraction_backend": _summarize_file_artifact(
                read_payloads,
                lambda artifacts: _mapping_or_empty(artifacts.get("backend_metadata_sidecars")).get("extraction_backend"),
            ),
        },
    }


def _accumulate_numeric_mapping(target: dict[str, float], raw_mapping: Any) -> None:
    if not isinstance(raw_mapping, Mapping):
        return
    for key, value in raw_mapping.items():
        if not isinstance(key, str):
            continue
        number = _finite_number(value)
        if number is not None:
            target[key] = round(target.get(key, 0.0) + number, 3)


def _build_matlab_summary(read_payloads: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for stage_id in get_ordered_stage_ids():
        metadata_status_counts: dict[str, int] = {}
        boundary_count = 0
        boundary_absent_count = 0
        session_count = 0
        session_absent_count = 0
        aggregate_seam_costs_ms: dict[str, float] = {}
        aggregate_session_counts: dict[str, float] = {}
        aggregate_session_timing_ms: dict[str, float] = {}
        total_boundary_call_count = 0
        total_engine_reused_calls = 0
        total_session_count = 0

        for payload in read_payloads:
            stage = _stage_payload_by_id(payload).get(stage_id, {})
            matlab = _mapping_or_empty(stage.get("matlab"))
            sources = matlab.get("metadata_sources")
            if isinstance(sources, list) and sources:
                for source in sources:
                    status = _read_status(_mapping_or_empty(source))
                    metadata_status_counts[status] = metadata_status_counts.get(status, 0) + 1
            else:
                metadata_status_counts["not_reported"] = metadata_status_counts.get("not_reported", 0) + 1

            boundary = matlab.get("boundary_instrumentation_summary")
            if isinstance(boundary, Mapping):
                boundary_count += 1
                total_boundary_call_count += int(_finite_number(boundary.get("call_count")) or 0)
                total_engine_reused_calls += int(_finite_number(boundary.get("engine_reused_calls")) or 0)
                _accumulate_numeric_mapping(aggregate_seam_costs_ms, boundary.get("aggregate_seam_costs_ms"))
            else:
                boundary_absent_count += 1

            session = matlab.get("session_lifecycle_summary")
            if isinstance(session, Mapping):
                session_count += 1
                total_session_count += int(_finite_number(session.get("session_count")) or 0)
                _accumulate_numeric_mapping(aggregate_session_counts, session.get("aggregate_counts"))
                _accumulate_numeric_mapping(aggregate_session_timing_ms, session.get("aggregate_timing_ms"))
            else:
                session_absent_count += 1

        summary[stage_id] = {
            "stage_id": stage_id,
            "metadata_source_read_status_counts": metadata_status_counts,
            "boundary_summary_count": boundary_count,
            "boundary_summary_absent_count": boundary_absent_count,
            "aggregate_boundary_call_count": total_boundary_call_count,
            "aggregate_engine_reused_calls": total_engine_reused_calls,
            "aggregate_seam_costs_ms": aggregate_seam_costs_ms,
            "session_summary_count": session_count,
            "session_summary_absent_count": session_absent_count,
            "aggregate_session_count": total_session_count,
            "aggregate_session_counts": aggregate_session_counts,
            "aggregate_session_timing_ms": aggregate_session_timing_ms,
        }
    return summary


def _rank_stage_aggregates(stage_aggregates: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        stage_aggregates.values(),
        key=lambda item: (-float(item.get("total_elapsed_wall_ms") or 0.0), int(item.get("order_index") or 0)),
    )
    return [dict(item) for item in ranked]


def _rank_slow_fovs(fov_records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    readable = [record for record in fov_records if record.get("telemetry_status") == "read"]
    ranked = sorted(
        readable,
        key=lambda item: (-float(item.get("total_stage_elapsed_wall_ms") or 0.0), int(item.get("fov_id") or 0)),
    )
    return [
        {
            "fov_id": int(record["fov_id"]),
            "total_stage_elapsed_wall_ms": record.get("total_stage_elapsed_wall_ms"),
            "run_total_elapsed_ms": record.get("run_total_elapsed_ms"),
            "slowest_stage": record.get("slowest_stage"),
            "telemetry_path": record.get("telemetry_path"),
        }
        for record in ranked
    ]


def build_performance_summary_payload(
    *,
    base_dir: Path,
    fov_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Build a deterministic run-level summary from Stage14 sidecars."""

    base_dir = Path(base_dir)
    sources = discover_performance_telemetry_sources(base_dir, fov_ids=fov_ids)
    read_payloads: list[dict[str, Any]] = []
    fov_records: list[dict[str, Any]] = []

    for source in sources:
        if not source.path.exists():
            fov_records.append(_build_absent_fov_record(source))
            continue
        raw_payload = _read_json_object(source.path)
        telemetry = _validate_telemetry_payload(
            raw_payload,
            path=source.path,
            expected_fov_id=source.fov_id,
        )
        read_payloads.append(telemetry)
        fov_records.append(_build_read_fov_record(source.path, telemetry))

    stage_aggregates = _build_stage_aggregates(read_payloads)
    slow_fov_rankings = _rank_slow_fovs(fov_records)
    total_measured_stage_ms = sum(
        float(record.get("total_stage_elapsed_wall_ms") or 0.0)
        for record in fov_records
        if record.get("telemetry_status") == "read"
    )
    run_total_values = [
        float(record["run_total_elapsed_ms"])
        for record in fov_records
        if isinstance(record.get("run_total_elapsed_ms"), (int, float))
    ]

    absent_fov_ids = [int(record["fov_id"]) for record in fov_records if record.get("telemetry_status") == "absent"]
    read_fov_ids = [int(record["fov_id"]) for record in fov_records if record.get("telemetry_status") == "read"]
    return {
        "schema_name": PERFORMANCE_SUMMARY_SCHEMA_NAME,
        "schema_version": PERFORMANCE_SUMMARY_SCHEMA_VERSION,
        "generated_at_utc": utc_now_iso(),
        "base_dir": str(base_dir),
        "inputs": {
            "telemetry_file_pattern": PERFORMANCE_TELEMETRY_GLOB,
            "requested_fov_ids": [int(value) for value in _dedupe_ints(list(fov_ids or []))],
            "discovered_fov_ids": [source.fov_id for source in sources],
            "read_fov_ids": read_fov_ids,
            "absent_fov_ids": absent_fov_ids,
            "present_telemetry_count": len(read_fov_ids),
            "absent_telemetry_count": len(absent_fov_ids),
        },
        "run": {
            "fov_count": len(fov_records),
            "present_fov_count": len(read_fov_ids),
            "absent_fov_count": len(absent_fov_ids),
            "total_measured_stage_elapsed_wall_ms": _round_float(total_measured_stage_ms),
            "total_run_elapsed_ms": _round_float(sum(run_total_values)) if run_total_values else None,
        },
        "stage_order": list(get_ordered_stage_ids()),
        "stage_aggregates": stage_aggregates,
        "stage_rankings": _rank_stage_aggregates(stage_aggregates),
        "slow_fov_rankings": slow_fov_rankings,
        "fovs": fov_records,
        "artifact_summary": _build_artifact_summary(read_payloads),
        "matlab_summary": _build_matlab_summary(read_payloads),
    }


def render_performance_summary_markdown(payload: Mapping[str, Any]) -> str:
    """Render a concise human-readable run-level performance summary."""

    inputs = _mapping_or_empty(payload.get("inputs"))
    run = _mapping_or_empty(payload.get("run"))
    lines = [
        "# PyStar Performance Telemetry Summary",
        "",
        "## Inputs",
        f"- **Base directory**: {payload.get('base_dir')}",
        f"- **Telemetry present**: {inputs.get('present_telemetry_count', 0)} FOV(s)",
        f"- **Telemetry absent**: {inputs.get('absent_telemetry_count', 0)} FOV(s)",
    ]
    absent_fov_ids = inputs.get("absent_fov_ids")
    if isinstance(absent_fov_ids, list) and absent_fov_ids:
        lines.append(f"- **Absent FOV IDs**: {', '.join(str(value) for value in absent_fov_ids)}")
    lines.extend(
        [
            "",
            "## Run Totals",
            f"- **Total measured stage wall time**: {run.get('total_measured_stage_elapsed_wall_ms')} ms",
            f"- **Total run elapsed time**: {run.get('total_run_elapsed_ms')} ms",
            "",
            "## Stage Ranking by Total Wall Time",
            "| Rank | Stage | Count | Total ms | Mean ms | Median ms | Min ms | Max ms | % measured stage time |",
            "|------|-------|-------|----------|---------|-----------|--------|--------|-----------------------|",
        ]
    )

    stage_rankings = payload.get("stage_rankings")
    if isinstance(stage_rankings, list):
        for rank, item in enumerate(stage_rankings, start=1):
            if not isinstance(item, Mapping):
                continue
            lines.append(
                "| {rank} | {stage} | {count} | {total} | {mean} | {median} | {min_v} | {max_v} | {percent} |".format(
                    rank=rank,
                    stage=item.get("stage_id"),
                    count=item.get("count"),
                    total=item.get("total_elapsed_wall_ms"),
                    mean=item.get("mean_elapsed_wall_ms"),
                    median=item.get("median_elapsed_wall_ms"),
                    min_v=item.get("min_elapsed_wall_ms"),
                    max_v=item.get("max_elapsed_wall_ms"),
                    percent=item.get("percent_of_total_measured_stage_time"),
                )
            )

    lines.extend(
        [
            "",
            "## Slowest FOVs",
            "| Rank | FOV | Total stage ms | Run total ms | Slowest stage | Slowest stage ms |",
            "|------|-----|----------------|--------------|---------------|------------------|",
        ]
    )
    slow_fovs = payload.get("slow_fov_rankings")
    if isinstance(slow_fovs, list):
        for rank, item in enumerate(slow_fovs[:10], start=1):
            if not isinstance(item, Mapping):
                continue
            slowest_stage = _mapping_or_empty(item.get("slowest_stage"))
            lines.append(
                "| {rank} | {fov} | {total} | {run_total} | {stage} | {stage_ms} |".format(
                    rank=rank,
                    fov=item.get("fov_id"),
                    total=item.get("total_stage_elapsed_wall_ms"),
                    run_total=item.get("run_total_elapsed_ms"),
                    stage=slowest_stage.get("stage_id"),
                    stage_ms=slowest_stage.get("elapsed_wall_ms"),
                )
            )

    artifact_summary = _mapping_or_empty(payload.get("artifact_summary"))
    spot_summary = _mapping_or_empty(artifact_summary.get("spot_table"))
    intensity_summary = _mapping_or_empty(artifact_summary.get("intensity_matrix"))
    decoded_summary = _mapping_or_empty(artifact_summary.get("decoded_outputs"))
    decoded_active = _mapping_or_empty(decoded_summary.get("active"))
    lines.extend(
        [
            "",
            "## Artifact Summary",
            "| Artifact | Present FOVs | Absent FOVs | Total size bytes | Total rows | Shape observations |",
            "|----------|--------------|-------------|------------------|------------|--------------------|",
            _artifact_markdown_row("spot_table", spot_summary),
            _artifact_markdown_row("intensity_matrix", intensity_summary),
            _artifact_markdown_row("decoded_active", decoded_active),
        ]
    )

    matlab_summary = _mapping_or_empty(payload.get("matlab_summary"))
    lines.extend(
        [
            "",
            "## MATLAB Boundary / Session Summary",
            "| Stage | Boundary summaries | Boundary calls | Session summaries | Aggregate session count | Metadata source statuses |",
            "|-------|--------------------|----------------|-------------------|-------------------------|--------------------------|",
        ]
    )
    for stage_id in get_ordered_stage_ids():
        item = _mapping_or_empty(matlab_summary.get(stage_id))
        lines.append(
            "| {stage} | {boundary_count} | {boundary_calls} | {session_count} | {aggregate_sessions} | {statuses} |".format(
                stage=stage_id,
                boundary_count=item.get("boundary_summary_count", 0),
                boundary_calls=item.get("aggregate_boundary_call_count", 0),
                session_count=item.get("session_summary_count", 0),
                aggregate_sessions=item.get("aggregate_session_count", 0),
                statuses=_format_status_counts(_mapping_or_empty(item.get("metadata_source_read_status_counts"))),
            )
        )

    return "\n".join(lines) + "\n"


def _format_status_counts(status_counts: Mapping[str, Any]) -> str:
    if not status_counts:
        return "none"
    return ", ".join(f"{key}={status_counts[key]}" for key in sorted(status_counts))


def _artifact_markdown_row(name: str, summary: Mapping[str, Any]) -> str:
    size_summary = _mapping_or_empty(summary.get("size_bytes"))
    row_summary = _mapping_or_empty(summary.get("row_count"))
    return "| {name} | {present} | {absent} | {size} | {rows} | {shapes} |".format(
        name=name,
        present=summary.get("present_count", 0),
        absent=summary.get("absent_count", 0),
        size=size_summary.get("total", 0.0),
        rows=row_summary.get("total", 0.0),
        shapes=summary.get("shape_observations", []),
    )


def write_performance_summary(
    *,
    base_dir: Path,
    fov_ids: Sequence[int] | None = None,
    output_json_path: Path | None = None,
    output_markdown_path: Path | None = None,
) -> tuple[Path, Path]:
    """Write run-level JSON and Markdown summaries and return their paths."""

    base_dir = Path(base_dir)
    json_path = output_json_path or (base_dir / "performance_summary.json")
    markdown_path = output_markdown_path or (base_dir / "performance_summary.md")
    payload = build_performance_summary_payload(base_dir=base_dir, fov_ids=fov_ids)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    write_backend_metadata(json_path, cast(dict[str, object], payload))
    temp_markdown_path = markdown_path.with_name(f"{markdown_path.name}.tmp")
    if temp_markdown_path.exists():
        temp_markdown_path.unlink()
    _ = temp_markdown_path.write_text(render_performance_summary_markdown(payload), encoding="utf-8")
    _ = temp_markdown_path.replace(markdown_path)
    return json_path, markdown_path
