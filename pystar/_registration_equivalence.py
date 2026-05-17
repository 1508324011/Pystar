"""Strict equivalence helpers for MATLAB local demons registration outputs.

This private module is a Stage18A harness for comparing a current serial
MATLAB local-demons tiled run with a future candidate executor.  It does not
dispatch providers, run MATLAB, change tile layouts, rewrite transform
manifests, or optimize anything.  It only compares already-produced requests,
tile flows, stitched flows, transform artifacts, and registration diagnostics.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from ._io_paths import get_flow_3d_sidecar_filename, get_transform_manifest_path
from ._registration_performance import (
    REGISTRATION_PERFORMANCE_SCHEMA_NAME,
    REGISTRATION_PERFORMANCE_SCHEMA_VERSION,
    load_registration_performance_diagnostics,
)
from .runtime_artifacts import Flow3DSidecarDescriptor, ReleaseContract, SCOPE_MODES, TransformEntry
from .serialization import json_safe, write_backend_metadata
from .tiling import TileSpec, stitch_tiles


TILE_SPEC_EQUIVALENCE_FIELDS = (
    "tile_index",
    "grid_position_yx",
    "grid_shape_yx",
    "region_origin_zyx",
    "region_shape_zyx",
    "write_origin_zyx",
    "write_shape_zyx",
    "write_offset_zyx",
    "full_volume_shape_zyx",
)

STAGE17_DIAGNOSTICS_SUMMARY_FIELDS = (
    "matlab_internal_timing_status",
    "matlab_internal_call_count",
    "matlab_internal_total_duration_ms",
    "matlab_internal_step_totals_ms",
    "matlab_internal_unaccounted_total_ms",
    "matlab_boundary_minus_internal_total_ms",
    "matlab_internal_dominant_step_counts",
    "matlab_internal_dominant_step_totals_ms",
)

_VOLATILE_REQUEST_KEYS = {
    "flow_output_path",
    "reference_volume_path",
    "moving_volume_path",
    "tmpdir",
    "tmpdir_path",
    "temporary_directory",
    "process_id",
    "pid",
    "session_id",
    "session_name",
    "started_at",
    "finished_at",
    "generated_at_utc",
    "timestamp",
    "execution_timestamp",
    "elapsed_wall_ms",
    "duration_ms",
    "total_duration_ms",
}

_VOLATILE_PAYLOAD_KEYS = {
    "backend_metadata",
    "runtime_context",
    "stage_outcomes",
    "generated_at_utc",
    "started_at",
    "finished_at",
    "start_time",
    "end_time",
    "execution_timestamp",
    "duration_seconds",
    "session_id",
    "session_name",
}

_REQUIRED_REQUEST_FIELDS = (
    "fov_id",
    "round_id",
    "reference_round",
    "provider",
    "method",
    "coverage_mode",
    "global_shift_already_applied",
)

_EXPECTED_REQUEST_VALUES = {
    "provider": "matlab",
    "method": "demons_3d",
    "global_shift_already_applied": True,
}

_VALID_COVERAGE_MODES = SCOPE_MODES

_REQUEST_RUNTIME_ENTRYPOINT_KEYS = (
    "entrypoint",
    "runtime_entrypoint",
    "matlab_entrypoint",
    "entrypoint_name",
)

_REQUEST_RUNTIME_MANIFEST_KEYS = (
    "manifest_sha256",
    "runtime_manifest_sha256",
    "runtime_manifest",
    "manifest_hash",
    "runtime_manifest_hash",
    "runtime_manifest_identity",
    "runtime_files",
    "runtime_file_records",
    "manifest",
)

_FLAT_COMPUTE_TILE_KEY_MAP = {
    "tile_index": "compute_tile_index",
    "grid_position_yx": "compute_tile_grid_position_yx",
    "grid_shape_yx": "compute_tile_grid_shape_yx",
    "region_origin_zyx": "compute_tile_origin_zyx",
    "region_shape_zyx": "compute_tile_shape_zyx",
    "write_origin_zyx": "compute_tile_write_origin_zyx",
    "write_shape_zyx": "compute_tile_write_shape_zyx",
    "write_offset_zyx": "compute_tile_write_offset_zyx",
    "full_volume_shape_zyx": "full_volume_shape_zyx",
}

_REFERENCE_VOLUME_SHAPE_KEYS = (
    "reference_volume_shape_zyx",
    "ref_volume_shape_zyx",
    "reference_shape_zyx",
    "fixed_volume_shape_zyx",
)

_MOVING_VOLUME_SHAPE_KEYS = (
    "moving_volume_shape_zyx",
    "mov_volume_shape_zyx",
    "moving_shape_zyx",
)


def _jsonable(value: Any) -> Any:
    return json_safe(value)


def _sha256_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def hash_array_bytes(array: Any) -> str:
    """Return a raw byte hash for an array without changing its values."""

    arr = np.ascontiguousarray(np.asarray(array))
    return _sha256_bytes(arr.tobytes(order="C"))


def _array_finite_counts(arr: NDArray[Any]) -> tuple[int, int, int]:
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"flow array dtype must be numeric, got {arr.dtype}")
    finite = np.isfinite(arr)
    nan = np.isnan(arr)
    inf = np.isinf(arr)
    return int(finite.sum()), int(nan.sum()), int(inf.sum())


@dataclass(frozen=True)
class FlowArrayFingerprint:
    shape: tuple[int, ...]
    dtype: str
    finite_count: int
    nan_count: int
    inf_count: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": [int(value) for value in self.shape],
            "dtype": self.dtype,
            "finite_count": int(self.finite_count),
            "nan_count": int(self.nan_count),
            "inf_count": int(self.inf_count),
            "sha256": self.sha256,
        }


def fingerprint_flow_array(array: Any) -> FlowArrayFingerprint:
    """Fingerprint one local-flow array while preserving raw dtype/bytes."""

    arr = np.ascontiguousarray(np.asarray(array))
    finite_count, nan_count, inf_count = _array_finite_counts(arr)
    return FlowArrayFingerprint(
        shape=tuple(int(value) for value in arr.shape),
        dtype=str(arr.dtype),
        finite_count=finite_count,
        nan_count=nan_count,
        inf_count=inf_count,
        sha256=hash_array_bytes(arr),
    )


@dataclass(frozen=True)
class FlowArrayDiff:
    passed: bool
    shape_equal: bool
    dtype_equal: bool
    finite_mask_equal: bool
    hash_equal: bool
    max_abs_diff: float | None
    baseline: FlowArrayFingerprint | None
    candidate: FlowArrayFingerprint | None
    differences: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "shape_equal": self.shape_equal,
            "dtype_equal": self.dtype_equal,
            "finite_mask_equal": self.finite_mask_equal,
            "hash_equal": self.hash_equal,
            "max_abs_diff": self.max_abs_diff,
            "baseline": None if self.baseline is None else self.baseline.to_dict(),
            "candidate": None if self.candidate is None else self.candidate.to_dict(),
            "differences": list(self.differences),
        }


def compare_flow_arrays(
    baseline: Any,
    candidate: Any,
    *,
    max_abs_diff_allowed: float = 0.0,
    label: str = "flow array",
) -> FlowArrayDiff:
    """Compare two flow arrays with strict zero-difference defaults."""

    differences: list[str] = []
    try:
        baseline_arr = np.asarray(baseline)
        candidate_arr = np.asarray(candidate)
        baseline_fp = fingerprint_flow_array(baseline_arr)
        candidate_fp = fingerprint_flow_array(candidate_arr)
    except Exception as exc:
        return FlowArrayDiff(
            passed=False,
            shape_equal=False,
            dtype_equal=False,
            finite_mask_equal=False,
            hash_equal=False,
            max_abs_diff=None,
            baseline=None,
            candidate=None,
            differences=(f"{label} fingerprint failed: {exc}",),
        )

    shape_equal = baseline_fp.shape == candidate_fp.shape
    dtype_equal = baseline_fp.dtype == candidate_fp.dtype
    hash_equal = baseline_fp.sha256 == candidate_fp.sha256
    finite_mask_equal = False
    max_abs_diff: float | None = None

    if not shape_equal:
        differences.append(f"{label} shape mismatch: {baseline_fp.shape} != {candidate_fp.shape}")
    if not dtype_equal:
        differences.append(f"{label} dtype mismatch: {baseline_fp.dtype!r} != {candidate_fp.dtype!r}")

    if shape_equal:
        baseline_finite = np.isfinite(baseline_arr)
        candidate_finite = np.isfinite(candidate_arr)
        finite_mask_equal = bool(np.array_equal(baseline_finite, candidate_finite))
        if not finite_mask_equal:
            differences.append(f"{label} finite mask mismatch")
        if baseline_arr.size == 0:
            max_abs_diff = 0.0
        else:
            diff = np.abs(baseline_arr.astype(np.float64, copy=False) - candidate_arr.astype(np.float64, copy=False))
            max_abs_diff = round(float(np.nanmax(diff)), 12)
        if max_abs_diff is not None and (not math.isfinite(max_abs_diff) or max_abs_diff > max_abs_diff_allowed):
            differences.append(
                f"{label} max_abs_diff {max_abs_diff} exceeds allowed {max_abs_diff_allowed}"
            )

    if not hash_equal:
        differences.append(f"{label} raw byte hash mismatch: {baseline_fp.sha256} != {candidate_fp.sha256}")

    passed = (
        shape_equal
        and dtype_equal
        and finite_mask_equal
        and hash_equal
        and max_abs_diff is not None
        and math.isfinite(max_abs_diff)
        and max_abs_diff <= max_abs_diff_allowed
    )
    return FlowArrayDiff(
        passed=passed,
        shape_equal=shape_equal,
        dtype_equal=dtype_equal,
        finite_mask_equal=finite_mask_equal,
        hash_equal=hash_equal,
        max_abs_diff=max_abs_diff,
        baseline=baseline_fp,
        candidate=candidate_fp,
        differences=tuple(differences),
    )


def _tuple_ints(value: Any) -> tuple[int, ...]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"expected list/tuple of integers, got {value!r}")
    return tuple(int(item) for item in value)


def _tile_mapping(tile: TileSpec | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(tile, TileSpec):
        return tile.as_dict()
    if isinstance(tile, Mapping):
        return dict(tile)
    raise ValueError(f"tile must be TileSpec or mapping, got {type(tile)}")


def _tile_spec_from_any(tile: TileSpec | Mapping[str, Any]) -> TileSpec:
    if isinstance(tile, TileSpec):
        return tile
    payload = _tile_mapping(tile)
    return TileSpec(
        tile_index=int(payload["tile_index"]),
        grid_position_yx=cast(tuple[int, int], _tuple_ints(payload["grid_position_yx"])),
        grid_shape_yx=cast(tuple[int, int], _tuple_ints(payload["grid_shape_yx"])),
        region_origin_zyx=cast(tuple[int, int, int], _tuple_ints(payload["region_origin_zyx"])),
        region_shape_zyx=cast(tuple[int, int, int], _tuple_ints(payload["region_shape_zyx"])),
        write_origin_zyx=cast(tuple[int, int, int], _tuple_ints(payload["write_origin_zyx"])),
        write_shape_zyx=cast(tuple[int, int, int], _tuple_ints(payload["write_shape_zyx"])),
        write_offset_zyx=cast(tuple[int, int, int], _tuple_ints(payload["write_offset_zyx"])),
        full_volume_shape_zyx=cast(tuple[int, int, int], _tuple_ints(payload["full_volume_shape_zyx"])),
    )


def _normalized_tile_identity(tile: TileSpec | Mapping[str, Any]) -> dict[str, Any]:
    payload = _tile_mapping(tile)
    normalized: dict[str, Any] = {}
    for field_name in TILE_SPEC_EQUIVALENCE_FIELDS:
        if field_name not in payload:
            raise ValueError(f"tile is missing required layout field {field_name!r}")
        value = payload[field_name]
        if field_name == "tile_index":
            normalized[field_name] = int(value)
        else:
            normalized[field_name] = [int(item) for item in _tuple_ints(value)]
    return normalized


@dataclass(frozen=True)
class TileSpecDiff:
    passed: bool
    tile_index: int | None
    differences: tuple[str, ...]
    baseline: dict[str, Any] | None
    candidate: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "tile_index": self.tile_index,
            "differences": list(self.differences),
            "baseline": self.baseline,
            "candidate": self.candidate,
        }


def compare_tile_specs(baseline: TileSpec | Mapping[str, Any], candidate: TileSpec | Mapping[str, Any]) -> TileSpecDiff:
    """Compare tile geometry exactly across the Stage18A layout contract."""

    try:
        baseline_payload = _normalized_tile_identity(baseline)
        candidate_payload = _normalized_tile_identity(candidate)
    except Exception as exc:
        return TileSpecDiff(
            passed=False,
            tile_index=None,
            differences=(f"tile layout normalization failed: {exc}",),
            baseline=None,
            candidate=None,
        )

    differences = [
        f"tile field {field_name!r} mismatch: {baseline_payload[field_name]!r} != {candidate_payload[field_name]!r}"
        for field_name in TILE_SPEC_EQUIVALENCE_FIELDS
        if baseline_payload[field_name] != candidate_payload[field_name]
    ]
    return TileSpecDiff(
        passed=not differences,
        tile_index=int(baseline_payload["tile_index"]),
        differences=tuple(differences),
        baseline=baseline_payload,
        candidate=candidate_payload,
    )


def _is_volatile_request_key(key: str) -> bool:
    key_lower = key.lower()
    if key_lower in _VOLATILE_REQUEST_KEYS:
        return True
    return key_lower.endswith("_ms") or "timing" in key_lower or "duration" in key_lower


def normalize_local_demons_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize request semantics while dropping temp paths/timing/session facts."""

    def normalize(value: Any) -> Any:
        if isinstance(value, Mapping):
            result: dict[str, Any] = {}
            for key, item in value.items():
                key_str = str(key)
                if _is_volatile_request_key(key_str):
                    continue
                result[key_str] = normalize(item)
            return result
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        return value

    normalized = cast(dict[str, Any], normalize(request))
    if "coverage_mode" not in normalized and "scope_mode" in normalized:
        normalized["coverage_mode"] = normalized["scope_mode"]
    if "method" not in normalized and "local_method" in normalized:
        normalized["method"] = normalized["local_method"]
    return normalized


def _nested_request_get(payload: Mapping[str, Any], key_path: Sequence[str]) -> Any:
    current: Any = payload
    for key in key_path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _first_request_value(payload: Mapping[str, Any], keys: Sequence[str | Sequence[str]]) -> Any:
    for key in keys:
        if isinstance(key, str):
            if key in payload:
                return payload[key]
            continue
        value = _nested_request_get(payload, key)
        if value is not None:
            return value
    return None


def _normalize_optional_int_tuple_for_request(value: Any, *, field_name: str) -> tuple[int, ...] | None:
    if value is None:
        return None
    try:
        return _tuple_ints(value)
    except Exception as exc:
        raise ValueError(f"{field_name} must be an integer sequence: {exc}") from exc


def _extract_compute_tile_metadata(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    compute_tile = payload.get("compute_tile")
    if isinstance(compute_tile, Mapping):
        return compute_tile

    flat_tile = {
        tile_field: payload[request_field]
        for tile_field, request_field in _FLAT_COMPUTE_TILE_KEY_MAP.items()
        if request_field in payload
    }
    if len(flat_tile) == len(_FLAT_COMPUTE_TILE_KEY_MAP):
        return flat_tile
    return None


def _collect_runtime_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    runtime = payload.get("runtime")
    runtime_mapping = runtime if isinstance(runtime, Mapping) else {}
    runtime_context = payload.get("runtime_context")
    runtime_context_mapping = runtime_context if isinstance(runtime_context, Mapping) else {}

    search_payloads = (payload, runtime_mapping, runtime_context_mapping)

    entrypoint = None
    for item in search_payloads:
        entrypoint = _first_request_value(item, _REQUEST_RUNTIME_ENTRYPOINT_KEYS)
        if entrypoint is not None:
            break

    manifest_identity = None
    for item in search_payloads:
        manifest_identity = _first_request_value(item, _REQUEST_RUNTIME_MANIFEST_KEYS)
        if manifest_identity is not None:
            break

    return {
        "entrypoint": _jsonable(entrypoint),
        "manifest_identity": _jsonable(manifest_identity),
    }


def _validate_one_local_demons_request(
    payload: Mapping[str, Any],
    *,
    label: str,
) -> tuple[dict[str, Any], list[str]]:
    differences: list[str] = []

    for field_name in _REQUIRED_REQUEST_FIELDS:
        if field_name not in payload:
            differences.append(f"{label} request missing required field {field_name!r}")

    for field_name, expected_value in _EXPECTED_REQUEST_VALUES.items():
        observed = payload.get(field_name)
        if observed != expected_value:
            differences.append(
                f"{label} request field {field_name!r} must be {expected_value!r}, got {observed!r}"
            )

    coverage_mode = payload.get("coverage_mode")
    if coverage_mode not in _VALID_COVERAGE_MODES:
        differences.append(
            f"{label} request coverage_mode must be one of {sorted(_VALID_COVERAGE_MODES)!r}, "
            f"got {coverage_mode!r}"
        )

    compute_tile = _extract_compute_tile_metadata(payload)
    normalized_tile: dict[str, Any] | None = None
    if isinstance(compute_tile, Mapping):
        try:
            normalized_tile = _normalized_tile_identity(compute_tile)
        except Exception as exc:
            differences.append(f"{label} request compute_tile metadata is invalid: {exc}")
    else:
        differences.append(f"{label} request missing compute_tile metadata mapping")

    runtime_identity = _collect_runtime_identity(payload)
    if runtime_identity["entrypoint"] is None:
        differences.append(f"{label} request missing runtime entrypoint identity")
    if runtime_identity["manifest_identity"] is None:
        differences.append(f"{label} request missing runtime manifest identity or file hashes")

    reference_shape = None
    moving_shape = None
    try:
        reference_shape = _normalize_optional_int_tuple_for_request(
            _first_request_value(payload, _REFERENCE_VOLUME_SHAPE_KEYS),
            field_name="reference volume shape",
        )
    except ValueError as exc:
        differences.append(f"{label} request {exc}")
    try:
        moving_shape = _normalize_optional_int_tuple_for_request(
            _first_request_value(payload, _MOVING_VOLUME_SHAPE_KEYS),
            field_name="moving volume shape",
        )
    except ValueError as exc:
        differences.append(f"{label} request {exc}")
    if reference_shape is None:
        differences.append(f"{label} request missing ref/moving volume shape: reference shape absent")
    if moving_shape is None:
        differences.append(f"{label} request missing ref/moving volume shape: moving shape absent")

    stable = {
        "fov_id": payload.get("fov_id"),
        "round_id": payload.get("round_id"),
        "reference_round": payload.get("reference_round"),
        "provider": payload.get("provider"),
        "method": payload.get("method"),
        "coverage_mode": coverage_mode,
        "global_shift_already_applied": payload.get("global_shift_already_applied"),
        "compute_tile": normalized_tile,
        "runtime": runtime_identity,
        "reference_volume_shape_zyx": None if reference_shape is None else [int(value) for value in reference_shape],
        "moving_volume_shape_zyx": None if moving_shape is None else [int(value) for value in moving_shape],
    }
    return stable, differences


@dataclass(frozen=True)
class RequestSemanticsDiff:
    passed: bool
    baseline_hash: str | None
    candidate_hash: str | None
    differences: tuple[str, ...]
    baseline: dict[str, Any] | None = None
    candidate: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "baseline_hash": self.baseline_hash,
            "candidate_hash": self.candidate_hash,
            "differences": list(self.differences),
            "baseline": self.baseline,
            "candidate": self.candidate,
        }


def _fingerprint_json_payload(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(_jsonable(dict(payload)), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(encoded)


def compare_local_demons_requests(
    baseline: Mapping[str, Any] | None,
    candidate: Mapping[str, Any] | None,
) -> RequestSemanticsDiff:
    """Compare MATLAB local demons request semantics, ignoring volatile facts."""

    if baseline is None and candidate is None:
        return RequestSemanticsDiff(passed=True, baseline_hash=None, candidate_hash=None, differences=())
    if baseline is None or candidate is None:
        return RequestSemanticsDiff(
            passed=False,
            baseline_hash=None,
            candidate_hash=None,
            differences=("one tile request is missing while the other is present",),
        )
    baseline_normalized = normalize_local_demons_request(baseline)
    candidate_normalized = normalize_local_demons_request(candidate)
    differences: list[str] = []

    baseline_stable, baseline_required_differences = _validate_one_local_demons_request(
        baseline_normalized,
        label="baseline",
    )
    candidate_stable, candidate_required_differences = _validate_one_local_demons_request(
        candidate_normalized,
        label="candidate",
    )
    differences.extend(baseline_required_differences)
    differences.extend(candidate_required_differences)

    baseline_hash = _fingerprint_json_payload(baseline_stable)
    candidate_hash = _fingerprint_json_payload(candidate_stable)
    if baseline_hash != candidate_hash:
        differences.append("stable local demons request semantics fingerprint mismatch")
    for key in baseline_stable:
        baseline_value = baseline_stable.get(key)
        candidate_value = candidate_stable.get(key)
        if baseline_value != candidate_value:
            differences.append(f"request field {key!r} mismatch: {baseline_value!r} != {candidate_value!r}")
    return RequestSemanticsDiff(
        passed=not differences,
        baseline_hash=baseline_hash,
        candidate_hash=candidate_hash,
        differences=tuple(differences),
        baseline=baseline_stable,
        candidate=candidate_stable,
    )


@dataclass(frozen=True)
class TileFlowResult:
    tile: TileSpec | Mapping[str, Any]
    flow_tile: Any
    request: Mapping[str, Any] | None = None


def _coerce_tile_flow_result(value: Any) -> TileFlowResult:
    if isinstance(value, TileFlowResult):
        return value
    if isinstance(value, Mapping):
        tile = value.get("tile")
        if tile is None:
            tile = {key: value[key] for key in TILE_SPEC_EQUIVALENCE_FIELDS if key in value}
        flow_tile = value.get("flow_tile", value.get("flow_3d", value.get("flow")))
        if flow_tile is None:
            raise ValueError("tile flow mapping must include flow_tile, flow_3d, or flow")
        request = value.get("request")
        return TileFlowResult(
            tile=cast(TileSpec | Mapping[str, Any], tile),
            flow_tile=flow_tile,
            request=cast(Mapping[str, Any] | None, request if isinstance(request, Mapping) else None),
        )
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        request = value[2] if len(value) >= 3 and isinstance(value[2], Mapping) else None
        return TileFlowResult(
            tile=cast(TileSpec | Mapping[str, Any], value[0]),
            flow_tile=value[1],
            request=cast(Mapping[str, Any] | None, request),
        )
    raise ValueError(f"unsupported tile flow result type: {type(value)}")


def _tile_index(result: TileFlowResult) -> int:
    return int(_normalized_tile_identity(result.tile)["tile_index"])


@dataclass(frozen=True)
class TileEquivalenceRecord:
    tile_index: int
    layout_equal: bool
    request_equal: bool | None
    flow_equal: bool
    max_abs_diff: float | None
    baseline_hash: str | None
    candidate_hash: str | None
    differences: tuple[str, ...]
    layout_diff: TileSpecDiff
    flow_diff: FlowArrayDiff
    request_diff: RequestSemanticsDiff | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tile_index": self.tile_index,
            "layout_equal": self.layout_equal,
            "request_equal": self.request_equal,
            "flow_equal": self.flow_equal,
            "max_abs_diff": self.max_abs_diff,
            "baseline_hash": self.baseline_hash,
            "candidate_hash": self.candidate_hash,
            "differences": list(self.differences),
            "layout_diff": self.layout_diff.to_dict(),
            "flow_diff": self.flow_diff.to_dict(),
            "request_diff": None if self.request_diff is None else self.request_diff.to_dict(),
        }


@dataclass(frozen=True)
class TiledFlowEquivalenceReport:
    passed: bool
    tile_count: int
    tile_records: tuple[TileEquivalenceRecord, ...]
    stitched_equal: bool
    stitched_max_abs_diff: float | None
    baseline_stitched_hash: str | None
    candidate_stitched_hash: str | None
    candidate_order_normalized: bool
    differences: tuple[str, ...]
    stitched_diff: FlowArrayDiff | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "tile_count": int(self.tile_count),
            "tile_records": [record.to_dict() for record in self.tile_records],
            "stitched_equal": self.stitched_equal,
            "stitched_max_abs_diff": self.stitched_max_abs_diff,
            "baseline_stitched_hash": self.baseline_stitched_hash,
            "candidate_stitched_hash": self.candidate_stitched_hash,
            "candidate_order_normalized": self.candidate_order_normalized,
            "differences": list(self.differences),
            "stitched_diff": None if self.stitched_diff is None else self.stitched_diff.to_dict(),
        }


def _sort_tile_results(values: Sequence[Any]) -> tuple[list[TileFlowResult], bool]:
    coerced = [_coerce_tile_flow_result(value) for value in values]
    original_order = [_tile_index(result) for result in coerced]
    sorted_results = sorted(coerced, key=_tile_index)
    sorted_order = [_tile_index(result) for result in sorted_results]
    return sorted_results, original_order != sorted_order


def _infer_full_shape_zyx(results: Sequence[TileFlowResult]) -> tuple[int, int, int]:
    if not results:
        raise ValueError("cannot infer full_shape_zyx from an empty tile result sequence")
    tile = _tile_spec_from_any(results[0].tile)
    return tile.full_volume_shape_zyx


def _stitch_result_flows(results: Sequence[TileFlowResult], *, full_shape_zyx: Sequence[int]) -> NDArray[Any]:
    tile_outputs = [(_tile_spec_from_any(result.tile), np.asarray(result.flow_tile)) for result in results]
    return cast(NDArray[Any], stitch_tiles(tile_outputs, full_shape_zyx=full_shape_zyx))


def compare_tiled_flow_outputs(
    baseline: Sequence[Any],
    candidate: Sequence[Any],
    *,
    full_shape_zyx: Sequence[int] | None = None,
    max_abs_diff_allowed: float = 0.0,
) -> TiledFlowEquivalenceReport:
    """Compare tiled local-demons outputs and their stitched ``flow_3d`` result."""

    differences: list[str] = []
    try:
        baseline_results, baseline_order_normalized = _sort_tile_results(baseline)
        candidate_results, candidate_order_normalized = _sort_tile_results(candidate)
    except Exception as exc:
        return TiledFlowEquivalenceReport(
            passed=False,
            tile_count=0,
            tile_records=(),
            stitched_equal=False,
            stitched_max_abs_diff=None,
            baseline_stitched_hash=None,
            candidate_stitched_hash=None,
            candidate_order_normalized=False,
            differences=(f"tile result normalization failed: {exc}",),
        )
    if baseline_order_normalized:
        differences.append("baseline tile results were not in tile_index order")
    if len(baseline_results) != len(candidate_results):
        differences.append(f"tile count mismatch: {len(baseline_results)} != {len(candidate_results)}")

    record_count = min(len(baseline_results), len(candidate_results))
    tile_records: list[TileEquivalenceRecord] = []
    for index in range(record_count):
        baseline_result = baseline_results[index]
        candidate_result = candidate_results[index]
        layout_diff = compare_tile_specs(baseline_result.tile, candidate_result.tile)
        flow_diff = compare_flow_arrays(
            baseline_result.flow_tile,
            candidate_result.flow_tile,
            max_abs_diff_allowed=max_abs_diff_allowed,
            label=f"tile {layout_diff.tile_index if layout_diff.tile_index is not None else index} flow_tile",
        )
        request_diff: RequestSemanticsDiff | None = None
        request_equal: bool | None = None
        if baseline_result.request is not None or candidate_result.request is not None:
            request_diff = compare_local_demons_requests(baseline_result.request, candidate_result.request)
            request_equal = request_diff.passed
        tile_differences = [
            *layout_diff.differences,
            *flow_diff.differences,
            *(request_diff.differences if request_diff is not None else ()),
        ]
        tile_records.append(
            TileEquivalenceRecord(
                tile_index=layout_diff.tile_index if layout_diff.tile_index is not None else index,
                layout_equal=layout_diff.passed,
                request_equal=request_equal,
                flow_equal=flow_diff.passed,
                max_abs_diff=flow_diff.max_abs_diff,
                baseline_hash=None if flow_diff.baseline is None else flow_diff.baseline.sha256,
                candidate_hash=None if flow_diff.candidate is None else flow_diff.candidate.sha256,
                differences=tuple(tile_differences),
                layout_diff=layout_diff,
                flow_diff=flow_diff,
                request_diff=request_diff,
            )
        )
        differences.extend(tile_differences)

    if full_shape_zyx is None:
        try:
            full_shape_zyx = _infer_full_shape_zyx(baseline_results)
        except Exception as exc:
            differences.append(f"stitched flow full_shape_zyx inference failed: {exc}")
            full_shape_zyx = None

    stitched_diff: FlowArrayDiff | None = None
    baseline_stitched_hash: str | None = None
    candidate_stitched_hash: str | None = None
    stitched_equal = False
    stitched_max_abs_diff: float | None = None
    if full_shape_zyx is not None and len(baseline_results) == len(candidate_results):
        try:
            baseline_stitched = _stitch_result_flows(baseline_results, full_shape_zyx=full_shape_zyx)
            candidate_stitched = _stitch_result_flows(candidate_results, full_shape_zyx=full_shape_zyx)
            stitched_diff = compare_flow_arrays(
                baseline_stitched,
                candidate_stitched,
                max_abs_diff_allowed=max_abs_diff_allowed,
                label="stitched flow_3d",
            )
            stitched_equal = stitched_diff.passed
            stitched_max_abs_diff = stitched_diff.max_abs_diff
            baseline_stitched_hash = None if stitched_diff.baseline is None else stitched_diff.baseline.sha256
            candidate_stitched_hash = None if stitched_diff.candidate is None else stitched_diff.candidate.sha256
            differences.extend(stitched_diff.differences)
        except Exception as exc:
            differences.append(f"stitched flow_3d comparison failed: {exc}")

    passed = (
        not differences
        and len(baseline_results) == len(candidate_results)
        and all(record.layout_equal and record.flow_equal and record.request_equal is not False for record in tile_records)
        and stitched_equal
    )
    return TiledFlowEquivalenceReport(
        passed=passed,
        tile_count=len(tile_records),
        tile_records=tuple(tile_records),
        stitched_equal=stitched_equal,
        stitched_max_abs_diff=stitched_max_abs_diff,
        baseline_stitched_hash=baseline_stitched_hash,
        candidate_stitched_hash=candidate_stitched_hash,
        candidate_order_normalized=candidate_order_normalized,
        differences=tuple(differences),
        stitched_diff=stitched_diff,
    )


@dataclass(frozen=True)
class DiagnosticsEquivalenceReport:
    passed: bool
    schema_compatible: bool
    timing_fields_ignored: bool
    differences: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "schema_compatible": self.schema_compatible,
            "timing_fields_ignored": self.timing_fields_ignored,
            "differences": list(self.differences),
        }


def _require_stage17_diagnostics_summary(payload: Mapping[str, Any], *, label: str) -> list[str]:
    differences: list[str] = []
    if payload.get("schema_name") != REGISTRATION_PERFORMANCE_SCHEMA_NAME:
        differences.append(
            f"{label} diagnostics schema_name mismatch: {payload.get('schema_name')!r}"
        )
    if payload.get("schema_version") != REGISTRATION_PERFORMANCE_SCHEMA_VERSION:
        differences.append(
            f"{label} diagnostics schema_version mismatch: {payload.get('schema_version')!r}"
        )
    summary = payload.get("summary")
    if not isinstance(summary, Mapping):
        return [*differences, f"{label} diagnostics summary must be a mapping"]
    missing = [field for field in STAGE17_DIAGNOSTICS_SUMMARY_FIELDS if field not in summary]
    if missing:
        differences.append(f"{label} diagnostics summary missing Stage17 fields: {missing}")
    slowest_rounds = summary.get("slowest_rounds")
    slowest_tiles = summary.get("slowest_tiles")
    if not isinstance(slowest_rounds, list):
        differences.append(f"{label} diagnostics summary.slowest_rounds must be a list")
        slowest_rounds = []
    if not isinstance(slowest_tiles, list):
        differences.append(f"{label} diagnostics summary.slowest_tiles must be a list")
        slowest_tiles = []
    for index, row in enumerate(slowest_rounds):
        if isinstance(row, Mapping) and "matlab_internal_total_duration_ms" not in row:
            differences.append(
                f"{label} diagnostics slowest_rounds[{index}] missing matlab_internal_total_duration_ms"
            )
    for index, row in enumerate(slowest_tiles):
        if isinstance(row, Mapping) and summary.get("matlab_internal_call_count", 0) and "matlab_internal_total_duration_ms" not in row:
            # Non-tiled local runs have no slowest tile rows.  When a slowest tile
            # row exists in an internal-timing run, keep the context explicit.
            differences.append(
                f"{label} diagnostics slowest_tiles[{index}] missing matlab_internal_total_duration_ms"
            )
    return differences


def compare_registration_diagnostics_schema(
    baseline_path: Path,
    candidate_path: Path,
    *,
    expected_fov_id: int | None = None,
) -> DiagnosticsEquivalenceReport:
    """Compare Stage16/17 diagnostics schemas while ignoring timing values."""

    differences: list[str] = []
    try:
        baseline_payload = load_registration_performance_diagnostics(
            Path(baseline_path),
            expected_fov_id=expected_fov_id,
        )
    except Exception as exc:
        differences.append(f"baseline diagnostics load/validation failed: {exc}")
        baseline_payload = None
    try:
        candidate_payload = load_registration_performance_diagnostics(
            Path(candidate_path),
            expected_fov_id=expected_fov_id,
        )
    except Exception as exc:
        differences.append(f"candidate diagnostics load/validation failed: {exc}")
        candidate_payload = None

    if baseline_payload is not None:
        differences.extend(_require_stage17_diagnostics_summary(baseline_payload, label="baseline"))
    if candidate_payload is not None:
        differences.extend(_require_stage17_diagnostics_summary(candidate_payload, label="candidate"))

    return DiagnosticsEquivalenceReport(
        passed=not differences,
        schema_compatible=not differences,
        timing_fields_ignored=True,
        differences=tuple(differences),
    )


def _load_manifest_payload(base_dir: Path, fov_id: int) -> dict[Any, Any]:
    manifest_path = get_transform_manifest_path(base_dir, fov_id)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Transform manifest not found: {manifest_path}")
    payload = np.load(manifest_path, allow_pickle=True).item()
    if not isinstance(payload, dict):
        raise ValueError(f"Transform manifest is malformed: expected dict payload, got {type(payload)}")
    return cast(dict[Any, Any], payload)


def _round_entry(manifest_payload: Mapping[Any, Any], round_id: int) -> Mapping[str, Any]:
    for key in (round_id, str(round_id)):
        value = manifest_payload.get(key)
        if isinstance(value, Mapping):
            return cast(Mapping[str, Any], value)
    raise KeyError(f"Transform manifest has no round entry for round {round_id}")


def _strip_volatile_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            key_str = str(key)
            key_lower = key_str.lower()
            if key_lower in _VOLATILE_PAYLOAD_KEYS or key_lower.endswith("_ms") or "timing" in key_lower:
                continue
            normalized[key_str] = _strip_volatile_payload(item)
        return normalized
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_strip_volatile_payload(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _normalize_round_entry_for_equivalence(round_payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = {
        str(key): value
        for key, value in round_payload.items()
        if str(key) not in {"backend_metadata"}
    }
    return cast(dict[str, Any], _jsonable(_strip_volatile_payload(normalized)))


def _normalize_contract(payload: Any) -> Any:
    return _jsonable(_strip_volatile_payload(payload))


def _validate_round_entry_contract(
    round_payload: Mapping[str, Any],
    *,
    label: str,
    round_id: int,
) -> list[str]:
    differences: list[str] = []
    for field_name in ("_scope", "_semantics", "flow_3d"):
        if field_name not in round_payload:
            differences.append(f"{label} round entry missing required field {field_name!r}")
    try:
        TransformEntry.from_legacy(
            int(round_id),
            round_payload,
            field_name=f"{label} transform round {round_id}",
        )
    except Exception as exc:
        differences.append(f"{label} round entry contract invalid: {exc}")
    return differences


def _validate_release_contract_payload(payload: Any, *, label: str) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        contract = ReleaseContract.from_legacy(
            payload,
            field_name=label,
            strict=True,
        ).to_legacy()
    except Exception as exc:
        return None, [f"{label} contract invalid: {exc}"]
    return cast(dict[str, Any], _normalize_contract(contract)), []


@dataclass(frozen=True)
class ArtifactEquivalenceReport:
    passed: bool
    sidecar_contract_equal: bool
    manifest_semantics_equal: bool
    diagnostics_schema_compatible: bool
    timing_fields_ignored: bool
    differences: tuple[str, ...]
    sidecar_flow_diff: FlowArrayDiff | None = None
    diagnostics_report: DiagnosticsEquivalenceReport | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "sidecar_contract_equal": self.sidecar_contract_equal,
            "manifest_semantics_equal": self.manifest_semantics_equal,
            "diagnostics_schema_compatible": self.diagnostics_schema_compatible,
            "timing_fields_ignored": self.timing_fields_ignored,
            "differences": list(self.differences),
            "sidecar_flow_diff": None if self.sidecar_flow_diff is None else self.sidecar_flow_diff.to_dict(),
            "diagnostics_report": None if self.diagnostics_report is None else self.diagnostics_report.to_dict(),
        }


def compare_transform_artifacts(
    baseline_dir: Path,
    candidate_dir: Path,
    *,
    fov_id: int,
    round_id: int,
    baseline_diagnostics_path: Path | None = None,
    candidate_diagnostics_path: Path | None = None,
) -> ArtifactEquivalenceReport:
    """Compare sidecar/manifest semantics for one round without rewriting artifacts."""

    differences: list[str] = []
    sidecar_contract_equal = False
    manifest_semantics_equal = False
    sidecar_flow_diff: FlowArrayDiff | None = None

    try:
        baseline_manifest = _load_manifest_payload(Path(baseline_dir), int(fov_id))
        candidate_manifest = _load_manifest_payload(Path(candidate_dir), int(fov_id))
        baseline_round = _round_entry(baseline_manifest, int(round_id))
        candidate_round = _round_entry(candidate_manifest, int(round_id))
        baseline_descriptor = baseline_round.get("flow_3d")
        candidate_descriptor = candidate_round.get("flow_3d")
        expected_sidecar_name = get_flow_3d_sidecar_filename(int(fov_id), int(round_id))

        descriptor_differences: list[str] = []
        baseline_descriptor_contract: dict[str, Any] | None = None
        candidate_descriptor_contract: dict[str, Any] | None = None
        for label, descriptor in (("baseline", baseline_descriptor), ("candidate", candidate_descriptor)):
            if not isinstance(descriptor, Mapping):
                descriptor_differences.append(f"{label} round flow_3d is not a sidecar descriptor mapping")
                continue
            try:
                validated_descriptor = Flow3DSidecarDescriptor.from_legacy(
                    descriptor,
                    field_name=f"{label} round flow_3d",
                ).to_legacy()
            except Exception as exc:
                descriptor_differences.append(f"{label} sidecar descriptor contract invalid: {exc}")
                continue
            if label == "baseline":
                baseline_descriptor_contract = validated_descriptor
            else:
                candidate_descriptor_contract = validated_descriptor
            if descriptor.get("path") != expected_sidecar_name:
                descriptor_differences.append(
                    f"{label} sidecar path mismatch: {descriptor.get('path')!r} != {expected_sidecar_name!r}"
                )
        if baseline_descriptor_contract is not None and candidate_descriptor_contract is not None:
            for field_name in ("storage", "path", "shape", "dtype"):
                if baseline_descriptor_contract.get(field_name) != candidate_descriptor_contract.get(field_name):
                    descriptor_differences.append(
                        f"sidecar descriptor field {field_name!r} mismatch: "
                        f"{baseline_descriptor_contract.get(field_name)!r} != "
                        f"{candidate_descriptor_contract.get(field_name)!r}"
                    )
        baseline_sidecar = get_transform_manifest_path(Path(baseline_dir), int(fov_id)).parent / expected_sidecar_name
        candidate_sidecar = get_transform_manifest_path(Path(candidate_dir), int(fov_id)).parent / expected_sidecar_name
        if not baseline_sidecar.exists():
            descriptor_differences.append(f"baseline sidecar missing: {baseline_sidecar}")
        if not candidate_sidecar.exists():
            descriptor_differences.append(f"candidate sidecar missing: {candidate_sidecar}")
        if baseline_sidecar.exists() and candidate_sidecar.exists():
            sidecar_flow_diff = compare_flow_arrays(
                np.load(baseline_sidecar, allow_pickle=False),
                np.load(candidate_sidecar, allow_pickle=False),
                label="flow_3d sidecar",
            )
            descriptor_differences.extend(sidecar_flow_diff.differences)
        sidecar_contract_equal = not descriptor_differences
        differences.extend(descriptor_differences)

        baseline_round_semantics = _normalize_round_entry_for_equivalence(baseline_round)
        candidate_round_semantics = _normalize_round_entry_for_equivalence(candidate_round)
        baseline_contract = _normalize_contract(baseline_manifest.get("_contract"))
        candidate_contract = _normalize_contract(candidate_manifest.get("_contract"))
        baseline_provenance = baseline_manifest.get("_provenance")
        candidate_provenance = candidate_manifest.get("_provenance")
        baseline_provenance_contract = _normalize_contract(
            baseline_provenance.get("release_contract") if isinstance(baseline_provenance, Mapping) else None
        )
        candidate_provenance_contract = _normalize_contract(
            candidate_provenance.get("release_contract") if isinstance(candidate_provenance, Mapping) else None
        )
        manifest_differences: list[str] = []
        manifest_differences.extend(
            _validate_round_entry_contract(
                baseline_round,
                label="baseline",
                round_id=int(round_id),
            )
        )
        manifest_differences.extend(
            _validate_round_entry_contract(
                candidate_round,
                label="candidate",
                round_id=int(round_id),
            )
        )
        if baseline_round_semantics != candidate_round_semantics:
            manifest_differences.append("round entry semantic payload drifted")
        baseline_contract_validated, baseline_contract_differences = _validate_release_contract_payload(
            baseline_contract,
            label="baseline top-level _contract",
        )
        candidate_contract_validated, candidate_contract_differences = _validate_release_contract_payload(
            candidate_contract,
            label="candidate top-level _contract",
        )
        manifest_differences.extend(baseline_contract_differences)
        manifest_differences.extend(candidate_contract_differences)
        if baseline_contract_validated != candidate_contract_validated:
            manifest_differences.append("top-level _contract payload drifted")
        baseline_provenance_contract_validated, baseline_provenance_contract_differences = _validate_release_contract_payload(
            baseline_provenance_contract,
            label="baseline _provenance.release_contract",
        )
        candidate_provenance_contract_validated, candidate_provenance_contract_differences = _validate_release_contract_payload(
            candidate_provenance_contract,
            label="candidate _provenance.release_contract",
        )
        manifest_differences.extend(baseline_provenance_contract_differences)
        manifest_differences.extend(candidate_provenance_contract_differences)
        if baseline_provenance_contract_validated != candidate_provenance_contract_validated:
            manifest_differences.append("_provenance.release_contract payload drifted")
        if baseline_contract_validated != baseline_provenance_contract_validated:
            manifest_differences.append("baseline _contract and _provenance.release_contract drifted")
        if candidate_contract_validated != candidate_provenance_contract_validated:
            manifest_differences.append("candidate _contract and _provenance.release_contract drifted")
        if ("_provenance" in baseline_manifest) != ("_provenance" in candidate_manifest):
            manifest_differences.append("_provenance presence drifted")
        if ("_contract" in baseline_manifest) != ("_contract" in candidate_manifest):
            manifest_differences.append("_contract presence drifted")
        manifest_semantics_equal = not manifest_differences
        differences.extend(manifest_differences)
    except Exception as exc:
        differences.append(f"transform artifact comparison failed: {exc}")

    diagnostics_report: DiagnosticsEquivalenceReport | None = None
    diagnostics_schema_compatible = True
    if baseline_diagnostics_path is not None or candidate_diagnostics_path is not None:
        if baseline_diagnostics_path is None or candidate_diagnostics_path is None:
            diagnostics_schema_compatible = False
            differences.append("one diagnostics path is missing while the other is present")
        else:
            diagnostics_report = compare_registration_diagnostics_schema(
                baseline_diagnostics_path,
                candidate_diagnostics_path,
                expected_fov_id=int(fov_id),
            )
            diagnostics_schema_compatible = diagnostics_report.passed
            differences.extend(diagnostics_report.differences)

    passed = sidecar_contract_equal and manifest_semantics_equal and diagnostics_schema_compatible and not differences
    return ArtifactEquivalenceReport(
        passed=passed,
        sidecar_contract_equal=sidecar_contract_equal,
        manifest_semantics_equal=manifest_semantics_equal,
        diagnostics_schema_compatible=diagnostics_schema_compatible,
        timing_fields_ignored=True,
        differences=tuple(differences),
        sidecar_flow_diff=sidecar_flow_diff,
        diagnostics_report=diagnostics_report,
    )


@dataclass(frozen=True)
class RegistrationEquivalenceReport:
    passed: bool
    tiled_flow: TiledFlowEquivalenceReport | None = None
    artifacts: ArtifactEquivalenceReport | None = None
    diagnostics: DiagnosticsEquivalenceReport | None = None
    differences: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "tiled_flow": None if self.tiled_flow is None else self.tiled_flow.to_dict(),
            "artifacts": None if self.artifacts is None else self.artifacts.to_dict(),
            "diagnostics": None if self.diagnostics is None else self.diagnostics.to_dict(),
            "differences": list(self.differences),
        }


def build_registration_equivalence_report(
    *,
    tiled_flow: TiledFlowEquivalenceReport | None = None,
    artifacts: ArtifactEquivalenceReport | None = None,
    diagnostics: DiagnosticsEquivalenceReport | None = None,
) -> RegistrationEquivalenceReport:
    """Combine structured Stage18A comparison reports."""

    differences: list[str] = []
    reports = [report for report in (tiled_flow, artifacts, diagnostics) if report is not None]
    for report in reports:
        differences.extend(cast(Any, report).differences)
    passed = bool(reports) and all(bool(cast(Any, report).passed) for report in reports) and not differences
    return RegistrationEquivalenceReport(
        passed=passed,
        tiled_flow=tiled_flow,
        artifacts=artifacts,
        diagnostics=diagnostics,
        differences=tuple(differences),
    )


def write_equivalence_report(path: Path, report: RegistrationEquivalenceReport) -> Path:
    """Write a JSON-safe equivalence report through the shared metadata helper."""

    write_backend_metadata(Path(path), cast(dict[str, object], report.to_dict()))
    return Path(path)
