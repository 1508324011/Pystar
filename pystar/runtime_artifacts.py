"""Typed compatibility models for persisted PyStar runtime artifacts.

These dataclasses name the artifact contracts that already exist on disk while
keeping the public representation legacy-compatible.  The boundary is explicit:
legacy dictionaries come in, typed objects validate the operational facts, and
legacy dictionaries go back out for existing callers and notebooks.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np


FLOW_3D_SIDECAR_STORAGE = "round_level_sidecar_npy"
LEGACY_SCHEMA_VERSION = 0
RUNTIME_ARTIFACT_SCHEMA_VERSION = 1

FIELD_SEMANTICS_REPRESENTATIONS = {"residual", "total", "unknown"}
FIELD_SEMANTICS_COMPOSITIONS = {"sequential_global_then_local", "independent", "unknown"}
FIELD_SEMANTICS_STATUSES = {"settled", "provisional", "unknown"}

SCOPE_MODES = {"full_fov", "tile_local"}
SCOPE_STATUSES = {"valid", "degraded", "invalid"}
RELEASE_GATE_STATUSES = {"valid", "degraded", "invalid", "debug_only"}

_SCHEMA_KEYS = ("schema_version", "_schema_version")
_MISSING = object()


def _field_name(name: str | None, default: str) -> str:
    return default if name is None else name


def _coerce_int_tuple(
    value: Any,
    *,
    field_name: str,
    expected_length: int | None = None,
) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list/tuple of integers")
    if expected_length is not None and len(value) != expected_length:
        raise ValueError(f"{field_name} must contain {expected_length} integers")

    coerced: list[int] = []
    for item in value:
        if not isinstance(item, (int, np.integer)):
            raise ValueError(f"{field_name} entries must be integers, got {item!r}")
        coerced.append(int(item))
    return tuple(coerced)


def _validate_numeric_vector(value: Any, *, field_name: str, length: int) -> None:
    if isinstance(value, np.ndarray):
        if value.shape != (length,):
            raise ValueError(f"{field_name} must have shape ({length},), got {value.shape}")
        if not np.issubdtype(value.dtype, np.number):
            raise ValueError(f"{field_name} must contain numeric values")
        return

    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a numeric vector of length {length}")
    if len(value) != length:
        raise ValueError(f"{field_name} must contain {length} numeric values")
    for item in value:
        if not isinstance(item, (int, float, np.integer, np.floating)):
            raise ValueError(f"{field_name} entries must be numeric, got {item!r}")


def _copy_extra(payload: Mapping[str, Any], known_keys: set[str]) -> dict[Any, Any]:
    return {key: value for key, value in payload.items() if str(key) not in known_keys}


def _is_round_key(key: Any) -> bool:
    if isinstance(key, (bool, np.bool_)):
        return False
    if isinstance(key, (int, np.integer)):
        return True
    if isinstance(key, str):
        return key.isdecimal()
    return False


@dataclass(frozen=True)
class FieldSemantics:
    """Operational semantics for persisted transform fields."""

    representation: str = "unknown"
    composition: str = "unknown"
    status: str = "unknown"
    recorded_at: str | None = None
    extra: dict[Any, Any] = field(default_factory=dict)

    @classmethod
    def unknown(cls, *, recorded_at: str | None = None) -> "FieldSemantics":
        return cls(recorded_at=recorded_at)

    @classmethod
    def from_legacy(
        cls,
        payload: Any,
        *,
        field_name: str = "field_semantics",
        recorded_at: str | None = None,
    ) -> "FieldSemantics":
        if payload is None:
            return cls.unknown(recorded_at=recorded_at)
        if not isinstance(payload, Mapping):
            raise ValueError(f"{field_name} must be a mapping")

        representation = payload.get("representation", "unknown")
        composition = payload.get("composition", "unknown")
        status = payload.get("status", "unknown")

        if representation not in FIELD_SEMANTICS_REPRESENTATIONS:
            raise ValueError(
                f"{field_name}.representation must be one of {sorted(FIELD_SEMANTICS_REPRESENTATIONS)}, "
                f"got {representation!r}"
            )
        if composition not in FIELD_SEMANTICS_COMPOSITIONS:
            raise ValueError(
                f"{field_name}.composition must be one of {sorted(FIELD_SEMANTICS_COMPOSITIONS)}, "
                f"got {composition!r}"
            )
        if status not in FIELD_SEMANTICS_STATUSES:
            raise ValueError(
                f"{field_name}.status must be one of {sorted(FIELD_SEMANTICS_STATUSES)}, got {status!r}"
            )

        recorded_value = payload.get("recorded_at", recorded_at)
        if recorded_value is not None and (not isinstance(recorded_value, str) or not recorded_value.strip()):
            raise ValueError(f"{field_name}.recorded_at must be a non-empty string when present")

        return cls(
            representation=str(representation),
            composition=str(composition),
            status=str(status),
            recorded_at=recorded_value,
            extra=_copy_extra(payload, {"representation", "composition", "status", "recorded_at"}),
        )

    def to_legacy(self) -> dict[str, Any]:
        payload = dict(self.extra)
        payload.update(
            {
                "representation": self.representation,
                "composition": self.composition,
                "status": self.status,
            }
        )
        if self.recorded_at is not None:
            payload["recorded_at"] = self.recorded_at
        return payload


@dataclass(frozen=True)
class ScopeMetadata:
    """Explicit coverage facts for a persisted transform entry."""

    coverage_mode: str
    region_origin_zyx: tuple[int, int, int]
    region_shape_zyx: tuple[int, int, int]
    full_volume_shape_zyx: tuple[int, int, int]
    tile_grid_shape_yx: tuple[int, int] | None = None
    tile_index: int | None = None
    extra: dict[Any, Any] = field(default_factory=dict)

    @classmethod
    def from_legacy(cls, payload: Any, *, field_name: str = "scope") -> "ScopeMetadata":
        if not isinstance(payload, Mapping):
            raise ValueError(f"{field_name} must be a mapping")

        coverage_mode = payload.get("coverage_mode")
        if coverage_mode not in SCOPE_MODES:
            raise ValueError(f"{field_name}.coverage_mode must be one of {sorted(SCOPE_MODES)}, got {coverage_mode!r}")

        region_origin_zyx = _coerce_int_tuple(
            payload.get("region_origin_zyx"),
            field_name=f"{field_name}.region_origin_zyx",
            expected_length=3,
        )
        region_shape_zyx = _coerce_int_tuple(
            payload.get("region_shape_zyx"),
            field_name=f"{field_name}.region_shape_zyx",
            expected_length=3,
        )
        full_volume_shape_zyx = _coerce_int_tuple(
            payload.get("full_volume_shape_zyx"),
            field_name=f"{field_name}.full_volume_shape_zyx",
            expected_length=3,
        )

        if any(value < 0 for value in region_origin_zyx):
            raise ValueError(f"{field_name}.region_origin_zyx must contain non-negative integers")
        if any(value <= 0 for value in region_shape_zyx):
            raise ValueError(f"{field_name}.region_shape_zyx must contain positive integers")
        if any(value <= 0 for value in full_volume_shape_zyx):
            raise ValueError(f"{field_name}.full_volume_shape_zyx must contain positive integers")

        for origin, size, full_size, axis_name in zip(
            region_origin_zyx,
            region_shape_zyx,
            full_volume_shape_zyx,
            ("z", "y", "x"),
        ):
            if origin + size > full_size:
                raise ValueError(
                    f"{field_name} {axis_name}-axis region exceeds full volume bounds: "
                    f"origin={origin}, size={size}, full={full_size}"
                )

        known = {
            "coverage_mode",
            "region_origin_zyx",
            "region_shape_zyx",
            "full_volume_shape_zyx",
            "tile_grid_shape_yx",
            "tile_index",
        }
        extra = _copy_extra(payload, known)
        tile_grid_shape_yx = None
        tile_index = None

        if coverage_mode == "tile_local":
            tile_grid_shape_yx = _coerce_int_tuple(
                payload.get("tile_grid_shape_yx"),
                field_name=f"{field_name}.tile_grid_shape_yx",
                expected_length=2,
            )
            if any(value <= 0 for value in tile_grid_shape_yx):
                raise ValueError(f"{field_name}.tile_grid_shape_yx must contain positive integers")

            raw_tile_index = payload.get("tile_index")
            if not isinstance(raw_tile_index, (int, np.integer)) or int(raw_tile_index) <= 0:
                raise ValueError(f"{field_name}.tile_index must be a positive integer for tile_local coverage")
            tile_index = int(raw_tile_index)
            if tile_index > int(tile_grid_shape_yx[0] * tile_grid_shape_yx[1]):
                raise ValueError(
                    f"{field_name}.tile_index={tile_index} exceeds tile grid capacity "
                    f"{int(tile_grid_shape_yx[0] * tile_grid_shape_yx[1])}"
                )
        else:
            if "tile_grid_shape_yx" in payload:
                extra["tile_grid_shape_yx"] = payload["tile_grid_shape_yx"]
            if "tile_index" in payload:
                extra["tile_index"] = payload["tile_index"]

        return cls(
            coverage_mode=str(coverage_mode),
            region_origin_zyx=(int(region_origin_zyx[0]), int(region_origin_zyx[1]), int(region_origin_zyx[2])),
            region_shape_zyx=(int(region_shape_zyx[0]), int(region_shape_zyx[1]), int(region_shape_zyx[2])),
            full_volume_shape_zyx=(
                int(full_volume_shape_zyx[0]),
                int(full_volume_shape_zyx[1]),
                int(full_volume_shape_zyx[2]),
            ),
            tile_grid_shape_yx=(None if tile_grid_shape_yx is None else (int(tile_grid_shape_yx[0]), int(tile_grid_shape_yx[1]))),
            tile_index=tile_index,
            extra=extra,
        )

    def contains(self, coords_zyx: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords_zyx)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"coords_zyx must have shape (N, 3), got {coords.shape}")
        z0, y0, x0 = self.region_origin_zyx
        dz, dy, dx = self.region_shape_zyx
        return (
            (coords[:, 0] >= z0)
            & (coords[:, 0] < z0 + dz)
            & (coords[:, 1] >= y0)
            & (coords[:, 1] < y0 + dy)
            & (coords[:, 2] >= x0)
            & (coords[:, 2] < x0 + dx)
        )

    def to_legacy(self) -> dict[str, Any]:
        payload = dict(self.extra)
        payload.update(
            {
                "coverage_mode": self.coverage_mode,
                "region_origin_zyx": list(self.region_origin_zyx),
                "region_shape_zyx": list(self.region_shape_zyx),
                "full_volume_shape_zyx": list(self.full_volume_shape_zyx),
            }
        )
        if self.coverage_mode == "tile_local":
            if self.tile_grid_shape_yx is None or self.tile_index is None:
                raise ValueError("tile_local scope requires tile_grid_shape_yx and tile_index")
            payload["tile_grid_shape_yx"] = list(self.tile_grid_shape_yx)
            payload["tile_index"] = int(self.tile_index)
        else:
            if "tile_grid_shape_yx" in self.extra:
                payload["tile_grid_shape_yx"] = self.extra["tile_grid_shape_yx"]
            if "tile_index" in self.extra:
                payload["tile_index"] = self.extra["tile_index"]
        return payload


@dataclass(frozen=True)
class Flow3DSidecarDescriptor:
    """Round-level sidecar descriptor for spilled dense 3D flow arrays."""

    storage: str
    path: str
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    extra: dict[Any, Any] = field(default_factory=dict)

    @classmethod
    def from_legacy(cls, payload: Any, *, field_name: str = "flow_3d") -> "Flow3DSidecarDescriptor":
        if not isinstance(payload, Mapping):
            raise ValueError(f"{field_name} sidecar descriptor must be a mapping")

        storage = payload.get("storage")
        if storage != FLOW_3D_SIDECAR_STORAGE:
            raise ValueError(f"{field_name}.storage must be {FLOW_3D_SIDECAR_STORAGE!r}, got {storage!r}")

        path = payload.get("path")
        if not isinstance(path, str) or not path.strip():
            raise ValueError(f"{field_name}.path must be a non-empty sidecar filename")
        if path in {".", ".."} or path.startswith("/") or "/" in path or "\\" in path:
            raise ValueError(f"{field_name}.path must be a direct filename under transforms/: {path}")

        shape = None
        raw_shape = payload.get("shape")
        if raw_shape is not None:
            shape = _coerce_int_tuple(raw_shape, field_name=f"{field_name}.shape")
            if len(shape) != 4:
                raise ValueError(f"{field_name}.shape must describe 4D [3, Z, Y, X] data")
            if shape[0] != 3:
                raise ValueError(f"{field_name}.shape first dimension must contain 3 displacement components")
            if any(value <= 0 for value in shape):
                raise ValueError(f"{field_name}.shape must contain positive integers")

        dtype = payload.get("dtype")
        if dtype is not None and (not isinstance(dtype, str) or not dtype.strip()):
            raise ValueError(f"{field_name}.dtype must be a non-empty string when present")

        return cls(
            storage=str(storage),
            path=path,
            shape=shape,
            dtype=dtype,
            extra=_copy_extra(payload, {"storage", "path", "shape", "dtype"}),
        )

    def to_legacy(self) -> dict[str, Any]:
        payload = dict(self.extra)
        payload.update({"storage": self.storage, "path": self.path})
        if self.shape is not None:
            payload["shape"] = list(self.shape)
        if self.dtype is not None:
            payload["dtype"] = self.dtype
        return payload


@dataclass(frozen=True)
class TransformEntry:
    """One round transform payload inside ``transforms_fov_{fov_id}.npy``."""

    round_id: int | None
    global_shift_3d: Any
    global_corr: Any = _MISSING
    flow_2d: Any = None
    flow_3d: Any = None
    final_corr: Any = _MISSING
    is_reference_round: Any = _MISSING
    round_id_value: Any = _MISSING
    field_semantics: FieldSemantics = field(default_factory=FieldSemantics.unknown)
    scope: ScopeMetadata | None = None
    backend_metadata: Any = _MISSING
    user_metadata: dict[str, Any] = field(default_factory=dict)
    extra: dict[Any, Any] = field(default_factory=dict)

    @classmethod
    def from_legacy(
        cls,
        round_key: Any,
        payload: Any,
        *,
        field_name: str | None = None,
    ) -> "TransformEntry":
        name = _field_name(field_name, f"transform round {round_key}")
        if not isinstance(payload, Mapping):
            raise ValueError(f"{name} must be a mapping")
        if "global_shift_3d" not in payload:
            raise ValueError(f"{name} is missing required field global_shift_3d")

        try:
            round_id = int(round_key)
        except (TypeError, ValueError):
            round_id = None

        global_shift_3d = payload["global_shift_3d"]
        _validate_numeric_vector(global_shift_3d, field_name=f"{name}.global_shift_3d", length=3)

        flow_3d = payload.get("flow_3d")
        if isinstance(flow_3d, Mapping):
            flow_3d = Flow3DSidecarDescriptor.from_legacy(flow_3d, field_name=f"{name}.flow_3d")
        elif isinstance(flow_3d, np.ndarray):
            if flow_3d.ndim != 4:
                raise ValueError(f"{name}.flow_3d ndarray must be 4D [3, Z, Y, X], got shape {flow_3d.shape}")
            if flow_3d.shape[0] != 3:
                raise ValueError(f"{name}.flow_3d first dimension must contain 3 displacement components")
        elif flow_3d is not None:
            raise ValueError(f"{name}.flow_3d must be None, ndarray, or sidecar descriptor mapping")

        flow_2d = payload.get("flow_2d")
        if isinstance(flow_2d, np.ndarray):
            if flow_2d.ndim != 3:
                raise ValueError(f"{name}.flow_2d ndarray must be 3D [2, Y, X], got shape {flow_2d.shape}")
            if flow_2d.shape[0] != 2:
                raise ValueError(f"{name}.flow_2d first dimension must contain 2 displacement components")
        elif flow_2d is not None:
            raise ValueError(f"{name}.flow_2d must be None or ndarray")

        scope_payload = payload.get("_scope")
        scope = None if scope_payload is None else ScopeMetadata.from_legacy(scope_payload, field_name=f"{name}._scope")
        semantics = FieldSemantics.from_legacy(payload.get("_semantics"), field_name=f"{name}._semantics")

        user_metadata = payload.get("user_metadata", payload.get("_user_metadata", payload.get("_user_meta", {})))
        if user_metadata is None:
            user_metadata = {}
        if not isinstance(user_metadata, Mapping):
            raise ValueError(f"{name}.user_metadata must be a mapping when present")

        known = {
            "global_shift_3d",
            "global_corr",
            "flow_2d",
            "flow_3d",
            "final_corr",
            "is_reference_round",
            "round_id",
            "_semantics",
            "_scope",
            "backend_metadata",
            "user_metadata",
            "_user_metadata",
            "_user_meta",
        }
        return cls(
            round_id=round_id,
            global_shift_3d=global_shift_3d,
            global_corr=payload.get("global_corr", _MISSING),
            flow_2d=flow_2d,
            flow_3d=flow_3d,
            final_corr=payload.get("final_corr", _MISSING),
            is_reference_round=payload.get("is_reference_round", _MISSING),
            round_id_value=payload.get("round_id", _MISSING),
            field_semantics=semantics,
            scope=scope,
            backend_metadata=payload.get("backend_metadata", _MISSING),
            user_metadata=dict(user_metadata),
            extra=_copy_extra(payload, known),
        )

    def to_legacy(self) -> dict[str, Any]:
        payload = dict(self.extra)
        payload["global_shift_3d"] = self.global_shift_3d
        if self.global_corr is not _MISSING:
            payload["global_corr"] = self.global_corr
        payload["flow_2d"] = self.flow_2d
        payload["flow_3d"] = self.flow_3d.to_legacy() if isinstance(self.flow_3d, Flow3DSidecarDescriptor) else self.flow_3d
        if self.final_corr is not _MISSING:
            payload["final_corr"] = self.final_corr
        if self.is_reference_round is not _MISSING:
            payload["is_reference_round"] = self.is_reference_round
        if self.round_id_value is not _MISSING:
            payload["round_id"] = self.round_id_value
        if self.backend_metadata is not _MISSING:
            payload["backend_metadata"] = self.backend_metadata
        if self.user_metadata:
            payload["user_metadata"] = dict(self.user_metadata)
        payload["_semantics"] = self.field_semantics.to_legacy()
        if self.scope is not None:
            payload["_scope"] = self.scope.to_legacy()
        return payload


@dataclass(frozen=True)
class ReleaseContract:
    """Release-facing contract wrapper with optional strict validation."""

    payload: dict[str, Any]

    @classmethod
    def from_legacy(
        cls,
        payload: Any,
        *,
        field_name: str = "release_contract",
        strict: bool = False,
    ) -> "ReleaseContract":
        if not isinstance(payload, Mapping):
            raise ValueError(f"{field_name} must be a mapping")
        contract = dict(payload)

        if strict:
            required = {
                "requested_scope_mode",
                "delivered_coverage",
                "scope_valid",
                "scope_status",
                "requested_intent",
                "delivered_capability",
                "field_semantics_contract",
                "release_gate",
            }
            missing = sorted(required.difference(contract))
            if missing:
                raise ValueError(f"{field_name} is missing required fields: {missing}")

            for key in ("requested_intent", "delivered_capability", "field_semantics_contract", "release_gate"):
                if not isinstance(contract.get(key), Mapping):
                    raise ValueError(f"{field_name}.{key} must be a mapping")

        requested_scope_mode = contract.get("requested_scope_mode")
        if requested_scope_mode is not None and requested_scope_mode not in SCOPE_MODES:
            raise ValueError(f"{field_name}.requested_scope_mode must be one of {sorted(SCOPE_MODES)}")
        delivered_coverage = contract.get("delivered_coverage")
        if delivered_coverage is not None and delivered_coverage not in SCOPE_MODES:
            raise ValueError(f"{field_name}.delivered_coverage must be one of {sorted(SCOPE_MODES)}")
        scope_status = contract.get("scope_status")
        if scope_status is not None and scope_status not in SCOPE_STATUSES:
            raise ValueError(f"{field_name}.scope_status must be one of {sorted(SCOPE_STATUSES)}")
        if "scope_valid" in contract and not isinstance(contract.get("scope_valid"), bool):
            raise ValueError(f"{field_name}.scope_valid must be a boolean")

        release_gate = contract.get("release_gate")
        if release_gate is not None:
            if not isinstance(release_gate, Mapping):
                raise ValueError(f"{field_name}.release_gate must be a mapping")
            status = release_gate.get("status")
            if status is not None and status not in RELEASE_GATE_STATUSES:
                raise ValueError(f"{field_name}.release_gate.status must be one of {sorted(RELEASE_GATE_STATUSES)}")

        return cls(payload=contract)

    def to_legacy(self) -> dict[str, Any]:
        return dict(self.payload)


@dataclass(frozen=True)
class TransformManifest:
    """Per-FOV transform manifest with typed round entries and legacy IO."""

    fov_id: int | None
    entries: tuple[TransformEntry, ...]
    provenance: dict[str, Any] | None = None
    release_contract: ReleaseContract | None = None
    schema_version: int = LEGACY_SCHEMA_VERSION
    schema_key: str | None = None
    metadata: dict[Any, Any] = field(default_factory=dict)
    user_metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_legacy(
        cls,
        payload: Any,
        *,
        fov_id: int | None = None,
        validate_release_contract: bool = False,
    ) -> "TransformManifest":
        if not isinstance(payload, Mapping):
            raise ValueError(f"Transform manifest must be a mapping, got {type(payload)}")

        schema_key = None
        schema_version = LEGACY_SCHEMA_VERSION
        for key in _SCHEMA_KEYS:
            if key in payload:
                schema_key = key
                raw_version = payload[key]
                if not isinstance(raw_version, (int, np.integer)):
                    raise ValueError(f"Transform manifest {key} must be an integer")
                schema_version = int(raw_version)
                break
        if schema_version < LEGACY_SCHEMA_VERSION:
            raise ValueError(f"Unsupported negative transform manifest schema_version: {schema_version}")

        entries: list[TransformEntry] = []
        metadata: dict[Any, Any] = {}
        provenance = None
        release_contract = None
        user_metadata: dict[str, Any] = {}

        for key, value in payload.items():
            if key in _SCHEMA_KEYS:
                continue
            if key == "_provenance":
                if not isinstance(value, Mapping):
                    raise ValueError("Transform manifest _provenance must be a mapping")
                provenance = dict(value)
                continue
            if key == "_contract":
                release_contract = ReleaseContract.from_legacy(
                    value,
                    field_name="Transform manifest _contract",
                    strict=validate_release_contract,
                )
                continue
            if key in {"user_metadata", "_user_metadata", "_user_meta"}:
                if not isinstance(value, Mapping):
                    raise ValueError(f"Transform manifest {key} must be a mapping")
                user_metadata.update(dict(value))
                continue

            if _is_round_key(key):
                entries.append(TransformEntry.from_legacy(key, value))
                continue

            metadata[key] = value

        entries.sort(key=lambda entry: (entry.round_id is None, -1 if entry.round_id is None else entry.round_id))
        return cls(
            fov_id=fov_id,
            entries=tuple(entries),
            provenance=provenance,
            release_contract=release_contract,
            schema_version=schema_version,
            schema_key=schema_key,
            metadata=metadata,
            user_metadata=user_metadata,
        )

    def to_legacy(self, *, include_schema_version: bool | None = None) -> dict[Any, Any]:
        payload: dict[Any, Any] = {}
        if include_schema_version is None:
            include_schema_version = self.schema_key is not None
        if include_schema_version:
            payload[self.schema_key or "schema_version"] = int(self.schema_version)

        payload.update(self.metadata)
        for entry in self.entries:
            if entry.round_id is None:
                raise ValueError("Cannot dump TransformEntry with non-numeric legacy round key")
            payload[int(entry.round_id)] = entry.to_legacy()

        if self.user_metadata:
            payload["user_metadata"] = dict(self.user_metadata)
        if self.provenance is not None:
            payload["_provenance"] = dict(self.provenance)
        if self.release_contract is not None:
            payload["_contract"] = self.release_contract.to_legacy()
        return payload

    def round_entry(self, round_id: int) -> TransformEntry:
        for entry in self.entries:
            if entry.round_id == round_id:
                return entry
        raise KeyError(f"Transform manifest has no round entry for round {round_id}")
