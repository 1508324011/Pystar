# pyright: reportDeprecated=false, reportExplicitAny=false, reportAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnusedCallResult=false, reportImplicitStringConcatenation=false, reportUnnecessaryCast=false
"""Private transform manifest and flow sidecar I/O helpers.

This module owns filesystem persistence and hydration for transform manifests and
round-level ``flow_3d`` sidecars while the stable public import surface remains
``pystar.io``.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
from numpy.typing import NDArray

from ._io_paths import (
    get_flow_3d_sidecar_filename,
    get_fov_output_structure,
    get_provenance_summary_path,
    get_transform_manifest_path,
)
from .runtime_artifacts import (
    FLOW_3D_SIDECAR_STORAGE,
    Flow3DSidecarDescriptor,
    TransformEntry,
    TransformManifest,
)

_BackfillRequestedIntent = Callable[[dict[str, Any]], None]
_BackfillFieldSemantics = Callable[[dict[str, Any], Mapping[Any, Any]], None]
_ValidateProvenance = Callable[[dict[str, Any]], None]
_ValidateRoundScopeAlignment = Callable[[Mapping[Any, Any], Mapping[Any, Any]], None]
_BuildProvenanceSummary = Callable[[int, Mapping[Any, Any], Mapping[str, Any]], str]

_backfill_requested_intent_diagnostic_defaults: _BackfillRequestedIntent | None = None
_backfill_field_semantics_contract: _BackfillFieldSemantics | None = None
_validate_provenance_schema: _ValidateProvenance | None = None
_validate_round_scope_contract_alignment: _ValidateRoundScopeAlignment | None = None
_build_provenance_summary_markdown: _BuildProvenanceSummary | None = None


def configure_transform_manifest_io(
    *,
    backfill_requested_intent_diagnostic_defaults: _BackfillRequestedIntent,
    backfill_field_semantics_contract: _BackfillFieldSemantics,
    validate_provenance_schema: _ValidateProvenance,
    validate_round_scope_contract_alignment: _ValidateRoundScopeAlignment,
    build_provenance_summary_markdown: _BuildProvenanceSummary,
) -> None:
    """Install provenance helpers supplied by the public ``pystar.io`` facade."""

    global _backfill_requested_intent_diagnostic_defaults
    global _backfill_field_semantics_contract
    global _validate_provenance_schema
    global _validate_round_scope_contract_alignment
    global _build_provenance_summary_markdown

    _backfill_requested_intent_diagnostic_defaults = backfill_requested_intent_diagnostic_defaults
    _backfill_field_semantics_contract = backfill_field_semantics_contract
    _validate_provenance_schema = validate_provenance_schema
    _validate_round_scope_contract_alignment = validate_round_scope_contract_alignment
    _build_provenance_summary_markdown = build_provenance_summary_markdown


def _require_provenance_helpers() -> tuple[
    _BackfillRequestedIntent,
    _BackfillFieldSemantics,
    _ValidateProvenance,
    _ValidateRoundScopeAlignment,
    _BuildProvenanceSummary,
]:
    helpers = (
        _backfill_requested_intent_diagnostic_defaults,
        _backfill_field_semantics_contract,
        _validate_provenance_schema,
        _validate_round_scope_contract_alignment,
        _build_provenance_summary_markdown,
    )
    if any(helper is None for helper in helpers):
        raise RuntimeError(
            "Transform manifest provenance helpers are not configured. "
            "Import and use these APIs through pystar.io instead of pystar._io_transforms."
        )
    return cast(
        tuple[
            _BackfillRequestedIntent,
            _BackfillFieldSemantics,
            _ValidateProvenance,
            _ValidateRoundScopeAlignment,
            _BuildProvenanceSummary,
        ],
        helpers,
    )


def is_round_transform_entry(value: object) -> bool:
    return isinstance(value, dict) and "global_shift_3d" in value


_is_round_transform_entry = is_round_transform_entry


def write_provenance_summary(base_dir: Path, fov_id: int, summary_markdown: str) -> Path:
    _ = get_fov_output_structure(base_dir, fov_id)
    summary_path = get_provenance_summary_path(base_dir, fov_id)
    temp_path = summary_path.with_suffix(".md.tmp")
    if summary_path.exists():
        summary_path.unlink()
    temp_path.write_text(summary_markdown, encoding="utf-8")
    temp_path.replace(summary_path)
    return summary_path


def persist_flow_3d_sidecar(
    base_dir: Path,
    fov_id: int,
    round_id: int,
    flow_3d: NDArray[Any],
) -> dict[str, Any]:
    """Persist one dense round-level ``flow_3d`` sidecar and return its descriptor."""

    _ = get_fov_output_structure(base_dir, fov_id)
    transforms_dir = get_transform_manifest_path(base_dir, fov_id).parent

    flow_3d_arr = np.asarray(flow_3d)
    if flow_3d_arr.ndim != 4:
        raise ValueError(
            f"flow_3d for round {round_id} must be 4D [3, Z, Y, X], got shape {flow_3d_arr.shape}"
        )
    if flow_3d_arr.shape[0] != 3:
        raise ValueError(
            f"flow_3d for round {round_id} first dimension must contain 3 displacement components, "
            f"got shape {flow_3d_arr.shape}"
        )

    sidecar_name = get_flow_3d_sidecar_filename(fov_id, round_id)
    sidecar_path = transforms_dir / sidecar_name
    temp_path = sidecar_path.with_suffix(f"{sidecar_path.suffix}.tmp")
    if temp_path.exists():
        temp_path.unlink()

    with temp_path.open("wb") as handle:
        np.save(handle, flow_3d_arr, allow_pickle=False)
    temp_path.replace(sidecar_path)

    return {
        "storage": FLOW_3D_SIDECAR_STORAGE,
        "path": sidecar_name,
        "shape": list(flow_3d_arr.shape),
        "dtype": str(flow_3d_arr.dtype),
    }


def _validate_flow_3d_sidecar_descriptor(
    flow_3d: Mapping[str, Any],
    *,
    round_key: Any,
    transforms_dir: Path,
) -> tuple[dict[str, Any], Path]:
    descriptor_model = Flow3DSidecarDescriptor.from_legacy(
        flow_3d,
        field_name=f"flow_3d manifest for round {round_key}",
    )
    descriptor = cast(dict[str, Any], descriptor_model.to_legacy())
    sidecar_path = transforms_dir / descriptor_model.path
    if not sidecar_path.exists():
        raise FileNotFoundError(
            f"flow_3d sidecar referenced by transform manifest is missing: {sidecar_path}"
        )

    return descriptor, sidecar_path


def _load_flow_3d_sidecar_array(
    descriptor: Mapping[str, Any],
    *,
    round_key: Any,
    sidecar_path: Path,
) -> NDArray[Any]:
    flow_3d_arr = np.load(sidecar_path, allow_pickle=False)
    if flow_3d_arr.ndim != 4:
        raise ValueError(f"flow_3d sidecar for round {round_key} must be 4D, got shape {flow_3d_arr.shape}")
    if flow_3d_arr.shape[0] != 3:
        raise ValueError(
            f"flow_3d sidecar for round {round_key} first dimension must contain 3 displacement components, "
            f"got shape {flow_3d_arr.shape}"
        )

    expected_shape = descriptor.get("shape")
    if expected_shape is not None and not isinstance(expected_shape, (list, tuple)):
        raise ValueError(f"flow_3d sidecar shape metadata for round {round_key} must be a list/tuple")
    if expected_shape is not None and tuple(expected_shape) != tuple(flow_3d_arr.shape):
        raise ValueError(
            f"flow_3d sidecar shape mismatch for round {round_key}: "
            f"manifest={expected_shape}, actual={list(flow_3d_arr.shape)}"
        )

    expected_dtype = descriptor.get("dtype")
    if expected_dtype is not None and str(flow_3d_arr.dtype) != expected_dtype:
        raise ValueError(
            f"flow_3d sidecar dtype mismatch for round {round_key}: "
            f"manifest={expected_dtype}, actual={flow_3d_arr.dtype}"
        )

    return cast(NDArray[Any], flow_3d_arr)


def materialize_round_transform_entry(
    base_dir: Path,
    fov_id: int,
    round_id: int,
    transform_data: Mapping[str, Any],
    *,
    hydrate_flow_3d: bool = True,
) -> dict[str, Any]:
    """Validate one round transform entry and optionally hydrate persisted flow sidecars."""

    manifest_path = get_transform_manifest_path(base_dir, fov_id)
    transforms_dir = manifest_path.parent

    round_payload = cast(
        dict[str, Any],
        TransformEntry.from_legacy(
            round_id,
            transform_data,
            field_name=f"transform round {round_id}",
        ).to_legacy(),
    )

    flow_3d = round_payload.get("flow_3d")
    if isinstance(flow_3d, Mapping):
        descriptor, sidecar_path = _validate_flow_3d_sidecar_descriptor(
            flow_3d,
            round_key=round_id,
            transforms_dir=transforms_dir,
        )
        if hydrate_flow_3d:
            round_payload["flow_3d"] = _load_flow_3d_sidecar_array(
                descriptor,
                round_key=round_id,
                sidecar_path=sidecar_path,
            )
        else:
            round_payload["flow_3d"] = descriptor
    elif flow_3d is not None and not isinstance(flow_3d, np.ndarray):
        raise ValueError(f"Unsupported flow_3d payload type for round {round_id}: {type(flow_3d)}")
    else:
        round_payload.setdefault("flow_3d", None)

    return round_payload


def save_transform_manifest(
    base_dir: Path,
    fov_id: int,
    transforms: dict[Any, Any],
    provenance: Optional[dict[str, Any]] = None,
) -> Path:
    """Persist the transform manifest and spill dense 3D flow arrays into sidecars."""

    manifest_model = TransformManifest.from_legacy(transforms, fov_id=fov_id)
    transforms = cast(dict[Any, Any], manifest_model.to_legacy())
    release_contract = transforms.get("_contract")
    if release_contract is not None and not isinstance(release_contract, Mapping):
        raise ValueError("Transform manifest _contract must be a mapping when present")

    if provenance is not None:
        provenance_release_contract = provenance.get("release_contract")
        if not isinstance(provenance_release_contract, Mapping):
            raise ValueError("Transform manifest _provenance.release_contract must be a mapping")
        if release_contract is not None and dict(release_contract) != dict(provenance_release_contract):
            raise ValueError(
                "Transform manifest persisted _contract must match _provenance.release_contract; "
                "manifest metadata drifted at the I/O boundary"
            )

        (
            backfill_requested_intent_diagnostic_defaults,
            backfill_field_semantics_contract,
            validate_provenance_schema,
            validate_round_scope_contract_alignment,
            build_provenance_summary_markdown,
        ) = _require_provenance_helpers()
        provenance = copy.deepcopy(provenance)
        backfill_requested_intent_diagnostic_defaults(provenance)
        backfill_field_semantics_contract(provenance, transforms)
        validate_provenance_schema(provenance)
    else:
        validate_round_scope_contract_alignment = None
        build_provenance_summary_markdown = None

    _ = get_fov_output_structure(base_dir, fov_id)
    manifest_path = get_transform_manifest_path(base_dir, fov_id)
    transforms_dir = manifest_path.parent

    manifest_payload: dict[Any, Any] = {}
    referenced_sidecars: set[Path] = set()
    sidecar_payloads: dict[Path, NDArray[Any]] = {}

    for round_key, transform_data in transforms.items():
        if not _is_round_transform_entry(transform_data):
            manifest_payload[round_key] = transform_data
            continue

        try:
            round_id = int(round_key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Round transform entry must use a numeric key, got {round_key!r}") from exc

        round_payload = cast(
            dict[str, Any],
            TransformEntry.from_legacy(
                round_id,
                transform_data,
                field_name=f"transform round {round_id}",
            ).to_legacy(),
        )
        flow_3d = round_payload.get("flow_3d")

        if isinstance(flow_3d, Mapping):
            descriptor, sidecar_path = _validate_flow_3d_sidecar_descriptor(
                flow_3d,
                round_key=round_id,
                transforms_dir=transforms_dir,
            )
            referenced_sidecars.add(sidecar_path)
            round_payload["flow_3d"] = descriptor
            manifest_payload[round_id] = round_payload
            continue

        if flow_3d is None:
            round_payload.setdefault("flow_3d", None)
            manifest_payload[round_id] = round_payload
            continue

        flow_3d_arr = np.asarray(flow_3d)
        if flow_3d_arr.ndim != 4:
            raise ValueError(
                f"flow_3d for round {round_id} must be 4D [3, Z, Y, X], got shape {flow_3d_arr.shape}"
            )
        if flow_3d_arr.shape[0] != 3:
            raise ValueError(
                f"flow_3d for round {round_id} first dimension must contain 3 displacement components, "
                f"got shape {flow_3d_arr.shape}"
            )

        sidecar_name = get_flow_3d_sidecar_filename(fov_id, round_id)
        sidecar_path = transforms_dir / sidecar_name
        descriptor = {
            "storage": FLOW_3D_SIDECAR_STORAGE,
            "path": sidecar_name,
            "shape": list(flow_3d_arr.shape),
            "dtype": str(flow_3d_arr.dtype),
        }
        referenced_sidecars.add(sidecar_path)
        sidecar_payloads[sidecar_path] = flow_3d_arr
        round_payload["flow_3d"] = descriptor
        manifest_payload[round_id] = round_payload

    if provenance is not None:
        cast(_BackfillFieldSemantics, _backfill_field_semantics_contract)(provenance, manifest_payload)
        cast(_ValidateProvenance, _validate_provenance_schema)(provenance)
        cast(_ValidateRoundScopeAlignment, validate_round_scope_contract_alignment)(
            manifest_payload,
            provenance["release_contract"],
        )
        manifest_payload["_provenance"] = provenance
        manifest_payload["_contract"] = provenance["release_contract"]

    manifest_payload = cast(
        dict[Any, Any],
        TransformManifest.from_legacy(
            manifest_payload,
            fov_id=fov_id,
            validate_release_contract=provenance is not None,
        ).to_legacy(),
    )

    summary_markdown = None
    if provenance is not None:
        summary_markdown = cast(_BuildProvenanceSummary, build_provenance_summary_markdown)(
            fov_id,
            manifest_payload,
            provenance,
        )

    for sidecar_path, flow_3d_arr in sidecar_payloads.items():
        temp_path = sidecar_path.with_suffix(f"{sidecar_path.suffix}.tmp")
        if temp_path.exists():
            temp_path.unlink()
        with temp_path.open("wb") as handle:
            np.save(handle, flow_3d_arr, allow_pickle=False)
        temp_path.replace(sidecar_path)

    for stale_path in transforms_dir.glob(f"transforms_fov_{fov_id}_round_*_flow_3d.npy"):
        if stale_path not in referenced_sidecars:
            stale_path.unlink()

    np.save(manifest_path, cast(Any, manifest_payload))
    if summary_markdown is not None:
        write_provenance_summary(base_dir, fov_id, summary_markdown)
    return manifest_path


def load_transform_manifest(
    base_dir: Path,
    fov_id: int,
    load_provenance: bool = False,
    *,
    hydrate_flow_3d: bool = True,
) -> dict[Any, Any]:
    """Load the transform manifest and optionally hydrate ``flow_3d`` sidecars."""

    manifest_path = get_transform_manifest_path(base_dir, fov_id)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Transform manifest not found: {manifest_path}. Run registration first!")

    transforms = np.load(manifest_path, allow_pickle=True).item()
    if not isinstance(transforms, dict):
        raise ValueError(f"Transform manifest is malformed: expected dict payload, got {type(transforms)}")

    manifest_model = TransformManifest.from_legacy(transforms, fov_id=fov_id)
    transforms = cast(dict[Any, Any], manifest_model.to_legacy())

    provenance = transforms.get("_provenance")
    release_contract = transforms.get("_contract")
    if release_contract is not None and not isinstance(release_contract, Mapping):
        raise ValueError("Transform manifest _contract must be a mapping when present")

    if provenance is not None:
        (
            backfill_requested_intent_diagnostic_defaults,
            backfill_field_semantics_contract,
            validate_provenance_schema,
            validate_round_scope_contract_alignment,
            _,
        ) = _require_provenance_helpers()
        backfill_requested_intent_diagnostic_defaults(provenance)
        backfill_field_semantics_contract(provenance, transforms)
        validate_provenance_schema(provenance)

        provenance_release_contract = provenance.get("release_contract")
        if not isinstance(provenance_release_contract, Mapping):
            raise ValueError("Transform manifest _provenance.release_contract must be a mapping")

        if release_contract is None:
            if load_provenance:
                raise ValueError(
                    "Transform manifest persisted _provenance but is missing matching _contract; "
                    "load_transform_manifest(load_provenance=True) requires both metadata fields"
                )
        elif dict(release_contract) != dict(provenance_release_contract):
            raise ValueError(
                "Transform manifest _contract must match _provenance.release_contract; "
                "manifest metadata drifted at the I/O boundary"
            )

        if release_contract is not None:
            release_contract = cast(Mapping[Any, Any], release_contract)
    else:
        validate_round_scope_contract_alignment = None

    materialized: dict[Any, Any] = {}

    for round_key, transform_data in transforms.items():
        if round_key in {"_provenance", "_contract"}:
            continue

        if not _is_round_transform_entry(transform_data):
            materialized[round_key] = transform_data
            continue

        try:
            round_id = int(round_key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Round transform entry must use a numeric key, got {round_key!r}") from exc

        round_payload = materialize_round_transform_entry(
            base_dir,
            fov_id,
            round_id,
            transform_data,
            hydrate_flow_3d=hydrate_flow_3d,
        )

        materialized[round_id] = round_payload

    if load_provenance and provenance is not None:
        cast(_ValidateRoundScopeAlignment, validate_round_scope_contract_alignment)(
            materialized,
            cast(Mapping[Any, Any], provenance["release_contract"]),
        )
        materialized["_contract"] = copy.deepcopy(cast(Mapping[Any, Any], release_contract))
        materialized["_provenance"] = provenance

    return materialized
