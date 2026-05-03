import copy
from collections.abc import Mapping, MutableMapping
import dask.array as da
import dask
import numpy as np
import tifffile
import xarray as xr
from pathlib import Path
from typing import List, Optional, Dict, Any, cast
from datetime import datetime, timezone
from numpy.typing import NDArray
from .infrastructure import ExperimentConfig
from .runtime_artifacts import TransformManifest


FLOW_3D_SIDECAR_STORAGE = "round_level_sidecar_npy"
RELEASE_GATE_STATUSES = {"valid", "degraded", "invalid", "debug_only"}
SCOPE_MODES = {"full_fov", "tile_local"}
SCOPE_STATUSES = {"valid", "degraded", "invalid"}
FIELD_SEMANTICS_REPRESENTATIONS = {"residual", "total", "unknown"}
FIELD_SEMANTICS_COMPOSITIONS = {"sequential_global_then_local", "independent", "unknown"}
FIELD_SEMANTICS_STATUSES = {"settled", "provisional", "unknown"}

REGISTRATION_PROFILE_FACTS: Dict[str, Dict[str, Any]] = {
    "demons_3d": {
        "name": "flow_3d_mainline",
        "declared_transform_capabilities": ["global_shift_3d", "flow_3d"],
        "supports_image_warp_mainline": True,
    },
    "optical_flow": {
        "name": "flow_2d_legacy_optical_flow",
        "declared_transform_capabilities": ["global_shift_3d", "flow_2d"],
        "supports_image_warp_mainline": False,
    },
    "bspline": {
        "name": "flow_2d_legacy_bspline",
        "declared_transform_capabilities": ["global_shift_3d", "flow_2d"],
        "supports_image_warp_mainline": False,
    },
}

# ==============================================================================
# SECTION 0: Provenance Schema and Validation
# ==============================================================================

PROVENANCE_VERSION = "1.0.0"

# Fixed Phase 1 RC Facts
RC_FACTS = {
    "preprocessing_backend": "native_pystar",
    "registration_backend": "native_pystar",
    "accelerator": "cpu"
}

EXECUTION_ENVELOPE_ALLOWED_VALUES = {
    "preprocessing_backend": {"native_pystar", "matlab_extracted", "provider_dispatch"},
    "registration_backend": {"native_pystar", "matlab_extracted", "provider_dispatch"},
    "accelerator": {"cpu"},
}

MATLAB_STAGE_SUPPORT_STATUSES = {"debug_only", "not_selected"}
MATLAB_STAGE_REQUIRED_PROMOTION_BLOCKERS = (
    "representative_benchmark_recovery_pending",
    "production_verification_pending",
)

LOCAL_ACCEPTANCE_MODE = "correlation_diagnostic_only"
FINAL_CORR_METRIC = "selected_transform_mip_corr"

MATLAB_STAGE_CONFIG_SURFACES: Dict[str, List[str]] = {
    "preprocessing": [
        "pipeline.preprocessing.sequence[].provider",
        "providers.matlab.preprocessing",
    ],
    "registration": [
        "pipeline.registration.global.provider",
        "pipeline.registration.local.provider",
        "providers.matlab.registration",
    ],
    "spot_finding": [
        "pipeline.spot_finding.provider",
        "providers.matlab.spot_finding",
    ],
    "extraction": [
        "pipeline.extraction.provider",
        "providers.matlab.extraction",
        "pipeline.extraction.transform_application_mode",
    ],
}

MATLAB_STAGE_PYTHON_OWNED_ARTIFACTS: Dict[str, List[str]] = {
    "preprocessing": [
        "Position{fov}/output_pystar/clean_data/clean_fov_{fov}_round_{round}_ch_{channel}.tif",
        "Position{fov}/output_pystar/qc_reports/preprocessing_provenance.yaml",
    ],
    "registration": [
        "Position{fov}/output_pystar/transforms/transforms_fov_{fov}.npy",
        "Position{fov}/output_pystar/transforms/transforms_fov_{fov}_round_{round}_flow_3d.npy",
        "Position{fov}/output_pystar/qc_reports/provenance_summary.md",
    ],
    "spot_finding": [
        "Position{fov}/output_pystar/spots/spots_fov_{fov}.csv",
        "Position{fov}/output_pystar/qc_reports/spot_finding_backend_fov_{fov}.json",
    ],
    "extraction": [
        "Position{fov}/output_pystar/extraction/intensity_matrix_fov_{fov}.npy",
        "Position{fov}/output_pystar/qc_reports/extraction_backend_fov_{fov}.json",
    ],
}


def _build_matlab_stage_contract_entry(
    *,
    matlab_requested: bool,
    declared_intent: str,
    config_surface: List[str],
    python_owned_artifacts: List[str],
) -> Dict[str, Any]:
    contract: Dict[str, Any] = {
        "declared_intent": declared_intent,
        "matlab_requested": matlab_requested,
        "config_surface": list(config_surface),
        "artifact_owner": "python_pystar",
        "python_owned_artifacts": list(python_owned_artifacts),
        "failure_contract": "fail_loud_no_fallback",
    }
    if matlab_requested:
        contract["current_support_status"] = "debug_only"
        contract["promotion_blockers"] = [
            "representative_benchmark_recovery_pending",
            "production_verification_pending",
        ]
    else:
        contract["current_support_status"] = "not_selected"
        contract["promotion_blockers"] = []
    return contract


def _build_matlab_stage_contracts(
    config: ExperimentConfig,
    execution_envelope: Mapping[str, str],
    registration_profile: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    _ = execution_envelope  # reserved for future envelope-specific narration
    preprocessing_mode = config.pipeline.preprocessing_provider_mode()
    registration_mode = config.pipeline.registration_provider_mode()
    return {
        "preprocessing": _build_matlab_stage_contract_entry(
            matlab_requested=config.pipeline.uses_matlab_preprocessing(),
            declared_intent=preprocessing_mode,
            config_surface=MATLAB_STAGE_CONFIG_SURFACES["preprocessing"],
            python_owned_artifacts=MATLAB_STAGE_PYTHON_OWNED_ARTIFACTS["preprocessing"],
        ),
        "registration": _build_matlab_stage_contract_entry(
            matlab_requested=(
                registration_profile.get("global_provider") == "matlab"
                or registration_profile.get("local_provider") == "matlab"
            ),
            declared_intent=registration_mode,
            config_surface=MATLAB_STAGE_CONFIG_SURFACES["registration"],
            python_owned_artifacts=MATLAB_STAGE_PYTHON_OWNED_ARTIFACTS["registration"],
        ),
        "spot_finding": _build_matlab_stage_contract_entry(
            matlab_requested=config.pipeline.uses_matlab_spot_finding(),
            declared_intent=config.pipeline.spot_finding.provider,
            config_surface=MATLAB_STAGE_CONFIG_SURFACES["spot_finding"],
            python_owned_artifacts=MATLAB_STAGE_PYTHON_OWNED_ARTIFACTS["spot_finding"],
        ),
        "extraction": _build_matlab_stage_contract_entry(
            matlab_requested=config.pipeline.uses_matlab_extraction(),
            declared_intent=config.pipeline.extraction.provider,
            config_surface=MATLAB_STAGE_CONFIG_SURFACES["extraction"],
            python_owned_artifacts=MATLAB_STAGE_PYTHON_OWNED_ARTIFACTS["extraction"],
        ),
    }


def _validate_matlab_stage_contracts(stage_contracts: Any) -> None:
    if not isinstance(stage_contracts, Mapping):
        raise ValueError("Provenance requested intent matlab_stage_contracts must be a mapping when present")

    required_stages = tuple(MATLAB_STAGE_CONFIG_SURFACES)
    missing_stages = set(required_stages).difference(stage_contracts.keys())
    if missing_stages:
        raise ValueError(
            "Provenance requested intent matlab_stage_contracts is missing stages: "
            f"{sorted(missing_stages)}"
        )

    for stage_name in required_stages:
        contract = stage_contracts.get(stage_name)
        if not isinstance(contract, Mapping):
            raise ValueError(f"matlab_stage_contracts.{stage_name} must be a mapping")
        declared_intent = contract.get("declared_intent")
        if not isinstance(declared_intent, str) or not declared_intent:
            raise ValueError(f"matlab_stage_contracts.{stage_name}.declared_intent must be a non-empty string")
        if not isinstance(contract.get("matlab_requested"), bool):
            raise ValueError(f"matlab_stage_contracts.{stage_name}.matlab_requested must be a boolean")
        current_support_status = contract.get("current_support_status")
        if current_support_status not in MATLAB_STAGE_SUPPORT_STATUSES:
            raise ValueError(
                f"matlab_stage_contracts.{stage_name}.current_support_status must be one of "
                f"{sorted(MATLAB_STAGE_SUPPORT_STATUSES)}, got {current_support_status!r}"
            )
        if contract.get("artifact_owner") != "python_pystar":
            raise ValueError(f"matlab_stage_contracts.{stage_name}.artifact_owner must be 'python_pystar'")
        if contract.get("failure_contract") != "fail_loud_no_fallback":
            raise ValueError(
                f"matlab_stage_contracts.{stage_name}.failure_contract must be 'fail_loud_no_fallback'"
            )
        config_surface = contract.get("config_surface")
        if (
            not isinstance(config_surface, list)
            or not config_surface
            or not all(isinstance(item, str) and item for item in config_surface)
        ):
            raise ValueError(
                f"matlab_stage_contracts.{stage_name}.config_surface must be a non-empty string list"
            )
        artifacts = contract.get("python_owned_artifacts")
        if (
            not isinstance(artifacts, list)
            or not artifacts
            or not all(isinstance(item, str) and item for item in artifacts)
        ):
            raise ValueError(
                f"matlab_stage_contracts.{stage_name}.python_owned_artifacts must be a non-empty string list"
            )
        promotion_blockers = contract.get("promotion_blockers")
        if (
            not isinstance(promotion_blockers, list)
            or not all(isinstance(item, str) and item for item in promotion_blockers)
        ):
            raise ValueError(
                f"matlab_stage_contracts.{stage_name}.promotion_blockers must be a string list"
            )

        matlab_requested = bool(contract.get("matlab_requested"))
        if matlab_requested:
            if current_support_status != "debug_only":
                raise ValueError(
                    f"matlab_stage_contracts.{stage_name}.current_support_status must stay 'debug_only' "
                    "while the MATLAB provider seam remains benchmark-blocked"
                )
            missing_blockers = [
                blocker
                for blocker in MATLAB_STAGE_REQUIRED_PROMOTION_BLOCKERS
                if blocker not in promotion_blockers
            ]
            if missing_blockers:
                raise ValueError(
                    f"matlab_stage_contracts.{stage_name}.promotion_blockers is missing required blockers: "
                    f"{missing_blockers}"
                )
        else:
            if current_support_status != "not_selected":
                raise ValueError(
                    f"matlab_stage_contracts.{stage_name}.current_support_status must be 'not_selected' "
                    "when the MATLAB seam is not requested"
                )
            if promotion_blockers:
                raise ValueError(
                    f"matlab_stage_contracts.{stage_name}.promotion_blockers must be empty when matlab_requested=false"
                )


def build_matlab_stage_contracts_from_config(config: ExperimentConfig) -> Dict[str, Dict[str, Any]]:
    execution_envelope = build_execution_envelope(config)
    registration_profile = derive_registration_profile(config)
    return _build_matlab_stage_contracts(config, execution_envelope, registration_profile)


def get_matlab_stage_contract(config: ExperimentConfig, stage_name: str) -> Dict[str, Any]:
    stage_contracts = build_matlab_stage_contracts_from_config(config)
    if stage_name not in stage_contracts:
        raise ValueError(
            f"Unknown MATLAB stage contract name {stage_name!r}; expected one of {sorted(stage_contracts)}"
        )
    return dict(stage_contracts[stage_name])


def _backfill_requested_intent_diagnostic_defaults(provenance: MutableMapping[str, Any]) -> None:
    """Backfill new diagnostic-only intent fields for legacy 1.0.0 manifests.

    New writes must include these fields explicitly via ``build_release_contract``.
    This helper exists only to preserve read compatibility for older persisted
    1.0.0 provenance payloads that predate the explicit diagnostic metadata.
    """
    release_contract = provenance.get("release_contract")
    if not isinstance(release_contract, Mapping):
        return

    requested_intent = release_contract.get("requested_intent")
    if not isinstance(requested_intent, dict):
        return

    requested_intent.setdefault("local_acceptance_mode", LOCAL_ACCEPTANCE_MODE)
    requested_intent.setdefault("final_corr_metric", FINAL_CORR_METRIC)
    requested_intent.setdefault("final_corr_diagnostic_only", True)
    requested_intent.setdefault("final_corr_release_gate", False)

    runtime_context = provenance.get("runtime_context")
    if not isinstance(runtime_context, dict):
        return

    config_reference = runtime_context.get("config_reference")
    if not isinstance(config_reference, dict):
        return

    key_parameters = config_reference.get("key_parameters")
    if not isinstance(key_parameters, dict):
        return

    key_parameters.setdefault("local_acceptance_mode", LOCAL_ACCEPTANCE_MODE)
    key_parameters.setdefault("final_corr_metric", FINAL_CORR_METRIC)
    key_parameters.setdefault("final_corr_diagnostic_only", True)
    key_parameters.setdefault("final_corr_release_gate", False)


def _default_field_semantics_payload(*, recorded_at: Optional[str] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "representation": "unknown",
        "composition": "unknown",
        "status": "unknown",
    }
    if recorded_at is not None:
        payload["recorded_at"] = recorded_at
    return payload


def _coerce_field_semantics_payload(
    value: Any,
    *,
    field_name: str,
    recorded_at: Optional[str] = None,
) -> Dict[str, Any]:
    if value is None:
        return _default_field_semantics_payload(recorded_at=recorded_at)
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")

    representation = value.get("representation", "unknown")
    composition = value.get("composition", "unknown")
    status = value.get("status", "unknown")

    if representation not in FIELD_SEMANTICS_REPRESENTATIONS:
        raise ValueError(
            f"{field_name}.representation must be one of {sorted(FIELD_SEMANTICS_REPRESENTATIONS)}, got {representation!r}"
        )
    if composition not in FIELD_SEMANTICS_COMPOSITIONS:
        raise ValueError(
            f"{field_name}.composition must be one of {sorted(FIELD_SEMANTICS_COMPOSITIONS)}, got {composition!r}"
        )
    if status not in FIELD_SEMANTICS_STATUSES:
        raise ValueError(
            f"{field_name}.status must be one of {sorted(FIELD_SEMANTICS_STATUSES)}, got {status!r}"
        )

    payload: Dict[str, Any] = {
        "representation": representation,
        "composition": composition,
        "status": status,
    }
    recorded_value = value.get("recorded_at", recorded_at)
    if recorded_value is not None:
        if not isinstance(recorded_value, str) or not recorded_value.strip():
            raise ValueError(f"{field_name}.recorded_at must be a non-empty ISO timestamp string when present")
        payload["recorded_at"] = recorded_value
    return payload


def _field_semantics_from_config(value: Any, *, field_name: str) -> Dict[str, Any]:
    if value is None:
        return _default_field_semantics_payload()
    return _coerce_field_semantics_payload(
        {
            "representation": getattr(value, "representation", "unknown"),
            "composition": getattr(value, "composition", "unknown"),
            "status": getattr(value, "status", "unknown"),
        },
        field_name=field_name,
    )


def _field_semantics_match(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (
        left.get("representation") == right.get("representation")
        and left.get("composition") == right.get("composition")
        and left.get("status") == right.get("status")
    )


def _infer_field_semantics_support(transforms: Mapping[Any, Any]) -> Dict[str, Any]:
    has_global_shift = False
    has_flow_2d = False
    has_flow_3d = False

    for _, transform_data in transforms.items():
        if not _is_round_transform_entry(transform_data):
            continue

        if transform_data.get("global_shift_3d") is not None:
            has_global_shift = True

        flow_2d = transform_data.get("flow_2d")
        if isinstance(flow_2d, np.ndarray):
            has_flow_2d = True

        flow_3d = transform_data.get("flow_3d")
        if isinstance(flow_3d, np.ndarray):
            has_flow_3d = True
        elif isinstance(flow_3d, Mapping) and flow_3d.get("storage") == FLOW_3D_SIDECAR_STORAGE:
            has_flow_3d = True

    return {
        "has_global_shift": has_global_shift,
        "has_flow_2d": has_flow_2d,
        "has_flow_3d": has_flow_3d,
        "supports_composition": has_global_shift and has_flow_3d,
    }


def _summarize_round_field_semantics(
    transforms: Mapping[Any, Any],
    declared: Mapping[str, Any],
) -> Dict[str, Any]:
    rounds_with_semantics: List[int] = []
    rounds_missing_semantics: List[int] = []
    rounds_mismatching_semantics: List[int] = []

    for round_key, transform_data in transforms.items():
        if not _is_round_transform_entry(transform_data):
            continue

        round_id = int(round_key)
        semantics_payload = transform_data.get("_semantics")
        if semantics_payload is None:
            rounds_missing_semantics.append(round_id)
            continue

        normalized = _coerce_field_semantics_payload(
            semantics_payload,
            field_name=f"transform round {round_id} _semantics",
        )
        rounds_with_semantics.append(round_id)
        if not _field_semantics_match(normalized, declared):
            rounds_mismatching_semantics.append(round_id)

    if rounds_mismatching_semantics:
        state = "mismatch"
    elif rounds_missing_semantics and not rounds_with_semantics:
        state = "not_recorded"
    elif rounds_missing_semantics:
        state = "partial"
    else:
        state = "match"

    return {
        "state": state,
        "rounds_with_semantics": sorted(rounds_with_semantics),
        "rounds_missing_semantics": sorted(rounds_missing_semantics),
        "rounds_mismatching_semantics": sorted(rounds_mismatching_semantics),
    }


def validate_field_semantics_contract(field_semantics_contract: Mapping[Any, Any]) -> Dict[str, Any]:
    declared = _coerce_field_semantics_payload(
        field_semantics_contract.get("declared"),
        field_name="field_semantics_contract.declared",
    )
    expected = _coerce_field_semantics_payload(
        field_semantics_contract.get("expected"),
        field_name="field_semantics_contract.expected",
    )
    inferred = _require_mapping(
        field_semantics_contract.get("inferred"),
        "field_semantics_contract.inferred",
    )
    consistency_check = _require_mapping(
        field_semantics_contract.get("consistency_check"),
        "field_semantics_contract.consistency_check",
    )

    errors: List[str] = []
    if declared["representation"] == "unknown":
        errors.append("Field representation is unknown - extraction semantics remain unresolved")
    if declared["composition"] == "unknown":
        errors.append("Field composition is unknown - global/local composition remains unresolved")
    if expected["representation"] == "unknown":
        errors.append("Extraction-expected field representation is unknown")
    if expected["composition"] == "unknown":
        errors.append("Extraction-expected field composition is unknown")

    if consistency_check.get("registration_vs_pipeline") == "mismatch":
        errors.append("Registration-declared field semantics do not match pipeline/extraction expectations")

    if consistency_check.get("persisted_rounds_vs_declared") == "mismatch":
        errors.append("Persisted round-level _semantics do not match declared registration field semantics")

    if expected["composition"] != "unknown" and not isinstance(inferred.get("supports_composition"), bool):
        errors.append("field_semantics_contract.inferred.supports_composition must be a boolean")

    if expected["composition"] == "sequential_global_then_local" and not inferred.get("supports_composition"):
        errors.append(
            "Expected sequential_global_then_local composition, but delivered transforms do not provide both global_shift_3d and flow_3d"
        )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
    }


def build_field_semantics_contract(
    config: ExperimentConfig,
    transforms: Mapping[Any, Any],
) -> Dict[str, Any]:
    declared = _field_semantics_from_config(
        config.pipeline.registration.field_semantics,
        field_name="pipeline.registration.field_semantics",
    )
    expected = _field_semantics_from_config(
        config.pipeline.field_semantics,
        field_name="pipeline.field_semantics",
    )
    inferred = _infer_field_semantics_support(transforms)
    persisted_rounds = _summarize_round_field_semantics(transforms, declared)

    contract = {
        "declared": declared,
        "expected": expected,
        "inferred": inferred,
        "consistency_check": {
            "registration_vs_pipeline": "match" if _field_semantics_match(declared, expected) else "mismatch",
            "persisted_rounds_vs_declared": persisted_rounds["state"],
            "rounds_with_semantics": persisted_rounds["rounds_with_semantics"],
            "rounds_missing_semantics": persisted_rounds["rounds_missing_semantics"],
            "rounds_mismatching_semantics": persisted_rounds["rounds_mismatching_semantics"],
        },
    }
    contract["validation"] = validate_field_semantics_contract(contract)
    return contract


def _backfill_field_semantics_contract(
    provenance: MutableMapping[str, Any],
    transforms: Mapping[Any, Any],
) -> None:
    release_contract = provenance.get("release_contract")
    if not isinstance(release_contract, dict):
        return

    field_semantics_contract = release_contract.get("field_semantics_contract")
    if not isinstance(field_semantics_contract, dict):
        field_semantics_contract = {}
        release_contract["field_semantics_contract"] = field_semantics_contract

    field_semantics_contract["declared"] = _coerce_field_semantics_payload(
        field_semantics_contract.get("declared"),
        field_name="release_contract.field_semantics_contract.declared",
    )
    field_semantics_contract["expected"] = _coerce_field_semantics_payload(
        field_semantics_contract.get("expected"),
        field_name="release_contract.field_semantics_contract.expected",
    )
    existing_inferred = field_semantics_contract.get("inferred")
    inferred_from_transforms = _infer_field_semantics_support(transforms)
    if isinstance(existing_inferred, Mapping):
        extra_inferred = {
            key: value
            for key, value in existing_inferred.items()
            if key not in inferred_from_transforms
        }
        inferred_from_transforms.update(extra_inferred)
    field_semantics_contract["inferred"] = inferred_from_transforms

    round_summary = _summarize_round_field_semantics(
        transforms,
        cast(Mapping[str, Any], field_semantics_contract["declared"]),
    )
    field_semantics_contract["consistency_check"] = {
        "registration_vs_pipeline": (
            "match"
            if _field_semantics_match(
                cast(Mapping[str, Any], field_semantics_contract["declared"]),
                cast(Mapping[str, Any], field_semantics_contract["expected"]),
            )
            else "mismatch"
        ),
        "persisted_rounds_vs_declared": round_summary["state"],
        "rounds_with_semantics": round_summary["rounds_with_semantics"],
        "rounds_missing_semantics": round_summary["rounds_missing_semantics"],
        "rounds_mismatching_semantics": round_summary["rounds_mismatching_semantics"],
    }

    field_semantics_contract["validation"] = validate_field_semantics_contract(field_semantics_contract)

    runtime_context = provenance.get("runtime_context")
    if isinstance(runtime_context, dict):
        config_reference = runtime_context.get("config_reference")
        if isinstance(config_reference, dict):
            key_parameters = config_reference.get("key_parameters")
            if isinstance(key_parameters, dict):
                key_parameters.setdefault(
                    "field_semantics",
                    dict(field_semantics_contract["expected"]),
                )
                key_parameters.setdefault(
                    "registration_field_semantics",
                    dict(field_semantics_contract["declared"]),
                )


def build_execution_envelope(config: Optional[ExperimentConfig] = None) -> Dict[str, str]:
    if config is None:
        envelope = dict(RC_FACTS)
    else:
        preprocessing_mode = config.pipeline.preprocessing_provider_mode()
        registration_mode = config.pipeline.registration_provider_mode()

        if preprocessing_mode == "native_only":
            preprocessing_backend = "native_pystar"
        elif preprocessing_mode == "matlab_only":
            preprocessing_backend = "matlab_extracted"
        else:
            preprocessing_backend = "provider_dispatch"

        if registration_mode == "native_only":
            registration_backend = "native_pystar"
        elif registration_mode == "matlab_only":
            registration_backend = "matlab_extracted"
        else:
            registration_backend = "provider_dispatch"

        envelope = {
            "preprocessing_backend": preprocessing_backend,
            "registration_backend": registration_backend,
            "accelerator": config.pipeline.accelerator,
        }
    validate_execution_envelope(envelope)
    return envelope


def derive_registration_profile(config: ExperimentConfig) -> Dict[str, Any]:
    reg_cfg = config.pipeline.registration
    global_provider = reg_cfg.global_provider
    local_provider = reg_cfg.local_provider

    if not reg_cfg.enable_local:
        if global_provider == "matlab":
            return {
                "name": "matlab_global_shift_only_experimental",
                "global_method": reg_cfg.global_stage.method,
                "global_provider": global_provider,
                "local_method": None,
                "local_provider": None,
                "declared_transform_capabilities": ["global_shift_3d"],
                "supports_image_warp_mainline": False,
                "backend_mode_status": "experimental_global_only",
            }
        return {
            "name": "global_shift_only",
            "global_method": reg_cfg.global_stage.method,
            "global_provider": global_provider,
            "local_method": None,
            "local_provider": None,
            "declared_transform_capabilities": ["global_shift_3d"],
            "supports_image_warp_mainline": False,
            "backend_mode_status": "mainline_native",
        }

    if local_provider == "matlab":
        return {
            "name": "matlab_demons_3d_experimental",
            "global_method": reg_cfg.global_stage.method,
            "global_provider": global_provider,
            "local_method": reg_cfg.local_method,
            "local_provider": local_provider,
            "declared_transform_capabilities": ["global_shift_3d", "flow_3d"],
            "supports_image_warp_mainline": False,
            "backend_mode_status": "experimental_local_kernel_swap",
        }

    profile = REGISTRATION_PROFILE_FACTS.get(reg_cfg.local_method)
    if profile is None:
        raise ValueError(f"Unsupported registration profile for local method: {reg_cfg.local_method!r}")
    backend_mode_status = "phase1_mainline"
    if global_provider != "native":
        backend_mode_status = "experimental_global_only"

    return {
        **profile,
        "global_method": reg_cfg.global_stage.method,
        "global_provider": global_provider,
        "local_method": reg_cfg.local_method,
        "local_provider": local_provider,
        "backend_mode_status": backend_mode_status,
    }


def summarize_delivered_capability(
    transforms: Mapping[Any, Any],
    reference_round: int,
) -> Dict[str, Any]:
    round_rows: Dict[str, Dict[str, Any]] = {}
    delivered_capabilities = {"global_shift_3d"}
    round_ids: List[int] = []
    flow_2d_rounds: List[int] = []
    flow_3d_rounds: List[int] = []
    missing_flow_3d_rounds: List[int] = []

    for round_key, transform_data in transforms.items():
        if not _is_round_transform_entry(transform_data):
            continue

        round_id = int(round_key)
        round_ids.append(round_id)
        flow_2d = transform_data.get("flow_2d")
        flow_3d = transform_data.get("flow_3d")
        has_flow_2d = isinstance(flow_2d, np.ndarray)
        has_flow_3d = isinstance(flow_3d, np.ndarray) or (
            isinstance(flow_3d, Mapping)
            and flow_3d.get("storage") == FLOW_3D_SIDECAR_STORAGE
            and isinstance(flow_3d.get("path"), str)
            and bool(flow_3d.get("path"))
        )
        is_reference_round = bool(transform_data.get("is_reference_round", round_id == reference_round))

        if has_flow_2d:
            delivered_capabilities.add("flow_2d")
            flow_2d_rounds.append(round_id)
        if has_flow_3d:
            delivered_capabilities.add("flow_3d")
            flow_3d_rounds.append(round_id)
        if not is_reference_round and not has_flow_3d:
            missing_flow_3d_rounds.append(round_id)

        round_rows[str(round_id)] = {
            "is_reference_round": is_reference_round,
            "has_flow_2d": has_flow_2d,
            "has_flow_3d": has_flow_3d,
        }

    return {
        "reference_round": reference_round,
        "round_ids": sorted(round_ids),
        "delivered_transform_capabilities": sorted(delivered_capabilities),
        "flow_2d_rounds": sorted(flow_2d_rounds),
        "flow_3d_rounds": sorted(flow_3d_rounds),
        "missing_flow_3d_rounds": sorted(missing_flow_3d_rounds),
        "per_round": round_rows,
    }


def _coerce_scope_int_tuple(value: Any, *, field_name: str, length: int) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list/tuple of {length} integers")
    if len(value) != length:
        raise ValueError(f"{field_name} must contain {length} integers")

    coerced: list[int] = []
    for item in value:
        if not isinstance(item, (int, np.integer)):
            raise ValueError(f"{field_name} entries must be integers, got {item!r}")
        coerced.append(int(item))
    return tuple(coerced)


def _normalize_round_scope_metadata(scope_payload: Any, *, field_name: str) -> Dict[str, Any]:
    if not isinstance(scope_payload, Mapping):
        raise ValueError(f"{field_name} must be a mapping")

    coverage_mode = scope_payload.get("coverage_mode")
    if coverage_mode not in SCOPE_MODES:
        raise ValueError(f"{field_name}.coverage_mode must be one of {sorted(SCOPE_MODES)}, got {coverage_mode!r}")

    region_origin_zyx = _coerce_scope_int_tuple(
        scope_payload.get("region_origin_zyx"),
        field_name=f"{field_name}.region_origin_zyx",
        length=3,
    )
    region_shape_zyx = _coerce_scope_int_tuple(
        scope_payload.get("region_shape_zyx"),
        field_name=f"{field_name}.region_shape_zyx",
        length=3,
    )
    full_volume_shape_zyx = _coerce_scope_int_tuple(
        scope_payload.get("full_volume_shape_zyx"),
        field_name=f"{field_name}.full_volume_shape_zyx",
        length=3,
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

    normalized: Dict[str, Any] = {
        "coverage_mode": str(coverage_mode),
        "region_origin_zyx": region_origin_zyx,
        "region_shape_zyx": region_shape_zyx,
        "full_volume_shape_zyx": full_volume_shape_zyx,
    }

    tile_grid_shape_yx = scope_payload.get("tile_grid_shape_yx")
    tile_index = scope_payload.get("tile_index")
    if coverage_mode == "tile_local":
        tile_grid_shape = _coerce_scope_int_tuple(
            tile_grid_shape_yx,
            field_name=f"{field_name}.tile_grid_shape_yx",
            length=2,
        )
        if any(value <= 0 for value in tile_grid_shape):
            raise ValueError(f"{field_name}.tile_grid_shape_yx must contain positive integers")
        if not isinstance(tile_index, (int, np.integer)) or int(tile_index) <= 0:
            raise ValueError(f"{field_name}.tile_index must be a positive integer for tile_local coverage")
        tile_index_int = int(tile_index)
        if tile_index_int > int(tile_grid_shape[0] * tile_grid_shape[1]):
            raise ValueError(
                f"{field_name}.tile_index={tile_index_int} exceeds tile grid capacity "
                f"{int(tile_grid_shape[0] * tile_grid_shape[1])}"
            )
        normalized["tile_grid_shape_yx"] = tile_grid_shape
        normalized["tile_index"] = tile_index_int

    return normalized


def _extract_consistent_round_scope_metadata(
    transforms: Mapping[Any, Any],
    *,
    require_all_rounds: bool,
) -> Optional[Dict[str, Any]]:
    round_scope_metadata: Optional[Dict[str, Any]] = None
    missing_rounds: list[int] = []

    for round_key, transform_data in transforms.items():
        if not _is_round_transform_entry(transform_data):
            continue

        scope_payload = transform_data.get("_scope")
        if scope_payload is None:
            missing_rounds.append(int(round_key))
            continue

        normalized = _normalize_round_scope_metadata(
            scope_payload,
            field_name=f"transform round {int(round_key)} _scope",
        )
        if round_scope_metadata is None:
            round_scope_metadata = normalized
            continue
        if normalized != round_scope_metadata:
            raise ValueError(
                f"Transform manifest mixes incompatible round scope metadata: round {int(round_key)} differs from earlier rounds"
            )

    if require_all_rounds and missing_rounds:
        raise ValueError(
            "Transform manifest is missing explicit _scope metadata for rounds: "
            + ", ".join(str(round_id) for round_id in sorted(missing_rounds))
        )

    return round_scope_metadata


def infer_delivered_scope_coverage(transforms: Mapping[Any, Any]) -> str:
    round_entries = [
        round_key
        for round_key, transform_data in transforms.items()
        if _is_round_transform_entry(transform_data)
    ]
    if not round_entries:
        raise ValueError(
            "Cannot derive delivered scope coverage from an empty transform bundle"
        )

    round_scope_metadata = _extract_consistent_round_scope_metadata(
        transforms,
        require_all_rounds=True,
    )
    if round_scope_metadata is None:
        raise ValueError("Cannot derive delivered scope coverage without explicit round _scope metadata")
    return str(round_scope_metadata["coverage_mode"])


def build_scope_contract(
    config: ExperimentConfig,
    transforms: Mapping[Any, Any],
) -> Dict[str, Any]:
    requested_scope_mode = config.pipeline.scope_mode
    delivered_coverage = infer_delivered_scope_coverage(transforms)
    scope_valid = requested_scope_mode == delivered_coverage
    scope_status = "valid" if scope_valid else "invalid"
    return {
        "requested_scope_mode": requested_scope_mode,
        "delivered_coverage": delivered_coverage,
        "scope_valid": scope_valid,
        "scope_status": scope_status,
    }


def validate_scope_contract(
    release_contract: Mapping[Any, Any],
    *,
    expected_scope_mode: Optional[str] = None,
) -> Dict[str, Any]:
    requested_scope_mode = release_contract.get("requested_scope_mode")
    if requested_scope_mode not in SCOPE_MODES:
        raise ValueError(
            "Release contract must declare requested_scope_mode as 'full_fov' or 'tile_local'"
        )

    delivered_coverage = release_contract.get("delivered_coverage")
    if delivered_coverage not in SCOPE_MODES:
        raise ValueError(
            "Release contract must declare delivered_coverage as 'full_fov' or 'tile_local'"
        )

    scope_valid = release_contract.get("scope_valid")
    if not isinstance(scope_valid, bool):
        raise ValueError("Release contract must declare scope_valid as a boolean")

    scope_status = release_contract.get("scope_status")
    if scope_status not in SCOPE_STATUSES:
        raise ValueError(
            "Release contract must declare scope_status as 'valid', 'degraded', or 'invalid'"
        )

    requested_intent = release_contract.get("requested_intent")
    if not isinstance(requested_intent, Mapping):
        raise ValueError("Release contract must include requested_intent mapping")

    requested_intent_scope = requested_intent.get("scope_mode")
    if requested_intent_scope not in SCOPE_MODES:
        raise ValueError(
            "Release contract requested_intent.scope_mode must be 'full_fov' or 'tile_local'"
        )
    if requested_intent_scope != requested_scope_mode:
        raise ValueError(
            "Release contract requested_scope_mode must match requested_intent.scope_mode"
        )

    if expected_scope_mode is not None and requested_scope_mode != expected_scope_mode:
        raise ValueError(
            f"Release contract requested_scope_mode={requested_scope_mode!r} does not match config scope_mode={expected_scope_mode!r}"
        )

    if scope_valid and delivered_coverage != requested_scope_mode:
        raise ValueError(
            "Release contract cannot mark scope_valid=True when delivered_coverage differs from requested_scope_mode"
        )
    if not scope_valid and delivered_coverage == requested_scope_mode:
        raise ValueError(
            "Release contract cannot mark scope_valid=False when delivered_coverage matches requested_scope_mode"
        )
    if scope_valid and scope_status != "valid":
        raise ValueError(
            "Release contract cannot mark scope_status as degraded/invalid when scope_valid=True"
        )
    if not scope_valid and scope_status == "valid":
        raise ValueError(
            "Release contract cannot mark scope_status='valid' when scope_valid=False"
        )

    return {
        "requested_scope_mode": requested_scope_mode,
        "delivered_coverage": delivered_coverage,
        "scope_valid": scope_valid,
        "scope_status": scope_status,
    }


def _validate_round_scope_contract_alignment(
    transforms: Mapping[Any, Any],
    release_contract: Mapping[Any, Any],
) -> None:
    scope_contract = validate_scope_contract(release_contract)
    delivered_coverage = scope_contract["delivered_coverage"]
    round_scope_metadata = _extract_consistent_round_scope_metadata(
        transforms,
        require_all_rounds=delivered_coverage == "tile_local",
    )

    if round_scope_metadata is None:
        return
    if round_scope_metadata["coverage_mode"] != delivered_coverage:
        raise ValueError(
            "Round _scope coverage_mode does not match release_contract.delivered_coverage: "
            f"{round_scope_metadata['coverage_mode']!r} != {delivered_coverage!r}"
        )


def _format_scope_region(scope_metadata: Mapping[str, Any]) -> str:
    origin = scope_metadata.get("region_origin_zyx")
    shape = scope_metadata.get("region_shape_zyx")
    if not isinstance(origin, tuple) or not isinstance(shape, tuple):
        raise ValueError("Scope metadata is missing normalized region_origin_zyx/region_shape_zyx tuples")

    z0, y0, x0 = origin
    dz, dy, dx = shape
    return f"z[{z0}:{z0 + dz}) y[{y0}:{y0 + dy}) x[{x0}:{x0 + dx})"


def build_release_contract(config: ExperimentConfig, transforms: Mapping[Any, Any]) -> Dict[str, Any]:
    requested_mode = config.pipeline.extraction.transform_application_mode
    reference_round = int(config.pipeline.registration.reference_round)
    execution_envelope = build_execution_envelope(config)
    registration_profile = derive_registration_profile(config)
    matlab_stage_contracts = _build_matlab_stage_contracts(config, execution_envelope, registration_profile)
    delivered_capability = summarize_delivered_capability(transforms, reference_round)
    scope_contract = build_scope_contract(config, transforms)
    field_semantics_contract = build_field_semantics_contract(config, transforms)

    reasons: List[str] = []
    status = "valid"
    non_mainline_execution = not is_phase1_mainline_execution_envelope(execution_envelope)
    uses_matlab_preprocessing = config.pipeline.uses_matlab_preprocessing()
    uses_matlab_global = registration_profile.get("global_provider") == "matlab"
    uses_matlab_local = registration_profile.get("local_provider") == "matlab"
    experimental_matlab_registration = uses_matlab_global or uses_matlab_local
    preprocessing_steps = [
        {
            "method": step.method,
            "provider": step.provider,
        }
        for step in config.pipeline.preprocessing.sequence
    ]
    if requested_mode == "coordinate_mapping":
        application_intent = "legacy_debug_path"
    elif experimental_matlab_registration:
        if uses_matlab_local:
            application_intent = "experimental_local_flow_provider_seam"
        elif uses_matlab_global and registration_profile.get("local_method") is not None:
            application_intent = "experimental_mixed_provider_dispatch"
        else:
            application_intent = "experimental_global_shift_only_provider_seam"
    elif non_mainline_execution:
        application_intent = "non_mainline_image_warp_review"
    else:
        application_intent = "formal_rc_mainline"

    if requested_mode == "coordinate_mapping":
        status = "debug_only"
        reasons.append(
            "coordinate_mapping is legacy diagnostic-only and excluded from formal Phase 1 RC evidence"
        )
    else:
        declared_capabilities = registration_profile["declared_transform_capabilities"]
        if "flow_3d" not in declared_capabilities:
            reasons.append(
                "image_warp mainline requires a registration profile that declares flow_3d capability"
            )
            if not non_mainline_execution:
                status = "invalid"
        if delivered_capability["flow_2d_rounds"]:
            reasons.append(
                "image_warp mainline does not support flow_2d delivery; use coordinate_mapping for legacy diagnostics"
            )
            if not non_mainline_execution:
                status = "invalid"
        missing_rounds = delivered_capability["missing_flow_3d_rounds"]
        if missing_rounds:
            reasons.append(
                f"image_warp mainline requires persisted flow_3d for every non-reference round; missing rounds: {missing_rounds}"
            )
            if not non_mainline_execution:
                status = "invalid"

    if not scope_contract["scope_valid"]:
        reasons.append(
            f"requested scope_mode {scope_contract['requested_scope_mode']!r} does not match delivered coverage {scope_contract['delivered_coverage']!r}"
        )
        status = "invalid"

    if uses_matlab_preprocessing:
        reasons.append(
            "preprocessing.sequence selects provider='matlab' as an explicit provider seam; canonical clean_data artifacts remain Python-owned and this run stays debug_only until representative benchmark recovery and production verification widen the support promise"
        )
        if status == "valid":
            status = "debug_only"

    if non_mainline_execution:
        reasons.append(
            "Execution envelope is non-mainline for Phase 1 RC: "
            f"preprocessing_backend={execution_envelope['preprocessing_backend']!r}, "
            f"registration_backend={execution_envelope['registration_backend']!r}, "
            f"accelerator={execution_envelope['accelerator']!r}"
        )
        if status == "valid":
            status = "debug_only"

    if experimental_matlab_registration:
        if uses_matlab_local:
            reasons.append(
                "registration method/provider intent uses an experimental MATLAB kernel-swap seam; "
                "Python-owned manifests/provenance remain authoritative and this run is not a Phase 1 mainline claim"
            )
        elif uses_matlab_global and registration_profile.get("local_method") is not None:
            reasons.append(
                "registration method/provider intent mixes MATLAB global alignment with native local refinement; "
                "provider_dispatch remains experimental/debug_only and Python-owned manifests/provenance remain authoritative"
            )
        else:
            reasons.append(
                "registration method/provider intent uses an experimental MATLAB global-shift-only seam; "
                "Python-owned manifests/provenance remain authoritative and local flow artifacts are not yet produced"
            )
        if status == "valid":
            status = "debug_only"

    expected_flow_3d_rounds = [
        round_id
        for round_id in delivered_capability["round_ids"]
        if round_id != reference_round
    ]
    gate0_required = requested_mode == "image_warp"
    gate0_passed = gate0_required and status == "valid"

    return {
        **scope_contract,
        "requested_intent": {
            "scope_mode": config.pipeline.scope_mode,
            "transform_application_mode": requested_mode,
            "application_intent": application_intent,
            "preprocessing_provider_mode": config.pipeline.preprocessing_provider_mode(),
            "preprocessing_steps": preprocessing_steps,
            "spot_finding_provider": config.pipeline.spot_finding.provider,
            "extraction_provider": config.pipeline.extraction.provider,
            "matlab_stage_contracts": matlab_stage_contracts,
            "backend_mode_status": registration_profile.get("backend_mode_status", "unknown"),
            "local_acceptance_mode": LOCAL_ACCEPTANCE_MODE,
            "final_corr_metric": FINAL_CORR_METRIC,
            "final_corr_diagnostic_only": True,
            "final_corr_release_gate": False,
            "registration_profile": registration_profile,
            "execution_envelope": execution_envelope,
        },
        "delivered_capability": {
            **delivered_capability,
            "expected_flow_3d_rounds": expected_flow_3d_rounds,
        },
        "field_semantics_contract": field_semantics_contract,
        "release_gate": {
            "status": status,
            "reasons": reasons,
            "gate0": {
                "required": gate0_required,
                "passed": gate0_passed,
                "sidecar_storage": FLOW_3D_SIDECAR_STORAGE,
                "expected_flow_3d_rounds": expected_flow_3d_rounds,
                "delivered_flow_3d_rounds": delivered_capability["flow_3d_rounds"],
                "missing_flow_3d_rounds": delivered_capability["missing_flow_3d_rounds"],
            },
        },
    }


def validate_execution_envelope(envelope: Dict[str, str]) -> None:
    """
    Validate that execution envelope uses explicitly supported values.
    Phase 1 RC mainline remains fixed by RC_FACTS, but non-mainline execution
    envelopes such as matlab_extracted preprocessing are still allowed as
    explicit debug/non-mainline runs.
    """
    for key, allowed_values in EXECUTION_ENVELOPE_ALLOWED_VALUES.items():
        actual_value = envelope.get(key)
        assert actual_value in allowed_values, (
            f"Execution envelope mismatch: '{key}' is '{actual_value}', "
            f"expected one of {sorted(allowed_values)}"
        )


def is_phase1_mainline_execution_envelope(envelope: Mapping[str, str]) -> bool:
    return all(envelope.get(key) == expected_value for key, expected_value in RC_FACTS.items())


def create_provenance(
    pipeline_version: str,
    environment_hash: str,
    stage_outcomes: Optional[Dict[str, Dict[str, Any]]] = None,
    release_contract: Optional[Dict[str, Any]] = None,
    config_reference: Optional[Dict[str, Any]] = None,
    software_versions: Optional[Dict[str, Any]] = None,
    hardware_context: Optional[Dict[str, Any]] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    duration_seconds: Optional[float] = None,
    execution_envelope: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Create a provenance dictionary with fixed RC facts and runtime context.
    
    Parameters
    ----------
    pipeline_version : str
        Version string of the pipeline (e.g., "0.1.0")
    environment_hash : str
        Hash representing the runtime environment
    stage_outcomes : dict, optional
        Dict mapping stage name to outcome dict with 'completed' and 'timestamp' keys
        
    Returns
    -------
    provenance : dict
        Complete provenance structure
    """
    if release_contract is None:
        raise ValueError("create_provenance requires an explicit release_contract payload")

    # Validate RC facts
    if execution_envelope is None:
        execution_envelope = dict(RC_FACTS)
    validate_execution_envelope(execution_envelope)
    
    # Create runtime context
    runtime_context: Dict[str, Any] = {
        "pipeline_version": pipeline_version,
        "execution_timestamp": datetime.now(timezone.utc).isoformat(),
        "environment_hash": environment_hash,
    }
    if start_time is not None:
        runtime_context["start_time"] = start_time
    if end_time is not None:
        runtime_context["end_time"] = end_time
    if duration_seconds is not None:
        runtime_context["duration_seconds"] = duration_seconds
    if software_versions is not None:
        runtime_context["software_versions"] = software_versions
    if hardware_context is not None:
        runtime_context["hardware_context"] = hardware_context
    if config_reference is not None:
        runtime_context["config_reference"] = config_reference
    
    # Default stage outcomes if not provided
    if stage_outcomes is None:
        stage_outcomes = {}
    
    provenance = {
        "provenance_version": PROVENANCE_VERSION,
        "execution_envelope": execution_envelope,
        "runtime_context": runtime_context,
        "stage_outcomes": stage_outcomes,
        "release_contract": release_contract,
    }
    
    return provenance


def validate_provenance_schema(provenance: Dict[str, Any]) -> None:
    """
    Validate provenance schema version and required fields.
    Raises ValueError if schema is invalid or unknown version.
    """
    # Check version
    version = provenance.get("provenance_version")
    if version is None:
        raise ValueError("Provenance missing required field: 'provenance_version'")
    
    if version != PROVENANCE_VERSION:
        raise ValueError(
            f"Unknown provenance schema version: '{version}', "
            f"expected '{PROVENANCE_VERSION}'"
        )
    
    # Check required top-level keys
    required_keys = [
        "provenance_version",
        "execution_envelope",
        "runtime_context",
        "stage_outcomes",
        "release_contract",
    ]
    for key in required_keys:
        if key not in provenance:
            raise ValueError(f"Provenance missing required field: '{key}'")
    
    # Validate execution envelope
    validate_execution_envelope(provenance.get("execution_envelope", {}))

    release_contract = provenance.get("release_contract")
    if not isinstance(release_contract, Mapping):
        raise ValueError("Provenance field 'release_contract' must be a mapping")

    requested_intent = release_contract.get("requested_intent")
    if not isinstance(requested_intent, Mapping):
        raise ValueError("Provenance field 'release_contract.requested_intent' must be a mapping")

    _ = validate_scope_contract(release_contract)

    transform_application_mode = requested_intent.get("transform_application_mode")
    if transform_application_mode not in {"coordinate_mapping", "image_warp"}:
        raise ValueError(
            "Provenance requested intent must declare transform_application_mode as 'coordinate_mapping' or 'image_warp'"
        )

    application_intent = requested_intent.get("application_intent")
    if not isinstance(application_intent, str) or not application_intent.strip():
        raise ValueError("Provenance requested intent must declare a non-empty application_intent")

    local_acceptance_mode = requested_intent.get("local_acceptance_mode")
    if local_acceptance_mode != LOCAL_ACCEPTANCE_MODE:
        raise ValueError(
            f"Provenance requested intent must declare local_acceptance_mode={LOCAL_ACCEPTANCE_MODE!r}"
        )

    final_corr_metric = requested_intent.get("final_corr_metric")
    if final_corr_metric != FINAL_CORR_METRIC:
        raise ValueError(
            f"Provenance requested intent must declare final_corr_metric={FINAL_CORR_METRIC!r}"
        )

    if requested_intent.get("final_corr_diagnostic_only") is not True:
        raise ValueError("Provenance requested intent must declare final_corr_diagnostic_only=True")

    if requested_intent.get("final_corr_release_gate") is not False:
        raise ValueError("Provenance requested intent must declare final_corr_release_gate=False")

    matlab_stage_contracts = requested_intent.get("matlab_stage_contracts")
    if matlab_stage_contracts is not None:
        _validate_matlab_stage_contracts(matlab_stage_contracts)

    requested_envelope = requested_intent.get("execution_envelope")
    if not isinstance(requested_envelope, Mapping):
        raise ValueError("Provenance requested intent must include execution_envelope")
    requested_preprocessing_backend = requested_envelope.get("preprocessing_backend")
    requested_registration_backend = requested_envelope.get("registration_backend")
    requested_accelerator = requested_envelope.get("accelerator")
    if not isinstance(requested_preprocessing_backend, str):
        raise ValueError("Provenance requested execution_envelope.preprocessing_backend must be a string")
    if not isinstance(requested_registration_backend, str):
        raise ValueError("Provenance requested execution_envelope.registration_backend must be a string")
    if not isinstance(requested_accelerator, str):
        raise ValueError("Provenance requested execution_envelope.accelerator must be a string")
    validate_execution_envelope({
        "preprocessing_backend": requested_preprocessing_backend,
        "registration_backend": requested_registration_backend,
        "accelerator": requested_accelerator,
    })

    registration_profile = requested_intent.get("registration_profile")
    if not isinstance(registration_profile, Mapping):
        raise ValueError("Provenance requested intent must include registration_profile mapping")

    profile_name = registration_profile.get("name")
    if not isinstance(profile_name, str) or not profile_name.strip():
        raise ValueError("Provenance registration profile must declare a non-empty name")

    declared_capabilities = registration_profile.get("declared_transform_capabilities")
    if not isinstance(declared_capabilities, list) or not declared_capabilities:
        raise ValueError("Provenance registration profile must declare non-empty transform capabilities")

    delivered_capability = release_contract.get("delivered_capability")
    if not isinstance(delivered_capability, Mapping):
        raise ValueError("Provenance field 'release_contract.delivered_capability' must be a mapping")

    delivered_capabilities = delivered_capability.get("delivered_transform_capabilities")
    if not isinstance(delivered_capabilities, list) or not delivered_capabilities:
        raise ValueError("Provenance delivered capability must declare delivered_transform_capabilities")

    field_semantics_contract = release_contract.get("field_semantics_contract")
    if not isinstance(field_semantics_contract, Mapping):
        raise ValueError("Provenance field 'release_contract.field_semantics_contract' must be a mapping")

    _ = _coerce_field_semantics_payload(
        field_semantics_contract.get("declared"),
        field_name="release_contract.field_semantics_contract.declared",
    )
    _ = _coerce_field_semantics_payload(
        field_semantics_contract.get("expected"),
        field_name="release_contract.field_semantics_contract.expected",
    )

    inferred = field_semantics_contract.get("inferred")
    if not isinstance(inferred, Mapping):
        raise ValueError("Provenance field_semantics_contract must include inferred mapping")
    for key in ("has_global_shift", "has_flow_2d", "has_flow_3d", "supports_composition"):
        value = inferred.get(key)
        if not isinstance(value, bool):
            raise ValueError(f"Provenance field_semantics_contract.inferred.{key} must be a boolean")

    consistency_check = field_semantics_contract.get("consistency_check")
    if not isinstance(consistency_check, Mapping):
        raise ValueError("Provenance field_semantics_contract must include consistency_check mapping")

    validation = field_semantics_contract.get("validation")
    if not isinstance(validation, Mapping):
        raise ValueError("Provenance field_semantics_contract must include validation mapping")
    validation_errors = validation.get("errors")
    if not isinstance(validation_errors, list):
        raise ValueError("Provenance field_semantics_contract.validation.errors must be a list")

    release_gate = release_contract.get("release_gate")
    if not isinstance(release_gate, Mapping):
        raise ValueError("Provenance field 'release_contract.release_gate' must be a mapping")

    gate_status = release_gate.get("status")
    if gate_status not in RELEASE_GATE_STATUSES:
        raise ValueError(
            f"Provenance release gate status must be one of {sorted(RELEASE_GATE_STATUSES)}, got {gate_status!r}"
        )

    reasons = release_gate.get("reasons")
    if not isinstance(reasons, list):
        raise ValueError("Provenance release gate must include a list of reasons")

    gate0 = release_gate.get("gate0")
    if not isinstance(gate0, Mapping):
        raise ValueError("Provenance release gate must include gate0 details")

def _build_fov_output_paths(base_dir: Path, fov_id: int) -> Dict[str, Path]:
    """Return canonical per-FOV output paths without mutating the filesystem."""

    fov_root = base_dir / f"Position{fov_id}" / "output_pystar"
    return {
        "root": fov_root,
        "transforms": fov_root / "transforms",
        "spots": fov_root / "spots",
        "extraction": fov_root / "extraction",
        "decoded": fov_root / "decoded",
        "qc": fov_root / "qc_reports",
        "cleaned": fov_root / "clean_data",
    }


def get_fov_output_structure(base_dir: Path, fov_id: int) -> Dict[str, Path]:
    """
    统一管理目录结构的逻辑。
    Good Taste: 如果你想改文件夹名，只改这里一行，全项目生效。
    """
    dirs = _build_fov_output_paths(base_dir, fov_id)
    
    for p in dirs.values():
        try:
            p.mkdir(parents=True, exist_ok=True)
        except FileExistsError:
            # Benchmark/replay bundles may intentionally replace canonical stage
            # directories (for example clean_data/) with symlinks to frozen bundle
            # inputs. Treat an existing symlink-to-directory as satisfying the
            # path contract instead of crashing while touching unrelated sibling
            # paths such as transforms/.
            if p.is_symlink() and p.exists() and p.is_dir():
                continue
            raise

    return dirs


def get_transform_manifest_path(base_dir: Path, fov_id: int) -> Path:
    paths = _build_fov_output_paths(base_dir, fov_id)
    return paths["transforms"] / f"transforms_fov_{fov_id}.npy"


def get_provenance_summary_path(base_dir: Path, fov_id: int) -> Path:
    paths = _build_fov_output_paths(base_dir, fov_id)
    return paths["qc"] / "provenance_summary.md"


def get_flow_3d_sidecar_filename(fov_id: int, round_id: int) -> str:
    return f"transforms_fov_{fov_id}_round_{round_id}_flow_3d.npy"


def _pick_first(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            value = mapping[key]
            if value is not None:
                return value
    return None


def _require_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Provenance summary requires mapping field '{field_name}'")
    return cast(Mapping[str, Any], value)


def _require_value(value: Any, field_name: str) -> Any:
    if value is None:
        raise ValueError(f"Provenance summary requires field '{field_name}'")
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"Provenance summary field '{field_name}' must not be empty")
    return value


def _require_first(mapping: Mapping[str, Any], field_name: str, *keys: str) -> Any:
    return _require_value(_pick_first(mapping, *keys), field_name)


def _parse_iso_datetime(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    if not isinstance(value, str) or not value.strip():
        return None

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _format_timestamp(value: Any) -> str:
    if value is None:
        return "Not recorded in formal provenance"
    parsed = _parse_iso_datetime(value)
    if parsed is None:
        return str(value)
    return parsed.isoformat()


def _require_timestamp_value(value: Any, field_name: str) -> Any:
    required_value = _require_value(value, field_name)
    if _parse_iso_datetime(required_value) is None:
        raise ValueError(
            f"Provenance summary field '{field_name}' must be an ISO 8601 timestamp, got {required_value!r}"
        )
    return required_value


def _format_duration_value(duration_value: Any, start_value: Any, end_value: Any) -> str:
    if duration_value is not None:
        try:
            total_seconds = int(round(float(duration_value)))
        except (TypeError, ValueError):
            return str(duration_value)

        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        parts = []
        if hours:
            parts.append(f"{hours}h")
        if hours or minutes:
            parts.append(f"{minutes}m")
        parts.append(f"{seconds}s")
        return " ".join(parts)

    start_dt = _parse_iso_datetime(start_value)
    end_dt = _parse_iso_datetime(end_value)
    if start_dt is None or end_dt is None:
        return "Not recorded in formal provenance"

    return _format_duration_value((end_dt - start_dt).total_seconds(), None, None)


def _require_duration_value(duration_value: Any, start_value: Any, end_value: Any) -> Any:
    if duration_value is not None:
        try:
            float(duration_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Provenance summary field 'runtime_context.duration_seconds' must be numeric when present"
            ) from exc
        return duration_value

    start_dt = _parse_iso_datetime(start_value)
    end_dt = _parse_iso_datetime(end_value)
    if start_dt is None or end_dt is None:
        raise ValueError(
            "Provenance summary requires runtime_context.duration_seconds or parseable start/end timestamps"
        )
    return (end_dt - start_dt).total_seconds()


def _format_scalar(value: Any) -> str:
    if value is None:
        return "Not recorded in formal provenance"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        rendered = f"{value:.4f}".rstrip("0").rstrip(".")
        return rendered or "0"
    if isinstance(value, (list, tuple, set)):
        return ", ".join(_format_scalar(item) for item in value)
    if isinstance(value, Mapping):
        return "; ".join(f"{key}={_format_scalar(item)}" for key, item in value.items())
    return str(value)


def _format_memory_value(hardware_context: Mapping[str, Any], runtime_context: Mapping[str, Any]) -> str:
    memory_bytes = _pick_first(
        hardware_context,
        "memory_available_bytes",
        "available_memory_bytes",
        "memory_bytes",
    )
    if memory_bytes is None:
        memory_bytes = _pick_first(
            runtime_context,
            "memory_available_bytes",
            "available_memory_bytes",
            "memory_bytes",
        )
    if memory_bytes is not None:
        try:
            gib = float(memory_bytes) / (1024 ** 3)
        except (TypeError, ValueError):
            return _format_scalar(memory_bytes)
        return f"{gib:.2f} GiB"

    memory_gb = _pick_first(
        hardware_context,
        "memory_available_gb",
        "available_memory_gb",
        "memory_gb",
    )
    if memory_gb is None:
        memory_gb = _pick_first(
            runtime_context,
            "memory_available_gb",
            "available_memory_gb",
            "memory_gb",
        )
    if memory_gb is not None:
        return f"{_format_scalar(memory_gb)} GB"

    return "Not recorded in formal provenance"


def _extract_software_versions(runtime_context: Mapping[str, Any]) -> Dict[str, Any]:
    versions: Dict[str, Any] = {}
    software_versions = runtime_context.get("software_versions")
    if isinstance(software_versions, Mapping):
        versions.update(software_versions)

    pipeline_version = runtime_context.get("pipeline_version")
    if pipeline_version is not None:
        versions.setdefault("PyStar", pipeline_version)

    for key, value in runtime_context.items():
        if key == "pipeline_version" or not key.endswith("_version") or value is None:
            continue
        component = key.removesuffix("_version").replace("_", " ").title()
        versions.setdefault(component, value)

    return versions


def _extract_config_reference(runtime_context: Mapping[str, Any]) -> Dict[str, Any]:
    config_reference = runtime_context.get("config_reference")
    if not isinstance(config_reference, Mapping):
        config_reference = {}

    key_parameters = _pick_first(
        config_reference,
        "key_parameters",
        "parameter_summary",
        "parameters",
    )
    if key_parameters is None:
        key_parameters = _pick_first(
            runtime_context,
            "key_parameters",
            "parameter_summary",
            "parameters",
        )

    return {
        "config_file": _pick_first(
            config_reference,
            "config_path",
            "config_file",
            "path",
        ) or _pick_first(runtime_context, "config_path", "config_file"),
        "config_hash": _pick_first(
            config_reference,
            "config_hash",
            "config_sha256",
            "hash",
        ) or _pick_first(runtime_context, "config_hash", "config_sha256"),
        "key_parameters": key_parameters,
    }


def _extract_round_summary(
    manifest_payload: Mapping[Any, Any],
    provenance: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    runtime_context = _require_mapping(provenance.get("runtime_context"), "runtime_context")
    stage_outcomes = _require_mapping(provenance.get("stage_outcomes"), "stage_outcomes")

    raw_round_summary = None
    for container in (stage_outcomes, runtime_context):
        raw_round_summary = _pick_first(
            container,
            "round_summary",
            "rounds",
            "round_statuses",
            "round_outcomes",
        )
        if raw_round_summary is not None:
            break

    if raw_round_summary is None:
        raise ValueError(
            "Provenance summary requires round-level execution summary under stage_outcomes.round_summary"
        )

    summary_by_round: Dict[int, Dict[str, Any]] = {}
    if isinstance(raw_round_summary, Mapping):
        items = raw_round_summary.items()
    elif isinstance(raw_round_summary, list):
        items = []
        for entry in raw_round_summary:
            if not isinstance(entry, Mapping):
                raise ValueError("Round summary entries must be mappings")
            round_key = _pick_first(entry, "round", "round_id")
            items.append((round_key, entry))
    else:
        raise ValueError("Round summary must be a mapping or a list of mappings")

    for round_key, round_info in items:
        if not isinstance(round_info, Mapping):
            raise ValueError(f"Round summary entry for round {round_key!r} must be a mapping")
        try:
            round_id = int(round_key)
        except (TypeError, ValueError):
            raise ValueError(f"Round summary key must be numeric, got {round_key!r}") from None

        status = round_info.get("status")
        if status is None:
            completed = round_info.get("completed")
            if completed is True:
                status = "completed"
            elif completed is False:
                status = "failed"

        summary_by_round[round_id] = {
            "status": _require_value(status, f"stage_outcomes.round_summary[{round_id}].status"),
            "start_time": _require_timestamp_value(
                _pick_first(
                    round_info,
                    "start_time",
                    "started_at",
                    "execution_start_time",
                ),
                f"stage_outcomes.round_summary[{round_id}].start_time",
            ),
            "end_time": _require_timestamp_value(
                _pick_first(
                    round_info,
                    "end_time",
                    "ended_at",
                    "execution_end_time",
                    "timestamp",
                ),
                f"stage_outcomes.round_summary[{round_id}].end_time",
            ),
        }

    manifest_rounds: List[int] = []
    for round_key, transform_data in manifest_payload.items():
        if not _is_round_transform_entry(transform_data):
            continue
        try:
            manifest_rounds.append(int(round_key))
        except (TypeError, ValueError):
            raise ValueError(f"Transform manifest round key must be numeric, got {round_key!r}") from None

    if not manifest_rounds:
        raise ValueError("Provenance summary requires at least one persisted round in the transform manifest")

    round_rows: List[Dict[str, Any]] = []
    for round_id in sorted(manifest_rounds):
        if round_id not in summary_by_round:
            raise ValueError(
                f"Provenance summary missing round-level execution summary for persisted round {round_id}"
            )
        info = summary_by_round[round_id]
        round_rows.append({
            "round": round_id,
            "status": info["status"],
            "start_time": info["start_time"],
            "end_time": info["end_time"],
        })

    return round_rows


def generate_field_semantics_summary(field_semantics_contract: Mapping[str, Any]) -> str:
    declared = _coerce_field_semantics_payload(
        field_semantics_contract.get("declared"),
        field_name="field_semantics_contract.declared",
    )
    expected = _coerce_field_semantics_payload(
        field_semantics_contract.get("expected"),
        field_name="field_semantics_contract.expected",
    )
    inferred = _require_mapping(
        field_semantics_contract.get("inferred"),
        "field_semantics_contract.inferred",
    )
    consistency_check = _require_mapping(
        field_semantics_contract.get("consistency_check"),
        "field_semantics_contract.consistency_check",
    )
    validation = _require_mapping(
        field_semantics_contract.get("validation"),
        "field_semantics_contract.validation",
    )

    lines = [
        "## Field Semantics",
        "",
        "| Field | Registration Declared | Extraction Expected |",
        "|-------|----------------------|---------------------|",
        f"| Status | {_format_scalar(declared.get('status'))} | {_format_scalar(expected.get('status'))} |",
        f"| Representation | {_format_scalar(declared.get('representation'))} | {_format_scalar(expected.get('representation'))} |",
        f"| Composition | {_format_scalar(declared.get('composition'))} | {_format_scalar(expected.get('composition'))} |",
    ]

    lines.extend([
        "",
        "### Inferred from Delivered Transforms",
        "| Check | Value |",
        "|-------|-------|",
        f"| Has global shift | {_format_scalar(inferred.get('has_global_shift'))} |",
        f"| Has flow 2D | {_format_scalar(inferred.get('has_flow_2d'))} |",
        f"| Has flow 3D | {_format_scalar(inferred.get('has_flow_3d'))} |",
        f"| Supports composition | {_format_scalar(inferred.get('supports_composition'))} |",
    ])

    lines.extend([
        "",
        "### Consistency Check",
        "| Check | Value |",
        "|-------|-------|",
        f"| Registration vs Extraction | {_format_scalar(consistency_check.get('registration_vs_pipeline'))} |",
        f"| Persisted rounds vs Declared | {_format_scalar(consistency_check.get('persisted_rounds_vs_declared'))} |",
        f"| Rounds with _semantics | {_format_scalar(consistency_check.get('rounds_with_semantics'))} |",
        f"| Rounds missing _semantics | {_format_scalar(consistency_check.get('rounds_missing_semantics'))} |",
        f"| Rounds mismatching _semantics | {_format_scalar(consistency_check.get('rounds_mismatching_semantics'))} |",
    ])

    validation_errors = validation.get("errors")
    if isinstance(validation_errors, list) and validation_errors:
        lines.extend(["", "### Validation Notes"])
        for error in validation_errors:
            lines.append(f"- {_format_scalar(error)}")

    expected_status = expected.get("status", "unknown")
    if expected_status == "settled":
        interpretation = (
            "Field semantics are recorded as settled. Registration and extraction are expected to follow a stable, validated contract."
        )
    elif expected_status == "provisional":
        interpretation = (
            "Field semantics are provisional. The current representation/composition contract is a working hypothesis and should remain benchmark-auditable."
        )
    else:
        interpretation = (
            "Field semantics are unknown. Registration and extraction metadata stay explicit, but semantic confidence is not yet closed."
        )

    lines.extend(["", "### Interpretation", f"- {interpretation}"])
    return "\n".join(lines)


def build_provenance_summary_markdown(
    fov_id: int,
    manifest_payload: Mapping[Any, Any],
    provenance: Mapping[str, Any],
) -> str:
    runtime_context = _require_mapping(provenance.get("runtime_context"), "runtime_context")
    execution_envelope = _require_mapping(provenance.get("execution_envelope"), "execution_envelope")
    release_contract = _require_mapping(provenance.get("release_contract"), "release_contract")
    requested_intent = _require_mapping(
        release_contract.get("requested_intent"),
        "release_contract.requested_intent",
    )
    registration_profile = _require_mapping(
        requested_intent.get("registration_profile"),
        "release_contract.requested_intent.registration_profile",
    )
    delivered_capability = _require_mapping(
        release_contract.get("delivered_capability"),
        "release_contract.delivered_capability",
    )
    field_semantics_contract = _require_mapping(
        release_contract.get("field_semantics_contract"),
        "release_contract.field_semantics_contract",
    )
    release_gate = _require_mapping(
        release_contract.get("release_gate"),
        "release_contract.release_gate",
    )
    scope_contract = validate_scope_contract(release_contract)

    software_versions = _extract_software_versions(runtime_context)
    if "PyStar" not in software_versions:
        raise ValueError("Provenance summary requires runtime_context.pipeline_version")
    if len(software_versions) < 2:
        raise ValueError(
            "Provenance summary requires at least one dependency version in runtime_context.software_versions"
        )

    config_reference = _extract_config_reference(runtime_context)
    hardware_context = runtime_context.get("hardware_context")
    if not isinstance(hardware_context, Mapping):
        hardware_context = runtime_context.get("hardware")
    hardware_context = _require_mapping(hardware_context, "runtime_context.hardware_context")

    start_time = _require_timestamp_value(_pick_first(
        runtime_context,
        "start_time",
        "processing_start_time",
        "execution_start_time",
        "started_at",
    ), "runtime_context.start_time")
    end_time = _require_timestamp_value(_pick_first(
        runtime_context,
        "end_time",
        "processing_end_time",
        "execution_end_time",
        "ended_at",
    ), "runtime_context.end_time")
    duration = _require_duration_value(_pick_first(
        runtime_context,
        "duration_seconds",
        "elapsed_seconds",
        "duration",
    ), start_time, end_time)
    execution_timestamp = _pick_first(runtime_context, "execution_timestamp", "recorded_at")
    if execution_timestamp is not None:
        execution_timestamp = _require_timestamp_value(execution_timestamp, "runtime_context.execution_timestamp")

    cpu_count = _require_first(
        hardware_context,
        "runtime_context.hardware_context.cpu_count",
        "cpu_count",
        "cpus_available",
        "cpu_available",
    )
    memory_value = _format_memory_value(hardware_context, runtime_context)
    if memory_value == "Not recorded in formal provenance":
        raise ValueError(
            "Provenance summary requires runtime_context.hardware_context memory availability metadata"
        )

    config_file = _require_value(config_reference.get("config_file"), "runtime_context.config_reference.config_path")
    config_hash = _require_value(config_reference.get("config_hash"), "runtime_context.config_reference.config_hash")
    key_parameters = _require_value(
        config_reference.get("key_parameters"),
        "runtime_context.config_reference.key_parameters",
    )
    if isinstance(key_parameters, Mapping) and not key_parameters:
        raise ValueError("Provenance summary key parameter mapping must not be empty")

    lines = [
        "# Provenance Summary",
        "",
        "## Execution Overview",
        f"- **FOV**: Position{fov_id}",
        f"- **Start Time**: {_format_timestamp(start_time)}",
        f"- **End Time**: {_format_timestamp(end_time)}",
        f"- **Duration**: {_format_duration_value(duration, start_time, end_time)}",
    ]
    if execution_timestamp is not None:
        lines.append(f"- **Recorded At**: {_format_timestamp(execution_timestamp)}")

    environment_hash = runtime_context.get("environment_hash")
    if environment_hash is not None:
        lines.append(f"- **Environment Hash**: {_format_scalar(environment_hash)}")

    lines.extend(["", "## Software Versions", "| Component | Version |", "|-----------|---------|"])
    for component, version in software_versions.items():
        lines.append(f"| {component} | {_format_scalar(version)} |")

    lines.extend(["", "## Execution Environment", "| Resource | Value |", "|----------|-------|"])
    lines.append(f"| CPUs Available | {_format_scalar(cpu_count)} |")
    lines.append(f"| Memory Available | {memory_value} |")
    lines.append(f"| Accelerator | {_format_scalar(execution_envelope.get('accelerator'))} |")

    lines.extend(["", "## Backend Configuration", "| Stage | Backend |", "|-------|---------|"])
    lines.append(
        f"| Preprocessing | {_format_scalar(execution_envelope.get('preprocessing_backend'))} |"
    )
    lines.append(
        f"| Registration | {_format_scalar(execution_envelope.get('registration_backend'))} |"
    )

    registration_backend_details = runtime_context.get("registration_backend_details")
    if isinstance(registration_backend_details, Mapping):
        lines.extend(["", "## Registration Backend Runtime", "| Field | Value |", "|-------|-------|"])
        lines.append(f"| Mode Status | {_format_scalar(registration_backend_details.get('mode_status'))} |")
        lines.append(f"| Runtime Path | {_format_scalar(registration_backend_details.get('runtime_path'))} |")
        lines.append(f"| Runtime Manifest | {_format_scalar(registration_backend_details.get('runtime_manifest'))} |")
        lines.append(f"| Entry Point | {_format_scalar(registration_backend_details.get('entrypoint'))} |")
        boundary_summary = registration_backend_details.get("boundary_instrumentation_summary")
        if isinstance(boundary_summary, Mapping):
            aggregate_seam_costs = boundary_summary.get("aggregate_seam_costs_ms")
            session_lifecycle_summary = boundary_summary.get("session_lifecycle_summary")
            lines.extend(["", "### Registration Boundary Instrumentation", "| Field | Value |", "|-------|-------|"])
            lines.append(f"| Call Count | {_format_scalar(boundary_summary.get('call_count'))} |")
            lines.append(f"| Engine Reused Calls | {_format_scalar(boundary_summary.get('engine_reused_calls'))} |")
            lines.append(f"| Total Boundary Duration (ms) | {_format_scalar(boundary_summary.get('total_duration_ms'))} |")
            if isinstance(aggregate_seam_costs, Mapping):
                for field_name in (
                    "engine_bootstrap_ms",
                    "runtime_file_validation_ms",
                    "input_staging_ms",
                    "matlab_call_ms",
                    "result_validation_ms",
                    "canonical_persistence_ms",
                    "teardown_ms",
                ):
                    lines.append(
                        f"| {field_name} | {_format_scalar(aggregate_seam_costs.get(field_name))} |"
                    )
            if isinstance(session_lifecycle_summary, Mapping):
                aggregate_counts = session_lifecycle_summary.get("aggregate_counts")
                aggregate_timing_ms = session_lifecycle_summary.get("aggregate_timing_ms")
                lines.extend(["", "### Registration Session Lifecycle", "| Field | Value |", "|-------|-------|"])
                lines.append(f"| Session Count | {_format_scalar(session_lifecycle_summary.get('session_count'))} |")
                lines.append(f"| Sessions With Bootstrap | {_format_scalar(session_lifecycle_summary.get('sessions_with_bootstrap'))} |")
                lines.append(f"| Sessions With Reuse | {_format_scalar(session_lifecycle_summary.get('sessions_with_reuse'))} |")
                lines.append(
                    f"| Sessions With Teardown Warning | {_format_scalar(session_lifecycle_summary.get('sessions_with_teardown_warning'))} |"
                )
                if isinstance(aggregate_counts, Mapping):
                    for field_name in (
                        "engine_bootstrap_count",
                        "engine_reuse_count",
                        "runtime_file_validation_count",
                        "runtime_file_validation_reuse_count",
                        "addpath_call_count",
                        "teardown_count",
                        "teardown_warning_count",
                    ):
                        lines.append(f"| {field_name} | {_format_scalar(aggregate_counts.get(field_name))} |")
                if isinstance(aggregate_timing_ms, Mapping):
                    for field_name in (
                        "configure_environment_ms",
                        "engine_module_import_ms",
                        "factory_resolution_ms",
                        "runtime_file_validation_ms",
                        "start_matlab_ms",
                        "addpath_ms",
                        "engine_bootstrap_ms",
                        "teardown_ms",
                    ):
                        lines.append(f"| {field_name} | {_format_scalar(aggregate_timing_ms.get(field_name))} |")

    declared_capabilities = _require_value(
        registration_profile.get("declared_transform_capabilities"),
        "release_contract.requested_intent.registration_profile.declared_transform_capabilities",
    )
    delivered_capabilities = _require_value(
        delivered_capability.get("delivered_transform_capabilities"),
        "release_contract.delivered_capability.delivered_transform_capabilities",
    )
    release_status = _require_value(
        release_gate.get("status"),
        "release_contract.release_gate.status",
    )
    requested_scope_mode = scope_contract["requested_scope_mode"]
    delivered_coverage = scope_contract["delivered_coverage"]
    scope_valid = scope_contract["scope_valid"]
    scope_status = scope_contract["scope_status"]
    round_scope_metadata = _extract_consistent_round_scope_metadata(
        manifest_payload,
        require_all_rounds=delivered_coverage == "tile_local",
    )

    lines.extend(["", "## Scope Contract", "| Field | Value |", "|-------|-------|"])
    lines.append(f"| Requested Scope | {_format_scalar(requested_scope_mode)} |")
    lines.append(f"| Delivered Coverage | {_format_scalar(delivered_coverage)} |")
    lines.append(f"| Scope Valid | {_format_scalar(scope_valid)} |")
    lines.append(f"| Scope Status | {_format_scalar(scope_status)} |")
    if round_scope_metadata is not None:
        lines.append(f"| Scope Region | {_format_scope_region(round_scope_metadata)} |")
        if round_scope_metadata["coverage_mode"] == "tile_local":
            tile_grid_shape = round_scope_metadata.get("tile_grid_shape_yx")
            tile_index = round_scope_metadata.get("tile_index")
            if isinstance(tile_grid_shape, tuple) and tile_index is not None:
                total_tiles = int(tile_grid_shape[0] * tile_grid_shape[1])
                lines.append(f"| Tile Index | {_format_scalar(f'{tile_index}/{total_tiles}')} |")

    if scope_valid:
        scope_narrative = (
            f"Registration requested {requested_scope_mode} scope and delivered {delivered_coverage}; extraction can treat this artifact as scope-valid."
        )
    else:
        scope_narrative = (
            f"Registration requested {requested_scope_mode} scope but delivered {delivered_coverage}; this artifact is not valid evidence for the requested scope."
        )
    if round_scope_metadata is not None and round_scope_metadata["coverage_mode"] == "tile_local":
        scope_narrative += f" Runtime tile coverage is {_format_scope_region(round_scope_metadata)}."
    lines.extend(["", "### Scope Narrative", f"- {scope_narrative}"])

    lines.extend(["", "## Release Contract", "| Field | Value |", "|-------|-------|"])
    lines.append(f"| Scope Mode | {_format_scalar(requested_intent.get('scope_mode'))} |")
    lines.append(
        f"| Transform Application Mode | {_format_scalar(requested_intent.get('transform_application_mode'))} |"
    )
    lines.append(f"| Application Intent | {_format_scalar(requested_intent.get('application_intent'))} |")
    backend_mode_status = requested_intent.get('backend_mode_status')
    if backend_mode_status is not None:
        lines.append(f"| Backend Mode Status | {_format_scalar(backend_mode_status)} |")
    lines.append(f"| Local Acceptance Mode | {_format_scalar(requested_intent.get('local_acceptance_mode'))} |")
    lines.append(f"| Persisted final_corr Metric | {_format_scalar(requested_intent.get('final_corr_metric'))} |")
    lines.append(f"| final_corr Diagnostic Only | {_format_scalar(requested_intent.get('final_corr_diagnostic_only'))} |")
    lines.append(f"| final_corr Release Gate | {_format_scalar(requested_intent.get('final_corr_release_gate'))} |")
    lines.append(f"| Registration Profile | {_format_scalar(registration_profile.get('name'))} |")
    lines.append(f"| Declared Transform Capability | {_format_scalar(declared_capabilities)} |")
    lines.append(f"| Delivered Transform Capability | {_format_scalar(delivered_capabilities)} |")
    lines.append(f"| Release Status | {_format_scalar(release_status)} |")

    matlab_stage_contracts = requested_intent.get("matlab_stage_contracts")
    if isinstance(matlab_stage_contracts, Mapping) and matlab_stage_contracts:
        lines.extend([
            "",
            "### MATLAB Stage Contracts",
            "| Stage | Declared Intent | MATLAB Requested | Support Status | Failure Contract | Promotion Blockers | Python-Owned Artifacts |",
            "|-------|-----------------|------------------|----------------|------------------|--------------------|------------------------|",
        ])
        for stage_name in ("preprocessing", "registration", "spot_finding", "extraction"):
            contract = matlab_stage_contracts.get(stage_name)
            if not isinstance(contract, Mapping):
                continue
            lines.append(
                "| {stage} | {intent} | {requested} | {status} | {failure_contract} | {promotion_blockers} | {artifacts} |".format(
                    stage=stage_name,
                    intent=_format_scalar(contract.get("declared_intent")),
                    requested=_format_scalar(contract.get("matlab_requested")),
                    status=_format_scalar(contract.get("current_support_status")),
                    failure_contract=_format_scalar(contract.get("failure_contract")),
                    promotion_blockers=_format_scalar(contract.get("promotion_blockers")),
                    artifacts=_format_scalar(contract.get("python_owned_artifacts")),
                )
            )
        lines.extend([
            "",
            "### MATLAB Stage Contract Note",
            "- These rows describe stage-promotion / support state, not bundle-level transform legality.",
            "- A bundle-level `Release Status = valid` does not promote any MATLAB-requested stage beyond `debug_only`; promotion still requires representative benchmark recovery and production verification.",
        ])

    lines.extend([
        "",
        "### Diagnostic Metric Note",
        f"- Persisted `final_corr` stores {_format_scalar(requested_intent.get('final_corr_metric'))} for runtime QC and rollback diagnostics only.",
        "- It is diagnostic-only and must not be interpreted as a formal release gate.",
    ])

    lines.extend(["", *generate_field_semantics_summary(field_semantics_contract).splitlines()])

    gate0 = release_gate.get("gate0")
    if isinstance(gate0, Mapping):
        lines.extend(["", "## Gate 0 Contract", "| Check | Value |", "|-------|-------|"])
        lines.append(f"| Required | {_format_scalar(gate0.get('required'))} |")
        lines.append(f"| Passed | {_format_scalar(gate0.get('passed'))} |")
        lines.append(f"| Sidecar Storage | {_format_scalar(gate0.get('sidecar_storage'))} |")
        lines.append(
            f"| Expected 3D Flow Rounds | {_format_scalar(gate0.get('expected_flow_3d_rounds'))} |"
        )
        lines.append(
            f"| Delivered 3D Flow Rounds | {_format_scalar(gate0.get('delivered_flow_3d_rounds'))} |"
        )
        lines.append(
            f"| Missing 3D Flow Rounds | {_format_scalar(gate0.get('missing_flow_3d_rounds'))} |"
        )

    gate_reasons = release_gate.get("reasons")
    if isinstance(gate_reasons, list) and gate_reasons:
        lines.extend(["", "### Release Gate Reasons"])
        for reason in gate_reasons:
            lines.append(f"- {_format_scalar(reason)}")

    lines.extend([
        "",
        "## Configuration Reference",
        f"- **Config File**: {_format_scalar(config_file)}",
        f"- **Config Hash**: {_format_scalar(config_hash)}",
    ])

    if isinstance(key_parameters, Mapping) and key_parameters:
        lines.extend(["", "### Key Parameters", "| Parameter | Value |", "|-----------|-------|"])
        for key, value in key_parameters.items():
            lines.append(f"| {key} | {_format_scalar(value)} |")
    else:
        lines.extend(["", f"- **Key Parameters**: {_format_scalar(key_parameters)}"])

    round_rows = _extract_round_summary(manifest_payload, provenance)
    lines.extend([
        "",
        "## Round-Level Summary",
        "| Round | Status | Start Time | End Time |",
        "|-------|--------|------------|----------|",
    ])
    for row in round_rows:
        lines.append(
            "| {round} | {status} | {start_time} | {end_time} |".format(
                round=row["round"],
                status=_format_scalar(row["status"]),
                start_time=_format_timestamp(row["start_time"]),
                end_time=_format_timestamp(row["end_time"]),
            )
        )

    return "\n".join(lines) + "\n"


def write_provenance_summary(base_dir: Path, fov_id: int, summary_markdown: str) -> Path:
    _ = get_fov_output_structure(base_dir, fov_id)
    summary_path = get_provenance_summary_path(base_dir, fov_id)
    temp_path = summary_path.with_suffix(".md.tmp")
    if summary_path.exists():
        summary_path.unlink()
    temp_path.write_text(summary_markdown, encoding="utf-8")
    temp_path.replace(summary_path)
    return summary_path


def _is_round_transform_entry(value: object) -> bool:
    return isinstance(value, dict) and "global_shift_3d" in value


def _ensure_round_field_semantics(
    round_payload: MutableMapping[str, Any],
    *,
    field_name: str,
    recorded_at: Optional[str] = None,
) -> None:
    round_payload["_semantics"] = _coerce_field_semantics_payload(
        round_payload.get("_semantics"),
        field_name=field_name,
        recorded_at=recorded_at,
    )


def persist_flow_3d_sidecar(
    base_dir: Path,
    fov_id: int,
    round_id: int,
    flow_3d: NDArray[Any],
) -> Dict[str, Any]:
    """
    Persist one dense round-level `flow_3d` sidecar immediately and return the
    manifest descriptor.

    This supports the lower-peak-memory registration path where each round's
    3D flow can be written and released before the full manifest is finalized.
    """
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


def _is_direct_sidecar_filename(sidecar_rel: str) -> bool:
    sidecar_rel_path = Path(sidecar_rel)
    return (
        sidecar_rel not in {".", ".."}
        and not sidecar_rel_path.is_absolute()
        and sidecar_rel_path.parent == Path('.')
        and "/" not in sidecar_rel
        and "\\" not in sidecar_rel
    )


def save_transform_manifest(
    base_dir: Path, 
    fov_id: int, 
    transforms: Dict[Any, Any],
    provenance: Optional[Dict[str, Any]] = None
) -> Path:
    """
    持久化 transform manifest，并把大的 3D flow 写到 round-level sidecar。

    主 manifest 仍保持 `transforms_fov_{fov_id}.npy` 这一既有入口，
    每个 round 的 `flow_3d` 字段在磁盘上存为 sidecar 描述符，
    由下游 loader 再物化回 ndarray。
    
    Parameters
    ----------
    base_dir : Path
        Output base directory
    fov_id : int
        FOV identifier
    transforms : dict
        Transform data dictionary
    provenance : dict, optional
        Execution envelope provenance data. If provided, will be validated
        and stored under key '_provenance' in the manifest.
    """
    manifest_model = TransformManifest.from_legacy(transforms, fov_id=fov_id)
    transforms = cast(Dict[Any, Any], manifest_model.to_legacy())

    if provenance is not None:
        provenance = copy.deepcopy(provenance)
        _backfill_requested_intent_diagnostic_defaults(provenance)
        _backfill_field_semantics_contract(provenance, transforms)
        validate_provenance_schema(provenance)
    
    _ = get_fov_output_structure(base_dir, fov_id)
    manifest_path = get_transform_manifest_path(base_dir, fov_id)
    transforms_dir = manifest_path.parent

    manifest_payload: Dict[Any, Any] = {}
    referenced_sidecars: set[Path] = set()
    sidecar_payloads: Dict[Path, NDArray[Any]] = {}

    for round_key, transform_data in transforms.items():
        if not _is_round_transform_entry(transform_data):
            manifest_payload[round_key] = transform_data
            continue

        try:
            round_id = int(round_key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Round transform entry must use a numeric key, got {round_key!r}") from exc

        round_payload = dict(transform_data)
        _ensure_round_field_semantics(
            round_payload,
            field_name=f"transform round {round_id} _semantics",
        )
        flow_3d = round_payload.get("flow_3d")

        if isinstance(flow_3d, Mapping):
            storage_kind = flow_3d.get("storage")
            if storage_kind != FLOW_3D_SIDECAR_STORAGE:
                raise ValueError(
                    f"Unsupported flow_3d storage kind for round {round_id}: {storage_kind!r}"
                )

            sidecar_rel = flow_3d.get("path")
            if not isinstance(sidecar_rel, str) or not sidecar_rel:
                raise ValueError(
                    f"flow_3d manifest for round {round_id} is missing a valid sidecar path"
                )

            if not _is_direct_sidecar_filename(sidecar_rel):
                raise ValueError(
                    f"flow_3d sidecar path must be a direct filename under transforms/: {sidecar_rel}"
                )

            sidecar_path = transforms_dir / sidecar_rel
            if not sidecar_path.exists():
                raise FileNotFoundError(
                    f"flow_3d sidecar referenced by transform manifest is missing: {sidecar_path}"
                )

            referenced_sidecars.add(sidecar_path)
            round_payload["flow_3d"] = dict(flow_3d)
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
        _backfill_field_semantics_contract(provenance, manifest_payload)
        validate_provenance_schema(provenance)
        _validate_round_scope_contract_alignment(manifest_payload, provenance["release_contract"])
        manifest_payload["_provenance"] = provenance
        manifest_payload["_contract"] = provenance["release_contract"]

    manifest_payload = cast(
        Dict[Any, Any],
        TransformManifest.from_legacy(
            manifest_payload,
            fov_id=fov_id,
            validate_release_contract=provenance is not None,
        ).to_legacy(),
    )

    summary_markdown = None
    if provenance is not None:
        summary_markdown = build_provenance_summary_markdown(fov_id, manifest_payload, provenance)

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


def _validate_flow_3d_sidecar_descriptor(
    flow_3d: Mapping[str, Any],
    *,
    round_key: Any,
    transforms_dir: Path,
) -> tuple[Dict[str, Any], Path]:
    storage_kind = flow_3d.get("storage")
    if storage_kind != FLOW_3D_SIDECAR_STORAGE:
        raise ValueError(f"Unsupported flow_3d storage kind for round {round_key}: {storage_kind!r}")

    sidecar_rel = flow_3d.get("path")
    if not isinstance(sidecar_rel, str) or not sidecar_rel:
        raise ValueError(f"flow_3d manifest for round {round_key} is missing a valid sidecar path")

    if not _is_direct_sidecar_filename(sidecar_rel):
        raise ValueError(
            f"flow_3d sidecar path must be a direct filename under transforms/: {sidecar_rel}"
        )

    sidecar_path = transforms_dir / sidecar_rel
    if not sidecar_path.exists():
        raise FileNotFoundError(
            f"flow_3d sidecar referenced by transform manifest is missing: {sidecar_path}"
        )

    return dict(flow_3d), sidecar_path


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
) -> Dict[str, Any]:
    """Materialize one round transform entry from persisted manifest metadata."""
    manifest_path = get_transform_manifest_path(base_dir, fov_id)
    transforms_dir = manifest_path.parent

    round_payload = dict(transform_data)
    round_payload.setdefault("flow_2d", None)
    _ensure_round_field_semantics(
        round_payload,
        field_name=f"transform round {round_id} _semantics",
    )

    flow_3d = round_payload.get("flow_3d")
    if isinstance(flow_3d, Mapping):
        descriptor, sidecar_path = _validate_flow_3d_sidecar_descriptor(
            flow_3d,
            round_key=round_id,
            transforms_dir=transforms_dir,
        )
        round_payload["flow_3d"] = _load_flow_3d_sidecar_array(
            descriptor,
            round_key=round_id,
            sidecar_path=sidecar_path,
        )
    elif flow_3d is not None and not isinstance(flow_3d, np.ndarray):
        raise ValueError(f"Unsupported flow_3d payload type for round {round_id}: {type(flow_3d)}")
    else:
        round_payload.setdefault("flow_3d", None)

    return round_payload


def load_transform_manifest(
    base_dir: Path,
    fov_id: int,
    load_provenance: bool = False,
    *,
    hydrate_flow_3d: bool = True,
) -> Dict[Any, Any]:
    """
    加载 transform manifest；默认把 round-level `flow_3d` sidecar 物化回 ndarray。
    
    Parameters
    ----------
    base_dir : Path
        Output base directory
    fov_id : int
        FOV identifier
    load_provenance : bool
        If True, include provenance in returned dict under key '_provenance'.
        If False (default), provenance is validated but not returned (backward compatibility).
    hydrate_flow_3d : bool
        If True (default), eagerly materialize persisted `flow_3d` sidecars.
        If False, keep validated sidecar descriptors so callers can hydrate one
        round at a time.
        
    Returns
    -------
    dict
        Transform data. If load_provenance=True, includes '_provenance' key.
        When ``hydrate_flow_3d=False``, non-reference rounds may keep
        descriptor-style `flow_3d` payloads.
    """
    manifest_path = get_transform_manifest_path(base_dir, fov_id)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Transform manifest not found: {manifest_path}. Run registration first!")

    transforms = np.load(manifest_path, allow_pickle=True).item()
    if not isinstance(transforms, dict):
        raise ValueError(f"Transform manifest is malformed: expected dict payload, got {type(transforms)}")

    manifest_model = TransformManifest.from_legacy(transforms, fov_id=fov_id)
    transforms = cast(Dict[Any, Any], manifest_model.to_legacy())

    provenance = transforms.get("_provenance")
    if provenance is not None:
        _backfill_requested_intent_diagnostic_defaults(provenance)
        _backfill_field_semantics_contract(provenance, transforms)
        validate_provenance_schema(provenance)

    transforms_dir = manifest_path.parent
    materialized: Dict[Any, Any] = {}

    for round_key, transform_data in transforms.items():
        # Skip provenance key - it's handled separately
        if round_key in {"_provenance", "_contract"}:
            continue
            
        if not _is_round_transform_entry(transform_data):
            materialized[round_key] = transform_data
            continue

        try:
            round_id = int(round_key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Round transform entry must use a numeric key, got {round_key!r}") from exc

        round_payload = dict(transform_data)
        round_payload.setdefault("flow_2d", None)
        _ensure_round_field_semantics(
            round_payload,
            field_name=f"transform round {round_id} _semantics",
        )
        flow_3d = round_payload.get("flow_3d")

        if isinstance(flow_3d, Mapping):
            descriptor, sidecar_path = _validate_flow_3d_sidecar_descriptor(
                flow_3d,
                round_key=round_key,
                transforms_dir=transforms_dir,
            )
            if hydrate_flow_3d:
                round_payload["flow_3d"] = _load_flow_3d_sidecar_array(
                    descriptor,
                    round_key=round_key,
                    sidecar_path=sidecar_path,
                )
            else:
                round_payload["flow_3d"] = descriptor
        elif flow_3d is not None and not isinstance(flow_3d, np.ndarray):
            raise ValueError(f"Unsupported flow_3d payload type for round {round_key}: {type(flow_3d)}")
        else:
            round_payload.setdefault("flow_3d", None)

        materialized[round_id] = round_payload

    if load_provenance and provenance is not None:
        _validate_round_scope_contract_alignment(materialized, provenance["release_contract"])
        materialized["_provenance"] = provenance

    return materialized

class ImageLoader:
    """Load raw and cleaned image volumes under the PyStar path contract.

    Raw images are resolved from ``dataset.raw_data_path`` and
    ``dataset.filename_pattern``. Clean images are resolved from the canonical
    per-FOV output tree created by ``get_fov_output_structure``. The public
    ``load_fov`` method returns a lazy xarray volume with dimensions
    ``(round, channel, z, y, x)`` and physical coordinates in nanometers; missing
    channels declared absent from a round are represented as zero volumes so that
    downstream stages see a rectangular tensor.
    """

    def __init__(self, config: ExperimentConfig):
        self.cfg = config
        self.raw_path = self.cfg.dataset.raw_data_path
        self.dims = self.cfg.dataset.dimensions
        self.pattern = self.cfg.dataset.filename_pattern
        
    def _get_path(self, fov: int, round_id: int, channel_id: int) -> Path:
        """
        Resolve one raw TIFF path from the YAML filename pattern.

        ``filename_pattern`` receives ``round``, ``fov`` and ``ch`` placeholders.
        The loader tries both zero-padded and plain channel strings because Leica
        exports are not always consistent across experiments. Missing or
        ambiguous matches fail loudly; raw input paths must come from the config,
        not from hard-coded local overrides.
        """
        # 尝试两种补零格式：ch00 (常见) 和 ch0 (偶尔见)
        # 这是一个实用的 hack，避免因为文件名格式不对就崩溃
        candidates = []
        
        for ch_str in [f"{channel_id:02d}", f"{channel_id}"]:
            glob_pattern = self.pattern.format(
                round=round_id, 
                fov=fov, 
                ch=ch_str
            )
            found = list(self.raw_path.glob(glob_pattern))
            if found:
                candidates.extend(found)
                break # 找到了就停止

        if not candidates:
            # 构建一个失败时的提示路径
            debug_path = self.raw_path / self.pattern.format(round=round_id, fov=fov, ch=f"{channel_id}")
            raise FileNotFoundError(
                f" Data missing!\n"
                f"Looking for: R{round_id} / FOV{fov} / CH{channel_id}\n"
                f"Pattern tried: {debug_path}\n"
                f"Check your 'raw_data_path' and 'filename_pattern' in yaml."
            )
            
        if len(candidates) > 1:
            raise ValueError(
                f"Ambiguous pattern! Found multiple files for one channel:\n{candidates}"
            )
            
        return candidates[0]

    def get_clean_path(self, fov_id: int, round_id: int, channel_id: int) -> Path:
        """Return the canonical clean-image path for one FOV/round/channel."""
        base_dir = Path(self.cfg.pipeline.output.directory)
        paths = get_fov_output_structure(base_dir, fov_id)
        clean_dir = paths["cleaned"]
        return clean_dir / f"clean_fov_{fov_id}_round_{round_id}_ch_{channel_id}.tif"

    def load_clean_image(self, fov_id: int, round_id: int, channel_id: int) -> NDArray[Any]:
        """Read a preprocessed clean 3-D TIFF from the PyStar output tree."""
        path = self.get_clean_path(fov_id, round_id, channel_id)
        if not path.exists():
            raise FileNotFoundError(f"Clean image not found: {path}. Run preprocessing first!")
        return tifffile.imread(path)

    def _lazy_load_tiff(self, path: Path) -> da.Array:
        """Create a Dask array for one raw 3-D TIFF without loading pixels now."""
        # 使用 delayed 读取，不立即加载进内存
        def loader(p):
            return tifffile.imread(p).squeeze()

        shape = (self.dims['z'], self.dims['height'], self.dims['width'])
        dtype = np.uint8 # 我们目前数据是 8-bit
        
        sample = dask.delayed(loader)(path)
        # 从 Config 读取 chunk size
        chunks = (
            self.cfg.dataset.io_chunk_size['z'],
            self.cfg.dataset.io_chunk_size['y'],
            self.cfg.dataset.io_chunk_size['x']
        )
        
        # 告诉 Dask 怎么切，不要让它瞎猜
        arr = da.from_delayed(sample, shape=shape, dtype=dtype)
        return arr.rechunk(cast(Any, chunks)) 

    def load_fov(self, fov_id: int) -> xr.DataArray:
        """
        Load one FOV as a lazy rectangular ``round/channel/z/y/x`` array.

        The dataset config may declare that different rounds have different
        channel sets. Valid channels are loaded lazily from disk; unavailable
        channels are zero-padded so registration/preprocessing can index by the
        global channel list without special cases. Coordinate arrays are physical
        distances in nanometers, while pixel-level algorithms downstream still
        use integer ``z, y, x`` indices.
        """
        rounds_cfg = self.cfg.dataset.round_structure
        all_rounds = sorted(rounds_cfg.keys())
        all_channels = sorted(self.cfg.dataset.channel_roles.keys())
        
        round_stacks = []
        
        print(f"DEBUG: Loading FOV {fov_id} structure...", end="", flush=True)
        
        for r_id in all_rounds:
            valid_channels = rounds_cfg[r_id]
            channel_stacks = []
            
            for c_id in all_channels:
                if c_id in valid_channels:
                    # 真实加载
                    fpath = self._get_path(fov_id, r_id, c_id)
                    arr = self._lazy_load_tiff(fpath)
                else:
                    # 虚拟填充 (Padding)
                    arr = da.zeros(
                        (self.dims['z'], self.dims['height'], self.dims['width']), 
                        dtype=np.uint8
                    )
                
                channel_stacks.append(arr)
            
            # Stack channels -> (C, Z, Y, X)
            round_stacks.append(da.stack(channel_stacks))
            
        # Stack rounds -> (R, C, Z, Y, X)
        final_dask = da.stack(round_stacks)
        
        # 物理坐标
        z_coords = np.arange(self.dims['z']) * self.cfg.dataset.pixel_size_z_nm
        y_coords = np.arange(self.dims['height']) * self.cfg.dataset.pixel_size_xy_nm
        x_coords = np.arange(self.dims['width']) * self.cfg.dataset.pixel_size_xy_nm
        
        xarr = xr.DataArray(
            final_dask,
            coords={
                "round": all_rounds,
                "channel": all_channels,
                "z": z_coords,
                "y": y_coords,
                "x": x_coords,
            },
            dims=("round", "channel", "z", "y", "x"),
            name=f"fov_{fov_id}",
            attrs={
                "fov_id": fov_id,
                "valid_channels_map": rounds_cfg,
                "channel_roles": self.cfg.dataset.channel_roles   
            }
        )
        print(" Done.")
        return xarr
