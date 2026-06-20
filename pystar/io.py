# pyright: reportPrivateUsage=false, reportUnsupportedDunderAll=false, reportUnannotatedClassAttribute=false, reportUnknownMemberType=false, reportImplicitStringConcatenation=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportExplicitAny=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportPrivateImportUsage=false

from ._io_image_loader import ImageLoader as ImageLoader
from ._io_paths import (
    get_flow_3d_sidecar_filename as get_flow_3d_sidecar_filename,
    get_fov_output_structure as get_fov_output_structure,
    get_provenance_summary_path as get_provenance_summary_path,
    get_transform_manifest_path as get_transform_manifest_path,
)
from ._io_provenance import (
    EXECUTION_ENVELOPE_ALLOWED_VALUES as EXECUTION_ENVELOPE_ALLOWED_VALUES,
    FINAL_CORR_METRIC as FINAL_CORR_METRIC,
    LOCAL_ACCEPTANCE_MODE as LOCAL_ACCEPTANCE_MODE,
    MATLAB_STAGE_ARTIFACT_CONTRACT_SUPPORTED as MATLAB_STAGE_ARTIFACT_CONTRACT_SUPPORTED,
    MATLAB_STAGE_CONFIG_SURFACES as MATLAB_STAGE_CONFIG_SURFACES,
    MATLAB_STAGE_PYTHON_OWNED_ARTIFACTS as MATLAB_STAGE_PYTHON_OWNED_ARTIFACTS,
    MATLAB_STAGE_REQUIRED_PROMOTION_BLOCKERS as MATLAB_STAGE_REQUIRED_PROMOTION_BLOCKERS,
    MATLAB_STAGE_SUPPORT_STATUSES as MATLAB_STAGE_SUPPORT_STATUSES,
    PROVENANCE_VERSION as PROVENANCE_VERSION,
    RC_FACTS as RC_FACTS,
    REGISTRATION_PROFILE_FACTS as REGISTRATION_PROFILE_FACTS,
    build_execution_envelope as build_execution_envelope,
    build_field_semantics_contract as build_field_semantics_contract,
    build_matlab_stage_contracts_from_config as build_matlab_stage_contracts_from_config,
    build_provenance_summary_markdown as build_provenance_summary_markdown,
    build_release_contract as build_release_contract,
    build_scope_contract as build_scope_contract,
    create_provenance as create_provenance,
    derive_registration_profile as derive_registration_profile,
    generate_field_semantics_summary as generate_field_semantics_summary,
    get_matlab_stage_contract as get_matlab_stage_contract,
    infer_delivered_scope_coverage as infer_delivered_scope_coverage,
    is_phase1_mainline_execution_envelope as is_phase1_mainline_execution_envelope,
    summarize_delivered_capability as summarize_delivered_capability,
    backfill_field_semantics_contract as _backfill_field_semantics_contract,
    backfill_requested_intent_diagnostic_defaults as _backfill_requested_intent_diagnostic_defaults,
    validate_round_scope_contract_alignment as _validate_round_scope_contract_alignment,
    validate_execution_envelope as validate_execution_envelope,
    validate_field_semantics_contract as validate_field_semantics_contract,
    validate_provenance_schema as validate_provenance_schema,
    validate_scope_contract as validate_scope_contract,
)
from ._io_transforms import (
    configure_transform_manifest_io as _configure_transform_manifest_io,
    load_transform_manifest as load_transform_manifest,
    materialize_round_transform_entry as materialize_round_transform_entry,
    persist_flow_3d_sidecar as persist_flow_3d_sidecar,
    save_transform_manifest as save_transform_manifest,
    write_provenance_summary as write_provenance_summary,
)
from .runtime_artifacts import (
    FLOW_3D_SIDECAR_STORAGE,
    RELEASE_GATE_STATUSES,
    SCOPE_MODES,
    SCOPE_STATUSES,
    FieldSemantics,
    Flow3DSidecarDescriptor,
    ScopeMetadata,
    TransformEntry,
    TransformManifest,
)

# Keep these explicit module-level bindings so ``pystar.io`` remains the stable
# public facade for runtime-artifact compatibility names after Stage 5C.
_PUBLIC_RUNTIME_ARTIFACT_EXPORTS = (
    FLOW_3D_SIDECAR_STORAGE,
    RELEASE_GATE_STATUSES,
    SCOPE_MODES,
    SCOPE_STATUSES,
    FieldSemantics,
    Flow3DSidecarDescriptor,
    ScopeMetadata,
    TransformEntry,
    TransformManifest,
)

_configure_transform_manifest_io(
    backfill_requested_intent_diagnostic_defaults=_backfill_requested_intent_diagnostic_defaults,
    backfill_field_semantics_contract=_backfill_field_semantics_contract,
    validate_provenance_schema=validate_provenance_schema,
    validate_round_scope_contract_alignment=_validate_round_scope_contract_alignment,
    build_provenance_summary_markdown=build_provenance_summary_markdown,
)
