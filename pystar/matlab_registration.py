"""MATLAB-backed registration boundary for PyStar.

The Python registration engine can delegate global phase-correlation and local
3D Demons registration to repo-local MATLAB entrypoints.  This module stages
cleaned 3D volumes, passes explicit JSON plans, validates MATLAB metadata, and
normalizes shifts/flows into PyStar's transform schema: `global_shift_3d` in
`z, y, x` order and local `flow_3d` as `(component, z, y, x)` residual fields
after the global shift.  MATLAB outputs are accepted only when their declared
coordinate order and semantics match the PyStar contract.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, Mapping, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat

from .infrastructure import ExperimentConfig
from .matlab_engine_bootstrap import (
    MATLABSessionCapsule,
    MatlabSharedSessionOwner,
    create_matlab_boundary_trace,
    finalize_matlab_boundary_trace,
    load_matlab_engine_factory,
    record_matlab_boundary_phase,
    snapshot_matlab_session_lifecycle,
    summarize_matlab_boundary_traces,
)
from .matlab_runtime import (
    collect_runtime_file_records as _collect_runtime_file_records_from_manifest,
    format_exception_message as _format_exception_message,
    load_validated_runtime_manifest,
    resolve_repo_runtime_path,
    resolve_staged_output_path,
    trusted_matlab_runtime_root,
    validate_manifest_declares_entrypoint_file,
    validate_configured_entrypoint_contract,
    validate_runtime_step_metadata,
    write_staged_3d_volume_tiff,
    sha256_file,
)


MATLAB_REGISTRATION_RUNTIME_MANIFEST_NAME = "runtime_manifest.json"
MATLAB_SUPPORTED_SHIFT_ORDERS = {"z_y_x", "dz_dy_dx", "dy_dx_dz", "yxz"}
MATLAB_SUPPORTED_SHIFT_SEMANTICS = {"apply_to_moving_volume"}
MATLAB_SUPPORTED_LOCAL_FLOW_STORAGE_FORMATS = {"mat_v7"}
MATLAB_SUPPORTED_LOCAL_FLOW_LAYOUTS = {"y_x_z_components"}
MATLAB_SUPPORTED_LOCAL_FLOW_COMPONENT_ORDERS = {"dx_dy_dz"}
MATLAB_SUPPORTED_LOCAL_FLOW_SEMANTICS = {"apply_to_moving_volume"}
MATLAB_SUPPORTED_LOCAL_FLOW_COMPOSITIONS = {"residual_after_global_shift"}


def _build_common_registration_plan(
    config: ExperimentConfig,
    *,
    fov_id: int,
    round_id: int,
    reference_round: int,
    scope_descriptor: Mapping[str, Any],
    volume_shape_zyx: tuple[int, int, int],
    compute_tile: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build the shared MATLAB registration plan fields for global/local calls.

    The plan describes both the staged subvolume (`volume_shape_zyx`) and the
    scope of that subvolume in the full FOV (`scope_*`).  Tile-local registration
    adds `compute_tile_*` fields so MATLAB can report fields that PyStar can later
    stitch and annotate with exact write-region metadata.
    """

    matlab_cfg = config.providers.matlab.registration

    if matlab_cfg.volume_transfer_mode != "temporary_tiff":
        raise ValueError(
            "matlab_extracted registration currently supports only volume_transfer_mode='temporary_tiff'"
        )

    plan = {
        "fov_id": int(fov_id),
        "round_id": int(round_id),
        "reference_round": int(reference_round),
        "scope_mode": str(scope_descriptor.get("coverage_mode")),
        "scope_region_zyx": list(scope_descriptor.get("region_origin_zyx", [])),
        "scope_shape_zyx": list(scope_descriptor.get("region_shape_zyx", [])),
        "full_volume_shape_zyx": list(scope_descriptor.get("full_volume_shape_zyx", [])),
        "volume_shape_zyx": [int(value) for value in volume_shape_zyx],
        "downsample_factor": int(config.pipeline.registration.downsample_factor),
        "global_max_shift": int(config.pipeline.registration.global_max_shift),
        "volume_transfer_mode": matlab_cfg.volume_transfer_mode,
        "input_volume_dtype": matlab_cfg.input_volume_dtype,
        "use_gpu": bool(matlab_cfg.use_gpu),
    }

    if "tile_grid_shape_yx" in scope_descriptor:
        plan["tile_grid_shape_yx"] = list(scope_descriptor["tile_grid_shape_yx"])
    if "tile_index" in scope_descriptor:
        plan["tile_index"] = int(scope_descriptor["tile_index"])
    if compute_tile is not None:
        plan["compute_tile_index"] = int(compute_tile["tile_index"])
        plan["compute_tile_grid_position_yx"] = [int(value) for value in compute_tile["grid_position_yx"]]
        plan["compute_tile_grid_shape_yx"] = [int(value) for value in compute_tile["grid_shape_yx"]]
        plan["compute_tile_origin_zyx"] = [int(value) for value in compute_tile["region_origin_zyx"]]
        plan["compute_tile_shape_zyx"] = [int(value) for value in compute_tile["region_shape_zyx"]]
        plan["compute_tile_write_origin_zyx"] = [int(value) for value in compute_tile["write_origin_zyx"]]
        plan["compute_tile_write_shape_zyx"] = [int(value) for value in compute_tile["write_shape_zyx"]]
        plan["compute_tile_write_offset_zyx"] = [int(value) for value in compute_tile["write_offset_zyx"]]

    return plan


def build_matlab_registration_plan(
    config: ExperimentConfig,
    *,
    fov_id: int,
    round_id: int,
    reference_round: int,
    scope_descriptor: Mapping[str, Any],
    volume_shape_zyx: tuple[int, int, int],
) -> Dict[str, Any]:
    """Build the JSON plan for MATLAB global registration of one round."""

    return _build_common_registration_plan(
        config,
        fov_id=fov_id,
        round_id=round_id,
        reference_round=reference_round,
        scope_descriptor=scope_descriptor,
        volume_shape_zyx=volume_shape_zyx,
    )


def build_matlab_local_registration_plan(
    config: ExperimentConfig,
    *,
    fov_id: int,
    round_id: int,
    reference_round: int,
    scope_descriptor: Mapping[str, Any],
    volume_shape_zyx: tuple[int, int, int],
    compute_tile: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build the JSON plan for MATLAB local 3D Demons registration.

    MATLAB local registration is intentionally limited to `demons_3d` here.  The
    plan declares that the rigid global shift has already been applied and that
    the returned flow should be interpreted as a residual displacement field.
    """

    reg_cfg = config.pipeline.registration
    if reg_cfg.local_method != "demons_3d":
        raise ValueError(
            "MATLAB local registration plan currently supports only local_method='demons_3d'"
        )

    plan = _build_common_registration_plan(
        config,
        fov_id=fov_id,
        round_id=round_id,
        reference_round=reference_round,
        scope_descriptor=scope_descriptor,
        volume_shape_zyx=volume_shape_zyx,
        compute_tile=compute_tile,
    )
    plan.update(
        {
            "provider": "matlab",
            "method": "demons_3d",
            "coverage_mode": plan.get("scope_mode"),
            "local_method": "demons_3d",
            "iterations": int(reg_cfg.demons_3d.num_iter),
            "accumulated_field_smoothing": float(reg_cfg.demons_3d.smoothing_sigma),
            "expected_flow_shape_zyx": [int(value) for value in volume_shape_zyx],
            "global_shift_already_applied": True,
        }
    )
    if reg_cfg.demons_3d.pyramid_levels is not None:
        plan["pyramid_levels"] = int(reg_cfg.demons_3d.pyramid_levels)
    return plan


def _runtime_manifest_identity(runtime_manifest_path: Path) -> Dict[str, str]:
    """Return stable runtime manifest identity for request equivalence evidence."""

    return {
        "path": str(runtime_manifest_path),
        "sha256": sha256_file(runtime_manifest_path),
    }


def resolve_matlab_registration_runtime_path(config: ExperimentConfig) -> Path:
    """Resolve the configured MATLAB registration runtime directory."""

    return resolve_repo_runtime_path(
        config.providers.matlab.registration.runtime_path,
        config_label="providers.matlab.registration.runtime_path",
        trusted_root=trusted_matlab_runtime_root(),
    )


def load_matlab_registration_runtime_manifest(runtime_dir: Path) -> Dict[str, Any]:
    """Load and validate the MATLAB registration runtime manifest."""

    manifest = load_validated_runtime_manifest(
        runtime_dir,
        manifest_label="MATLAB registration runtime manifest",
        missing_hint="Expected repo-local manifest for matlab_extracted registration backend.",
        manifest_name=MATLAB_REGISTRATION_RUNTIME_MANIFEST_NAME,
        optional_string_fields=("local_entrypoint",),
    )
    local_entrypoint = manifest.get("local_entrypoint")
    if local_entrypoint is not None:
        validate_manifest_declares_entrypoint_file(
            manifest,
            str(local_entrypoint),
            manifest_label="MATLAB registration runtime manifest",
            expectation_label="configured local_entrypoint file",
        )

    return manifest


def _validate_runtime_entrypoint_contract(
    runtime_manifest: Mapping[str, Any],
    configured_entrypoint: str,
) -> None:
    validate_configured_entrypoint_contract(
        runtime_manifest,
        configured_entrypoint,
        config_label="providers.matlab.registration.entrypoint",
        manifest_label="MATLAB registration runtime manifest",
    )


def _load_matlab_engine_factory() -> Callable[[], Any]:
    factory, _factory_metrics = load_matlab_engine_factory(
        consumer="registration global/local provider='matlab'",
    )
    return factory


def _write_staged_volume_tiff(volume_path: Path, volume: Any) -> None:
    write_staged_3d_volume_tiff(volume_path, volume, owner_label="MATLAB registration")


class MATLABRegistrationBackend:
    """Execute MATLAB registration entrypoints under PyStar's transform contract.

    The backend owns a MATLAB session capsule and validates all runtime files.
    Global calls return rigid shifts normalized to `z, y, x`; local calls return
    residual 3D flow fields normalized to `(dz, dy, dx)` component order and
    `(component, z, y, x)` array layout.  Boundary metadata is kept so downstream
    reports can distinguish algorithm cost from MATLAB session/bootstrap cost.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        engine_factory: Optional[Callable[[], Any]] = None,
        matlab_session_owner: MatlabSharedSessionOwner | None = None,
    ) -> None:
        self.config = config
        self.engine_factory = engine_factory
        self.matlab_session_owner = matlab_session_owner
        self.runtime_dir = resolve_matlab_registration_runtime_path(config)
        self.runtime_manifest = load_matlab_registration_runtime_manifest(self.runtime_dir)
        self.entrypoint = config.providers.matlab.registration.entrypoint
        self.local_entrypoints = dict(config.providers.matlab.registration.local_entrypoints)
        self.local_entrypoint = self.local_entrypoints.get(
            "demons_3d",
            str(self.runtime_manifest.get("local_entrypoint", "")).strip(),
        )
        _validate_runtime_entrypoint_contract(self.runtime_manifest, self.entrypoint)
        manifest_local_entrypoint = str(self.runtime_manifest.get("local_entrypoint", "")).strip()
        if manifest_local_entrypoint and self.local_entrypoint and self.local_entrypoint != manifest_local_entrypoint:
            raise ValueError(
                "providers.matlab.registration.local_entrypoints['demons_3d'] must match the repo-local MATLAB runtime manifest. "
                f"Config local entrypoint={self.local_entrypoint!r}, manifest local entrypoint={manifest_local_entrypoint!r}"
            )
        self._session_capsule = MATLABSessionCapsule(
            consumer="registration provider='matlab'",
            runtime_dir=self.runtime_dir,
            entrypoint=self.entrypoint,
            engine_factory=engine_factory,
            engine_factory_consumer="registration global/local provider='matlab'",
            startup_failure_prefix="Failed to start MATLAB Engine for registration provider='matlab'",
            addpath_failure_prefix="Failed to add MATLAB registration runtime path",
            session_owner=matlab_session_owner,
            runtime_file_validator=self._collect_runtime_file_records,
        )

    @property
    def _engine(self) -> Any:
        return self._session_capsule.engine

    @property
    def _session_lifecycle(self) -> dict[str, Any]:
        return self._session_capsule.session_lifecycle

    @property
    def _session_lifecycle_summary(self) -> dict[str, Any] | None:
        return self._session_capsule.summarize_session_lifecycle()

    def close(self) -> None:
        """Close the owned MATLAB Engine session if one was started."""

        self._session_capsule.close()

    def _ensure_engine(self) -> Any:
        return self._session_capsule.ensure_engine()

    def _consume_last_engine_acquire(self) -> dict[str, Any]:
        return self._session_capsule.consume_last_engine_acquire()

    def _resolve_entrypoint_callable(self, entrypoint_name: str) -> Any:
        return self._session_capsule.resolve_callable(entrypoint_name)

    def _resolve_runtime_file_records(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return self._session_capsule.validate_runtime_files(self._collect_runtime_file_records)

    def _runtime_file_records_for_provenance(self) -> list[dict[str, Any]]:
        cached_records = self._session_capsule.peek_runtime_file_records()
        if cached_records is not None:
            return cached_records
        return self._collect_runtime_file_records()

    def _collect_runtime_file_records(self) -> list[dict[str, Any]]:
        return _collect_runtime_file_records_from_manifest(
            self.runtime_manifest,
            self.runtime_dir,
            missing_required_prefix="Required MATLAB registration runtime file is missing",
            missing_required_suffix="matlab_extracted registration cannot proceed.",
        )

    def _normalize_input_volume(self, volume: NDArray[Any]) -> NDArray[Any]:
        if volume.ndim != 3:
            raise ValueError(f"MATLAB registration expects a 3D volume, got ndim={volume.ndim}")

        matlab_cfg = self.config.providers.matlab.registration

        input_dtype = np.dtype(matlab_cfg.input_volume_dtype)
        arr = np.asarray(volume)
        if input_dtype == np.uint8:
            if arr.dtype == np.uint8:
                return arr
            if np.issubdtype(arr.dtype, np.floating):
                max_val = float(np.max(arr)) if arr.size else 0.0
                if max_val <= 1.0:
                    arr = np.clip(arr, 0.0, 1.0) * 255.0
                else:
                    arr = np.clip(arr, 0.0, 255.0)
                return np.rint(arr).astype(np.uint8)
            return np.clip(arr, 0, 255).astype(np.uint8)

        raise ValueError(f"Unsupported providers.matlab.registration.input_volume_dtype={matlab_cfg.input_volume_dtype!r}")

    def _validate_global_response_metadata(
        self,
        metadata: Mapping[str, Any],
        *,
        round_id: int,
        reference_round: int,
    ) -> None:
        shift_order = metadata.get("shift_order")
        if shift_order not in MATLAB_SUPPORTED_SHIFT_ORDERS:
            raise ValueError(
                "MATLAB registration metadata must declare a supported shift_order. "
                f"Got {shift_order!r}, supported={sorted(MATLAB_SUPPORTED_SHIFT_ORDERS)}"
            )

        shift_semantics = metadata.get("shift_semantics")
        if shift_semantics not in MATLAB_SUPPORTED_SHIFT_SEMANTICS:
            raise ValueError(
                "MATLAB registration metadata must declare supported shift_semantics. "
                f"Got {shift_semantics!r}, supported={sorted(MATLAB_SUPPORTED_SHIFT_SEMANTICS)}"
            )

        global_shift = metadata.get("global_shift")
        if not isinstance(global_shift, list) or len(global_shift) != 3:
            raise ValueError("MATLAB registration metadata must declare global_shift as a length-3 list")
        if any(not isinstance(value, (int, float)) for value in global_shift):
            raise ValueError("MATLAB registration metadata global_shift entries must be numeric")

        global_corr = metadata.get("global_corr")
        if not isinstance(global_corr, (int, float)):
            raise ValueError("MATLAB registration metadata must declare numeric global_corr")

        metadata_round_id = metadata.get("round_id")
        if not isinstance(metadata_round_id, (int, float)) or int(metadata_round_id) != round_id:
            raise ValueError(
                f"MATLAB registration metadata round_id mismatch: expected {round_id}, got {metadata_round_id!r}"
            )

        metadata_reference_round = metadata.get("reference_round")
        if not isinstance(metadata_reference_round, (int, float)) or int(metadata_reference_round) != reference_round:
            raise ValueError(
                "MATLAB registration metadata reference_round mismatch: "
                f"expected {reference_round}, got {metadata_reference_round!r}"
            )

        validate_runtime_step_metadata(
            metadata.get("steps"),
            missing_steps_message="MATLAB registration metadata must declare a non-empty steps list",
            step_label="MATLAB registration step",
        )

    def _validate_local_response_metadata(
        self,
        metadata: Mapping[str, Any],
        *,
        round_id: int,
        reference_round: int,
        tmpdir_path: Path,
        expected_shape_zyx: tuple[int, int, int],
    ) -> Path:
        metadata_round_id = metadata.get("round_id")
        if not isinstance(metadata_round_id, (int, float)) or int(metadata_round_id) != round_id:
            raise ValueError(
                f"MATLAB local-registration metadata round_id mismatch: expected {round_id}, got {metadata_round_id!r}"
            )

        metadata_reference_round = metadata.get("reference_round")
        if not isinstance(metadata_reference_round, (int, float)) or int(metadata_reference_round) != reference_round:
            raise ValueError(
                "MATLAB local-registration metadata reference_round mismatch: "
                f"expected {reference_round}, got {metadata_reference_round!r}"
            )

        storage_format = metadata.get("flow_storage_format")
        if storage_format not in MATLAB_SUPPORTED_LOCAL_FLOW_STORAGE_FORMATS:
            raise ValueError(
                "MATLAB local-registration metadata must declare a supported flow_storage_format. "
                f"Got {storage_format!r}, supported={sorted(MATLAB_SUPPORTED_LOCAL_FLOW_STORAGE_FORMATS)}"
            )

        flow_layout = metadata.get("flow_layout")
        if flow_layout not in MATLAB_SUPPORTED_LOCAL_FLOW_LAYOUTS:
            raise ValueError(
                "MATLAB local-registration metadata must declare a supported flow_layout. "
                f"Got {flow_layout!r}, supported={sorted(MATLAB_SUPPORTED_LOCAL_FLOW_LAYOUTS)}"
            )

        component_order = metadata.get("flow_component_order")
        if component_order not in MATLAB_SUPPORTED_LOCAL_FLOW_COMPONENT_ORDERS:
            raise ValueError(
                "MATLAB local-registration metadata must declare a supported flow_component_order. "
                f"Got {component_order!r}, supported={sorted(MATLAB_SUPPORTED_LOCAL_FLOW_COMPONENT_ORDERS)}"
            )

        flow_semantics = metadata.get("flow_semantics")
        if flow_semantics not in MATLAB_SUPPORTED_LOCAL_FLOW_SEMANTICS:
            raise ValueError(
                "MATLAB local-registration metadata must declare supported flow_semantics. "
                f"Got {flow_semantics!r}, supported={sorted(MATLAB_SUPPORTED_LOCAL_FLOW_SEMANTICS)}"
            )

        flow_composition = metadata.get("flow_composition")
        if flow_composition not in MATLAB_SUPPORTED_LOCAL_FLOW_COMPOSITIONS:
            raise ValueError(
                "MATLAB local-registration metadata must declare supported flow_composition. "
                f"Got {flow_composition!r}, supported={sorted(MATLAB_SUPPORTED_LOCAL_FLOW_COMPOSITIONS)}"
            )

        global_shift_already_applied = metadata.get("global_shift_already_applied")
        if not isinstance(global_shift_already_applied, bool) or not global_shift_already_applied:
            raise ValueError(
                "MATLAB local-registration metadata must declare global_shift_already_applied=true"
            )

        flow_variable = metadata.get("flow_variable")
        if not isinstance(flow_variable, str) or not flow_variable.strip():
            raise ValueError("MATLAB local-registration metadata must declare a non-empty flow_variable")

        flow_shape_yxz_component = metadata.get("flow_shape_yxz_component")
        if not isinstance(flow_shape_yxz_component, list) or len(flow_shape_yxz_component) != 4:
            raise ValueError(
                "MATLAB local-registration metadata must declare flow_shape_yxz_component as a length-4 list"
            )
        if any(not isinstance(value, (int, float)) or int(value) <= 0 for value in flow_shape_yxz_component):
            raise ValueError("MATLAB local-registration metadata flow_shape_yxz_component entries must be positive")

        expected_shape_yxz_component = [
            int(expected_shape_zyx[1]),
            int(expected_shape_zyx[2]),
            int(expected_shape_zyx[0]),
            3,
        ]
        normalized_shape = [int(value) for value in flow_shape_yxz_component]
        if normalized_shape != expected_shape_yxz_component:
            raise ValueError(
                "MATLAB local-registration flow_shape_yxz_component mismatch: "
                f"expected {expected_shape_yxz_component}, got {normalized_shape}"
            )

        flow_output_path = resolve_staged_output_path(
            metadata.get("flow_output_path"),
            tmpdir_path=tmpdir_path,
            missing_output_path_message="MATLAB local-registration metadata must declare a non-empty flow_output_path",
            outside_tmpdir_prefix="MATLAB local-registration flow output must stay inside the staged temporary directory",
            missing_output_prefix="MATLAB local-registration reported flow output that does not exist",
        )

        validate_runtime_step_metadata(
            metadata.get("steps"),
            missing_steps_message="MATLAB local-registration metadata must declare a non-empty steps list",
            step_label="MATLAB local-registration step",
        )

        return flow_output_path

    def _normalize_global_shift_zyx(self, metadata: Mapping[str, Any]) -> NDArray[np.float32]:
        shift_order = str(metadata["shift_order"])
        shift_values = np.asarray(metadata["global_shift"], dtype=np.float32)

        if shift_order in {"z_y_x", "dz_dy_dx"}:
            return np.asarray(shift_values, dtype=np.float32)

        if shift_order in {"dy_dx_dz", "yxz"}:
            return np.asarray([shift_values[2], shift_values[0], shift_values[1]], dtype=np.float32)

        raise ValueError(f"Unsupported MATLAB registration shift_order={shift_order!r}")

    def _load_local_flow_zyx(
        self,
        flow_output_path: Path,
        metadata: Mapping[str, Any],
        *,
        round_id: int,
    ) -> NDArray[np.float32]:
        flow_variable = str(metadata["flow_variable"])
        mat_payload = loadmat(flow_output_path)
        if flow_variable not in mat_payload:
            raise ValueError(
                f"MATLAB local-registration output is missing variable {flow_variable!r}: {flow_output_path}"
            )

        raw_flow = np.asarray(mat_payload[flow_variable])
        if raw_flow.ndim != 4:
            raise ValueError(
                f"MATLAB local-registration raw flow for round {round_id} must be 4D [Y, X, Z, 3], got {raw_flow.shape}"
            )
        if raw_flow.shape[-1] != 3:
            raise ValueError(
                f"MATLAB local-registration raw flow for round {round_id} must have 3 components, got {raw_flow.shape[-1]}"
            )

        expected_shape = tuple(int(value) for value in metadata["flow_shape_yxz_component"])
        if tuple(raw_flow.shape) != expected_shape:
            raise ValueError(
                f"MATLAB local-registration raw flow shape mismatch for round {round_id}: "
                f"metadata={list(expected_shape)}, actual={list(raw_flow.shape)}"
            )

        dx_yxz = np.asarray(raw_flow[..., 0], dtype=np.float32)
        dy_yxz = np.asarray(raw_flow[..., 1], dtype=np.float32)
        dz_yxz = np.asarray(raw_flow[..., 2], dtype=np.float32)

        dx_zyx = np.transpose(dx_yxz, (2, 0, 1))
        dy_zyx = np.transpose(dy_yxz, (2, 0, 1))
        dz_zyx = np.transpose(dz_yxz, (2, 0, 1))
        return np.stack([dz_zyx, dy_zyx, dx_zyx], axis=0).astype(np.float32, copy=False)

    def build_provenance_trace(self, round_results: Mapping[int, Mapping[str, Any]]) -> Dict[str, Any]:
        matlab_version = None
        normalized_round_results: Dict[str, Any] = {}
        mode_status = "experimental_global_only"

        for round_id, result in round_results.items():
            result_mapping = dict(result)
            metadata = result_mapping.get("matlab_metadata")
            if matlab_version is None and isinstance(metadata, Mapping):
                candidate_version = metadata.get("matlab_version")
                if isinstance(candidate_version, str) and candidate_version.strip():
                    matlab_version = candidate_version
            if matlab_version is None:
                local_flow = result_mapping.get("local_flow")
                if isinstance(local_flow, Mapping):
                    local_metadata = local_flow.get("matlab_metadata")
                    if isinstance(local_metadata, Mapping):
                        candidate_version = local_metadata.get("matlab_version")
                        if isinstance(candidate_version, str) and candidate_version.strip():
                            matlab_version = candidate_version
            if isinstance(result_mapping.get("local_flow"), Mapping) or result_mapping.get("mode") == "experimental_local_kernel_swap":
                mode_status = "experimental_local_kernel_swap"
            normalized_round_results[str(int(round_id))] = result_mapping

        trace: Dict[str, Any] = {
            "backend": "matlab_extracted",
            "mode_status": mode_status,
            "runtime_path": str(self.runtime_dir),
            "runtime_manifest": str(self.runtime_dir / MATLAB_REGISTRATION_RUNTIME_MANIFEST_NAME),
            "entrypoint": self.entrypoint,
            "runtime_files": self._runtime_file_records_for_provenance(),
            "round_results": normalized_round_results,
            "session_lifecycle": snapshot_matlab_session_lifecycle(self._session_lifecycle),
            "session_lifecycle_summary": self._session_lifecycle_summary,
        }
        boundary_traces: list[Mapping[str, Any]] = []
        for result_mapping in normalized_round_results.values():
            if not isinstance(result_mapping, Mapping):
                continue
            boundary_trace = result_mapping.get("boundary_instrumentation")
            if isinstance(boundary_trace, Mapping):
                boundary_traces.append(boundary_trace)
            local_flow = result_mapping.get("local_flow")
            if isinstance(local_flow, Mapping):
                local_boundary_trace = local_flow.get("boundary_instrumentation")
                if isinstance(local_boundary_trace, Mapping):
                    boundary_traces.append(local_boundary_trace)
        if boundary_traces:
            trace["boundary_instrumentation_summary"] = summarize_matlab_boundary_traces(boundary_traces)
        if self.local_entrypoint:
            trace["local_entrypoint"] = self.local_entrypoint
        if matlab_version is not None:
            trace["matlab_version"] = matlab_version
        return trace

    def compute_global_shift(
        self,
        reference_volume: NDArray[Any],
        moving_volume: NDArray[Any],
        *,
        fov_id: int,
        round_id: int,
        reference_round: int,
        scope_descriptor: Mapping[str, Any],
    ) -> Dict[str, Any]:
        boundary_trace = create_matlab_boundary_trace(
            stage_name="matlab_registration_global",
            runtime_dir=self.runtime_dir,
            entrypoint=self.entrypoint,
            session=self._session_lifecycle,
            call_scope={
                "fov_id": int(fov_id),
                "round_id": int(round_id),
                "reference_round": int(reference_round),
                "coverage_mode": scope_descriptor.get("coverage_mode"),
            },
        )
        ref_volume = self._normalize_input_volume(reference_volume)
        mov_volume = self._normalize_input_volume(moving_volume)
        request_payload = build_matlab_registration_plan(
            self.config,
            fov_id=fov_id,
            round_id=round_id,
            reference_round=reference_round,
            scope_descriptor=scope_descriptor,
            volume_shape_zyx=(int(ref_volume.shape[0]), int(ref_volume.shape[1]), int(ref_volume.shape[2])),
        )

        runtime_validation_started = time.perf_counter()
        runtime_files, runtime_validation_details = self._resolve_runtime_file_records()
        record_matlab_boundary_phase(
            boundary_trace,
            phase_name="runtime_file_validation",
            duration_ms=round((time.perf_counter() - runtime_validation_started) * 1000.0, 3),
            seam_cost_key="runtime_file_validation_ms",
            details={
                "runtime_file_count": len(runtime_files),
                **runtime_validation_details,
            },
        )

        matlab_callable = self._resolve_entrypoint_callable(self.entrypoint)
        engine_acquire = self._consume_last_engine_acquire()
        session_bootstrap = engine_acquire.get("session_bootstrap")
        if isinstance(session_bootstrap, Mapping):
            engine_bootstrap_ms_value = session_bootstrap.get("engine_bootstrap_ms")
            engine_bootstrap_ms = (
                float(engine_bootstrap_ms_value)
                if isinstance(engine_bootstrap_ms_value, (int, float))
                else 0.0
            )
            record_matlab_boundary_phase(
                boundary_trace,
                phase_name="engine_bootstrap",
                duration_ms=engine_bootstrap_ms,
                seam_cost_key="engine_bootstrap_ms",
                details=session_bootstrap,
            )

        with TemporaryDirectory(prefix=f"pystar_matlab_registration_fov{fov_id}_round{round_id}_") as tmpdir:
            tmpdir_path = Path(tmpdir)
            ref_path = tmpdir_path / f"reference_round_{reference_round}.tif"
            moving_path = tmpdir_path / f"moving_round_{round_id}.tif"
            input_staging_started = time.perf_counter()
            _write_staged_volume_tiff(ref_path, ref_volume)
            _write_staged_volume_tiff(moving_path, mov_volume)
            record_matlab_boundary_phase(
                boundary_trace,
                phase_name="input_staging",
                duration_ms=round((time.perf_counter() - input_staging_started) * 1000.0, 3),
                seam_cost_key="input_staging_ms",
                details={
                    "staged_inputs": [ref_path.name, moving_path.name],
                    "volume_shape_zyx": [int(ref_volume.shape[0]), int(ref_volume.shape[1]), int(ref_volume.shape[2])],
                },
            )

            matlab_call_started = time.perf_counter()
            try:
                metadata_json = matlab_callable(
                    str(ref_path),
                    str(moving_path),
                    json.dumps(request_payload, sort_keys=True),
                    nargout=1,
                )
            except Exception as exc:  # pragma: no cover - exact engine exception type depends on MATLAB install
                raise RuntimeError(
                    _format_exception_message(
                        f"MATLAB registration entrypoint '{self.entrypoint}' failed for FOV {fov_id} round {round_id}",
                        exc,
                    )
                ) from exc
            record_matlab_boundary_phase(
                boundary_trace,
                phase_name="matlab_call",
                duration_ms=round((time.perf_counter() - matlab_call_started) * 1000.0, 3),
                seam_cost_key="matlab_call_ms",
                details={"request_volume_shape_zyx": request_payload.get("volume_shape_zyx")},
            )

        if not isinstance(metadata_json, str):
            raise ValueError(
                f"MATLAB registration entrypoint '{self.entrypoint}' must return a JSON string metadata payload"
            )

        result_validation_started = time.perf_counter()
        try:
            metadata = json.loads(metadata_json)
        except json.JSONDecodeError as exc:
            raise ValueError(
                _format_exception_message(
                    f"MATLAB registration entrypoint '{self.entrypoint}' returned invalid JSON metadata",
                    exc,
                )
            ) from exc
        if not isinstance(metadata, dict):
            raise ValueError("MATLAB registration metadata payload must decode to a JSON object")

        self._validate_global_response_metadata(
            metadata,
            round_id=round_id,
            reference_round=reference_round,
        )
        global_shift_3d = self._normalize_global_shift_zyx(metadata)
        global_corr = float(metadata["global_corr"])
        record_matlab_boundary_phase(
            boundary_trace,
            phase_name="result_validation",
            duration_ms=round((time.perf_counter() - result_validation_started) * 1000.0, 3),
            seam_cost_key="result_validation_ms",
            details={
                "reported_step_count": len(metadata.get("steps", [])) if isinstance(metadata.get("steps"), list) else 0,
                "global_corr": global_corr,
            },
        )
        finalized_boundary_trace = finalize_matlab_boundary_trace(
            boundary_trace,
            session=self._session_lifecycle,
            engine_reused_this_call=bool(engine_acquire.get("engine_reused_this_call", False)),
        )

        return {
            "global_shift_3d": global_shift_3d,
            "global_corr": global_corr,
            "backend_metadata": {
                "backend": "matlab_extracted",
                "provider": "matlab",
                "mode": "experimental_global_shift_only",
                "runtime": {
                    "runtime_path": str(self.runtime_dir),
                    "runtime_manifest": str(self.runtime_dir / MATLAB_REGISTRATION_RUNTIME_MANIFEST_NAME),
                    "entrypoint": self.entrypoint,
                },
                "request": request_payload,
                "runtime_files": runtime_files,
                "matlab_metadata": metadata,
                "normalized_result": {
                    "global_shift_3d": [
                        float(global_shift_3d[0]),
                        float(global_shift_3d[1]),
                        float(global_shift_3d[2]),
                    ],
                    "global_corr": global_corr,
                },
                "boundary_instrumentation": finalized_boundary_trace,
                "session_lifecycle": snapshot_matlab_session_lifecycle(self._session_lifecycle),
                "session_lifecycle_summary": self._session_lifecycle_summary,
            },
        }

    def compute_local_flow(
        self,
        reference_volume: NDArray[Any],
        moving_volume: NDArray[Any],
        *,
        fov_id: int,
        round_id: int,
        reference_round: int,
        scope_descriptor: Mapping[str, Any],
        compute_tile: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        if not self.local_entrypoint:
            raise RuntimeError(
                "MATLAB registration runtime manifest does not declare local_entrypoint required for local demons kernel"
            )

        boundary_trace = create_matlab_boundary_trace(
            stage_name="matlab_registration_local",
            runtime_dir=self.runtime_dir,
            entrypoint=self.local_entrypoint,
            session=self._session_lifecycle,
            call_scope={
                "fov_id": int(fov_id),
                "round_id": int(round_id),
                "reference_round": int(reference_round),
                "coverage_mode": scope_descriptor.get("coverage_mode"),
                "compute_tile_index": None if compute_tile is None else compute_tile.get("tile_index"),
            },
        )

        ref_volume = self._normalize_input_volume(reference_volume)
        mov_volume = self._normalize_input_volume(moving_volume)
        reference_volume_shape_zyx = [int(value) for value in ref_volume.shape]
        moving_volume_shape_zyx = [int(value) for value in mov_volume.shape]
        request_payload = build_matlab_local_registration_plan(
            self.config,
            fov_id=fov_id,
            round_id=round_id,
            reference_round=reference_round,
            scope_descriptor=scope_descriptor,
            volume_shape_zyx=tuple(reference_volume_shape_zyx),
            compute_tile=compute_tile,
        )
        request_payload["reference_volume_shape_zyx"] = reference_volume_shape_zyx
        request_payload["moving_volume_shape_zyx"] = moving_volume_shape_zyx

        runtime_validation_started = time.perf_counter()
        runtime_files, runtime_validation_details = self._resolve_runtime_file_records()
        runtime_manifest_path = self.runtime_dir / MATLAB_REGISTRATION_RUNTIME_MANIFEST_NAME
        request_payload["runtime"] = {
            "runtime_path": str(self.runtime_dir),
            "runtime_manifest": str(runtime_manifest_path),
            "runtime_manifest_sha256": _runtime_manifest_identity(runtime_manifest_path)["sha256"],
            "entrypoint": self.local_entrypoint,
        }
        record_matlab_boundary_phase(
            boundary_trace,
            phase_name="runtime_file_validation",
            duration_ms=round((time.perf_counter() - runtime_validation_started) * 1000.0, 3),
            seam_cost_key="runtime_file_validation_ms",
            details={
                "runtime_file_count": len(runtime_files),
                **runtime_validation_details,
            },
        )

        matlab_callable = self._resolve_entrypoint_callable(self.local_entrypoint)
        engine_acquire = self._consume_last_engine_acquire()
        session_bootstrap = engine_acquire.get("session_bootstrap")
        if isinstance(session_bootstrap, Mapping):
            engine_bootstrap_ms_value = session_bootstrap.get("engine_bootstrap_ms")
            engine_bootstrap_ms = (
                float(engine_bootstrap_ms_value)
                if isinstance(engine_bootstrap_ms_value, (int, float))
                else 0.0
            )
            record_matlab_boundary_phase(
                boundary_trace,
                phase_name="engine_bootstrap",
                duration_ms=engine_bootstrap_ms,
                seam_cost_key="engine_bootstrap_ms",
                details=session_bootstrap,
            )

        with TemporaryDirectory(prefix=f"pystar_matlab_local_registration_fov{fov_id}_round{round_id}_") as tmpdir:
            tmpdir_path = Path(tmpdir)
            ref_path = tmpdir_path / f"reference_round_{reference_round}.tif"
            moving_path = tmpdir_path / f"moving_round_{round_id}.tif"
            flow_output_path = tmpdir_path / f"local_flow_round_{round_id}.mat"
            input_staging_started = time.perf_counter()
            _write_staged_volume_tiff(ref_path, ref_volume)
            _write_staged_volume_tiff(moving_path, mov_volume)
            request_payload["flow_output_path"] = str(flow_output_path)
            record_matlab_boundary_phase(
                boundary_trace,
                phase_name="input_staging",
                duration_ms=round((time.perf_counter() - input_staging_started) * 1000.0, 3),
                seam_cost_key="input_staging_ms",
                details={
                    "staged_inputs": [ref_path.name, moving_path.name],
                    "flow_output_name": flow_output_path.name,
                    "volume_shape_zyx": [int(ref_volume.shape[0]), int(ref_volume.shape[1]), int(ref_volume.shape[2])],
                },
            )

            matlab_call_started = time.perf_counter()
            try:
                metadata_json = matlab_callable(
                    str(ref_path),
                    str(moving_path),
                    json.dumps(request_payload, sort_keys=True),
                    nargout=1,
                )
            except Exception as exc:  # pragma: no cover - exact engine exception type depends on MATLAB install
                raise RuntimeError(
                    _format_exception_message(
                        f"MATLAB local registration entrypoint '{self.local_entrypoint}' failed for FOV {fov_id} round {round_id}",
                        exc,
                    )
                ) from exc
            record_matlab_boundary_phase(
                boundary_trace,
                phase_name="matlab_call",
                duration_ms=round((time.perf_counter() - matlab_call_started) * 1000.0, 3),
                seam_cost_key="matlab_call_ms",
                details={
                    "request_volume_shape_zyx": request_payload.get("volume_shape_zyx"),
                    "compute_tile_index": None if compute_tile is None else compute_tile.get("tile_index"),
                },
            )

            if not isinstance(metadata_json, str):
                raise ValueError(
                    f"MATLAB local registration entrypoint '{self.local_entrypoint}' must return a JSON string metadata payload"
                )

            result_validation_started = time.perf_counter()
            try:
                metadata = json.loads(metadata_json)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    _format_exception_message(
                        f"MATLAB local registration entrypoint '{self.local_entrypoint}' returned invalid JSON metadata",
                        exc,
                    )
                ) from exc
            if not isinstance(metadata, dict):
                raise ValueError("MATLAB local-registration metadata payload must decode to a JSON object")

            validated_flow_output_path = self._validate_local_response_metadata(
                metadata,
                round_id=round_id,
                reference_round=reference_round,
                tmpdir_path=tmpdir_path,
                expected_shape_zyx=(int(ref_volume.shape[0]), int(ref_volume.shape[1]), int(ref_volume.shape[2])),
            )
            flow_3d = self._load_local_flow_zyx(
                validated_flow_output_path,
                metadata,
                round_id=round_id,
            )
            record_matlab_boundary_phase(
                boundary_trace,
                phase_name="result_validation",
                duration_ms=round((time.perf_counter() - result_validation_started) * 1000.0, 3),
                seam_cost_key="result_validation_ms",
                details={
                    "reported_step_count": len(metadata.get("steps", [])) if isinstance(metadata.get("steps"), list) else 0,
                    "flow_shape": [int(value) for value in flow_3d.shape],
                },
            )

        finalized_boundary_trace = finalize_matlab_boundary_trace(
            boundary_trace,
            session=self._session_lifecycle,
            engine_reused_this_call=bool(engine_acquire.get("engine_reused_this_call", False)),
        )

        return {
            "flow_3d": flow_3d,
            "backend_metadata": {
                "provider": "matlab",
                "mode": "experimental_local_kernel_swap",
                "runtime": {
                    "runtime_path": str(self.runtime_dir),
                    "runtime_manifest": str(self.runtime_dir / MATLAB_REGISTRATION_RUNTIME_MANIFEST_NAME),
                    "entrypoint": self.local_entrypoint,
                },
                "request": request_payload,
                "runtime_files": runtime_files,
                "matlab_metadata": metadata,
                "normalized_result": {
                    "flow_3d_shape": [int(value) for value in flow_3d.shape],
                    "flow_3d_dtype": str(flow_3d.dtype),
                    "mean_abs_displacement": float(np.abs(flow_3d).mean()),
                },
                "boundary_instrumentation": finalized_boundary_trace,
                "session_lifecycle": snapshot_matlab_session_lifecycle(self._session_lifecycle),
                "session_lifecycle_summary": self._session_lifecycle_summary,
            },
        }


MATLABGlobalRegistrationBackend = MATLABRegistrationBackend
