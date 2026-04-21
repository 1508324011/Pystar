from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, Mapping, Optional

import numpy as np
import tifffile
from numpy.typing import NDArray
from scipy.io import loadmat

from .infrastructure import ExperimentConfig
from .matlab_engine_bootstrap import close_matlab_engine_best_effort, load_matlab_engine_module


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


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _format_exception_message(prefix: str, exc: Exception) -> str:
    detail = str(exc).strip()
    if detail:
        return f"{prefix}: {detail}"
    return f"{prefix} ({exc.__class__.__name__})"


def resolve_matlab_registration_runtime_path(config: ExperimentConfig) -> Path:
    matlab_cfg = config.providers.matlab.registration

    runtime_path = matlab_cfg.runtime_path
    if not runtime_path.is_absolute():
        runtime_path = _repo_root() / runtime_path
    return runtime_path.resolve()


def load_matlab_registration_runtime_manifest(runtime_dir: Path) -> Dict[str, Any]:
    manifest_path = runtime_dir / MATLAB_REGISTRATION_RUNTIME_MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"MATLAB registration runtime manifest is missing: {manifest_path}. "
            "Expected repo-local manifest for matlab_extracted registration backend."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"MATLAB registration runtime manifest must be a JSON object: {manifest_path}")

    required_files = manifest.get("required_files")
    optional_files = manifest.get("optional_files", [])
    entrypoint = manifest.get("entrypoint")
    local_entrypoint = manifest.get("local_entrypoint")
    if not isinstance(required_files, list) or not required_files:
        raise ValueError("MATLAB registration runtime manifest must declare a non-empty required_files list")
    if not isinstance(optional_files, list):
        raise ValueError("MATLAB registration runtime manifest optional_files must be a list")
    if not isinstance(entrypoint, str) or not entrypoint.strip():
        raise ValueError("MATLAB registration runtime manifest must declare a non-empty entrypoint")
    if local_entrypoint is not None and (not isinstance(local_entrypoint, str) or not local_entrypoint.strip()):
        raise ValueError("MATLAB registration runtime manifest local_entrypoint must be a non-empty string when present")

    for bucket_name, bucket in (("required_files", required_files), ("optional_files", optional_files)):
        for item in bucket:
            if not isinstance(item, dict):
                raise ValueError(f"MATLAB registration runtime manifest {bucket_name} entries must be JSON objects")
            for key in ("name", "source_path", "role"):
                value = item.get(key)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"MATLAB registration runtime manifest entry in {bucket_name} is missing non-empty '{key}'"
                    )

    declared_filenames = {
        item["name"]
        for bucket_name in ("required_files", "optional_files")
        for item in manifest.get(bucket_name, [])
        if isinstance(item, Mapping) and isinstance(item.get("name"), str)
    }
    if local_entrypoint is not None and f"{local_entrypoint}.m" not in declared_filenames:
        raise ValueError(
            "MATLAB registration runtime manifest must declare the configured local_entrypoint file. "
            f"Missing {local_entrypoint!r}.m in runtime manifest"
        )

    return manifest


def _validate_runtime_entrypoint_contract(
    runtime_manifest: Mapping[str, Any],
    configured_entrypoint: str,
) -> None:
    manifest_entrypoint = runtime_manifest.get("entrypoint")
    if configured_entrypoint != manifest_entrypoint:
        raise ValueError(
            "providers.matlab.registration.entrypoint must match the repo-local MATLAB runtime manifest. "
            f"Config entrypoint={configured_entrypoint!r}, manifest entrypoint={manifest_entrypoint!r}"
        )

    declared_filenames = {
        item["name"]
        for bucket_name in ("required_files", "optional_files")
        for item in runtime_manifest.get(bucket_name, [])
        if isinstance(item, Mapping) and isinstance(item.get("name"), str)
    }
    expected_entrypoint_file = f"{configured_entrypoint}.m"
    if expected_entrypoint_file not in declared_filenames:
        raise ValueError(
            "MATLAB registration runtime manifest must declare the configured entrypoint file. "
            f"Missing {expected_entrypoint_file!r} in runtime manifest"
        )


def _load_matlab_engine_factory() -> Callable[[], Any]:
    matlab_engine = load_matlab_engine_module(
        consumer="registration global/local provider='matlab'",
    )

    start_matlab = getattr(matlab_engine, "start_matlab", None)
    if start_matlab is None or not callable(start_matlab):
        raise RuntimeError("Imported 'matlab.engine' module does not expose callable start_matlab()")
    return start_matlab


class MATLABRegistrationBackend:
    def __init__(
        self,
        config: ExperimentConfig,
        *,
        engine_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.config = config
        self.engine_factory = engine_factory
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
        self._engine: Any = None

    def close(self) -> None:
        if self._engine is None:
            return
        try:
            close_matlab_engine_best_effort(
                self._engine,
                consumer="registration provider='matlab'",
            )
        finally:
            self._engine = None

    def _ensure_engine(self) -> Any:
        if self._engine is not None:
            return self._engine

        factory = self.engine_factory or _load_matlab_engine_factory()
        try:
            engine = factory()
        except Exception as exc:  # pragma: no cover - exact engine exception type depends on MATLAB install
            raise RuntimeError(
                _format_exception_message(
                    "Failed to start MATLAB Engine for registration provider='matlab'",
                    exc,
                )
            ) from exc

        try:
            engine.addpath(str(self.runtime_dir), nargout=0)
        except Exception as exc:  # pragma: no cover - exact engine exception type depends on MATLAB install
            try:
                engine.quit()
            except Exception:
                pass
            raise RuntimeError(
                _format_exception_message(
                    f"Failed to add MATLAB registration runtime path: {self.runtime_dir}",
                    exc,
                )
            ) from exc

        self._engine = engine
        return engine

    def _resolve_entrypoint_callable(self, entrypoint_name: str) -> Any:
        engine = self._ensure_engine()
        try:
            return getattr(engine, entrypoint_name)
        except AttributeError as exc:
            raise RuntimeError(
                f"MATLAB runtime path {self.runtime_dir} does not expose entrypoint '{entrypoint_name}'"
            ) from exc

    def _collect_runtime_file_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for bucket_name, required_default in (("required_files", True), ("optional_files", False)):
            for item in self.runtime_manifest.get(bucket_name, []):
                file_path = self.runtime_dir / item["name"]
                is_required = bool(item.get("required", required_default))
                is_used = is_required
                if is_used and not file_path.exists():
                    raise FileNotFoundError(
                        f"Required MATLAB registration runtime file is missing: {file_path}. "
                        "matlab_extracted registration cannot proceed."
                    )

                record = {
                    "name": item["name"],
                    "required": is_required,
                    "used": is_used,
                    "role": item["role"],
                    "source_path": item["source_path"],
                }
                if file_path.exists():
                    record["sha256"] = _sha256_file(file_path)
                records.append(record)

        return records

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

        steps = metadata.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ValueError("MATLAB registration metadata must declare a non-empty steps list")
        for index, step in enumerate(steps):
            if not isinstance(step, Mapping):
                raise ValueError(f"MATLAB registration step #{index} must be a mapping")
            name = step.get("name")
            duration_ms = step.get("duration_ms")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"MATLAB registration step #{index} is missing a non-empty name")
            if not isinstance(duration_ms, (int, float)) or duration_ms < 0:
                raise ValueError(
                    f"MATLAB registration step '{name}' must report a non-negative duration_ms"
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

        flow_output_path_value = metadata.get("flow_output_path")
        if not isinstance(flow_output_path_value, str) or not flow_output_path_value.strip():
            raise ValueError("MATLAB local-registration metadata must declare a non-empty flow_output_path")
        flow_output_path = Path(flow_output_path_value)
        if not flow_output_path.is_absolute():
            flow_output_path = tmpdir_path / flow_output_path
        flow_output_path = flow_output_path.resolve()
        if flow_output_path.parent != tmpdir_path.resolve():
            raise ValueError(
                "MATLAB local-registration flow output must stay inside the staged temporary directory: "
                f"{flow_output_path}"
            )
        if not flow_output_path.exists():
            raise FileNotFoundError(
                f"MATLAB local-registration reported flow output that does not exist: {flow_output_path}"
            )

        steps = metadata.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ValueError("MATLAB local-registration metadata must declare a non-empty steps list")
        for index, step in enumerate(steps):
            if not isinstance(step, Mapping):
                raise ValueError(f"MATLAB local-registration step #{index} must be a mapping")
            name = step.get("name")
            duration_ms = step.get("duration_ms")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"MATLAB local-registration step #{index} is missing a non-empty name")
            if not isinstance(duration_ms, (int, float)) or duration_ms < 0:
                raise ValueError(
                    f"MATLAB local-registration step '{name}' must report a non-negative duration_ms"
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
            "runtime_files": self._collect_runtime_file_records(),
            "round_results": normalized_round_results,
        }
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

        matlab_callable = self._resolve_entrypoint_callable(self.entrypoint)

        with TemporaryDirectory(prefix=f"pystar_matlab_registration_fov{fov_id}_round{round_id}_") as tmpdir:
            tmpdir_path = Path(tmpdir)
            ref_path = tmpdir_path / f"reference_round_{reference_round}.tif"
            moving_path = tmpdir_path / f"moving_round_{round_id}.tif"
            tifffile.imwrite(ref_path, ref_volume)
            tifffile.imwrite(moving_path, mov_volume)

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

        if not isinstance(metadata_json, str):
            raise ValueError(
                f"MATLAB registration entrypoint '{self.entrypoint}' must return a JSON string metadata payload"
            )

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
                "matlab_metadata": metadata,
                "normalized_result": {
                    "global_shift_3d": [
                        float(global_shift_3d[0]),
                        float(global_shift_3d[1]),
                        float(global_shift_3d[2]),
                    ],
                    "global_corr": global_corr,
                },
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

        ref_volume = self._normalize_input_volume(reference_volume)
        mov_volume = self._normalize_input_volume(moving_volume)
        request_payload = build_matlab_local_registration_plan(
            self.config,
            fov_id=fov_id,
            round_id=round_id,
            reference_round=reference_round,
            scope_descriptor=scope_descriptor,
            volume_shape_zyx=(int(ref_volume.shape[0]), int(ref_volume.shape[1]), int(ref_volume.shape[2])),
            compute_tile=compute_tile,
        )

        matlab_callable = self._resolve_entrypoint_callable(self.local_entrypoint)

        with TemporaryDirectory(prefix=f"pystar_matlab_local_registration_fov{fov_id}_round{round_id}_") as tmpdir:
            tmpdir_path = Path(tmpdir)
            ref_path = tmpdir_path / f"reference_round_{reference_round}.tif"
            moving_path = tmpdir_path / f"moving_round_{round_id}.tif"
            flow_output_path = tmpdir_path / f"local_flow_round_{round_id}.mat"
            tifffile.imwrite(ref_path, ref_volume)
            tifffile.imwrite(moving_path, mov_volume)
            request_payload["flow_output_path"] = str(flow_output_path)

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

            if not isinstance(metadata_json, str):
                raise ValueError(
                    f"MATLAB local registration entrypoint '{self.local_entrypoint}' must return a JSON string metadata payload"
                )

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
                "matlab_metadata": metadata,
                "normalized_result": {
                    "flow_3d_shape": [int(value) for value in flow_3d.shape],
                    "flow_3d_dtype": str(flow_3d.dtype),
                    "mean_abs_displacement": float(np.abs(flow_3d).mean()),
                },
            },
        }


MATLABGlobalRegistrationBackend = MATLABRegistrationBackend
