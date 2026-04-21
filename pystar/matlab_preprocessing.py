from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import numpy as np
import tifffile
import yaml

from .infrastructure import ExperimentConfig
from .io import (
    MATLAB_STAGE_CONFIG_SURFACES,
    MATLAB_STAGE_PYTHON_OWNED_ARTIFACTS,
    get_fov_output_structure,
    get_matlab_stage_contract,
)
from .matlab_engine_bootstrap import close_matlab_engine_best_effort, load_matlab_engine_module


PREPROCESSING_PROVENANCE_VERSION = "1.0"
MATLAB_RUNTIME_MANIFEST_NAME = "runtime_manifest.json"
MATLAB_SUPPORTED_SEQUENCE_METHODS = {
    "none",
    "min_max_normalize",
    "histogram_match",
    "morpho_reconstruction_contrast",
}
MATLAB_SUPPORTED_HISTOGRAM_SCOPES = {"inter_round", "intra_round"}


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _format_exception_message(prefix: str, exc: Exception) -> str:
    detail = str(exc).strip()
    if detail:
        return f"{prefix}: {detail}"
    return f"{prefix} ({exc.__class__.__name__})"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def write_preprocessing_provenance(base_dir: Path, fov_id: int, provenance: Mapping[str, Any]) -> Path:
    paths = get_fov_output_structure(base_dir, fov_id)
    output_path = paths["qc"] / "preprocessing_provenance.yaml"
    temp_path = output_path.with_suffix(".yaml.tmp")
    merged_provenance = dict(provenance)
    existing_contract = provenance.get("matlab_stage_contract")
    if isinstance(existing_contract, Mapping):
        stage_contract = dict(existing_contract)
    else:
        providers_used = provenance.get("providers_used")
        matlab_requested = bool(
            provenance.get("provider") == "matlab"
            or provenance.get("backend") in {"matlab_extracted", "provider_dispatch"}
            or provenance.get("provider_mode") in {"matlab_only", "mixed"}
            or (isinstance(providers_used, Sequence) and "matlab" in providers_used)
        )
        declared_intent = (
            provenance.get("provider_mode")
            or provenance.get("provider")
            or provenance.get("backend")
            or "unknown"
        )
        stage_contract = {
            "declared_intent": declared_intent,
            "matlab_requested": matlab_requested,
            "config_surface": list(MATLAB_STAGE_CONFIG_SURFACES["preprocessing"]),
            "artifact_owner": "python_pystar",
            "python_owned_artifacts": list(MATLAB_STAGE_PYTHON_OWNED_ARTIFACTS["preprocessing"]),
            "failure_contract": "fail_loud_no_fallback",
            "current_support_status": "debug_only" if matlab_requested else "not_selected",
            "promotion_blockers": (
                [
                    "representative_benchmark_recovery_pending",
                    "production_verification_pending",
                ]
                if matlab_requested
                else []
            ),
        }
    merged_provenance["matlab_stage_contract"] = stage_contract
    temp_path.write_text(
        yaml.safe_dump(merged_provenance, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    temp_path.replace(output_path)
    return output_path


def resolve_matlab_runtime_path(config: ExperimentConfig) -> Path:
    matlab_cfg = config.providers.matlab.preprocessing

    runtime_path = matlab_cfg.runtime_path
    if not runtime_path.is_absolute():
        runtime_path = _repo_root() / runtime_path
    return runtime_path.resolve()


def load_matlab_runtime_manifest(runtime_dir: Path) -> Dict[str, Any]:
    manifest_path = runtime_dir / MATLAB_RUNTIME_MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"MATLAB runtime manifest is missing: {manifest_path}. "
            "Expected repo-local manifest for matlab_extracted preprocessing backend."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"MATLAB runtime manifest must be a JSON object: {manifest_path}")

    required_files = manifest.get("required_files")
    optional_files = manifest.get("optional_files", [])
    entrypoint = manifest.get("entrypoint")
    if not isinstance(required_files, list) or not required_files:
        raise ValueError("MATLAB runtime manifest must declare a non-empty required_files list")
    if not isinstance(optional_files, list):
        raise ValueError("MATLAB runtime manifest optional_files must be a list")
    if not isinstance(entrypoint, str) or not entrypoint.strip():
        raise ValueError("MATLAB runtime manifest must declare a non-empty entrypoint")

    for bucket_name, bucket in (("required_files", required_files), ("optional_files", optional_files)):
        for item in bucket:
            if not isinstance(item, dict):
                raise ValueError(f"MATLAB runtime manifest {bucket_name} entries must be JSON objects")
            for key in ("name", "source_path", "role"):
                value = item.get(key)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"MATLAB runtime manifest entry in {bucket_name} is missing non-empty '{key}'"
                    )

    return manifest


def _validate_runtime_entrypoint_contract(
    runtime_manifest: Mapping[str, Any],
    configured_entrypoint: str,
) -> None:
    manifest_entrypoint = runtime_manifest.get("entrypoint")
    if configured_entrypoint != manifest_entrypoint:
        raise ValueError(
            "providers.matlab.preprocessing.entrypoint must match the repo-local MATLAB runtime manifest. "
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
            "MATLAB runtime manifest must declare the configured entrypoint file. "
            f"Missing {expected_entrypoint_file!r} in runtime manifest"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _validate_filename_pattern_for_matlab_backend(filename_pattern: str) -> None:
    required_tokens = ("round{round", "Position{fov}", "_ch{ch}")
    if all(token in filename_pattern for token in required_tokens):
        return
    raise ValueError(
        "matlab_extracted preprocessing currently supports Leica-style filename patterns containing "
        "'round{round', 'Position{fov}', and '_ch{ch}'. "
        f"Current pattern: {filename_pattern!r}"
    )


def _select_round_ids(config: ExperimentConfig, target_rounds: Optional[list[int]]) -> list[int]:
    all_rounds = sorted(int(round_id) for round_id in config.dataset.round_structure.keys())
    if target_rounds is None:
        return all_rounds

    requested = sorted({int(round_id) for round_id in target_rounds})
    selected = [round_id for round_id in requested if round_id in all_rounds]
    if not selected:
        raise ValueError(f"No valid round ids remain after filtering target_rounds={target_rounds!r}")
    return selected


def _resolve_uniform_seq_channels(config: ExperimentConfig, round_ids: list[int]) -> list[int]:
    round_channel_map: Dict[int, list[int]] = {}
    for round_id in round_ids:
        channel_ids = config.dataset.round_structure.get(round_id)
        if channel_ids is None:
            raise ValueError(f"Round {round_id} is missing from dataset.round_structure")
        seq_channels = sorted(
            int(channel_id)
            for channel_id in channel_ids
            if config.dataset.channel_roles.get(int(channel_id)) == "seq"
        )
        if not seq_channels:
            raise ValueError(f"Round {round_id} does not contain any seq channels for preprocessing")
        round_channel_map[round_id] = seq_channels

    canonical = round_channel_map[round_ids[0]]
    mismatching = {
        round_id: channels
        for round_id, channels in round_channel_map.items()
        if channels != canonical
    }
    if mismatching:
        raise ValueError(
            "matlab_extracted preprocessing v1 requires the same seq channel layout for every selected round. "
            f"Found per-round seq channel mismatch: {mismatching}"
        )
    return canonical


def build_matlab_preprocessing_plan(
    config: ExperimentConfig,
    fov_id: int,
    target_rounds: Optional[list[int]] = None,
    *,
    sequence: Optional[Sequence[Any]] = None,
    filename_pattern: Optional[str] = None,
    input_sub_dir: Optional[str] = None,
) -> Dict[str, Any]:
    matlab_cfg = config.providers.matlab.preprocessing
    active_sequence = list(sequence) if sequence is not None else [
        step for step in config.pipeline.preprocessing.sequence if getattr(step, "provider", "native") == "matlab"
    ]
    if not active_sequence:
        raise ValueError("No provider='matlab' preprocessing steps were selected for MATLAB execution")

    effective_filename_pattern = filename_pattern or config.dataset.filename_pattern
    _validate_filename_pattern_for_matlab_backend(effective_filename_pattern)

    round_ids = _select_round_ids(config, target_rounds)
    seq_channels = _resolve_uniform_seq_channels(config, round_ids)
    raw_sequence: list[dict[str, Any]] = []
    skipped_steps: list[dict[str, Any]] = []
    equalize_methods: list[str] = []
    apply_min_max_normalize = False
    morphology_enabled = False
    morphology_radius = int(matlab_cfg.morphology.radius)
    morphology_height = int(matlab_cfg.morphology.height)

    for index, step in enumerate(active_sequence):
        raw_sequence.append(
            {
                "index": index,
                "method": step.method,
                "provider": getattr(step, "provider", "matlab"),
                "params": dict(step.params),
            }
        )

        if step.method not in MATLAB_SUPPORTED_SEQUENCE_METHODS:
            raise ValueError(
                f"matlab_extracted preprocessing does not support step '{step.method}'. "
                f"Supported steps: {sorted(MATLAB_SUPPORTED_SEQUENCE_METHODS)}"
            )

        if step.method == "none":
            continue

        if step.method == "min_max_normalize":
            if apply_min_max_normalize:
                raise ValueError("matlab_extracted preprocessing supports at most one min_max_normalize step")
            apply_min_max_normalize = True
            continue

        if step.method == "histogram_match":
            scope = step.params.get("scope")
            if scope not in MATLAB_SUPPORTED_HISTOGRAM_SCOPES:
                raise ValueError(
                    "matlab_extracted histogram_match requires params.scope to be 'inter_round' or 'intra_round'"
                )
            if scope == "inter_round" and 1 not in round_ids:
                skipped_steps.append(
                    {
                        "method": step.method,
                        "scope": scope,
                        "reason": "reference_round_1_not_selected",
                    }
                )
                continue
            equalize_methods.append(scope)
            continue

        if step.method == "morpho_reconstruction_contrast":
            if morphology_enabled:
                raise ValueError(
                    "matlab_extracted preprocessing supports at most one morpho_reconstruction_contrast step"
                )
            morphology_enabled = True
            if "radius" in step.params:
                morphology_radius = int(step.params["radius"])
            if "height" in step.params:
                morphology_height = int(step.params["height"])
            if morphology_radius <= 0 or morphology_height <= 0:
                raise ValueError("matlab_extracted morphology parameters must be positive integers")

    return {
        "fov_id": int(fov_id),
        "sub_dir": input_sub_dir if input_sub_dir is not None else f"Position{int(fov_id)}",
        "round_ids": round_ids,
        "seq_channels": seq_channels,
        "reference_round": 1,
        "expected_z_slices": int(config.dataset.dimensions["z"]),
        "loader_input_format": matlab_cfg.loader_input_format,
        "loader_output_dtype": matlab_cfg.loader_output_dtype,
        "use_gpu": bool(matlab_cfg.use_gpu),
        "apply_min_max_normalize": apply_min_max_normalize,
        "equalize_methods": equalize_methods,
        "morphology": {
            "enabled": morphology_enabled,
            "method": matlab_cfg.morphology.method,
            "radius": morphology_radius,
            "height": morphology_height,
        },
        "raw_sequence": raw_sequence,
        "skipped_steps": skipped_steps,
        "filename_pattern": effective_filename_pattern,
    }


def _load_matlab_engine_factory() -> Callable[[], Any]:
    matlab_engine = load_matlab_engine_module(
        consumer="preprocessing step provider='matlab'",
    )

    start_matlab = getattr(matlab_engine, "start_matlab", None)
    if start_matlab is None or not callable(start_matlab):
        raise RuntimeError("Imported 'matlab.engine' module does not expose callable start_matlab()")
    return start_matlab


class MATLABPreprocessingBackend:
    def __init__(
        self,
        config: ExperimentConfig,
        *,
        engine_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.config = config
        self.engine_factory = engine_factory
        self.runtime_dir = resolve_matlab_runtime_path(config)
        self.runtime_manifest = load_matlab_runtime_manifest(self.runtime_dir)
        self.entrypoint = config.providers.matlab.preprocessing.entrypoint
        _validate_runtime_entrypoint_contract(self.runtime_manifest, self.entrypoint)
        self._engine: Any = None

    def close(self) -> None:
        if self._engine is None:
            return
        try:
            close_matlab_engine_best_effort(
                self._engine,
                consumer="preprocessing provider='matlab'",
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
                    "Failed to start MATLAB Engine for preprocessing provider='matlab'",
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
                    f"Failed to add MATLAB preprocessing runtime path: {self.runtime_dir}",
                    exc,
                    )
                ) from exc

        self._engine = engine
        return engine

    def _resolve_callable(self) -> Any:
        engine = self._ensure_engine()
        try:
            return getattr(engine, self.entrypoint)
        except AttributeError as exc:
            raise RuntimeError(
                f"MATLAB runtime path {self.runtime_dir} does not expose entrypoint '{self.entrypoint}'"
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
                        f"Required MATLAB runtime file is missing: {file_path}. "
                        "matlab_extracted preprocessing cannot proceed."
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

    def _validate_result_metadata(
        self,
        metadata: Mapping[str, Any],
        clean_output_dir: Path,
        plan: Mapping[str, Any],
    ) -> None:
        output_files = metadata.get("output_files")
        if not isinstance(output_files, list) or not output_files:
            raise ValueError("MATLAB preprocessing entrypoint did not report any output_files")

        expected_file_count = len(plan["round_ids"]) * len(plan["seq_channels"])
        if len(output_files) != expected_file_count:
            raise ValueError(
                "MATLAB preprocessing entrypoint returned the wrong number of clean output files: "
                f"expected {expected_file_count}, got {len(output_files)}"
            )

        output_shape = metadata.get("output_shape")
        if not isinstance(output_shape, list) or not output_shape:
            raise ValueError("MATLAB preprocessing entrypoint must report output_shape as a non-empty list")
        if any(not isinstance(dim, int) or dim <= 0 for dim in output_shape):
            raise ValueError("MATLAB preprocessing metadata output_shape must contain positive integer dimensions")
        if len(output_shape) >= 3 and output_shape[2] != int(plan["expected_z_slices"]):
            raise ValueError(
                "MATLAB preprocessing output_shape reports an unexpected z-depth: "
                f"expected {plan['expected_z_slices']}, got {output_shape[2]}"
            )

        steps = metadata.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ValueError("MATLAB preprocessing entrypoint did not report executed preprocessing steps")
        for index, step in enumerate(steps):
            if not isinstance(step, Mapping):
                raise ValueError(f"MATLAB preprocessing step #{index} must be a mapping")
            name = step.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"MATLAB preprocessing step #{index} is missing a non-empty name")
            duration_ms = step.get("duration_ms")
            if not isinstance(duration_ms, (int, float)) or duration_ms < 0:
                raise ValueError(
                    f"MATLAB preprocessing step '{name}' must report a non-negative duration_ms"
                )

        expected_dtype = np.dtype(str(plan["loader_output_dtype"]))
        clean_output_dir = clean_output_dir.resolve()
        expected_filenames = {
            f"clean_fov_{plan['fov_id']}_round_{round_id}_ch_{channel_id}.tif"
            for round_id in plan["round_ids"]
            for channel_id in plan["seq_channels"]
        }
        observed_filenames: set[str] = set()

        for output_file in output_files:
            if not isinstance(output_file, str) or not output_file:
                raise ValueError("MATLAB preprocessing metadata output_files entries must be non-empty strings")
            output_path = Path(output_file)
            if not output_path.is_absolute():
                output_path = clean_output_dir / output_path
            output_path = output_path.resolve()
            if output_path.parent != clean_output_dir:
                raise ValueError(
                    "MATLAB preprocessing output file must stay inside the canonical clean_data directory: "
                    f"{output_path}"
                )
            if not output_path.exists():
                raise FileNotFoundError(
                    f"MATLAB preprocessing reported output file that does not exist: {output_path}"
                )

            observed_filenames.add(output_path.name)

            clean_volume = tifffile.imread(output_path)
            if clean_volume.dtype != expected_dtype:
                raise ValueError(
                    "MATLAB preprocessing output dtype does not match the declared handoff contract: "
                    f"expected {expected_dtype}, got {clean_volume.dtype} for {output_path.name}"
                )
            if clean_volume.ndim != 3:
                raise ValueError(
                    "MATLAB preprocessing output must be a 3D TIFF stack readable by downstream stages: "
                    f"got ndim={clean_volume.ndim} for {output_path.name}"
                )
            if int(clean_volume.shape[0]) != int(plan["expected_z_slices"]):
                raise ValueError(
                    "MATLAB preprocessing output z-depth does not match dataset dimensions: "
                    f"expected {plan['expected_z_slices']}, got {clean_volume.shape[0]} for {output_path.name}"
                )

        if observed_filenames != expected_filenames:
            missing = sorted(expected_filenames - observed_filenames)
            unexpected = sorted(observed_filenames - expected_filenames)
            raise ValueError(
                "MATLAB preprocessing output filenames do not match the canonical clean_data handoff contract. "
                f"Missing={missing}, unexpected={unexpected}"
            )

    def execute_sequence(
        self,
        fov_id: int,
        *,
        sequence: Optional[Sequence[Any]] = None,
        input_root: Optional[Path] = None,
        input_filename_pattern: Optional[str] = None,
        output_dir: Optional[Path] = None,
        target_rounds: Optional[list[int]] = None,
        input_sub_dir: Optional[str] = None,
        segment_label: Optional[str] = None,
    ) -> Dict[str, Any]:
        plan = build_matlab_preprocessing_plan(
            self.config,
            fov_id,
            target_rounds=target_rounds,
            sequence=sequence,
            filename_pattern=input_filename_pattern,
            input_sub_dir=input_sub_dir,
        )
        runtime_files = self._collect_runtime_file_records()

        if output_dir is None:
            base_dir = Path(self.config.pipeline.output.directory)
            output_paths = get_fov_output_structure(base_dir, fov_id)
            clean_output_dir = output_paths["cleaned"]
        else:
            clean_output_dir = Path(output_dir)
            clean_output_dir.mkdir(parents=True, exist_ok=True)

        resolved_input_root = Path(input_root) if input_root is not None else Path(self.config.dataset.raw_data_path)
        plan_for_matlab = dict(plan)
        plan_for_matlab["clean_output_dir"] = str(clean_output_dir)
        plan_for_matlab["input_root"] = str(resolved_input_root)

        matlab_callable = self._resolve_callable()
        config_json = json.dumps(plan_for_matlab, sort_keys=True)
        started_at = _iso_utc_now()
        start_time = time.perf_counter()

        try:
            metadata_json = matlab_callable(
                str(resolved_input_root),
                plan_for_matlab["sub_dir"],
                str(clean_output_dir),
                config_json,
                nargout=1,
            )
        except Exception as exc:  # pragma: no cover - exact engine exception type depends on MATLAB install
            raise RuntimeError(
                _format_exception_message(
                    f"MATLAB preprocessing entrypoint '{self.entrypoint}' failed for FOV {fov_id}",
                    exc,
                )
            ) from exc

        finished_at = _iso_utc_now()
        duration_ms = round((time.perf_counter() - start_time) * 1000.0, 3)

        if not isinstance(metadata_json, str):
            raise ValueError(
                f"MATLAB preprocessing entrypoint '{self.entrypoint}' must return a JSON string metadata payload"
            )

        try:
            metadata = json.loads(metadata_json)
        except json.JSONDecodeError as exc:
            raise ValueError(
                _format_exception_message(
                    f"MATLAB preprocessing entrypoint '{self.entrypoint}' returned invalid JSON metadata",
                    exc,
                )
            ) from exc
        if not isinstance(metadata, dict):
            raise ValueError("MATLAB preprocessing metadata payload must decode to a JSON object")
        self._validate_result_metadata(metadata, clean_output_dir, plan)

        return {
            "version": PREPROCESSING_PROVENANCE_VERSION,
            "generated_at": finished_at,
            "fov_id": int(fov_id),
            "backend": "matlab_extracted",
            "provider": "matlab",
            "matlab_stage_contract": get_matlab_stage_contract(self.config, "preprocessing"),
            "segment_label": segment_label,
            "duration_ms": duration_ms,
            "started_at": started_at,
            "finished_at": finished_at,
            "input_contract": {
                "raw_data_path": str(resolved_input_root),
                "filename_pattern": plan["filename_pattern"],
                "sub_dir": plan["sub_dir"],
                "selected_rounds": plan["round_ids"],
                "seq_channels": plan["seq_channels"],
                "expected_z_slices": plan["expected_z_slices"],
            },
            "runtime": {
                "runtime_path": str(self.runtime_dir),
                "runtime_manifest": str(self.runtime_dir / MATLAB_RUNTIME_MANIFEST_NAME),
                "entrypoint": self.entrypoint,
            },
            "normalized_backend_config": {
                "loader_input_format": plan["loader_input_format"],
                "loader_output_dtype": plan["loader_output_dtype"],
                "use_gpu": plan["use_gpu"],
                "apply_min_max_normalize": plan["apply_min_max_normalize"],
                "equalize_methods": plan["equalize_methods"],
                "morphology": plan["morphology"],
                "skipped_steps": plan["skipped_steps"],
            },
            "raw_sequence": plan["raw_sequence"],
            "matlab_files": runtime_files,
            "matlab_metadata": metadata,
        }

    def preprocess_fov(self, fov_id: int, target_rounds: Optional[list[int]] = None) -> Dict[str, Any]:
        return self.execute_sequence(fov_id, target_rounds=target_rounds)
