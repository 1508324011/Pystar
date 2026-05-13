"""MATLAB-backed preprocessing boundary for PyStar.

Native preprocessing atoms are the default execution path.  When a preprocessing
step explicitly selects `provider: matlab`, this module converts the selected
step sequence into a small JSON plan, calls the repo-local MATLAB runtime, and
validates that MATLAB wrote the same canonical `clean_data/` TIFF artifacts that
the native sanitizer would produce.  The handoff is intentionally explicit:
unsupported preprocessing methods fail before MATLAB starts, and MATLAB errors
are never hidden behind a native fallback.
"""

from __future__ import annotations

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
from .matlab_engine_bootstrap import (
    MATLABSessionCapsule,
    MatlabSharedSessionOwner,
    create_matlab_boundary_trace,
    finalize_matlab_boundary_trace,
    load_matlab_engine_factory,
    record_matlab_boundary_phase,
    snapshot_matlab_session_lifecycle,
)
from .matlab_runtime import (
    collect_runtime_file_records as _collect_runtime_file_records_from_manifest,
    format_exception_message as _format_exception_message,
    load_validated_runtime_manifest,
    resolve_repo_runtime_path,
    trusted_matlab_runtime_root,
    validate_configured_entrypoint_contract,
    validate_runtime_step_metadata,
)


PREPROCESSING_PROVENANCE_VERSION = "1.0"
MATLAB_RUNTIME_MANIFEST_NAME = "runtime_manifest.json"
MATLAB_PREPROCESSING_PACKAGE_NAME = "pystar_preprocessing"
MATLAB_SUPPORTED_SEQUENCE_METHODS = {
    "none",
    "min_max_normalize",
    "histogram_match",
    "morpho_reconstruction_contrast",
}
MATLAB_SUPPORTED_HISTOGRAM_SCOPES = {"inter_round", "intra_round"}


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_preprocessing_provenance(base_dir: Path, fov_id: int, provenance: Mapping[str, Any]) -> Path:
    """Persist preprocessing provenance and the MATLAB-stage support contract.

    The provenance file is the downstream audit record for `clean_data/`: it says
    which provider path ran, which runtime files were used, and whether MATLAB was
    requested.  When the caller did not already include a stage contract, this
    helper derives one from the provider/backend fields so reports can still show
    the current support status and fail-loud boundary.
    """

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
    """Resolve the configured MATLAB preprocessing runtime inside the repo tree."""

    return resolve_repo_runtime_path(
        config.providers.matlab.preprocessing.runtime_path,
        config_label="providers.matlab.preprocessing.runtime_path",
        trusted_root=trusted_matlab_runtime_root(),
    )


def load_matlab_runtime_manifest(runtime_dir: Path) -> Dict[str, Any]:
    """Load and validate the preprocessing MATLAB runtime manifest."""

    manifest = load_validated_runtime_manifest(
        runtime_dir,
        manifest_label="MATLAB runtime manifest",
        missing_hint="Expected repo-local manifest for matlab_extracted preprocessing backend.",
        package_name=MATLAB_PREPROCESSING_PACKAGE_NAME,
        manifest_name=MATLAB_RUNTIME_MANIFEST_NAME,
    )
    return manifest


def _validate_runtime_entrypoint_contract(
    runtime_manifest: Mapping[str, Any],
    configured_entrypoint: str,
) -> None:
    validate_configured_entrypoint_contract(
        runtime_manifest,
        configured_entrypoint,
        config_label="providers.matlab.preprocessing.entrypoint",
        manifest_label="MATLAB runtime manifest",
    )


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
    """Convert selected preprocessing steps into a MATLAB execution plan.

    Only the MATLAB-runtime subset is supported here: `none`,
    `min_max_normalize`, `histogram_match`, and
    `morpho_reconstruction_contrast`.  The plan also freezes the selected rounds,
    sequencing channels, Leica-style raw filename pattern, expected z-depth, and
    output dtype so MATLAB can produce the exact canonical clean-image handoff
    that downstream PyStar stages consume.
    """

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
    factory, _factory_metrics = load_matlab_engine_factory(
        consumer="preprocessing step provider='matlab'",
    )
    return factory


class MATLABPreprocessingBackend:
    """Run MATLAB preprocessing and validate the canonical clean-image handoff.

    The backend owns one MATLAB Engine session capsule, validates the runtime
    manifest and entrypoint, stages an explicit plan, and checks every reported
    clean TIFF for filename, dtype, dimensionality, and z-depth.  The returned
    provenance includes boundary timings and MATLAB metadata; the image artifacts
    themselves remain Python-owned under the normal PyStar output schema.
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
        self.runtime_dir = resolve_matlab_runtime_path(config)
        self.runtime_manifest = load_matlab_runtime_manifest(self.runtime_dir)
        self.entrypoint = config.providers.matlab.preprocessing.entrypoint
        _validate_runtime_entrypoint_contract(self.runtime_manifest, self.entrypoint)
        self._session_capsule = MATLABSessionCapsule(
            consumer="preprocessing provider='matlab'",
            runtime_dir=self.runtime_dir,
            entrypoint=self.entrypoint,
            engine_factory=engine_factory,
            engine_factory_consumer="preprocessing step provider='matlab'",
            startup_failure_prefix="Failed to start MATLAB Engine for preprocessing provider='matlab'",
            addpath_failure_prefix="Failed to add MATLAB preprocessing runtime path",
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
        """Close the owned MATLAB Engine session if it was started."""

        self._session_capsule.close()

    def _ensure_engine(self) -> Any:
        return self._session_capsule.ensure_engine()

    def _consume_last_engine_acquire(self) -> dict[str, Any]:
        return self._session_capsule.consume_last_engine_acquire()

    def _resolve_callable(self) -> Any:
        return self._session_capsule.resolve_callable()

    def _resolve_runtime_file_records(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return self._session_capsule.validate_runtime_files(self._collect_runtime_file_records)

    def _collect_runtime_file_records(self) -> list[dict[str, Any]]:
        return _collect_runtime_file_records_from_manifest(
            self.runtime_manifest,
            self.runtime_dir,
            missing_required_prefix="Required MATLAB runtime file is missing",
            missing_required_suffix="matlab_extracted preprocessing cannot proceed.",
        )

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

        validate_runtime_step_metadata(
            metadata.get("steps"),
            missing_steps_message="MATLAB preprocessing entrypoint did not report executed preprocessing steps",
            step_label="MATLAB preprocessing step",
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
        """Execute one MATLAB preprocessing segment for a FOV.

        `sequence` may be a subset of the global preprocessing sequence when the
        native sanitizer splits mixed-provider execution into segments.  Optional
        `input_root` and `input_filename_pattern` allow a later MATLAB segment to
        consume intermediate native outputs while still writing the final clean
        images into the canonical `clean_data/` layout.
        """

        boundary_trace = create_matlab_boundary_trace(
            stage_name="matlab_preprocessing",
            runtime_dir=self.runtime_dir,
            entrypoint=self.entrypoint,
            session=self._session_lifecycle,
            call_scope={
                "fov_id": int(fov_id),
                "target_rounds": None if target_rounds is None else [int(round_id) for round_id in target_rounds],
                "segment_label": segment_label,
            },
        )
        plan = build_matlab_preprocessing_plan(
            self.config,
            fov_id,
            target_rounds=target_rounds,
            sequence=sequence,
            filename_pattern=input_filename_pattern,
            input_sub_dir=input_sub_dir,
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
        config_json = json.dumps(plan_for_matlab, sort_keys=True)
        started_at = _iso_utc_now()
        start_time = time.perf_counter()

        matlab_call_started = time.perf_counter()
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
        record_matlab_boundary_phase(
            boundary_trace,
            phase_name="matlab_call",
            duration_ms=round((time.perf_counter() - matlab_call_started) * 1000.0, 3),
            seam_cost_key="matlab_call_ms",
            details={
                "selected_round_count": len(plan["round_ids"]),
                "selected_channel_count": len(plan["seq_channels"]),
            },
        )

        finished_at = _iso_utc_now()
        duration_ms = round((time.perf_counter() - start_time) * 1000.0, 3)

        if not isinstance(metadata_json, str):
            raise ValueError(
                f"MATLAB preprocessing entrypoint '{self.entrypoint}' must return a JSON string metadata payload"
            )

        result_validation_started = time.perf_counter()
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
        record_matlab_boundary_phase(
            boundary_trace,
            phase_name="result_validation",
            duration_ms=round((time.perf_counter() - result_validation_started) * 1000.0, 3),
            seam_cost_key="result_validation_ms",
            details={
                "output_file_count": len(metadata.get("output_files", [])) if isinstance(metadata.get("output_files"), list) else 0,
                "step_count": len(metadata.get("steps", [])) if isinstance(metadata.get("steps"), list) else 0,
            },
        )
        finalized_boundary_trace = finalize_matlab_boundary_trace(
            boundary_trace,
            session=self._session_lifecycle,
            engine_reused_this_call=bool(engine_acquire.get("engine_reused_this_call", False)),
        )

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
            "boundary_instrumentation": finalized_boundary_trace,
            "session_lifecycle": snapshot_matlab_session_lifecycle(self._session_lifecycle),
            "session_lifecycle_summary": self._session_lifecycle_summary,
        }

    def preprocess_fov(self, fov_id: int, target_rounds: Optional[list[int]] = None) -> Dict[str, Any]:
        """Preprocess one FOV entirely through the MATLAB provider path."""

        return self.execute_sequence(fov_id, target_rounds=target_rounds)
