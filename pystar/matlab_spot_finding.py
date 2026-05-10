"""MATLAB-backed spot finding boundary for PyStar.

This module is the seam between Python-owned PyStar artifacts and the
repo-local MATLAB `max3d` runtime.  It stages one cleaned 3D volume at a time,
passes a small JSON execution plan to MATLAB, validates the returned metadata,
and normalizes the MATLAB CSV output back into PyStar's canonical
`z, y, x, intensity` spot table.  The MATLAB provider is deliberately explicit:
runtime files must come from `matlab_runtime/`, metadata must match the staged
call, and failures are raised rather than hidden behind a native fallback.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, Mapping, Optional

import numpy as np
import pandas as pd

from ._artifact_schemas import SpotTableSchema, validate_spot_table, wrap_table_read_error
from .infrastructure import ExperimentConfig
from .matlab_engine_bootstrap import (
    MATLABSessionCapsule,
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
    resolve_staged_output_path,
    trusted_matlab_runtime_root,
    validate_expected_3d_staged_volume_shape,
    validate_configured_entrypoint_contract,
    validate_runtime_step_metadata,
    write_staged_3d_volume_tiff,
)


MATLAB_SPOTFINDING_RUNTIME_MANIFEST_NAME = "runtime_manifest.json"
MATLAB_SPOTFINDING_REQUIRED_COLUMNS = ("z", "y", "x", "intensity")
MATLAB_SPOTFINDING_PACKAGE_NAME = "pystar_spotfinding"


def resolve_matlab_spotfinding_runtime_path(config: ExperimentConfig) -> Path:
    """Resolve and validate the repo-local MATLAB spot-finding runtime path."""

    return resolve_repo_runtime_path(
        config.providers.matlab.spot_finding.runtime_path,
        config_label="providers.matlab.spot_finding.runtime_path",
        trusted_root=trusted_matlab_runtime_root(),
    )


def load_matlab_spotfinding_runtime_manifest(runtime_dir: Path) -> Dict[str, Any]:
    """Load the MATLAB spot-finding runtime manifest and validate its schema."""

    manifest = load_validated_runtime_manifest(
        runtime_dir,
        manifest_label="MATLAB spot-finding runtime manifest",
        missing_hint="Expected repo-local manifest for MATLAB spot-finding provider.",
        package_name=MATLAB_SPOTFINDING_PACKAGE_NAME,
        manifest_name=MATLAB_SPOTFINDING_RUNTIME_MANIFEST_NAME,
    )
    return manifest


def _validate_runtime_entrypoint_contract(runtime_manifest: Mapping[str, Any], configured_entrypoint: str) -> None:
    validate_configured_entrypoint_contract(
        runtime_manifest,
        configured_entrypoint,
        config_label="providers.matlab.spot_finding.entrypoint",
        manifest_label="MATLAB spot-finding runtime manifest",
    )


def build_matlab_spotfinding_plan(
    config: ExperimentConfig,
    *,
    fov_id: int,
    round_id: int,
    channel_id: int,
    volume_shape_zyx: tuple[int, int, int],
) -> Dict[str, Any]:
    """Build the JSON-serializable plan consumed by the MATLAB `max3d` entrypoint.

    `volume_shape_zyx` records the staged Python volume shape before MATLAB sees
    it.  The returned plan intentionally carries both the PyStar algorithm name
    and `matlab_method="max3d"`: the former preserves pipeline provenance, while
    the latter tells the MATLAB runtime which kernel to execute.
    """

    matlab_cfg = config.providers.matlab.spot_finding
    peak_cfg = config.pipeline.spot_finding.peak_local_max
    return {
        "fov_id": int(fov_id),
        "round_id": int(round_id),
        "reference_round": int(config.pipeline.spot_finding.reference_round),
        "channel_id": int(channel_id),
        "algorithm": config.pipeline.spot_finding.algorithm,
        "matlab_method": "max3d",
        "threshold_rel": float(peak_cfg.threshold_rel),
        "min_distance": int(peak_cfg.min_distance),
        "exclude_border": bool(peak_cfg.exclude_border),
        "volume_shape_zyx": [int(value) for value in volume_shape_zyx],
        "volume_transfer_mode": matlab_cfg.volume_transfer_mode,
        "input_volume_dtype": matlab_cfg.input_volume_dtype,
    }


def _load_matlab_engine_factory() -> Callable[[], Any]:
    factory, _factory_metrics = load_matlab_engine_factory(
        consumer="spot_finding.provider='matlab'",
    )
    return factory


def _normalize_spot_dataframe(
    df: pd.DataFrame,
    *,
    fov_id: int,
    path: Path | str | None,
) -> pd.DataFrame:
    """Normalize MATLAB spot CSV columns to PyStar's `z, y, x, intensity` schema."""

    expected = SpotTableSchema().expected_description()

    if all(column in df.columns for column in MATLAB_SPOTFINDING_REQUIRED_COLUMNS):
        normalized = df.loc[:, list(MATLAB_SPOTFINDING_REQUIRED_COLUMNS)].copy()
    elif all(column in df.columns for column in ("x", "y", "z", "intensity")):
        normalized = df.loc[:, ["z", "y", "x", "intensity"]].copy()
    else:
        raise ValueError(
            f"FOV {fov_id} spot table schema error during matlab spot normalization at {path}: "
            "missing required columns for MATLAB spot output; expected either [z, y, x, intensity] "
            f"or [x, y, z, intensity]. Expected schema: {expected}"
        )

    validated = validate_spot_table(
        normalized,
        fov_id=fov_id,
        path=path,
        context="matlab spot normalization",
    )
    return validated.astype({"z": np.float32, "y": np.float32, "x": np.float32, "intensity": np.float32})


def _write_staged_volume_tiff(volume_path: Path, volume: Any) -> None:
    write_staged_3d_volume_tiff(volume_path, volume, owner_label="MATLAB spot-finding")


class MATLABSpotFindingBackend:
    """Execute the MATLAB spot-finding runtime behind PyStar's provider seam.

    A backend instance owns one MATLAB session capsule and validates all runtime
    files before the first call.  Each `find_spots` call stages a single cleaned
    3D reference-round/channel volume, calls the configured MATLAB entrypoint,
    reads the emitted CSV, and returns a normalized spot DataFrame plus boundary
    instrumentation.  The returned coordinates are PyStar reference-frame pixel
    coordinates in `z, y, x` order.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        engine_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.config = config
        self.engine_factory = engine_factory
        self.runtime_dir = resolve_matlab_spotfinding_runtime_path(config)
        self.runtime_manifest = load_matlab_spotfinding_runtime_manifest(self.runtime_dir)
        self.entrypoint = config.providers.matlab.spot_finding.entrypoint
        _validate_runtime_entrypoint_contract(self.runtime_manifest, self.entrypoint)
        self._session_capsule = MATLABSessionCapsule(
            consumer="spot_finding.provider='matlab'",
            runtime_dir=self.runtime_dir,
            entrypoint=self.entrypoint,
            engine_factory=engine_factory,
            engine_factory_consumer="spot_finding.provider='matlab'",
            startup_failure_prefix="Failed to start MATLAB Engine for spot_finding.provider='matlab'",
            addpath_failure_prefix="Failed to add MATLAB spot-finding runtime path",
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

    def _resolve_callable(self) -> Any:
        return self._session_capsule.resolve_callable()

    def _resolve_runtime_file_records(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return self._session_capsule.validate_runtime_files(self._collect_runtime_file_records)

    def _collect_runtime_file_records(self) -> list[dict[str, Any]]:
        return _collect_runtime_file_records_from_manifest(
            self.runtime_manifest,
            self.runtime_dir,
            missing_required_prefix="Required MATLAB spot-finding runtime file is missing",
            missing_required_suffix="MATLAB spot-finding provider cannot proceed.",
        )

    def _normalize_input_volume(self, volume: Any) -> Any:
        if volume.ndim != 3:
            raise ValueError(f"MATLAB spot-finding expects a 3D volume, got ndim={volume.ndim}")

        input_dtype = np.dtype(self.config.providers.matlab.spot_finding.input_volume_dtype)
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

        raise ValueError(
            f"Unsupported providers.matlab.spot_finding.input_volume_dtype={self.config.providers.matlab.spot_finding.input_volume_dtype!r}"
        )

    def _validate_response_metadata(
        self,
        metadata: Mapping[str, Any],
        *,
        tmpdir_path: Path,
        expected_shape_zyx: tuple[int, int, int],
        round_id: int,
        channel_id: int,
    ) -> Path:
        metadata_round_id = metadata.get("round_id")
        if not isinstance(metadata_round_id, (int, float)) or int(metadata_round_id) != round_id:
            raise ValueError(
                f"MATLAB spot-finding metadata round_id mismatch: expected {round_id}, got {metadata_round_id!r}"
            )

        metadata_reference_round = metadata.get("reference_round")
        expected_reference_round = int(self.config.pipeline.spot_finding.reference_round)
        if not isinstance(metadata_reference_round, (int, float)) or int(metadata_reference_round) != expected_reference_round:
            raise ValueError(
                "MATLAB spot-finding metadata reference_round mismatch: "
                f"expected {expected_reference_round}, got {metadata_reference_round!r}"
            )

        metadata_channel_id = metadata.get("channel_id")
        if not isinstance(metadata_channel_id, (int, float)) or int(metadata_channel_id) != channel_id:
            raise ValueError(
                f"MATLAB spot-finding metadata channel_id mismatch: expected {channel_id}, got {metadata_channel_id!r}"
            )

        validate_expected_3d_staged_volume_shape(
            metadata.get("volume_shape_zyx"),
            expected_shape_zyx=expected_shape_zyx,
            mismatch_prefix="MATLAB spot-finding metadata volume_shape_zyx mismatch",
        )

        output_path = resolve_staged_output_path(
            metadata.get("output_path"),
            tmpdir_path=tmpdir_path,
            missing_output_path_message="MATLAB spot-finding metadata must declare a non-empty output_path",
            outside_tmpdir_prefix="MATLAB spot-finding output must stay inside the staged temporary directory",
            missing_output_prefix="MATLAB spot-finding reported output that does not exist",
        )

        validate_runtime_step_metadata(
            metadata.get("steps"),
            missing_steps_message="MATLAB spot-finding metadata must declare a non-empty steps list",
            step_label="MATLAB spot-finding step",
        )

        return output_path

    def find_spots(
        self,
        volume: Any,
        *,
        fov_id: int,
        round_id: int,
        channel_id: int,
    ) -> Dict[str, Any]:
        """Run MATLAB `max3d` on one staged cleaned volume.

        Parameters identify the FOV/round/channel for provenance and metadata
        validation; `volume` must be a 3D array in `z, y, x` order.  The method
        returns `{spots, backend_metadata}` where `spots` is normalized to
        `z, y, x, intensity` float32 columns and `backend_metadata` records the
        MATLAB runtime, validated manifest files, MATLAB metadata payload, and
        per-call boundary timings.
        """

        boundary_trace = create_matlab_boundary_trace(
            stage_name="matlab_spot_finding",
            runtime_dir=self.runtime_dir,
            entrypoint=self.entrypoint,
            session=self._session_lifecycle,
            call_scope={
                "fov_id": int(fov_id),
                "round_id": int(round_id),
                "channel_id": int(channel_id),
            },
        )
        volume_for_matlab = self._normalize_input_volume(volume)
        plan = build_matlab_spotfinding_plan(
            self.config,
            fov_id=fov_id,
            round_id=round_id,
            channel_id=channel_id,
            volume_shape_zyx=(int(volume_for_matlab.shape[0]), int(volume_for_matlab.shape[1]), int(volume_for_matlab.shape[2])),
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

        with TemporaryDirectory(prefix=f"pystar_matlab_spotfinding_fov{fov_id}_round{round_id}_ch{channel_id}_") as tmpdir:
            tmpdir_path = Path(tmpdir)
            volume_path = tmpdir_path / f"spotfinding_input_fov_{fov_id}_round_{round_id}_ch_{channel_id}.tif"
            input_staging_started = time.perf_counter()
            _write_staged_volume_tiff(volume_path, volume_for_matlab)
            record_matlab_boundary_phase(
                boundary_trace,
                phase_name="input_staging",
                duration_ms=round((time.perf_counter() - input_staging_started) * 1000.0, 3),
                seam_cost_key="input_staging_ms",
                details={
                    "staged_input": volume_path.name,
                    "volume_shape_zyx": [int(volume_for_matlab.shape[0]), int(volume_for_matlab.shape[1]), int(volume_for_matlab.shape[2])],
                },
            )

            matlab_call_started = time.perf_counter()
            try:
                metadata_json = matlab_callable(
                    str(volume_path),
                    json.dumps(plan, sort_keys=True),
                    nargout=1,
                )
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    _format_exception_message(
                        f"MATLAB spot-finding entrypoint '{self.entrypoint}' failed for FOV {fov_id} round {round_id} channel {channel_id}",
                        exc,
                    )
                ) from exc
            record_matlab_boundary_phase(
                boundary_trace,
                phase_name="matlab_call",
                duration_ms=round((time.perf_counter() - matlab_call_started) * 1000.0, 3),
                seam_cost_key="matlab_call_ms",
                details={"volume_shape_zyx": plan.get("volume_shape_zyx")},
            )

            if not isinstance(metadata_json, str):
                raise ValueError(
                    f"MATLAB spot-finding entrypoint '{self.entrypoint}' must return a JSON string metadata payload"
                )

            result_validation_started = time.perf_counter()
            try:
                metadata = json.loads(metadata_json)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    _format_exception_message(
                        f"MATLAB spot-finding entrypoint '{self.entrypoint}' returned invalid JSON metadata",
                        exc,
                    )
                ) from exc
            if not isinstance(metadata, dict):
                raise ValueError("MATLAB spot-finding metadata payload must decode to a JSON object")

            output_path = self._validate_response_metadata(
                metadata,
                tmpdir_path=tmpdir_path.resolve(),
                expected_shape_zyx=(int(volume_for_matlab.shape[0]), int(volume_for_matlab.shape[1]), int(volume_for_matlab.shape[2])),
                round_id=round_id,
                channel_id=channel_id,
            )
            try:
                raw_spots_df = pd.read_csv(output_path)
            except Exception as exc:
                raise wrap_table_read_error(
                    exc,
                    "MATLAB spot output",
                    fov_id=fov_id,
                    path=output_path,
                    context="matlab spot normalization",
                    expected=SpotTableSchema().expected_description(),
                ) from exc

            spots_df = _normalize_spot_dataframe(
                raw_spots_df,
                fov_id=fov_id,
                path=output_path,
            )
            record_matlab_boundary_phase(
                boundary_trace,
                phase_name="result_validation",
                duration_ms=round((time.perf_counter() - result_validation_started) * 1000.0, 3),
                seam_cost_key="result_validation_ms",
                details={
                    "reported_step_count": len(metadata.get("steps", [])) if isinstance(metadata.get("steps"), list) else 0,
                    "spot_count": int(len(spots_df)),
                },
            )

        finalized_boundary_trace = finalize_matlab_boundary_trace(
            boundary_trace,
            session=self._session_lifecycle,
            engine_reused_this_call=bool(engine_acquire.get("engine_reused_this_call", False)),
        )

        return {
            "spots": spots_df,
            "backend_metadata": {
                "provider": "matlab",
                "runtime_path": str(self.runtime_dir),
                "runtime_manifest": str(self.runtime_dir / MATLAB_SPOTFINDING_RUNTIME_MANIFEST_NAME),
                "entrypoint": self.entrypoint,
                "runtime_files": runtime_files,
                "matlab_metadata": metadata,
                "normalized_result": {
                    "spot_count": int(len(spots_df)),
                    "columns": list(spots_df.columns),
                },
                "boundary_instrumentation": finalized_boundary_trace,
                "session_lifecycle": snapshot_matlab_session_lifecycle(self._session_lifecycle),
                "session_lifecycle_summary": self._session_lifecycle_summary,
            },
        }
