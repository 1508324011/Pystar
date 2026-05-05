"""MATLAB-backed signal extraction boundary for PyStar.

This module stages a cleaned 3D volume and a `z, y, x` coordinate table for the
repo-local MATLAB extraction runtime.  MATLAB computes the same per-spot box-sum
contract that the native extractor uses, then PyStar validates the returned
metadata and converts the result into the one-dimensional intensity vector
consumed by `SignalMiner`.  The boundary is explicit and fail-loud: runtime
files, output ordering, output length, and staging paths are all checked before
any intensities are accepted.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, Mapping, Optional

import numpy as np
import pandas as pd
import tifffile

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
    load_runtime_manifest_json,
    require_manifest_string,
    resolve_repo_runtime_path,
    trusted_matlab_runtime_root,
    validate_configured_entrypoint_contract,
    validate_runtime_manifest_file_buckets,
    validate_runtime_manifest_file_entries,
)


MATLAB_EXTRACTION_RUNTIME_MANIFEST_NAME = "runtime_manifest.json"
MATLAB_EXTRACTION_PACKAGE_NAME = "pystar_extraction"


def resolve_matlab_extraction_runtime_path(config: ExperimentConfig) -> Path:
    """Resolve and validate the repo-local MATLAB extraction runtime path."""

    return resolve_repo_runtime_path(
        config.providers.matlab.extraction.runtime_path,
        config_label="providers.matlab.extraction.runtime_path",
        trusted_root=trusted_matlab_runtime_root(),
    )


def load_matlab_extraction_runtime_manifest(runtime_dir: Path) -> Dict[str, Any]:
    """Load the MATLAB extraction runtime manifest and validate its schema."""

    manifest = load_runtime_manifest_json(
        runtime_dir,
        manifest_label="MATLAB extraction runtime manifest",
        missing_hint="Expected repo-local manifest for MATLAB extraction provider.",
        manifest_name=MATLAB_EXTRACTION_RUNTIME_MANIFEST_NAME,
    )
    required_files, optional_files = validate_runtime_manifest_file_buckets(
        manifest,
        manifest_label="MATLAB extraction runtime manifest",
    )
    _ = require_manifest_string(
        manifest,
        key="entrypoint",
        manifest_label="MATLAB extraction runtime manifest",
    )
    package_name = manifest.get("package_name")
    if package_name != MATLAB_EXTRACTION_PACKAGE_NAME:
        raise ValueError(
            "MATLAB extraction runtime manifest package_name mismatch: "
            f"expected {MATLAB_EXTRACTION_PACKAGE_NAME!r}, got {package_name!r}"
        )
    validate_runtime_manifest_file_entries(
        required_files=required_files,
        optional_files=optional_files,
        manifest_label="MATLAB extraction runtime manifest",
    )
    return manifest


def _validate_runtime_entrypoint_contract(runtime_manifest: Mapping[str, Any], configured_entrypoint: str) -> None:
    validate_configured_entrypoint_contract(
        runtime_manifest,
        configured_entrypoint,
        config_label="providers.matlab.extraction.entrypoint",
        manifest_label="MATLAB extraction runtime manifest",
    )


def build_matlab_extraction_plan(
    config: ExperimentConfig,
    *,
    fov_id: int,
    round_id: int,
    channel_id: int,
    n_spots: int,
    volume_shape_zyx: tuple[int, int, int],
    box_size: tuple[int, int, int],
    transform_application_mode: str,
) -> Dict[str, Any]:
    """Build the JSON-serializable execution plan for one MATLAB extraction call.

    `volume_shape_zyx`, `box_size_zyx`, and `n_spots` are repeated in the MATLAB
    metadata contract so PyStar can reject stale or truncated outputs.  The
    `transform_application_mode` is recorded because callers may pass either a
    moving-round volume with mapped coordinates or a reference-frame warped
    volume with original reference coordinates.
    """

    matlab_cfg = config.providers.matlab.extraction
    return {
        "fov_id": int(fov_id),
        "round_id": int(round_id),
        "channel_id": int(channel_id),
        "method": config.pipeline.extraction.method,
        "transform_application_mode": transform_application_mode,
        "n_spots": int(n_spots),
        "box_size_zyx": [int(value) for value in box_size],
        "volume_shape_zyx": [int(value) for value in volume_shape_zyx],
        "volume_transfer_mode": matlab_cfg.volume_transfer_mode,
        "coords_transfer_mode": matlab_cfg.coords_transfer_mode,
        "input_volume_dtype": matlab_cfg.input_volume_dtype,
    }


def _load_matlab_engine_factory() -> Callable[[], Any]:
    factory, _factory_metrics = load_matlab_engine_factory(
        consumer="extraction.provider='matlab'",
    )
    return factory


def _write_staged_volume_tiff(volume_path: Path, volume: Any) -> None:
    staged_volume = np.asarray(volume)
    if staged_volume.ndim != 3:
        raise ValueError(f"MATLAB extraction expects a 3D staged volume, got ndim={staged_volume.ndim}")

    with tifffile.TiffWriter(volume_path) as writer:
        for plane in staged_volume:
            writer.write(np.asarray(plane), photometric="minisblack", metadata=None)


class MATLABExtractionBackend:
    """Execute MATLAB box-sum extraction under PyStar's provider contract.

    A backend instance owns the MATLAB session, runtime manifest validation, and
    per-call boundary instrumentation.  The public `extract_intensities` method
    accepts a 3D volume plus an `(N, 3)` coordinate matrix in `z, y, x` order and
    returns an `(N,)` float32 intensity vector in the same spot order.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        engine_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.config = config
        self.engine_factory = engine_factory
        self.runtime_dir = resolve_matlab_extraction_runtime_path(config)
        self.runtime_manifest = load_matlab_extraction_runtime_manifest(self.runtime_dir)
        self.entrypoint = config.providers.matlab.extraction.entrypoint
        _validate_runtime_entrypoint_contract(self.runtime_manifest, self.entrypoint)
        self._session_capsule = MATLABSessionCapsule(
            consumer="extraction.provider='matlab'",
            runtime_dir=self.runtime_dir,
            entrypoint=self.entrypoint,
            engine_factory=engine_factory,
            engine_factory_consumer="extraction.provider='matlab'",
            startup_failure_prefix="Failed to start MATLAB Engine for extraction.provider='matlab'",
            addpath_failure_prefix="Failed to add MATLAB extraction runtime path",
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
            missing_required_prefix="Required MATLAB extraction runtime file is missing",
            missing_required_suffix="MATLAB extraction provider cannot proceed.",
        )

    def _normalize_input_volume(self, volume: Any) -> Any:
        if volume.ndim != 3:
            raise ValueError(f"MATLAB extraction expects a 3D volume, got ndim={volume.ndim}")

        input_dtype = np.dtype(self.config.providers.matlab.extraction.input_volume_dtype)
        arr = np.asarray(volume)
        if input_dtype == np.float32:
            return np.asarray(arr, dtype=np.float32)
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
            f"Unsupported providers.matlab.extraction.input_volume_dtype={self.config.providers.matlab.extraction.input_volume_dtype!r}"
        )

    def _validate_response_metadata(
        self,
        metadata: Mapping[str, Any],
        *,
        tmpdir_path: Path,
        round_id: int,
        channel_id: int,
        expected_count: int,
        expected_shape_zyx: tuple[int, int, int],
    ) -> Path:
        metadata_round_id = metadata.get("round_id")
        if not isinstance(metadata_round_id, (int, float)) or int(metadata_round_id) != round_id:
            raise ValueError(
                f"MATLAB extraction metadata round_id mismatch: expected {round_id}, got {metadata_round_id!r}"
            )

        metadata_channel_id = metadata.get("channel_id")
        if not isinstance(metadata_channel_id, (int, float)) or int(metadata_channel_id) != channel_id:
            raise ValueError(
                f"MATLAB extraction metadata channel_id mismatch: expected {channel_id}, got {metadata_channel_id!r}"
            )

        metadata_count = metadata.get("n_spots")
        if not isinstance(metadata_count, (int, float)) or int(metadata_count) != expected_count:
            raise ValueError(
                f"MATLAB extraction metadata n_spots mismatch: expected {expected_count}, got {metadata_count!r}"
            )

        volume_shape = metadata.get("volume_shape_zyx")
        if not isinstance(volume_shape, list) or [int(v) for v in volume_shape] != [int(v) for v in expected_shape_zyx]:
            raise ValueError(
                "MATLAB extraction metadata volume_shape_zyx mismatch: "
                f"expected {list(expected_shape_zyx)}, got {volume_shape!r}"
            )

        output_path_value = metadata.get("output_path")
        if not isinstance(output_path_value, str) or not output_path_value.strip():
            raise ValueError("MATLAB extraction metadata must declare a non-empty output_path")
        output_path = Path(output_path_value)
        if not output_path.is_absolute():
            output_path = tmpdir_path / output_path
        output_path = output_path.resolve()
        if output_path.parent != tmpdir_path.resolve():
            raise ValueError(
                "MATLAB extraction output must stay inside the staged temporary directory: "
                f"{output_path}"
            )
        if not output_path.exists():
            raise FileNotFoundError(
                f"MATLAB extraction reported output that does not exist: {output_path}"
            )

        steps = metadata.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ValueError("MATLAB extraction metadata must declare a non-empty steps list")
        for index, step in enumerate(steps):
            if not isinstance(step, Mapping):
                raise ValueError(f"MATLAB extraction step #{index} must be a mapping")
            name = step.get("name")
            duration_ms = step.get("duration_ms")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"MATLAB extraction step #{index} is missing a non-empty name")
            if not isinstance(duration_ms, (int, float)) or duration_ms < 0:
                raise ValueError(
                    f"MATLAB extraction step '{name}' must report a non-negative duration_ms"
                )

        return output_path

    def extract_intensities(
        self,
        volume: Any,
        coords_zyx: Any,
        *,
        fov_id: int,
        round_id: int,
        channel_id: int,
        box_size: tuple[int, int, int],
        transform_application_mode: str,
    ) -> Dict[str, Any]:
        """Extract one channel/round intensity vector through MATLAB.

        `coords_zyx` must already be expressed in the coordinate frame expected
        by `transform_application_mode`: moving-image coordinates for
        `coordinate_mapping`, or reference-frame coordinates when the volume has
        already been image-warped.  The returned dictionary contains the ordered
        intensity vector and boundary metadata; output `spot_index` must match
        `0..N-1` exactly so row order stays aligned with `spots_fov_<id>.csv`.
        """

        boundary_trace = create_matlab_boundary_trace(
            stage_name="matlab_extraction",
            runtime_dir=self.runtime_dir,
            entrypoint=self.entrypoint,
            session=self._session_lifecycle,
            call_scope={
                "fov_id": int(fov_id),
                "round_id": int(round_id),
                "channel_id": int(channel_id),
                "transform_application_mode": transform_application_mode,
            },
        )
        volume_for_matlab = self._normalize_input_volume(volume)
        coords = np.asarray(coords_zyx, dtype=np.float32)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"MATLAB extraction expects coords shaped (N, 3), got {coords.shape}")

        plan = build_matlab_extraction_plan(
            self.config,
            fov_id=fov_id,
            round_id=round_id,
            channel_id=channel_id,
            n_spots=int(len(coords)),
            volume_shape_zyx=(int(volume_for_matlab.shape[0]), int(volume_for_matlab.shape[1]), int(volume_for_matlab.shape[2])),
            box_size=box_size,
            transform_application_mode=transform_application_mode,
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

        with TemporaryDirectory(prefix=f"pystar_matlab_extraction_fov{fov_id}_round{round_id}_ch{channel_id}_") as tmpdir:
            tmpdir_path = Path(tmpdir)
            volume_path = tmpdir_path / f"extraction_input_fov_{fov_id}_round_{round_id}_ch_{channel_id}.tif"
            coords_path = tmpdir_path / f"coords_fov_{fov_id}_round_{round_id}_ch_{channel_id}.csv"
            input_staging_started = time.perf_counter()
            _write_staged_volume_tiff(volume_path, volume_for_matlab)
            pd.DataFrame(
                {
                    "spot_index": np.arange(len(coords), dtype=np.int64),
                    "z": coords[:, 0],
                    "y": coords[:, 1],
                    "x": coords[:, 2],
                }
            ).to_csv(coords_path, index=False)
            record_matlab_boundary_phase(
                boundary_trace,
                phase_name="input_staging",
                duration_ms=round((time.perf_counter() - input_staging_started) * 1000.0, 3),
                seam_cost_key="input_staging_ms",
                details={
                    "staged_inputs": [volume_path.name, coords_path.name],
                    "volume_shape_zyx": [int(volume_for_matlab.shape[0]), int(volume_for_matlab.shape[1]), int(volume_for_matlab.shape[2])],
                    "spot_count": int(len(coords)),
                },
            )

            matlab_call_started = time.perf_counter()
            try:
                metadata_json = matlab_callable(
                    str(volume_path),
                    str(coords_path),
                    json.dumps(plan, sort_keys=True),
                    nargout=1,
                )
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    _format_exception_message(
                        f"MATLAB extraction entrypoint '{self.entrypoint}' failed for FOV {fov_id} round {round_id} channel {channel_id}",
                        exc,
                    )
                ) from exc
            record_matlab_boundary_phase(
                boundary_trace,
                phase_name="matlab_call",
                duration_ms=round((time.perf_counter() - matlab_call_started) * 1000.0, 3),
                seam_cost_key="matlab_call_ms",
                details={
                    "volume_shape_zyx": plan.get("volume_shape_zyx"),
                    "spot_count": int(len(coords)),
                },
            )

            if not isinstance(metadata_json, str):
                raise ValueError(
                    f"MATLAB extraction entrypoint '{self.entrypoint}' must return a JSON string metadata payload"
                )

            result_validation_started = time.perf_counter()
            try:
                metadata = json.loads(metadata_json)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    _format_exception_message(
                        f"MATLAB extraction entrypoint '{self.entrypoint}' returned invalid JSON metadata",
                        exc,
                    )
                ) from exc
            if not isinstance(metadata, dict):
                raise ValueError("MATLAB extraction metadata payload must decode to a JSON object")

            output_path = self._validate_response_metadata(
                metadata,
                tmpdir_path=tmpdir_path.resolve(),
                round_id=round_id,
                channel_id=channel_id,
                expected_count=int(len(coords)),
                expected_shape_zyx=(int(volume_for_matlab.shape[0]), int(volume_for_matlab.shape[1]), int(volume_for_matlab.shape[2])),
            )
            output_df = pd.read_csv(output_path)
            if "spot_index" not in output_df.columns or "intensity" not in output_df.columns:
                raise ValueError("MATLAB extraction output must contain 'spot_index' and 'intensity' columns")
            spot_index = np.asarray(pd.to_numeric(output_df["spot_index"], errors="raise"), dtype=np.int64)
            expected_index = np.arange(len(coords), dtype=np.int64)
            if not np.array_equal(spot_index, expected_index):
                raise ValueError(
                    "MATLAB extraction output spot_index ordering mismatch: expected sequential indices "
                    f"0..{len(coords) - 1}"
                )
            intensities = np.asarray(pd.to_numeric(output_df["intensity"], errors="raise"), dtype=np.float32)
            if len(intensities) != len(coords):
                raise ValueError(
                    f"MATLAB extraction output length mismatch: expected {len(coords)}, got {len(intensities)}"
                )
            record_matlab_boundary_phase(
                boundary_trace,
                phase_name="result_validation",
                duration_ms=round((time.perf_counter() - result_validation_started) * 1000.0, 3),
                seam_cost_key="result_validation_ms",
                details={
                    "reported_step_count": len(metadata.get("steps", [])) if isinstance(metadata.get("steps"), list) else 0,
                    "spot_count": int(len(intensities)),
                },
            )

        finalized_boundary_trace = finalize_matlab_boundary_trace(
            boundary_trace,
            session=self._session_lifecycle,
            engine_reused_this_call=bool(engine_acquire.get("engine_reused_this_call", False)),
        )

        return {
            "intensities": intensities,
            "backend_metadata": {
                "provider": "matlab",
                "runtime_path": str(self.runtime_dir),
                "runtime_manifest": str(self.runtime_dir / MATLAB_EXTRACTION_RUNTIME_MANIFEST_NAME),
                "entrypoint": self.entrypoint,
                "runtime_files": runtime_files,
                "matlab_metadata": metadata,
                "normalized_result": {
                    "spot_count": int(len(intensities)),
                    "columns": list(output_df.columns),
                    "dtype": str(intensities.dtype),
                },
                "boundary_instrumentation": finalized_boundary_trace,
                "session_lifecycle": snapshot_matlab_session_lifecycle(self._session_lifecycle),
                "session_lifecycle_summary": self._session_lifecycle_summary,
            },
        }
