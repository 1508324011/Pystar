"""Private per-FOV performance telemetry helpers for the batch runner.

Stage14 is intentionally measurement-only.  This module reads existing runtime
artifacts and backend metadata after the normal sequential pipeline has
completed, then writes one JSON sidecar under the canonical ``qc_reports/``
directory.  It does not own stage execution, provider dispatch, release
contracts, retry behavior, or artifact schemas.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ._artifact_schemas import intensity_matrix_metadata_path
from ._io_paths import get_fov_output_structure, get_transform_manifest_path
from ._stage_contracts import get_ordered_stage_ids, get_stage_spec, validate_stage_ids
from .serialization import write_backend_metadata


# This module intentionally accepts config-like objects and persisted metadata
# mappings from several pipeline stages. Keep pyright noise local rather than
# weakening the project-wide fail-loud runtime checks below.
# pyright: reportExplicitAny=false, reportAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnnecessaryIsInstance=false


PERFORMANCE_TELEMETRY_SCHEMA_NAME = "pystar_performance_telemetry"
PERFORMANCE_TELEMETRY_SCHEMA_VERSION = 1


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp for telemetry metadata."""

    return datetime.now(timezone.utc).isoformat()


def record_stage_timing(
    stage_id: str,
    elapsed_wall_ms: float,
    *,
    status: str = "completed",
) -> dict[str, Any]:
    """Build one validated stage timing record for the canonical runner.

    The batch runner owns the wall-clock measurement.  This helper validates the
    stage ID against the immutable stage contract so malformed telemetry input
    fails loudly as an internal schema bug rather than producing ambiguous JSON.
    """

    stage = get_stage_spec(stage_id)
    elapsed_ms = float(elapsed_wall_ms)
    if not math.isfinite(elapsed_ms) or elapsed_ms < 0:
        raise ValueError(
            f"Performance telemetry schema error: elapsed_wall_ms for stage {stage_id!r} must be finite and non-negative"
        )
    if not isinstance(status, str) or not status.strip():
        raise ValueError(f"Performance telemetry schema error: status for stage {stage_id!r} must be non-empty")

    return {
        "stage_id": stage.stage_id,
        "order_index": stage.order_index,
        "display_label": stage.display_label,
        "elapsed_wall_ms": round(elapsed_ms, 3),
        "status": status.strip(),
    }


def _error_payload(exc: Exception) -> dict[str, str]:
    return {
        "type": exc.__class__.__name__,
        "message": str(exc),
    }


def _file_info(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "path": str(path),
        "exists": False,
        "size_bytes": None,
    }
    try:
        exists = path.exists()
    except OSError as exc:
        info["stat_error"] = _error_payload(exc)
        return info

    info["exists"] = bool(exists)
    if not exists:
        return info

    try:
        stat_result = path.stat()
    except OSError as exc:
        info["stat_error"] = _error_payload(exc)
        return info

    info["size_bytes"] = int(stat_result.st_size)
    info["is_file"] = path.is_file()
    return info


def _csv_row_count(path: Path) -> dict[str, Any]:
    info = _file_info(path)
    if not info["exists"]:
        info.update({"read_status": "absent", "row_count": None})
        return info

    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            line_count = sum(1 for _ in handle)
    except Exception as exc:
        info.update({"read_status": "error", "row_count": None, "read_error": _error_payload(exc)})
        return info

    info.update({"read_status": "read", "row_count": max(line_count - 1, 0)})
    return info


def _npy_array_summary(path: Path, *, allow_pickle: bool = False) -> dict[str, Any]:
    info = _file_info(path)
    if not info["exists"]:
        info.update({"read_status": "absent", "shape": None, "dtype": None})
        return info

    try:
        loaded = np.load(path, allow_pickle=allow_pickle, mmap_mode=None if allow_pickle else "r")
    except Exception as exc:
        info.update({"read_status": "error", "shape": None, "dtype": None, "read_error": _error_payload(exc)})
        return info

    try:
        shape = [int(value) for value in getattr(loaded, "shape", ())]
        dtype = str(getattr(loaded, "dtype", "unknown"))
    finally:
        close = getattr(loaded, "close", None)
        if callable(close):
            _ = close()

    info.update({"read_status": "read", "shape": shape, "dtype": dtype})
    return info


def _load_json_mapping(path: Path) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    info = _file_info(path)
    if not info["exists"]:
        info["read_status"] = "absent"
        return None, info

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        info.update({"read_status": "error", "read_error": _error_payload(exc)})
        return None, info

    if not isinstance(payload, dict):
        info.update(
            {
                "read_status": "error",
                "read_error": {
                    "type": "ValueError",
                    "message": "Telemetry source JSON must be an object",
                },
            }
        )
        return None, info

    info["read_status"] = "read"
    return dict(payload), info


def _load_yaml_mapping(path: Path) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    info = _file_info(path)
    if not info["exists"]:
        info["read_status"] = "absent"
        return None, info

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        info.update({"read_status": "error", "read_error": _error_payload(exc)})
        return None, info

    if not isinstance(payload, dict):
        info.update(
            {
                "read_status": "error",
                "read_error": {
                    "type": "ValueError",
                    "message": "Telemetry source YAML must be an object",
                },
            }
        )
        return None, info

    info["read_status"] = "read"
    return dict(payload), info


def _nested_mapping(payload: Mapping[str, Any] | None, keys: Sequence[str]) -> dict[str, Any] | None:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return dict(current) if isinstance(current, Mapping) else None


def _extract_session_summary(
    payload: Mapping[str, Any] | None,
    boundary_summary: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    direct = payload.get("session_lifecycle_summary") if isinstance(payload, Mapping) else None
    if isinstance(direct, Mapping):
        return dict(direct)

    nested = boundary_summary.get("session_lifecycle_summary") if isinstance(boundary_summary, Mapping) else None
    if isinstance(nested, Mapping):
        return dict(nested)
    return None


def _require_mapping_source(
    payload: dict[str, Any] | None,
    source_info: Mapping[str, Any],
    *,
    stage_id: str,
    provider: str | None,
    expected_description: str,
) -> None:
    """Require a selected MATLAB stage to expose its metadata source."""

    if provider is None:
        return
    if payload is not None:
        return

    path = source_info.get("path", "<unknown>")
    read_status = source_info.get("read_status")
    if read_status == "absent" or source_info.get("exists") is False:
        message = (
            f"Performance telemetry expected MATLAB {stage_id} metadata for provider {provider!r} "
            + f"at {path}: missing {expected_description}"
        )
        raise FileNotFoundError(message)

    message = (
        f"Performance telemetry could not read MATLAB {stage_id} metadata for provider {provider!r} "
        + f"at {path}: expected {expected_description}; error={source_info.get('read_error')!r}"
    )
    raise ValueError(message)


def _load_transform_manifest_root(path: Path) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    info = _file_info(path)
    if not info.get("exists"):
        info["read_status"] = "absent"
        return None, info

    try:
        raw_payload = np.load(path, allow_pickle=True).item()
        if not isinstance(raw_payload, dict):
            raise ValueError("transform manifest root is not a dictionary")
        info["read_status"] = "read"
        info["root_keys"] = [str(key) for key in raw_payload.keys()]
        return dict(raw_payload), info
    except Exception as exc:
        info["read_status"] = "error"
        info["read_error"] = _error_payload(exc)
        return None, info


def _provider_summary(config: Any) -> dict[str, Any]:
    pipeline = getattr(config, "pipeline", None)
    registration = getattr(pipeline, "registration", None)
    spot_finding = getattr(pipeline, "spot_finding", None)
    extraction = getattr(pipeline, "extraction", None)
    decoding = getattr(pipeline, "decoding", None)

    preprocessing_providers = None
    preprocessing_providers_used = getattr(pipeline, "preprocessing_providers_used", None)
    if callable(preprocessing_providers_used):
        provider_values = preprocessing_providers_used()
        if isinstance(provider_values, Sequence) and not isinstance(provider_values, (str, bytes)):
            preprocessing_providers = [str(value) for value in provider_values]
        else:
            preprocessing_providers = [str(provider_values)]
    else:
        preprocessing = getattr(pipeline, "preprocessing", None)
        sequence = getattr(preprocessing, "sequence", None)
        if isinstance(sequence, Sequence) and not isinstance(sequence, (str, bytes)):
            preprocessing_providers = sorted({str(getattr(step, "provider", "native")) for step in sequence}) or ["native"]

    preprocessing_provider_mode = getattr(pipeline, "preprocessing_provider_mode", None)
    registration_provider_mode = getattr(pipeline, "registration_provider_mode", None)

    return {
        "preprocessing": {
            "providers": preprocessing_providers or ["native"],
            "provider_mode": preprocessing_provider_mode() if callable(preprocessing_provider_mode) else None,
        },
        "registration": {
            "provider_mode": registration_provider_mode() if callable(registration_provider_mode) else None,
            "global_provider": getattr(registration, "global_provider", None),
            "local_provider": getattr(registration, "local_provider", None),
            "local_method": getattr(registration, "local_method", None),
            "enable_local": getattr(registration, "enable_local", None),
        },
        "spot_finding": {
            "provider": getattr(spot_finding, "provider", None),
            "algorithm": getattr(spot_finding, "algorithm", None),
        },
        "signal_extraction": {
            "provider": getattr(extraction, "provider", None),
            "method": getattr(extraction, "method", None),
            "transform_application_mode": getattr(extraction, "transform_application_mode", None),
        },
        "decoding": {
            "gating_mode": getattr(decoding, "gating_mode", None),
        },
    }


def _matlab_provider_by_stage(providers: Mapping[str, Any]) -> dict[str, str | None]:
    preprocessing = providers.get("preprocessing")
    registration = providers.get("registration")
    spot_finding = providers.get("spot_finding")
    signal_extraction = providers.get("signal_extraction")

    preprocessing_provider: str | None = None
    if isinstance(preprocessing, Mapping):
        preprocessing_providers = preprocessing.get("providers")
        if isinstance(preprocessing_providers, Sequence) and not isinstance(preprocessing_providers, (str, bytes)):
            provider_set = {str(provider) for provider in preprocessing_providers}
            preprocessing_provider = "matlab" if "matlab" in provider_set else None

    registration_provider: str | None = None
    if isinstance(registration, Mapping):
        if registration.get("global_provider") == "matlab" or registration.get("local_provider") == "matlab":
            registration_provider = "matlab"

    return {
        "preprocessing": preprocessing_provider,
        "registration": registration_provider,
        "spot_finding": (
            "matlab"
            if isinstance(spot_finding, Mapping) and spot_finding.get("provider") == "matlab"
            else None
        ),
        "signal_extraction": (
            "matlab"
            if isinstance(signal_extraction, Mapping) and signal_extraction.get("provider") == "matlab"
            else None
        ),
        "decoding": None,
    }


def _output_directory(config: Any) -> Path:
    pipeline = getattr(config, "pipeline", None)
    output = getattr(pipeline, "output", None)
    directory = getattr(output, "directory", None)
    if directory is None:
        raise ValueError("Performance telemetry schema error: config.pipeline.output.directory is required")
    return Path(directory)


def _config_summary(config: Any) -> dict[str, Any]:
    config_source_path = getattr(config, "config_source_path", None)
    return {
        "source_path": None if config_source_path is None else str(config_source_path),
        "config_sha256": getattr(config, "config_sha256", None),
        "output_directory": str(_output_directory(config)),
    }


def _artifact_summary(base_dir: Path, fov_id: int, *, require_completed_outputs: bool = False) -> dict[str, Any]:
    paths = get_fov_output_structure(base_dir, fov_id)
    transform_manifest = get_transform_manifest_path(base_dir, fov_id)
    flow_sidecars = sorted(paths["transforms"].glob(f"transforms_fov_{fov_id}_round_*_flow_3d.npy"))
    spots_path = paths["spots"] / f"spots_fov_{fov_id}.csv"
    intensity_matrix_path = paths["extraction"] / f"intensity_matrix_fov_{fov_id}.npy"
    intensity_metadata_path = intensity_matrix_metadata_path(intensity_matrix_path)
    decoded_path = paths["decoded"] / f"decoded_fov_{fov_id}.csv"
    goodreads_path = paths["decoded"] / f"decoded_fov_{fov_id}_goodreads.csv"
    pre_pattern_path = paths["decoded"] / f"decoded_fov_{fov_id}_pre_pattern_check.csv"

    artifacts: dict[str, Any] = {
        "output_paths": {key: str(value) for key, value in paths.items()},
        "transform_manifest": _file_info(transform_manifest),
        "flow_3d_sidecars": [_npy_array_summary(path) for path in flow_sidecars],
        "spot_table": _csv_row_count(spots_path),
        "intensity_matrix": _npy_array_summary(intensity_matrix_path),
        "intensity_matrix_metadata": _file_info(intensity_metadata_path),
        "decoded_outputs": {
            "active": _csv_row_count(decoded_path),
            "goodreads": _csv_row_count(goodreads_path),
            "pre_pattern_check": _csv_row_count(pre_pattern_path),
        },
        "backend_metadata_sidecars": {
            "preprocessing_provenance": _file_info(paths["qc"] / "preprocessing_provenance.yaml"),
            "spot_finding_backend": _file_info(paths["qc"] / f"spot_finding_backend_fov_{fov_id}.json"),
            "extraction_backend": _file_info(paths["qc"] / f"extraction_backend_fov_{fov_id}.json"),
        },
    }

    if require_completed_outputs:
        required_artifacts = {
            "transform_manifest": artifacts["transform_manifest"],
            "spot_table": artifacts["spot_table"],
            "intensity_matrix": artifacts["intensity_matrix"],
            "decoded_active": artifacts["decoded_outputs"]["active"],
            "decoded_goodreads": artifacts["decoded_outputs"]["goodreads"],
            "decoded_pre_pattern_check": artifacts["decoded_outputs"]["pre_pattern_check"],
        }
        for label, info in required_artifacts.items():
            if isinstance(info, Mapping) and info.get("exists") is True and info.get("read_status") != "error":
                continue
            if isinstance(info, Mapping) and info.get("read_status") == "error":
                message = (
                    f"Performance telemetry could not inspect required completed artifact {label} at "
                    f"{info.get('path')}: {info.get('read_error')}"
                )
                raise ValueError(
                    message
                )
            message = (
                f"Performance telemetry expected required completed artifact {label} at "
                f"{info.get('path') if isinstance(info, Mapping) else '<unknown>'}"
            )
            raise FileNotFoundError(
                message
            )

    return artifacts


def _stage_matlab_summary(
    base_dir: Path,
    fov_id: int,
    stage_id: str,
    *,
    matlab_provider: str | None = None,
) -> dict[str, Any]:
    paths = get_fov_output_structure(base_dir, fov_id)
    metadata_sources: list[dict[str, Any]] = []
    boundary_summary: dict[str, Any] | None = None
    session_summary: dict[str, Any] | None = None

    if stage_id == "preprocessing":
        source_path = paths["qc"] / "preprocessing_provenance.yaml"
        payload, source_info = _load_yaml_mapping(source_path)
        source_info["kind"] = "preprocessing_provenance"
        metadata_sources.append(source_info)
        _require_mapping_source(
            payload,
            source_info,
            stage_id=stage_id,
            provider=matlab_provider,
            expected_description="preprocessing provenance with boundary_instrumentation_summary",
        )
        candidate = payload.get("boundary_instrumentation_summary") if isinstance(payload, Mapping) else None
        boundary_summary = dict(candidate) if isinstance(candidate, Mapping) else None
        session_summary = _extract_session_summary(payload, boundary_summary)
    elif stage_id == "registration":
        source_path = get_transform_manifest_path(base_dir, fov_id)
        manifest, source_info = _load_transform_manifest_root(source_path)
        source_info["kind"] = "transform_manifest_registration_provenance"
        metadata_sources.append(source_info)
        registration_backend_details = _nested_mapping(
            manifest,
            ("_provenance", "runtime_context", "registration_backend_details"),
        )
        candidate = (
            registration_backend_details.get("boundary_instrumentation_summary")
            if isinstance(registration_backend_details, Mapping)
            else None
        )
        boundary_summary = dict(candidate) if isinstance(candidate, Mapping) else None
        session_summary = _extract_session_summary(registration_backend_details, boundary_summary)
    elif stage_id == "spot_finding":
        source_path = paths["qc"] / f"spot_finding_backend_fov_{fov_id}.json"
        payload, source_info = _load_json_mapping(source_path)
        source_info["kind"] = "spot_finding_backend_metadata"
        metadata_sources.append(source_info)
        _require_mapping_source(
            payload,
            source_info,
            stage_id=stage_id,
            provider=matlab_provider,
            expected_description="spot-finding backend metadata with boundary_instrumentation_summary",
        )
        candidate = payload.get("boundary_instrumentation_summary") if isinstance(payload, Mapping) else None
        boundary_summary = dict(candidate) if isinstance(candidate, Mapping) else None
        session_summary = _extract_session_summary(payload, boundary_summary)
    elif stage_id == "signal_extraction":
        source_path = paths["qc"] / f"extraction_backend_fov_{fov_id}.json"
        payload, source_info = _load_json_mapping(source_path)
        source_info["kind"] = "extraction_backend_metadata"
        metadata_sources.append(source_info)
        _require_mapping_source(
            payload,
            source_info,
            stage_id=stage_id,
            provider=matlab_provider,
            expected_description="extraction backend metadata with boundary_instrumentation_summary",
        )
        candidate = payload.get("boundary_instrumentation_summary") if isinstance(payload, Mapping) else None
        boundary_summary = dict(candidate) if isinstance(candidate, Mapping) else None
        session_summary = _extract_session_summary(payload, boundary_summary)

    return {
        "metadata_sources": metadata_sources,
        "boundary_instrumentation_summary": boundary_summary,
        "session_lifecycle_summary": session_summary,
    }


def _normalize_stage_timings(stage_timings: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(stage_timings, Sequence) or isinstance(stage_timings, (str, bytes)):
        raise ValueError("Performance telemetry schema error: stage_timings must be an ordered sequence")

    normalized: list[dict[str, Any]] = []
    for record in stage_timings:
        if not isinstance(record, Mapping):
            raise ValueError("Performance telemetry schema error: each stage timing record must be a mapping")
        stage_id = record.get("stage_id")
        if not isinstance(stage_id, str):
            raise ValueError("Performance telemetry schema error: each stage timing record must include string stage_id")
        elapsed = record.get("elapsed_wall_ms")
        if not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool):
            raise ValueError(
                f"Performance telemetry schema error: stage {stage_id!r} must include numeric elapsed_wall_ms"
            )
        normalized.append(record_stage_timing(stage_id, float(elapsed), status=str(record.get("status", "completed"))))

    stage_ids = tuple(record["stage_id"] for record in normalized)
    if stage_ids != get_ordered_stage_ids():
        _ = validate_stage_ids(stage_ids)
    return normalized


def build_performance_telemetry_payload(
    *,
    config: Any,
    fov_id: int,
    stage_timings: Sequence[Mapping[str, Any]],
    run_started_at_utc: str | None = None,
    run_finished_at_utc: str | None = None,
    total_elapsed_ms: float | None = None,
    require_completed_outputs: bool = False,
) -> dict[str, Any]:
    """Build the machine-readable per-FOV performance telemetry payload."""

    base_dir = _output_directory(config)
    normalized_timings = _normalize_stage_timings(stage_timings)
    providers = _provider_summary(config)
    artifacts = _artifact_summary(base_dir, int(fov_id), require_completed_outputs=require_completed_outputs)
    matlab_provider_by_stage = _matlab_provider_by_stage(providers)
    stages = []
    for timing in normalized_timings:
        stage_id = str(timing["stage_id"])
        stages.append(
            {
                **timing,
                "runner_calls": list(get_stage_spec(stage_id).runner_calls),
                "matlab": _stage_matlab_summary(
                    base_dir,
                    int(fov_id),
                    stage_id,
                    matlab_provider=(matlab_provider_by_stage.get(stage_id) if require_completed_outputs else None),
                ),
            }
        )

    if total_elapsed_ms is not None:
        total_elapsed_ms_value = float(total_elapsed_ms)
        if not math.isfinite(total_elapsed_ms_value) or total_elapsed_ms_value < 0:
            raise ValueError("Performance telemetry schema error: total_elapsed_ms must be finite and non-negative")

    return {
        "schema_name": PERFORMANCE_TELEMETRY_SCHEMA_NAME,
        "schema_version": PERFORMANCE_TELEMETRY_SCHEMA_VERSION,
        "generated_at_utc": utc_now_iso(),
        "fov_id": int(fov_id),
        "config": _config_summary(config),
        "providers": providers,
        "run": {
            "started_at_utc": run_started_at_utc,
            "finished_at_utc": run_finished_at_utc,
            "total_elapsed_ms": None if total_elapsed_ms is None else round(float(total_elapsed_ms), 3),
        },
        "stage_order": list(get_ordered_stage_ids()),
        "stages": stages,
        "artifacts": artifacts,
    }


def write_performance_telemetry(
    *,
    config: Any,
    fov_id: int,
    stage_timings: Sequence[Mapping[str, Any]],
    run_started_at_utc: str | None = None,
    run_finished_at_utc: str | None = None,
    total_elapsed_ms: float | None = None,
    require_completed_outputs: bool = False,
) -> Path:
    """Persist ``qc_reports/performance_fov_<fov_id>.json`` and return its path."""

    base_dir = _output_directory(config)
    paths = get_fov_output_structure(base_dir, int(fov_id))
    output_path = paths["qc"] / f"performance_fov_{int(fov_id)}.json"
    payload = build_performance_telemetry_payload(
        config=config,
        fov_id=int(fov_id),
        stage_timings=stage_timings,
        run_started_at_utc=run_started_at_utc,
        run_finished_at_utc=run_finished_at_utc,
        total_elapsed_ms=total_elapsed_ms,
        require_completed_outputs=require_completed_outputs,
    )
    write_backend_metadata(output_path, payload)
    return output_path
