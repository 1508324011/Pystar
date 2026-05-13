"""MATLAB Engine bootstrap, session lifecycle, and boundary timing helpers.

All MATLAB-backed providers share the same session-management rules: discover a
local MATLAB installation, import `matlab.engine`, start a session lazily, add the
repo-local runtime path, validate runtime files once per session, and record
boundary timings in a stage-neutral schema.  Keeping this logic in one module
prevents each provider from inventing slightly different fallback or provenance
behavior.
"""

from __future__ import annotations

import importlib
import hashlib
import json
import os
import platform
import re
import shutil
import sys
import time
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, cast

from .serialization import json_safe


MATLAB_ROOT_ENV_KEYS = (
    "PYSTAR_MATLAB_ROOT",
    "MATLAB_ROOT",
    "MATLAB_HOME",
    "MATLABHOME",
    "MWE_INSTALL",
)

MATLAB_BOUNDARY_SEAM_COST_KEYS = (
    "engine_bootstrap_ms",
    "runtime_file_validation_ms",
    "input_staging_ms",
    "matlab_call_ms",
    "result_validation_ms",
    "canonical_persistence_ms",
    "teardown_ms",
)

MATLAB_SESSION_TIMING_KEYS = (
    "configure_environment_ms",
    "engine_module_import_ms",
    "factory_resolution_ms",
    "find_matlab_ms",
    "connect_matlab_ms",
    "runtime_file_validation_ms",
    "start_matlab_ms",
    "share_engine_ms",
    "addpath_ms",
    "health_check_ms",
    "sentinel_ms",
    "engine_bootstrap_ms",
    "teardown_ms",
)

MATLAB_SESSION_COUNT_KEYS = (
    "engine_bootstrap_count",
    "engine_reuse_count",
    "runtime_file_validation_count",
    "runtime_file_validation_reuse_count",
    "addpath_call_count",
    "teardown_count",
    "teardown_warning_count",
)


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _elapsed_ms(start_time: float) -> float:
    return round((time.perf_counter() - start_time) * 1000.0, 3)


def _format_exception_message(prefix: str, exc: Exception) -> str:
    detail = str(exc).strip()
    if detail:
        return f"{prefix}: {detail}"
    return f"{prefix} ({exc.__class__.__name__})"


def _zeroed_metric_map(keys: Sequence[str]) -> dict[str, float]:
    return {key: 0.0 for key in keys}


def _metric_as_float(value: Any) -> float:
    return round(float(value), 3) if isinstance(value, (int, float)) else 0.0


def _count_as_int(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) else 0


def _earlier_iso_timestamp(current: Any, candidate: Any) -> str | None:
    if isinstance(current, str) and current and isinstance(candidate, str) and candidate:
        return min(current, candidate)
    if isinstance(current, str) and current:
        return current
    if isinstance(candidate, str) and candidate:
        return candidate
    return None


def _later_iso_timestamp(current: Any, candidate: Any) -> str | None:
    if isinstance(current, str) and current and isinstance(candidate, str) and candidate:
        return max(current, candidate)
    if isinstance(candidate, str) and candidate:
        return candidate
    if isinstance(current, str) and current:
        return current
    return None


def _session_timing_totals(snapshot: Mapping[str, Any] | None) -> dict[str, float]:
    totals = _zeroed_metric_map(MATLAB_SESSION_TIMING_KEYS)
    if not isinstance(snapshot, Mapping):
        return totals
    raw_totals = snapshot.get("aggregate_timing_ms")
    if not isinstance(raw_totals, Mapping):
        return totals
    for key in MATLAB_SESSION_TIMING_KEYS:
        totals[key] = _metric_as_float(raw_totals.get(key))
    return totals


def _accumulate_metric_map(target: dict[str, float], source: Mapping[str, Any], *, mode: str) -> None:
    for key in target:
        value = _metric_as_float(source.get(key))
        if mode == "sum":
            target[key] = round(target[key] + value, 3)
            continue
        if mode == "max":
            target[key] = round(max(target[key], value), 3)
            continue
        raise ValueError(f"Unsupported metric accumulation mode: {mode!r}")


def _session_snapshot_from_trace(trace: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for key in ("session_lifecycle_after", "session_lifecycle_before"):
        snapshot = trace.get(key)
        if isinstance(snapshot, Mapping):
            return snapshot
    return None


def _normalize_session_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any] | None:
    session_id = snapshot.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        return None

    normalized = {
        "schema_version": "1.0",
        "session_id": session_id,
        "consumer": snapshot.get("consumer"),
        "runtime_path": snapshot.get("runtime_path"),
        "entrypoint": snapshot.get("entrypoint"),
        "session_started_at": snapshot.get("session_started_at"),
        "session_last_used_at": snapshot.get("session_last_used_at"),
        "session_last_teardown_at": snapshot.get("session_last_teardown_at"),
        **{key: _count_as_int(snapshot.get(key)) for key in MATLAB_SESSION_COUNT_KEYS},
        "aggregate_timing_ms": _session_timing_totals(snapshot),
    }
    shared_session = snapshot.get("shared_session")
    if isinstance(shared_session, Mapping):
        normalized["shared_session"] = dict(shared_session)
    return normalized


def _merge_session_snapshot(current: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(current)
    merged["session_started_at"] = _earlier_iso_timestamp(
        current.get("session_started_at"),
        candidate.get("session_started_at"),
    )
    merged["session_last_used_at"] = _later_iso_timestamp(
        current.get("session_last_used_at"),
        candidate.get("session_last_used_at"),
    )
    merged["session_last_teardown_at"] = _later_iso_timestamp(
        current.get("session_last_teardown_at"),
        candidate.get("session_last_teardown_at"),
    )
    for key in MATLAB_SESSION_COUNT_KEYS:
        merged[key] = max(_count_as_int(current.get(key)), _count_as_int(candidate.get(key)))

    merged_timing = _session_timing_totals(current)
    _accumulate_metric_map(merged_timing, _session_timing_totals(candidate), mode="max")
    merged["aggregate_timing_ms"] = merged_timing
    current_shared = current.get("shared_session")
    candidate_shared = candidate.get("shared_session")
    if isinstance(candidate_shared, Mapping):
        merged["shared_session"] = dict(candidate_shared)
    elif isinstance(current_shared, Mapping):
        merged["shared_session"] = dict(current_shared)
    return merged


def summarize_matlab_session_snapshots(snapshots: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Merge raw session lifecycle snapshots by MATLAB session id."""

    merged_sessions: dict[str, dict[str, Any]] = {}
    for snapshot in snapshots:
        normalized = _normalize_session_snapshot(snapshot)
        if normalized is None:
            continue
        session_id = str(normalized["session_id"])
        existing = merged_sessions.get(session_id)
        merged_sessions[session_id] = normalized if existing is None else _merge_session_snapshot(existing, normalized)

    if not merged_sessions:
        return None

    sessions = [merged_sessions[session_id] for session_id in sorted(merged_sessions)]
    aggregate_counts = {key: 0 for key in MATLAB_SESSION_COUNT_KEYS}
    aggregate_timing_ms = _zeroed_metric_map(MATLAB_SESSION_TIMING_KEYS)
    consumers = sorted({str(item["consumer"]) for item in sessions if isinstance(item.get("consumer"), str) and item["consumer"]})
    runtime_paths = sorted({str(item["runtime_path"]) for item in sessions if isinstance(item.get("runtime_path"), str) and item["runtime_path"]})
    entrypoints = sorted({str(item["entrypoint"]) for item in sessions if isinstance(item.get("entrypoint"), str) and item["entrypoint"]})

    sessions_with_reuse = 0
    sessions_with_bootstrap = 0
    sessions_with_teardown_warning = 0
    for item in sessions:
        for key in MATLAB_SESSION_COUNT_KEYS:
            aggregate_counts[key] += _count_as_int(item.get(key))
        _accumulate_metric_map(aggregate_timing_ms, _session_timing_totals(item), mode="sum")
        if _count_as_int(item.get("engine_reuse_count")) > 0:
            sessions_with_reuse += 1
        if _count_as_int(item.get("engine_bootstrap_count")) > 0:
            sessions_with_bootstrap += 1
        if _count_as_int(item.get("teardown_warning_count")) > 0:
            sessions_with_teardown_warning += 1

    return {
        "schema_version": "1.0",
        "session_count": len(sessions),
        "session_ids": [str(item["session_id"]) for item in sessions],
        "consumers": consumers,
        "runtime_paths": runtime_paths,
        "entrypoints": entrypoints,
        "sessions_with_bootstrap": sessions_with_bootstrap,
        "sessions_with_reuse": sessions_with_reuse,
        "sessions_with_teardown_warning": sessions_with_teardown_warning,
        "aggregate_counts": aggregate_counts,
        "aggregate_timing_ms": aggregate_timing_ms,
        "sessions": sessions,
    }


def summarize_matlab_session_lifecycle(traces: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Summarize MATLAB Engine lifecycle data embedded in boundary traces."""

    snapshots = [snapshot for trace in traces for snapshot in [_session_snapshot_from_trace(trace)] if isinstance(snapshot, Mapping)]
    return summarize_matlab_session_snapshots(snapshots)


def merge_matlab_session_lifecycle_summaries(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Merge already-normalized lifecycle summaries across stages or FOVs."""

    session_snapshots: list[Mapping[str, Any]] = []
    for summary in summaries:
        if not isinstance(summary, Mapping):
            continue
        sessions = summary.get("sessions")
        if not isinstance(sessions, Sequence) or isinstance(sessions, (str, bytes)):
            continue
        for session in sessions:
            if isinstance(session, Mapping):
                session_snapshots.append(session)
    return summarize_matlab_session_snapshots(session_snapshots)


def _platform_architecture() -> str:
    system = platform.system()
    if system == "Linux":
        return "glnxa64"
    if system == "Windows":
        return "win64"
    if system == "Darwin":
        return "maca64" if platform.machine() == "arm64" else "maci64"
    raise RuntimeError(f"Unsupported platform for MATLAB Engine bootstrap: {system!r}")


def _iter_candidate_roots() -> list[Path]:
    candidates: list[Path] = []

    for env_key in MATLAB_ROOT_ENV_KEYS:
        raw_value = os.environ.get(env_key)
        if not raw_value:
            continue
        candidates.append(Path(raw_value).expanduser())

    matlab_cli = shutil.which("matlab")
    if matlab_cli is not None:
        matlab_path = Path(matlab_cli).expanduser().resolve()
        candidates.append(matlab_path.parent.parent)

    return candidates


def _engine_paths_for_root(matlab_root: Path) -> tuple[Path, Path]:
    engine_dist = matlab_root / "extern" / "engines" / "python" / "dist"
    engine_binary_dir = engine_dist / "matlab" / "engine" / _platform_architecture()
    return engine_dist, engine_binary_dir


def _is_valid_matlab_root(matlab_root: Path) -> bool:
    if not matlab_root.exists():
        return False
    engine_dist, engine_binary_dir = _engine_paths_for_root(matlab_root)
    return engine_dist.is_dir() and engine_binary_dir.is_dir()


def detect_matlab_root() -> Path | None:
    """Return the first MATLAB root with a usable Python Engine layout."""

    seen: set[Path] = set()
    for candidate in _iter_candidate_roots():
        try:
            resolved = candidate.resolve()
        except FileNotFoundError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if _is_valid_matlab_root(resolved):
            return resolved
    return None


def configure_matlab_engine_environment(*, matlab_root: Path | None = None) -> dict[str, Any]:
    """Expose MATLAB Engine paths to the current Python process.

    The function mutates `sys.path` and `MWE_INSTALL` only when a usable MATLAB
    root is found.  The returned dict is intentionally JSON-serializable so it can
    be embedded in provider diagnostics and CLI probes.
    """

    resolved_root = matlab_root.resolve() if matlab_root is not None else detect_matlab_root()
    status: dict[str, Any] = {
        "matlab_root": None,
        "engine_dist_path": None,
        "engine_binary_path": None,
        "configured": False,
        "added_paths": [],
    }
    if resolved_root is None:
        status["reason"] = (
            "No usable MATLAB root was detected from environment variables or the `matlab` executable on PATH."
        )
        return status

    engine_dist, engine_binary_dir = _engine_paths_for_root(resolved_root)
    if not engine_dist.is_dir() or not engine_binary_dir.is_dir():
        status["reason"] = (
            "Detected MATLAB root is missing Python Engine files: "
            f"{resolved_root}"
        )
        return status

    os.environ["MWE_INSTALL"] = str(resolved_root)

    added_paths: list[str] = []
    for path in (engine_binary_dir, engine_dist):
        path_str = str(path)
        if path_str in sys.path:
            continue
        sys.path.insert(0, path_str)
        added_paths.append(path_str)

    status.update(
        {
            "matlab_root": str(resolved_root),
            "engine_dist_path": str(engine_dist),
            "engine_binary_path": str(engine_binary_dir),
            "configured": True,
            "added_paths": added_paths,
        }
    )
    return status


def load_matlab_engine_factory(*, consumer: str) -> tuple[Callable[[], Any], dict[str, Any]]:
    """Resolve `matlab.engine.start_matlab` and report bootstrap discovery timings."""

    configure_started = time.perf_counter()
    status = configure_matlab_engine_environment()
    configure_environment_ms = _elapsed_ms(configure_started)

    import_started = time.perf_counter()
    try:
        matlab_engine = importlib.import_module("matlab.engine")
    except ImportError as exc:
        matlab_root = status.get("matlab_root")
        if matlab_root is None:
            hint = (
                "No MATLAB installation could be discovered from environment variables or `matlab` on PATH."
            )
        else:
            hint = (
                "Detected MATLAB root "
                f"{matlab_root}, but the Engine module is still unavailable."
            )
        raise RuntimeError(
            "MATLAB Engine for Python is unavailable. Configure the active Python environment before using "
            f"{consumer}. {hint}"
        ) from exc
    engine_module_import_ms = _elapsed_ms(import_started)

    resolve_started = time.perf_counter()
    start_matlab = getattr(matlab_engine, "start_matlab", None)
    if start_matlab is None or not callable(start_matlab):
        raise RuntimeError("Imported 'matlab.engine' module does not expose callable start_matlab()")
    factory_resolution_ms = _elapsed_ms(resolve_started)

    return start_matlab, {
        "consumer": consumer,
        "configured_environment": status,
        "configure_environment_ms": configure_environment_ms,
        "engine_module_import_ms": engine_module_import_ms,
        "factory_resolution_ms": factory_resolution_ms,
    }


def load_matlab_engine_module_with_metrics(*, consumer: str) -> tuple[Any, dict[str, Any]]:
    """Resolve the `matlab.engine` module and shared-session callables."""

    configure_started = time.perf_counter()
    status = configure_matlab_engine_environment()
    configure_environment_ms = _elapsed_ms(configure_started)

    import_started = time.perf_counter()
    try:
        matlab_engine = importlib.import_module("matlab.engine")
    except ImportError as exc:
        matlab_root = status.get("matlab_root")
        if matlab_root is None:
            hint = (
                "No MATLAB installation could be discovered from environment variables or `matlab` on PATH."
            )
        else:
            hint = (
                "Detected MATLAB root "
                f"{matlab_root}, but the Engine module is still unavailable."
            )
        raise RuntimeError(
            "MATLAB Engine for Python is unavailable. Configure the active Python environment before using "
            f"{consumer}. {hint}"
        ) from exc
    engine_module_import_ms = _elapsed_ms(import_started)

    resolve_started = time.perf_counter()
    for function_name in ("find_matlab", "connect_matlab", "start_matlab"):
        candidate = getattr(matlab_engine, function_name, None)
        if candidate is None or not callable(candidate):
            raise RuntimeError(f"Imported 'matlab.engine' module does not expose callable {function_name}()")
    factory_resolution_ms = _elapsed_ms(resolve_started)

    return matlab_engine, {
        "consumer": consumer,
        "configured_environment": status,
        "configure_environment_ms": configure_environment_ms,
        "engine_module_import_ms": engine_module_import_ms,
        "factory_resolution_ms": factory_resolution_ms,
    }


def load_matlab_engine_module(*, consumer: str) -> Any:
    """Import the `matlab.engine` module after environment configuration."""

    status = configure_matlab_engine_environment()
    try:
        return importlib.import_module("matlab.engine")
    except ImportError as exc:
        matlab_root = status.get("matlab_root")
        if matlab_root is None:
            hint = (
                "No MATLAB installation could be discovered from environment variables or `matlab` on PATH."
            )
        else:
            hint = (
                "Detected MATLAB root "
                f"{matlab_root}, but the Engine module is still unavailable."
            )
        raise RuntimeError(
            "MATLAB Engine for Python is unavailable. Configure the active Python environment before using "
            f"{consumer}. {hint}"
        ) from exc


def probe_matlab_engine_environment() -> dict[str, Any]:
    """Return a diagnostic snapshot for `scripts/check_matlab_engine.py`."""

    status = configure_matlab_engine_environment()
    probe: dict[str, Any] = {
        "python_executable": sys.executable,
        "pythonpath": os.environ.get("PYTHONPATH"),
        "mwe_install": os.environ.get("MWE_INSTALL"),
        **status,
    }
    try:
        matlab_engine = importlib.import_module("matlab.engine")
        probe["available"] = True
        probe["module"] = getattr(matlab_engine, "__file__", None)
    except Exception as exc:  # pragma: no cover - depends on local MATLAB install
        probe["available"] = False
        probe["error"] = str(exc)
        matlab_root = status.get("matlab_root")
        if matlab_root is None:
            probe["next_step"] = (
                "Expose a MATLAB installation via PATH or set PYSTAR_MATLAB_ROOT/MATLAB_ROOT before running MATLAB-backed scenarios."
            )
        else:
            probe["next_step"] = (
                "Verify the detected MATLAB root contains a compatible Engine package and that the current Python "
                f"version matches the local MATLAB release. Detected root: {matlab_root}"
            )
    return probe


def close_matlab_engine_best_effort(engine: Any, *, consumer: str) -> str | None:
    """Try to close a MATLAB Engine without invalidating completed work.

    MATLAB can occasionally raise teardown/process-termination errors after the
    provider call already completed and artifacts were validated.  In that case we
    return a warning string for provenance instead of reclassifying the successful
    stage as failed.
    """

    if engine is None:
        return None

    try:
        engine.quit()
        return None
    except Exception as exc:  # pragma: no cover - depends on MATLAB runtime behavior
        detail = str(exc).strip() or exc.__class__.__name__
        if isinstance(exc, SystemError) and "cannot be terminated" in detail.lower():
            message = (
                "MATLAB Engine teardown reported a process-termination issue after "
                f"{consumer} completed. Dropping the engine handle and relying on worker-process "
                f"reclamation instead of reclassifying the completed run as failed. Original error: {detail}"
            )
        else:
            message = (
                "MATLAB Engine teardown failed after "
                f"{consumer} completed. Dropping the engine handle without reclassifying the completed "
                f"run as failed. Original error: {detail}"
            )
        warnings.warn(message)
        return message


PYSTAR_SENTINEL_SCHEMA_VERSION = "1.0"
PYSTAR_SENTINEL_APPDATA_KEY = "PyStarSession"
MATLAB_SHARED_SESSION_NAME_MAX_LENGTH = 63
_GENERATED_SHARED_SESSION_PREFIX = "pystar"
_TRUNCATED_SESSION_COMPONENT_DIGEST_LENGTH = 6
_MIN_GENERATED_SESSION_COMPONENT_LENGTH = 8


def _pystar_source_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _pystar_matlab_runtime_root() -> Path:
    return (_pystar_source_root() / "matlab_runtime").resolve()


def _sanitize_session_name_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "config"


def _truncate_session_name_component(component: str, *, max_length: int) -> str:
    if max_length < 1:
        raise ValueError("Generated MATLAB shared-session name has no room for a deterministic identity component")
    if len(component) <= max_length:
        return component

    digest = hashlib.sha256(component.encode("utf-8")).hexdigest()[:_TRUNCATED_SESSION_COMPONENT_DIGEST_LENGTH]
    if max_length <= len(digest):
        return digest[:max_length]

    prefix_length = max_length - len(digest) - 1
    prefix = component[:prefix_length].rstrip("_")
    if not prefix:
        return digest[:max_length]
    return f"{prefix}_{digest}"


def _generated_shared_session_name(config_stem: str, config_hash: str, run_id: str) -> str:
    prefix = _GENERATED_SHARED_SESSION_PREFIX
    config_stem = _sanitize_session_name_component(config_stem)
    config_hash = _sanitize_session_name_component(config_hash)
    run_id = _sanitize_session_name_component(run_id)
    generated = f"{prefix}_{config_stem}_{config_hash}_{run_id}"
    if len(generated) <= MATLAB_SHARED_SESSION_NAME_MAX_LENGTH:
        return generated

    component_budget = MATLAB_SHARED_SESSION_NAME_MAX_LENGTH - len(prefix) - len(config_hash) - 3
    if component_budget < 2:
        raise ValueError(
            "Generated MATLAB shared-session name cannot fit MATLAB namelengthmax while preserving "
            f"the deterministic config hash {config_hash!r}"
        )

    min_stem_budget = min(_MIN_GENERATED_SESSION_COMPONENT_LENGTH, max(1, component_budget // 2))
    max_run_id_budget = component_budget - min_stem_budget
    if len(run_id) > max_run_id_budget:
        run_id = _truncate_session_name_component(run_id, max_length=max_run_id_budget)

    config_stem_budget = component_budget - len(run_id)
    config_stem = _truncate_session_name_component(config_stem, max_length=config_stem_budget)
    return f"{prefix}_{config_stem}_{config_hash}_{run_id}"


def _validate_shared_session_name(name: str, *, label: str = "MATLAB shared-session name") -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{label} must be a non-empty string")
    normalized = name.strip()
    if len(normalized) > MATLAB_SHARED_SESSION_NAME_MAX_LENGTH:
        raise ValueError(
            f"{label} must be at most {MATLAB_SHARED_SESSION_NAME_MAX_LENGTH} characters for MATLAB "
            f"namelengthmax/shareEngine compatibility; got {len(normalized)} characters"
        )
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", normalized):
        raise ValueError(
            f"{label} must contain only ASCII letters, digits, and underscores, and must start with a letter; "
            f"got {name!r}"
        )
    return normalized


def _config_hash8(config: Any) -> str:
    raw_hash = getattr(config, "config_sha256", None)
    if not isinstance(raw_hash, str) or not raw_hash.strip():
        return "nohash"
    digest = raw_hash.strip()
    if digest.startswith("sha256:"):
        digest = digest.split(":", 1)[1]
    digest = re.sub(r"[^A-Fa-f0-9]", "", digest)
    return digest[:8].lower() if digest else "nohash"


def _config_stem(config: Any) -> str:
    config_source_path = getattr(config, "config_source_path", None)
    if config_source_path is None:
        return "config"
    try:
        stem = Path(config_source_path).stem
    except TypeError:
        stem = "config"
    return _sanitize_session_name_component(stem)


def _run_id() -> tuple[str, str]:
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    slurm_array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if slurm_job_id and slurm_array_task_id:
        return (
            f"slurm_{_sanitize_session_name_component(slurm_job_id)}_{_sanitize_session_name_component(slurm_array_task_id)}",
            "slurm",
        )
    return f"pid_{os.getpid()}", "pid"


def resolve_matlab_shared_session_name(config: Any) -> dict[str, str]:
    """Return the exact deterministic shared MATLAB Engine session identity."""

    shared_cfg = getattr(getattr(getattr(config, "providers", None), "matlab", None), "shared_session", None)
    configured_name = getattr(shared_cfg, "name", None)
    if isinstance(configured_name, str) and configured_name.strip():
        return {
            "name": _validate_shared_session_name(
                configured_name,
                label="providers.matlab.shared_session.name",
            ),
            "name_source": "configured",
            "run_id_source": "configured",
        }

    run_id, run_id_source = _run_id()
    generated = _generated_shared_session_name(_config_stem(config), _config_hash8(config), run_id)
    return {
        "name": _validate_shared_session_name(generated, label="generated MATLAB shared-session name"),
        "name_source": "generated",
        "run_id_source": run_id_source,
    }


def should_use_shared_matlab_session(config: Any) -> bool:
    """Return true when shared MATLAB sessions are both enabled and needed."""

    shared_cfg = getattr(getattr(getattr(config, "providers", None), "matlab", None), "shared_session", None)
    if not bool(getattr(shared_cfg, "enabled", False)):
        return False
    pipeline = getattr(config, "pipeline", None)
    if pipeline is None:
        return False
    for method_name in (
        "uses_matlab_preprocessing",
        "uses_matlab_registration",
        "uses_matlab_spot_finding",
        "uses_matlab_extraction",
    ):
        method = getattr(pipeline, method_name, None)
        if callable(method) and bool(method()):
            return True
    return False


def _matlab_single_quote(value: str) -> str:
    return value.replace("'", "''")


def _share_engine_with_name(engine: Any, session_name: str) -> None:
    command = f"matlab.engine.shareEngine('{_matlab_single_quote(session_name)}')"
    engine.eval(command, nargout=0)


def _declares_python_attribute(engine: Any, attribute_name: str) -> bool:
    try:
        if attribute_name in vars(engine):
            return True
    except TypeError:
        pass

    for cls in getattr(type(engine), "__mro__", (type(engine),)):
        try:
            if attribute_name in vars(cls):
                return True
        except TypeError:
            continue
    return False


def _declares_true_python_attribute(engine: Any, attribute_name: str) -> bool:
    try:
        if vars(engine).get(attribute_name) is True:
            return True
    except TypeError:
        pass

    for cls in getattr(type(engine), "__mro__", (type(engine),)):
        try:
            if vars(cls).get(attribute_name) is True:
                return True
        except TypeError:
            continue
    return False


def _uses_explicit_fake_sentinel_seam(engine: Any) -> bool:
    return _declares_true_python_attribute(engine, "_pystar_fake_engine") or _declares_true_python_attribute(
        engine,
        "_pystar_fake_sentinel_seam",
    )


def _fake_sentinel_callable(engine: Any, attribute_name: str) -> Callable[..., Any] | None:
    if not (_uses_explicit_fake_sentinel_seam(engine) or _declares_python_attribute(engine, attribute_name)):
        return None
    candidate = getattr(engine, attribute_name, None)
    return candidate if callable(candidate) else None


def _read_pystar_sentinel(engine: Any) -> dict[str, Any] | None:
    fake_reader = _fake_sentinel_callable(engine, "_pystar_get_sentinel")
    if fake_reader is not None:
        sentinel = fake_reader()
        return dict(sentinel) if isinstance(sentinel, Mapping) else None

    key = _matlab_single_quote(PYSTAR_SENTINEL_APPDATA_KEY)
    has_sentinel = engine.eval(f"isappdata(0, '{key}')", nargout=1)
    if not bool(has_sentinel):
        return None
    payload = engine.eval(f"jsonencode(getappdata(0, '{key}'))", nargout=1)
    if not isinstance(payload, str) or not payload.strip():
        return None
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError("MATLAB shared-session sentinel exists but is not valid JSON") from exc
    if not isinstance(decoded, Mapping):
        raise RuntimeError("MATLAB shared-session sentinel must decode to a JSON object")
    return dict(decoded)


def _write_pystar_sentinel(engine: Any, sentinel: Mapping[str, Any]) -> None:
    fake_writer = _fake_sentinel_callable(engine, "_pystar_set_sentinel")
    if fake_writer is not None:
        fake_writer(dict(sentinel))
        return

    key = _matlab_single_quote(PYSTAR_SENTINEL_APPDATA_KEY)
    payload = _matlab_single_quote(json.dumps(json_safe(dict(sentinel)), sort_keys=True))
    engine.eval(
        f"setappdata(0, '{key}', jsondecode('{payload}'))",
        nargout=0,
    )


def _sentinel_identity(session_name: str) -> dict[str, str]:
    return {
        "sentinel_schema_version": PYSTAR_SENTINEL_SCHEMA_VERSION,
        "session_name": session_name,
        "pystar_source_root": str(_pystar_source_root().resolve()),
        "matlab_runtime_root": str(_pystar_matlab_runtime_root()),
    }


def _safe_config_reference(config: Any) -> dict[str, str | None]:
    config_source_path = getattr(config, "config_source_path", None)
    config_hash = getattr(config, "config_sha256", None)
    return {
        "config_source_path": None if config_source_path is None else str(config_source_path),
        "config_hash": config_hash if isinstance(config_hash, str) else None,
    }


def _shared_metadata_from_acquire_record(record: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "shared_session_enabled",
        "shared_session_name",
        "shared_session_name_source",
        "shared_session_lifetime",
        "shared_session_mode",
        "shared_session_owner_id",
        "engine_acquire_mode",
        "attached_existing",
        "started_owned",
        "claimed_existing_without_sentinel",
        "health_check_status",
        "health_check_timestamp_utc",
        "health_check_duration_ms",
        "sentinel_schema_version",
        "sentinel_identity_match",
        "pystar_source_root",
        "matlab_runtime_root",
        "config_source_path",
        "config_hash",
        "run_id_source",
    )
    return {key: record.get(key) for key in keys if key in record}


def _record_shared_session_on_lifecycle(session: dict[str, Any], record: Mapping[str, Any]) -> None:
    session["shared_session"] = _shared_metadata_from_acquire_record(record)


def record_matlab_session_shared_owner_acquire(
    session: dict[str, Any],
    acquire_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Record a borrowed shared-owner acquisition on a stage-local lifecycle."""

    measured_at = str(acquire_record.get("measured_at") or _iso_utc_now())
    details = dict(acquire_record)
    engine_acquire_mode = str(details.get("engine_acquire_mode") or "owner_reuse")
    if engine_acquire_mode == "owner_reuse":
        session["engine_reuse_count"] = int(session.get("engine_reuse_count", 0)) + 1
        session["last_reuse"] = {
            "measured_at": measured_at,
            "engine_reused": True,
            "engine_acquire_mode": engine_acquire_mode,
            "shared_session_name": details.get("shared_session_name"),
        }
    else:
        session["engine_bootstrap_count"] = int(session.get("engine_bootstrap_count", 0)) + 1
        session["last_bootstrap"] = details
        if session.get("session_started_at") is None:
            session["session_started_at"] = measured_at

    if _metric_as_float(details.get("addpath_ms")) > 0:
        session["addpath_call_count"] = int(session.get("addpath_call_count", 0)) + 1

    session["session_last_used_at"] = measured_at
    timing_totals = _session_timing_totals(session)
    for key in MATLAB_SESSION_TIMING_KEYS:
        if key == "teardown_ms":
            continue
        timing_totals[key] = round(timing_totals[key] + _metric_as_float(details.get(key)), 3)
    session["aggregate_timing_ms"] = timing_totals
    _record_shared_session_on_lifecycle(session, details)
    return details


class MatlabSharedSessionOwner:
    """Own or borrow one deterministic named MATLAB Engine session."""

    def __init__(
        self,
        *,
        session_name: str,
        name_source: str,
        lifetime: str,
        health_check_timeout_s: float,
        config_reference: Mapping[str, str | None] | None = None,
        run_id_source: str = "unknown",
        fov_id: int | None = None,
        engine_module_loader: Callable[[str], tuple[Any, Mapping[str, Any]]] | None = None,
    ) -> None:
        self.session_name = _validate_shared_session_name(session_name)
        if lifetime not in {"run", "fov"}:
            raise ValueError(f"MATLAB shared-session lifetime must be 'run' or 'fov', got {lifetime!r}")
        if health_check_timeout_s <= 0:
            raise ValueError("MATLAB shared-session health_check_timeout_s must be positive")
        self.name_source = name_source
        self.lifetime = lifetime
        self.health_check_timeout_s = float(health_check_timeout_s)
        self.config_reference = dict(config_reference or {})
        self.run_id_source = run_id_source
        self.fov_id = None if fov_id is None else int(fov_id)
        self.owner_id = uuid.uuid4().hex
        self.engine_module_loader = engine_module_loader or (
            lambda consumer: load_matlab_engine_module_with_metrics(consumer=consumer)
        )
        self.engine: Any = None
        self.mode: str = "inactive"
        self._runtime_dirs_added: set[str] = set()
        self.last_acquire_record: dict[str, Any] | None = None
        self.last_teardown: dict[str, Any] | None = None

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        fov_id: int | None = None,
        engine_module_loader: Callable[[str], tuple[Any, Mapping[str, Any]]] | None = None,
    ) -> "MatlabSharedSessionOwner":
        shared_cfg = getattr(getattr(getattr(config, "providers", None), "matlab", None), "shared_session", None)
        identity = resolve_matlab_shared_session_name(config)
        return cls(
            session_name=identity["name"],
            name_source=identity["name_source"],
            lifetime=str(getattr(shared_cfg, "lifetime", "run")),
            health_check_timeout_s=float(getattr(shared_cfg, "health_check_timeout_s", 30.0)),
            config_reference=_safe_config_reference(config),
            run_id_source=identity["run_id_source"],
            fov_id=fov_id,
            engine_module_loader=engine_module_loader,
        )

    def __enter__(self) -> "MatlabSharedSessionOwner":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback_obj: Any) -> None:
        self.close()

    def _base_acquire_record(self, *, consumer: str, runtime_dir: Path, entrypoint: str) -> dict[str, Any]:
        identity = _sentinel_identity(self.session_name)
        return {
            "schema_version": "1.0",
            "measured_at": _iso_utc_now(),
            "consumer": consumer,
            "fov_id": self.fov_id,
            "runtime_path": str(runtime_dir),
            "entrypoint": entrypoint,
            "shared_session_enabled": True,
            "shared_session_name": self.session_name,
            "shared_session_name_source": self.name_source,
            "shared_session_lifetime": self.lifetime,
            "shared_session_owner_id": self.owner_id,
            "run_id_source": self.run_id_source,
            "sentinel_schema_version": PYSTAR_SENTINEL_SCHEMA_VERSION,
            "pystar_source_root": identity["pystar_source_root"],
            "matlab_runtime_root": identity["matlab_runtime_root"],
            **self.config_reference,
        }

    def _apply_runtime_path(self, engine: Any, runtime_dir: Path, *, addpath_failure_prefix: str) -> float:
        runtime_key = str(runtime_dir.resolve())
        if runtime_key in self._runtime_dirs_added:
            return 0.0
        addpath_started = time.perf_counter()
        try:
            engine.addpath(str(runtime_dir), nargout=0)
        except Exception as exc:
            raise RuntimeError(
                _format_exception_message(
                    f"{addpath_failure_prefix}: {runtime_dir}",
                    exc,
                )
            ) from exc
        addpath_ms = _elapsed_ms(addpath_started)
        self._runtime_dirs_added.add(runtime_key)
        return addpath_ms

    def _resolve_entrypoint(self, engine: Any, entrypoint: str, runtime_dir: Path) -> Any:
        try:
            return getattr(engine, entrypoint)
        except AttributeError as exc:
            raise RuntimeError(
                f"MATLAB shared session '{self.session_name}' with runtime path {runtime_dir} does not expose entrypoint '{entrypoint}'"
            ) from exc

    def _run_health_check(
        self,
        engine: Any,
        *,
        runtime_dir: Path,
        entrypoint: str,
        mode: str,
    ) -> dict[str, Any]:
        health_started = time.perf_counter()
        liveness_started = time.perf_counter()
        try:
            version_callable = getattr(engine, "version", None)
            matlab_version = version_callable(nargout=1) if callable(version_callable) else engine.eval("version", nargout=1)
        except Exception as exc:
            raise RuntimeError(
                f"MATLAB shared session '{self.session_name}' failed liveness/version health check"
            ) from exc
        liveness_ms = _elapsed_ms(liveness_started)

        entrypoint_started = time.perf_counter()
        _ = self._resolve_entrypoint(engine, entrypoint, runtime_dir)
        entrypoint_resolution_ms = _elapsed_ms(entrypoint_started)

        sentinel_started = time.perf_counter()
        expected_identity = _sentinel_identity(self.session_name)
        observed_sentinel = _read_pystar_sentinel(engine)
        claimed_existing_without_sentinel = False
        sentinel_identity_match = True
        if observed_sentinel is not None:
            mismatches = {
                key: {"expected": expected_value, "observed": observed_sentinel.get(key)}
                for key, expected_value in expected_identity.items()
                if observed_sentinel.get(key) != expected_value
            }
            if mismatches:
                sentinel_identity_match = False
                raise RuntimeError(
                    "MATLAB shared-session sentinel identity mismatch for "
                    f"session '{self.session_name}'. Mismatches: {mismatches}. "
                    "Use a different providers.matlab.shared_session.name or close the stale MATLAB session."
                )
        else:
            claimed_existing_without_sentinel = mode == "attached_existing"
            _write_pystar_sentinel(engine, expected_identity)
        sentinel_ms = _elapsed_ms(sentinel_started)
        health_check_ms = _elapsed_ms(health_started)
        if health_check_ms / 1000.0 > self.health_check_timeout_s:
            raise RuntimeError(
                f"MATLAB shared session '{self.session_name}' health check exceeded timeout "
                f"{self.health_check_timeout_s:.3f}s"
            )
        return {
            "health_check_status": "passed",
            "health_check_timestamp_utc": _iso_utc_now(),
            "health_check_ms": health_check_ms,
            "health_check_duration_ms": health_check_ms,
            "health_check_timeout_s": self.health_check_timeout_s,
            "health_check_liveness_ms": liveness_ms,
            "health_check_entrypoint_resolution_ms": entrypoint_resolution_ms,
            "sentinel_ms": sentinel_ms,
            "sentinel_identity_match": sentinel_identity_match,
            "claimed_existing_without_sentinel": claimed_existing_without_sentinel,
            "matlab_version": matlab_version,
        }

    def _cleanup_failed_owned_start(self, engine: Any, record: dict[str, Any]) -> None:
        cleanup_started = time.perf_counter()
        warning_message = close_matlab_engine_best_effort(
            engine,
            consumer=f"MATLAB shared session '{self.session_name}' failed startup cleanup",
        )
        record["teardown_action"] = "cleanup_after_failed_start"
        record["teardown_ms"] = _elapsed_ms(cleanup_started)
        record["teardown_warning_count"] = 1 if warning_message else 0
        record["teardown_warning_message"] = warning_message

    def ensure_engine(
        self,
        *,
        consumer: str,
        runtime_dir: Path,
        entrypoint: str,
        startup_failure_prefix: str,
        addpath_failure_prefix: str,
    ) -> tuple[Any, dict[str, Any]]:
        """Attach/start the exact named session and verify it for one stage."""

        runtime_dir = Path(runtime_dir)
        if self.engine is not None:
            record = self._base_acquire_record(consumer=consumer, runtime_dir=runtime_dir, entrypoint=entrypoint)
            addpath_ms = self._apply_runtime_path(
                self.engine,
                runtime_dir,
                addpath_failure_prefix=addpath_failure_prefix,
            )
            health = self._run_health_check(
                self.engine,
                runtime_dir=runtime_dir,
                entrypoint=entrypoint,
                mode=cast(str, self.mode),
            )
            record.update(
                {
                    "engine_acquire_mode": "owner_reuse",
                    "shared_session_mode": self.mode,
                    "attached_existing": self.mode == "attached_existing",
                    "started_owned": self.mode == "started_owned",
                    "addpath_ms": addpath_ms,
                    "engine_bootstrap_ms": round(addpath_ms + float(health.get("health_check_duration_ms", 0.0)), 3),
                    **health,
                }
            )
            self.last_acquire_record = record
            return self.engine, dict(record)

        record = self._base_acquire_record(consumer=consumer, runtime_dir=runtime_dir, entrypoint=entrypoint)
        engine_module: Any = None
        factory_metrics: Mapping[str, Any] | None = None
        try:
            engine_module, factory_metrics = self.engine_module_loader(consumer)
        except TypeError:
            # Backward-compatible test seam for simple zero-argument factories.
            engine_module, factory_metrics = cast(Any, self.engine_module_loader)()
        if isinstance(factory_metrics, Mapping):
            record.update({key: factory_metrics.get(key) for key in factory_metrics})

        find_started = time.perf_counter()
        try:
            existing_names = tuple(str(name) for name in engine_module.find_matlab())
        except Exception as exc:
            raise RuntimeError(
                _format_exception_message(
                    f"Failed to list MATLAB shared sessions before using '{self.session_name}'",
                    exc,
                )
            ) from exc
        find_matlab_ms = _elapsed_ms(find_started)
        record["find_matlab_ms"] = find_matlab_ms
        record["available_shared_sessions"] = list(existing_names)

        engine: Any
        mode: str
        connect_matlab_ms = 0.0
        start_matlab_ms = 0.0
        share_engine_ms = 0.0
        if self.session_name in existing_names:
            connect_started = time.perf_counter()
            try:
                engine = engine_module.connect_matlab(self.session_name)
            except Exception as exc:
                raise RuntimeError(
                    _format_exception_message(
                        f"Failed to connect to MATLAB shared session '{self.session_name}'",
                        exc,
                    )
                ) from exc
            connect_matlab_ms = _elapsed_ms(connect_started)
            mode = "attached_existing"
            engine_acquire_mode = "connect_existing"
        else:
            start_started = time.perf_counter()
            try:
                engine = engine_module.start_matlab()
            except Exception as exc:
                raise RuntimeError(
                    _format_exception_message(
                        startup_failure_prefix,
                        exc,
                    )
                ) from exc
            start_matlab_ms = _elapsed_ms(start_started)
            mode = "started_owned"
            engine_acquire_mode = "cold_start"
            share_started = time.perf_counter()
            try:
                _share_engine_with_name(engine, self.session_name)
            except Exception as exc:
                self._cleanup_failed_owned_start(engine, record)
                raise RuntimeError(
                    _format_exception_message(
                        f"Failed to share newly started MATLAB Engine as '{self.session_name}'",
                        exc,
                    )
                ) from exc
            share_engine_ms = _elapsed_ms(share_started)

        addpath_ms = 0.0
        try:
            addpath_ms = self._apply_runtime_path(
                engine,
                runtime_dir,
                addpath_failure_prefix=addpath_failure_prefix,
            )
            health = self._run_health_check(
                engine,
                runtime_dir=runtime_dir,
                entrypoint=entrypoint,
                mode=mode,
            )
        except Exception:
            if mode == "started_owned":
                self._cleanup_failed_owned_start(engine, record)
            raise

        record.update(
            {
                "shared_session_mode": mode,
                "engine_acquire_mode": engine_acquire_mode,
                "attached_existing": mode == "attached_existing",
                "started_owned": mode == "started_owned",
                "connect_matlab_ms": connect_matlab_ms,
                "start_matlab_ms": start_matlab_ms,
                "share_engine_ms": share_engine_ms,
                "addpath_ms": addpath_ms,
                **health,
            }
        )
        record["engine_bootstrap_ms"] = round(
            _metric_as_float(record.get("configure_environment_ms"))
            + _metric_as_float(record.get("engine_module_import_ms"))
            + _metric_as_float(record.get("factory_resolution_ms"))
            + find_matlab_ms
            + connect_matlab_ms
            + start_matlab_ms
            + share_engine_ms
            + addpath_ms
            + float(health.get("health_check_duration_ms", 0.0)),
            3,
        )
        self.engine = engine
        self.mode = mode
        self.last_acquire_record = record
        return self.engine, dict(record)

    def close(self) -> dict[str, Any] | None:
        """Close only PyStar-owned sessions; never quit attached sessions."""

        if self.engine is None:
            return None
        record: dict[str, Any] = {
            "schema_version": "1.0",
            "measured_at": _iso_utc_now(),
            "shared_session_name": self.session_name,
            "shared_session_mode": self.mode,
            "shared_session_owner_id": self.owner_id,
            "shared_session_lifetime": self.lifetime,
        }
        teardown_started = time.perf_counter()
        if self.mode == "started_owned":
            warning_message = close_matlab_engine_best_effort(
                self.engine,
                consumer=f"MATLAB shared session '{self.session_name}'",
            )
            record["teardown_action"] = "quit_owned"
            record["warning_message"] = warning_message
            record["teardown_warning_count"] = 1 if warning_message else 0
        else:
            record["teardown_action"] = "borrowed_noop"
            record["warning_message"] = None
            record["teardown_warning_count"] = 0
        record["teardown_ms"] = _elapsed_ms(teardown_started)
        self.engine = None
        self.mode = "inactive"
        self._runtime_dirs_added.clear()
        self.last_teardown = record
        return dict(record)


class MATLABSessionCapsule:
    """Lazy MATLAB Engine session wrapper shared by all MATLAB providers.

    A provider creates one capsule per backend instance.  The capsule starts the
    engine on first use, adds the repo-local runtime directory, caches runtime
    file validation results, and records lifecycle counters/timings for later
    boundary reports.  It contains no provider-specific algorithm logic.
    """

    def __init__(
        self,
        *,
        consumer: str,
        runtime_dir: Path,
        entrypoint: str,
        engine_factory: Callable[[], Any] | None = None,
        engine_factory_consumer: str | None = None,
        startup_failure_prefix: str,
        addpath_failure_prefix: str,
        session_owner: MatlabSharedSessionOwner | None = None,
        runtime_file_validator: Callable[[], Sequence[Mapping[str, Any]]] | None = None,
    ) -> None:
        self.consumer = consumer
        self.runtime_dir = runtime_dir
        self.entrypoint = entrypoint
        self.engine_factory = engine_factory
        self.engine_factory_consumer = engine_factory_consumer or consumer
        self.startup_failure_prefix = startup_failure_prefix
        self.addpath_failure_prefix = addpath_failure_prefix
        self.session_owner = session_owner
        self.runtime_file_validator = runtime_file_validator
        self.engine: Any = None
        self.session_lifecycle = create_matlab_session_lifecycle(
            consumer=consumer,
            runtime_dir=runtime_dir,
            entrypoint=entrypoint,
        )
        self._last_engine_acquire: dict[str, Any] | None = None
        self._validated_runtime_files: list[dict[str, Any]] | None = None

    def close(self) -> None:
        """Best-effort close and reset cached runtime validation state."""

        if self.engine is None:
            self._validated_runtime_files = None
            return

        if self.session_owner is not None:
            teardown_started = time.perf_counter()
            teardown_details = record_matlab_session_teardown(
                self.session_lifecycle,
                teardown_ms=_elapsed_ms(teardown_started),
                warning_message=None,
            )
            teardown_details["teardown_action"] = "borrowed_noop"
            latest_owner_record = self.session_owner.last_acquire_record
            if isinstance(latest_owner_record, Mapping):
                _record_shared_session_on_lifecycle(self.session_lifecycle, latest_owner_record)
            self.engine = None
            self._last_engine_acquire = None
            self._validated_runtime_files = None
            return

        warning_message: str | None = None
        teardown_started = time.perf_counter()
        try:
            warning_message = close_matlab_engine_best_effort(
                self.engine,
                consumer=self.consumer,
            )
        finally:
            record_matlab_session_teardown(
                self.session_lifecycle,
                teardown_ms=_elapsed_ms(teardown_started),
                warning_message=warning_message,
            )
            self.engine = None
            self._last_engine_acquire = None
            self._validated_runtime_files = None

    def ensure_engine(self) -> Any:
        """Start or reuse a MATLAB Engine session for one provider call."""

        if self.engine is not None:
            self._last_engine_acquire = {
                "engine_reused_this_call": True,
                "session_bootstrap": None,
            }
            record_matlab_session_reuse(self.session_lifecycle)
            return self.engine

        if self.session_owner is not None:
            if self._validated_runtime_files is None and self.runtime_file_validator is not None:
                self.validate_runtime_files(self.runtime_file_validator)
            engine, acquire_record = self.session_owner.ensure_engine(
                consumer=self.consumer,
                runtime_dir=self.runtime_dir,
                entrypoint=self.entrypoint,
                startup_failure_prefix=self.startup_failure_prefix,
                addpath_failure_prefix=self.addpath_failure_prefix,
            )
            session_bootstrap = record_matlab_session_shared_owner_acquire(
                self.session_lifecycle,
                acquire_record,
            )
            engine_acquire_mode = str(acquire_record.get("engine_acquire_mode") or "owner_reuse")
            self.engine = engine
            self._last_engine_acquire = {
                "engine_reused_this_call": engine_acquire_mode in {"connect_existing", "owner_reuse"},
                "session_bootstrap": session_bootstrap,
            }
            return self.engine

        factory_metrics: Mapping[str, Any] | None = None
        if self.engine_factory is None:
            factory, factory_metrics = load_matlab_engine_factory(
                consumer=self.engine_factory_consumer,
            )
        else:
            factory = self.engine_factory

        engine_started = time.perf_counter()
        try:
            engine = factory()
        except Exception as exc:  # pragma: no cover - exact engine exception type depends on MATLAB install
            raise RuntimeError(
                _format_exception_message(
                    self.startup_failure_prefix,
                    exc,
                )
            ) from exc
        start_matlab_ms = _elapsed_ms(engine_started)

        addpath_started = time.perf_counter()
        try:
            engine.addpath(str(self.runtime_dir), nargout=0)
        except Exception as exc:  # pragma: no cover - exact engine exception type depends on MATLAB install
            try:
                engine.quit()
            except Exception:
                pass
            raise RuntimeError(
                _format_exception_message(
                    f"{self.addpath_failure_prefix}: {self.runtime_dir}",
                    exc,
                )
            ) from exc
        addpath_ms = _elapsed_ms(addpath_started)

        self.engine = engine
        self._last_engine_acquire = {
            "engine_reused_this_call": False,
            "session_bootstrap": record_matlab_session_bootstrap(
                self.session_lifecycle,
                factory_metrics=factory_metrics,
                start_matlab_ms=start_matlab_ms,
                addpath_ms=addpath_ms,
            ),
        }
        return self.engine

    def resolve_callable(self, entrypoint_name: str | None = None) -> Any:
        """Return the MATLAB entrypoint function from the active engine."""

        engine = self.ensure_engine()
        target_entrypoint = self.entrypoint if entrypoint_name is None else entrypoint_name
        try:
            return getattr(engine, target_entrypoint)
        except AttributeError as exc:
            raise RuntimeError(
                f"MATLAB runtime path {self.runtime_dir} does not expose entrypoint '{target_entrypoint}'"
            ) from exc

    def consume_last_engine_acquire(self) -> dict[str, Any]:
        """Return and clear bootstrap/reuse details for the most recent acquire."""

        state = dict(self._last_engine_acquire or {})
        self._last_engine_acquire = None
        return state

    def validate_runtime_files(
        self,
        validator: Callable[[], Sequence[Mapping[str, Any]]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Validate provider runtime files once per session and cache records."""

        if self._validated_runtime_files is not None:
            reuse_details = record_matlab_session_runtime_file_validation(
                self.session_lifecycle,
                validation_ms=0.0,
                runtime_file_count=len(self._validated_runtime_files),
                cache_reused=True,
            )
            return [dict(item) for item in self._validated_runtime_files], reuse_details

        validation_started = time.perf_counter()
        records = validator()
        validation_ms = _elapsed_ms(validation_started)
        normalized_records = [dict(item) for item in records]
        self._validated_runtime_files = normalized_records
        validation_details = record_matlab_session_runtime_file_validation(
            self.session_lifecycle,
            validation_ms=validation_ms,
            runtime_file_count=len(normalized_records),
            cache_reused=False,
        )
        return [dict(item) for item in normalized_records], validation_details

    def peek_runtime_file_records(self) -> list[dict[str, Any]] | None:
        """Return cached runtime file records without triggering validation."""

        if self._validated_runtime_files is None:
            return None
        return [dict(item) for item in self._validated_runtime_files]

    def summarize_session_lifecycle(self) -> dict[str, Any] | None:
        """Summarize this capsule's lifecycle in the shared reporting schema."""

        return summarize_matlab_session_snapshots(
            [snapshot_matlab_session_lifecycle(self.session_lifecycle)]
        )


def create_matlab_session_lifecycle(
    *,
    consumer: str,
    runtime_dir: Path | None,
    entrypoint: str | None,
) -> dict[str, Any]:
    """Create a mutable lifecycle record for one MATLAB session capsule."""

    return {
        "schema_version": "1.0",
        "consumer": consumer,
        "session_id": uuid.uuid4().hex,
        "runtime_path": None if runtime_dir is None else str(runtime_dir),
        "entrypoint": entrypoint,
        "session_started_at": None,
        "session_last_used_at": None,
        "session_last_teardown_at": None,
        "engine_bootstrap_count": 0,
        "engine_reuse_count": 0,
        "runtime_file_validation_count": 0,
        "runtime_file_validation_reuse_count": 0,
        "addpath_call_count": 0,
        "teardown_count": 0,
        "teardown_warning_count": 0,
        "aggregate_timing_ms": _zeroed_metric_map(MATLAB_SESSION_TIMING_KEYS),
        "last_bootstrap": None,
        "last_reuse": None,
        "last_runtime_file_validation": None,
        "last_teardown": None,
    }


def snapshot_matlab_session_lifecycle(session: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe copy of a mutable MATLAB session lifecycle record."""

    return {
        "schema_version": session.get("schema_version"),
        "consumer": session.get("consumer"),
        "session_id": session.get("session_id"),
        "runtime_path": session.get("runtime_path"),
        "entrypoint": session.get("entrypoint"),
        "session_started_at": session.get("session_started_at"),
        "session_last_used_at": session.get("session_last_used_at"),
        "session_last_teardown_at": session.get("session_last_teardown_at"),
        "engine_bootstrap_count": session.get("engine_bootstrap_count", 0),
        "engine_reuse_count": session.get("engine_reuse_count", 0),
        "runtime_file_validation_count": session.get("runtime_file_validation_count", 0),
        "runtime_file_validation_reuse_count": session.get("runtime_file_validation_reuse_count", 0),
        "addpath_call_count": session.get("addpath_call_count", 0),
        "teardown_count": session.get("teardown_count", 0),
        "teardown_warning_count": session.get("teardown_warning_count", 0),
        "aggregate_timing_ms": _session_timing_totals(session),
        "last_bootstrap": session.get("last_bootstrap"),
        "last_reuse": session.get("last_reuse"),
        "last_runtime_file_validation": session.get("last_runtime_file_validation"),
        "last_teardown": session.get("last_teardown"),
        **(
            {"shared_session": dict(session["shared_session"])}
            if isinstance(session.get("shared_session"), Mapping)
            else {}
        ),
    }


def record_matlab_session_bootstrap(
    session: dict[str, Any],
    *,
    factory_metrics: Mapping[str, Any] | None,
    start_matlab_ms: float,
    addpath_ms: float,
) -> dict[str, Any]:
    """Record engine start/addpath timings on a session lifecycle record."""

    measured_at = _iso_utc_now()
    bootstrap_details = {
        "measured_at": measured_at,
        "configure_environment_ms": float((factory_metrics or {}).get("configure_environment_ms", 0.0) or 0.0),
        "engine_module_import_ms": float((factory_metrics or {}).get("engine_module_import_ms", 0.0) or 0.0),
        "factory_resolution_ms": float((factory_metrics or {}).get("factory_resolution_ms", 0.0) or 0.0),
        "start_matlab_ms": float(start_matlab_ms),
        "addpath_ms": float(addpath_ms),
        "engine_bootstrap_ms": round(
            float((factory_metrics or {}).get("configure_environment_ms", 0.0) or 0.0)
            + float((factory_metrics or {}).get("engine_module_import_ms", 0.0) or 0.0)
            + float((factory_metrics or {}).get("factory_resolution_ms", 0.0) or 0.0)
            + float(start_matlab_ms)
            + float(addpath_ms),
            3,
        ),
        "configured_environment": None if factory_metrics is None else dict(factory_metrics.get("configured_environment", {})),
    }
    session["engine_bootstrap_count"] = int(session.get("engine_bootstrap_count", 0)) + 1
    session["addpath_call_count"] = int(session.get("addpath_call_count", 0)) + 1
    if session.get("session_started_at") is None:
        session["session_started_at"] = measured_at
    session["session_last_used_at"] = measured_at
    timing_totals = _session_timing_totals(session)
    _accumulate_metric_map(timing_totals, bootstrap_details, mode="sum")
    session["aggregate_timing_ms"] = timing_totals
    session["last_bootstrap"] = bootstrap_details
    return bootstrap_details


def record_matlab_session_reuse(session: dict[str, Any]) -> dict[str, Any]:
    """Record that an existing MATLAB Engine session served another call."""

    measured_at = _iso_utc_now()
    reuse_details = {
        "measured_at": measured_at,
        "engine_reused": True,
    }
    session["engine_reuse_count"] = int(session.get("engine_reuse_count", 0)) + 1
    session["session_last_used_at"] = measured_at
    session["last_reuse"] = reuse_details
    return reuse_details


def record_matlab_session_runtime_file_validation(
    session: dict[str, Any],
    *,
    validation_ms: float,
    runtime_file_count: int,
    cache_reused: bool,
) -> dict[str, Any]:
    """Record runtime-file validation or cache reuse for a MATLAB session."""

    measured_at = _iso_utc_now()
    validation_details = {
        "measured_at": measured_at,
        "runtime_file_count": int(runtime_file_count),
        "validation_ms": float(validation_ms),
        "cache_reused": bool(cache_reused),
        "validation_scope": "stage_local_matlab_session",
    }
    if cache_reused:
        session["runtime_file_validation_reuse_count"] = int(
            session.get("runtime_file_validation_reuse_count", 0)
        ) + 1
    else:
        session["runtime_file_validation_count"] = int(
            session.get("runtime_file_validation_count", 0)
        ) + 1
        timing_totals = _session_timing_totals(session)
        timing_totals["runtime_file_validation_ms"] = round(
            timing_totals["runtime_file_validation_ms"] + float(validation_ms),
            3,
        )
        session["aggregate_timing_ms"] = timing_totals
    session["last_runtime_file_validation"] = validation_details
    return validation_details


def record_matlab_session_teardown(
    session: dict[str, Any],
    *,
    teardown_ms: float,
    warning_message: str | None,
) -> dict[str, Any]:
    """Record MATLAB Engine teardown timing and any teardown warning."""

    measured_at = _iso_utc_now()
    teardown_details = {
        "measured_at": measured_at,
        "teardown_ms": float(teardown_ms),
        "warning_message": warning_message,
    }
    session["teardown_count"] = int(session.get("teardown_count", 0)) + 1
    if warning_message:
        session["teardown_warning_count"] = int(session.get("teardown_warning_count", 0)) + 1
    session["session_last_teardown_at"] = measured_at
    timing_totals = _session_timing_totals(session)
    timing_totals["teardown_ms"] = round(timing_totals["teardown_ms"] + float(teardown_ms), 3)
    session["aggregate_timing_ms"] = timing_totals
    session["last_teardown"] = teardown_details
    return teardown_details


def create_matlab_boundary_trace(
    *,
    stage_name: str,
    runtime_dir: Path,
    entrypoint: str,
    session: Mapping[str, Any],
    call_scope: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Start a per-call MATLAB boundary trace for a provider stage."""

    return {
        "schema_version": "1.0",
        "stage_name": stage_name,
        "runtime_path": str(runtime_dir),
        "entrypoint": entrypoint,
        "call_scope": {} if call_scope is None else dict(call_scope),
        "started_at": _iso_utc_now(),
        "finished_at": None,
        "total_duration_ms": 0.0,
        "engine_reused_this_call": False,
        "session_lifecycle_before": snapshot_matlab_session_lifecycle(session),
        "session_lifecycle_after": None,
        "phase_timings_ms": {},
        "phase_details": {},
        "seam_costs_ms": {key: 0.0 for key in MATLAB_BOUNDARY_SEAM_COST_KEYS},
        "_perf_started": time.perf_counter(),
    }


def record_matlab_boundary_phase(
    trace: dict[str, Any],
    *,
    phase_name: str,
    duration_ms: float,
    seam_cost_key: str | None = None,
    details: Mapping[str, Any] | None = None,
) -> None:
    """Add one measured phase to a MATLAB boundary trace."""

    normalized_duration = round(float(duration_ms), 3)
    trace.setdefault("phase_timings_ms", {})[phase_name] = normalized_duration
    if details is not None:
        trace.setdefault("phase_details", {})[phase_name] = dict(details)
    if seam_cost_key is not None:
        seam_costs = trace.setdefault("seam_costs_ms", {})
        seam_costs[seam_cost_key] = round(float(seam_costs.get(seam_cost_key, 0.0)) + normalized_duration, 3)


def finalize_matlab_boundary_trace(
    trace: dict[str, Any],
    *,
    session: Mapping[str, Any],
    engine_reused_this_call: bool,
) -> dict[str, Any]:
    """Finalize a boundary trace with total duration and session-after snapshot."""

    perf_started = trace.pop("_perf_started", None)
    total_duration_ms = _elapsed_ms(perf_started) if isinstance(perf_started, (int, float)) else 0.0
    trace["finished_at"] = _iso_utc_now()
    trace["total_duration_ms"] = total_duration_ms
    trace["engine_reused_this_call"] = bool(engine_reused_this_call)
    trace["session_lifecycle_after"] = snapshot_matlab_session_lifecycle(session)
    return trace


def summarize_matlab_boundary_traces(traces: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate per-call MATLAB boundary traces for one stage or FOV."""

    valid_traces = [trace for trace in traces if isinstance(trace, Mapping)]
    aggregate_seam_costs = {key: 0.0 for key in MATLAB_BOUNDARY_SEAM_COST_KEYS}
    max_seam_costs = {key: 0.0 for key in MATLAB_BOUNDARY_SEAM_COST_KEYS}
    total_duration_ms = 0.0
    engine_reused_calls = 0
    stage_counts: dict[str, int] = {}

    for trace in valid_traces:
        total_duration_ms += float(trace.get("total_duration_ms", 0.0) or 0.0)
        if trace.get("engine_reused_this_call") is True:
            engine_reused_calls += 1
        stage_name = trace.get("stage_name")
        if isinstance(stage_name, str) and stage_name:
            stage_counts[stage_name] = stage_counts.get(stage_name, 0) + 1

        seam_costs = trace.get("seam_costs_ms")
        if not isinstance(seam_costs, Mapping):
            continue
        for key in MATLAB_BOUNDARY_SEAM_COST_KEYS:
            value = seam_costs.get(key)
            if not isinstance(value, (int, float)):
                continue
            aggregate_seam_costs[key] = round(aggregate_seam_costs[key] + float(value), 3)
            max_seam_costs[key] = round(max(max_seam_costs[key], float(value)), 3)

    call_count = len(valid_traces)
    average_seam_costs = {
        key: round((aggregate_seam_costs[key] / call_count), 3) if call_count else 0.0
        for key in MATLAB_BOUNDARY_SEAM_COST_KEYS
    }
    summary = {
        "schema_version": "1.0",
        "call_count": call_count,
        "engine_reused_calls": engine_reused_calls,
        "stage_counts": stage_counts,
        "aggregate_seam_costs_ms": aggregate_seam_costs,
        "average_seam_costs_ms": average_seam_costs,
        "max_seam_costs_ms": max_seam_costs,
        "total_duration_ms": round(total_duration_ms, 3),
    }
    session_lifecycle_summary = summarize_matlab_session_lifecycle(valid_traces)
    if session_lifecycle_summary is not None:
        summary["session_lifecycle_summary"] = session_lifecycle_summary
    return summary


def _boundary_summary_extra_seam_costs(summary: Mapping[str, Any]) -> dict[str, float]:
    extras = _zeroed_metric_map(MATLAB_BOUNDARY_SEAM_COST_KEYS)
    canonical_persistence_ms = summary.get("fov_canonical_persistence_ms")
    if isinstance(canonical_persistence_ms, (int, float)):
        extras["canonical_persistence_ms"] = round(float(canonical_persistence_ms), 3)
    return extras


def merge_matlab_boundary_summaries(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Merge boundary summaries from multiple stages while preserving seam costs."""

    valid_summaries = [summary for summary in summaries if isinstance(summary, Mapping)]
    if not valid_summaries:
        return None

    aggregate_seam_costs = {key: 0.0 for key in MATLAB_BOUNDARY_SEAM_COST_KEYS}
    max_seam_costs = {key: 0.0 for key in MATLAB_BOUNDARY_SEAM_COST_KEYS}
    stage_counts: dict[str, int] = {}
    total_duration_ms = 0.0
    call_count = 0
    engine_reused_calls = 0

    for summary in valid_summaries:
        total_duration_ms += _metric_as_float(summary.get("total_duration_ms"))
        call_count += _count_as_int(summary.get("call_count"))
        engine_reused_calls += _count_as_int(summary.get("engine_reused_calls"))

        raw_stage_counts = summary.get("stage_counts")
        if isinstance(raw_stage_counts, Mapping):
            for key, value in raw_stage_counts.items():
                if not isinstance(key, str) or not key:
                    continue
                stage_counts[key] = stage_counts.get(key, 0) + _count_as_int(value)

        seam_costs = summary.get("aggregate_seam_costs_ms")
        if isinstance(seam_costs, Mapping):
            _accumulate_metric_map(aggregate_seam_costs, seam_costs, mode="sum")
        max_costs = summary.get("max_seam_costs_ms")
        if isinstance(max_costs, Mapping):
            _accumulate_metric_map(max_seam_costs, max_costs, mode="max")
        extra_costs = _boundary_summary_extra_seam_costs(summary)
        _accumulate_metric_map(aggregate_seam_costs, extra_costs, mode="sum")
        _accumulate_metric_map(max_seam_costs, extra_costs, mode="max")

    merged_summary = {
        "schema_version": "1.0",
        "call_count": call_count,
        "engine_reused_calls": engine_reused_calls,
        "stage_counts": stage_counts,
        "aggregate_seam_costs_ms": aggregate_seam_costs,
        "average_seam_costs_ms": {
            key: round((aggregate_seam_costs[key] / call_count), 3) if call_count else 0.0
            for key in MATLAB_BOUNDARY_SEAM_COST_KEYS
        },
        "max_seam_costs_ms": max_seam_costs,
        "total_duration_ms": round(total_duration_ms, 3),
    }

    session_summaries = [
        session_summary
        for summary in valid_summaries
        for session_summary in [summary.get("session_lifecycle_summary")]
        if isinstance(session_summary, Mapping)
    ]
    merged_session_summary = merge_matlab_session_lifecycle_summaries(session_summaries)
    if merged_session_summary is not None:
        merged_summary["session_lifecycle_summary"] = merged_session_summary
    return merged_summary
