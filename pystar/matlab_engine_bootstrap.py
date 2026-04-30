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
import os
import platform
import shutil
import sys
import time
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


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
    "runtime_file_validation_ms",
    "start_matlab_ms",
    "addpath_ms",
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

    return {
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
    ) -> None:
        self.consumer = consumer
        self.runtime_dir = runtime_dir
        self.entrypoint = entrypoint
        self.engine_factory = engine_factory
        self.engine_factory_consumer = engine_factory_consumer or consumer
        self.startup_failure_prefix = startup_failure_prefix
        self.addpath_failure_prefix = addpath_failure_prefix
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
