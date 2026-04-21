from __future__ import annotations

import importlib
import os
import platform
import shutil
import sys
import warnings
from pathlib import Path
from typing import Any


MATLAB_ROOT_ENV_KEYS = (
    "PYSTAR_MATLAB_ROOT",
    "MATLAB_ROOT",
    "MATLAB_HOME",
    "MATLABHOME",
    "MWE_INSTALL",
)


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


def load_matlab_engine_module(*, consumer: str) -> Any:
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
