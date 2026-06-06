#!/usr/bin/env python
"""Stage30a validation helper for native preprocessing volume workers.

This script is validation-only infrastructure.  It reuses the existing native
preprocessing calibration profile harness, runs or loads isolated worker-count
profiles, compares canonical clean TIFF outputs exactly, and writes a JSON plus
Markdown evidence report.  It intentionally does not change production
preprocessing algorithms or persisted scientific artifact contracts.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import traceback
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _prefer_source_root_for_pystar(source_root: Path) -> None:
    """Ensure validation-worktree sitecustomize cannot pin an older pystar package."""

    source_root_text = str(source_root)
    sys.path[:] = [entry for entry in sys.path if entry != source_root_text]
    sys.path.insert(0, source_root_text)

    pystar_module = sys.modules.get("pystar")
    module_file = getattr(pystar_module, "__file__", None)
    if module_file is None:
        return

    try:
        _ = Path(str(module_file)).resolve().relative_to(source_root.resolve())
    except ValueError:
        for module_name in list(sys.modules):
            if module_name == "pystar" or module_name.startswith("pystar."):
                del sys.modules[module_name]


_prefer_source_root_for_pystar(REPO_ROOT)

from pystar.infrastructure import ExperimentConfig, load_config
from pystar.serialization import write_backend_metadata
from scripts.profile_native_preprocessing_calibration import (
    CALIBRATION_PROFILE_SCHEMA_NAME,
    CALIBRATION_PROFILE_SCHEMA_VERSION,
    HISTOGRAM_REAL_MATCH_ATTRIBUTION_SCHEMA_NAME,
    HISTOGRAM_REAL_MATCH_ATTRIBUTION_SCHEMA_VERSION,
    _assert_path_within,
    _reject_symlink_components,
    profile_native_preprocessing_calibration,
)


STAGE30A_SWEEP_SCHEMA_NAME = "pystar_stage30a_native_volume_worker_sweep_validation"
STAGE30A_SWEEP_SCHEMA_VERSION = 1
STAGE30A_REPORT_FILENAMES = {
    "json": "stage30a_native_volume_worker_sweep_validation.json",
    "markdown": "stage30a_native_volume_worker_sweep_validation.md",
}
STAGE30A_OUTPUT_MARKER = ".pystar_stage30a_native_volume_worker_sweep_validation"
STAGE32_POLICY_SCHEMA_NAME = "pystar_stage32_native_volume_worker_policy_validation"
STAGE32_POLICY_SCHEMA_VERSION = 1
STAGE32_REPORT_FILENAMES = {
    "json": "stage32_native_volume_worker_policy_validation.json",
    "markdown": "stage32_native_volume_worker_policy_validation.md",
}
STAGE32_OUTPUT_MARKER = ".pystar_stage32_native_volume_worker_policy_validation"
STAGE33_FOV1_POLICY_SCHEMA_NAME = "pystar_stage33_fov1_native_volume_worker_policy_validation"
STAGE33_FOV1_POLICY_SCHEMA_VERSION = 1
STAGE33_FOV1_REPORT_FILENAMES = {
    "json": "stage33_fov1_native_volume_worker_policy_validation.json",
    "markdown": "stage33_fov1_native_volume_worker_policy_validation.md",
}
STAGE33_FOV1_OUTPUT_MARKER = ".pystar_stage33_fov1_native_volume_worker_policy_validation"
STAGE33_FOV1_POLICY_FOV_IDS = (1,)
STAGE32_POLICY_BASELINE_WORKER = 1
DEFAULT_STAGE32_POLICY_CANDIDATE_WORKERS = (4,)
STAGE32_SURFACE_SCOPES = ("full", "limited")
REDACTED_CONFIG_VALUE = "<redacted>"
SECRET_CONFIG_KEY_TERMS = (
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "access_key",
    "private_key",
    "credential",
    "auth",
    "bearer",
    "dsn",
    "url",
)
DEFAULT_BASELINE_COMMIT = "fec32e7c755de8510afe0ad603b2ecc071cf452b"
DEFAULT_EXPECTED_WORKERS = (1, 2, 3, 4)
CANONICAL_CLEAN_CONTRACT = (
    "Position{fov_id}/output_pystar/clean_data/"
    "clean_fov_{fov_id}_round_{round_id}_ch_{channel_id}.tif"
)
CANONICAL_OUTPUT_DIRS = (
    "output_pystar",
    "transforms",
    "spots",
    "extraction",
    "decoded",
    "qc_reports",
    "clean_data",
)


def _sweep_contract(
    stage32_policy: bool,
    *,
    stage33_fov1_policy: bool = False,
) -> dict[str, object]:
    if stage32_policy and stage33_fov1_policy:
        raise ValueError("Stage32 and Stage33 FOV1 policy modes are mutually exclusive")
    if stage33_fov1_policy:
        return {
            "schema_name": STAGE33_FOV1_POLICY_SCHEMA_NAME,
            "schema_version": STAGE33_FOV1_POLICY_SCHEMA_VERSION,
            "report_filenames": STAGE33_FOV1_REPORT_FILENAMES,
            "output_marker": STAGE33_FOV1_OUTPUT_MARKER,
            "label": "Stage33 FOV1",
            "description": "FOV1-only native volume worker policy validation",
        }
    if stage32_policy:
        return {
            "schema_name": STAGE32_POLICY_SCHEMA_NAME,
            "schema_version": STAGE32_POLICY_SCHEMA_VERSION,
            "report_filenames": STAGE32_REPORT_FILENAMES,
            "output_marker": STAGE32_OUTPUT_MARKER,
            "label": "Stage32",
            "description": "native volume worker policy validation",
        }
    return {
        "schema_name": STAGE30A_SWEEP_SCHEMA_NAME,
        "schema_version": STAGE30A_SWEEP_SCHEMA_VERSION,
        "report_filenames": STAGE30A_REPORT_FILENAMES,
        "output_marker": STAGE30A_OUTPUT_MARKER,
        "label": "Stage30a",
        "description": "native volume worker sweep validation",
    }


def _sweep_contract_for_payload(payload: Mapping[str, object]) -> dict[str, object]:
    schema_name = payload.get("schema_name")
    return _sweep_contract(
        schema_name == STAGE32_POLICY_SCHEMA_NAME,
        stage33_fov1_policy=schema_name == STAGE33_FOV1_POLICY_SCHEMA_NAME,
    )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _read_json_mapping(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON profile at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object profile at {path}; got {type(value).__name__}")
    return cast(dict[str, object], value)


def _read_yaml_mapping(path: Path) -> dict[str, object]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Malformed YAML config at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping config at {path}; got {type(value).__name__}")
    return cast(dict[str, object], value)


def _is_secret_config_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return any(term in normalized for term in SECRET_CONFIG_KEY_TERMS)


def _redact_config_for_report(value: object, *, key: str = "") -> object:
    if key and _is_secret_config_key(key):
        return REDACTED_CONFIG_VALUE
    if isinstance(value, Mapping):
        return {
            str(child_key): _redact_config_for_report(child_value, key=str(child_key))
            for child_key, child_value in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_redact_config_for_report(item) for item in value]
    return value


def _current_git_value(repo_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            env={**os.environ, "GIT_MASTER": os.environ.get("GIT_MASTER", "1")},
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value or None


def _git_dirty(repo_root: Path) -> bool | None:
    status = _current_git_value(repo_root, "status", "--porcelain")
    if status is None:
        return None
    return bool(status.strip())


def _parse_positive_int_list(raw_value: str | None, *, field_name: str) -> tuple[int, ...]:
    if raw_value is None:
        raise ValueError(f"{field_name} must be provided")
    values: list[int] = []
    seen: set[int] = set()
    for raw_token in raw_value.split(","):
        token = raw_token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(f"Invalid {field_name} value {token!r}; expected comma-separated integers") from exc
        if value <= 0:
            raise ValueError(f"Invalid {field_name} value {value!r}; expected positive integers")
        if value not in seen:
            values.append(value)
            seen.add(value)
    if not values:
        raise ValueError(f"{field_name} must include at least one integer")
    return tuple(values)


def _parse_nonnegative_int_list(raw_value: str | None, *, field_name: str) -> tuple[int, ...] | None:
    if raw_value is None:
        return None
    values: list[int] = []
    seen: set[int] = set()
    for raw_token in raw_value.split(","):
        token = raw_token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(f"Invalid {field_name} value {token!r}; expected comma-separated integers") from exc
        if value < 0:
            raise ValueError(f"Invalid {field_name} value {value!r}; expected non-negative integers")
        if value not in seen:
            values.append(value)
            seen.add(value)
    if not values:
        raise ValueError(f"{field_name} must include at least one integer when provided")
    return tuple(values)


def _profile_label_for_worker(worker_count: int) -> str:
    return f"workers_{worker_count}"


def _parse_existing_profile_specs(raw_values: Sequence[str]) -> dict[str, Path]:
    profiles: dict[str, Path] = {}
    for raw_value in raw_values:
        if "=" not in raw_value:
            raise ValueError(
                "--existing-profile expects LABEL=PATH, for example "
                "--existing-profile 2=/path/native_preprocessing_calibration_profile.json"
            )
        raw_label, raw_path = raw_value.split("=", 1)
        label = raw_label.strip()
        if not label:
            raise ValueError(f"Existing profile label must be non-empty: {raw_value!r}")
        if label.isdigit():
            label = _profile_label_for_worker(int(label))
        if label in profiles:
            raise ValueError(f"Duplicate existing profile label: {label}")
        path = Path(raw_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Existing profile JSON does not exist: {path}")
        profiles[label] = path
    return profiles


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        _ = path.relative_to(parent)
    except ValueError:
        return False
    return True


def _paths_overlap(first: Path, second: Path) -> bool:
    first_resolved = first.expanduser().resolve(strict=False)
    second_resolved = second.expanduser().resolve(strict=False)
    return (
        first_resolved == second_resolved
        or _is_relative_to(first_resolved, second_resolved)
        or _is_relative_to(second_resolved, first_resolved)
    )


def _reject_sweep_production_overlap(
    *,
    output_dir: Path,
    production_root_base: Path,
    base_production_output_dir: Path,
) -> None:
    _reject_symlink_components(production_root_base, field_name="--production-root-base")
    base_production_resolved = base_production_output_dir.expanduser().resolve(strict=False)
    if _paths_overlap(output_dir, production_root_base):
        raise ValueError(
            "--production-root-base must be isolated from the Stage30a/Stage32/Stage33 evidence "
            f"--output-dir: {production_root_base} vs {output_dir}"
        )
    if _paths_overlap(output_dir, base_production_resolved):
        raise ValueError(
            "--output-dir must be a dedicated Stage30a/Stage32/Stage33 evidence directory that does not "
            "overlap the base config pipeline.output.directory: "
            f"{output_dir} vs {base_production_resolved}"
        )
    if _paths_overlap(production_root_base, base_production_resolved):
        raise ValueError(
            "--production-root-base must point to isolated validation run outputs and must not "
            "overlap the base config pipeline.output.directory: "
            f"{production_root_base} vs {base_production_resolved}"
        )


def _prepare_sweep_output_dir(
    output_dir: Path,
    *,
    stage32_policy: bool = False,
    stage33_fov1_policy: bool = False,
) -> Path:
    contract = _sweep_contract(stage32_policy, stage33_fov1_policy=stage33_fov1_policy)
    marker_name = str(contract["output_marker"])
    label = str(contract["label"])
    output_path = output_dir.expanduser()
    _reject_symlink_components(output_path, field_name="--output-dir")
    if output_path.exists() and not output_path.is_dir():
        raise ValueError(f"--output-dir must be a directory: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(output_path, field_name="--output-dir")
    output_resolved = output_path.resolve(strict=True)
    unsafe_roots = {Path("/").resolve(), REPO_ROOT.resolve(), Path.home().resolve()}
    if output_resolved in unsafe_roots:
        raise ValueError(
            f"--output-dir must be a dedicated {label} evidence directory, "
            f"not an unsafe root: {output_path}"
        )

    marker_path = output_path / marker_name
    if marker_path.exists():
        if marker_path.is_symlink() or not marker_path.is_file():
            raise ValueError(f"{label} output marker is invalid: {marker_path}")
        marker_text = marker_path.read_text(encoding="utf-8")
        expected = _sweep_output_marker_text(
            schema_name=str(contract["schema_name"]),
            schema_version=int(cast(int, contract["schema_version"])),
        )
        if marker_text != expected:
            raise ValueError(f"{label} output marker schema drifted: {marker_path}")
    else:
        entries = list(output_path.iterdir())
        if entries:
            raise ValueError(
                f"--output-dir exists and is not an empty {label} sweep output directory. "
                f"Choose an empty/dedicated directory or one containing {marker_name}: {output_path}"
            )
        _ = marker_path.write_text(
            _sweep_output_marker_text(
                schema_name=str(contract["schema_name"]),
                schema_version=int(cast(int, contract["schema_version"])),
            ),
            encoding="utf-8",
        )
    return output_resolved


def _sweep_output_marker_text(*, schema_name: str, schema_version: int) -> str:
    return f"schema_name={schema_name}\nschema_version={schema_version}\n"


def _worker_config_payload(
    *,
    base_config: Mapping[str, object],
    worker_count: int,
    production_output_root: Path,
) -> dict[str, object]:
    payload = dict(base_config)
    pipeline = payload.get("pipeline")
    if not isinstance(pipeline, dict):
        raise ValueError("Config missing required 'pipeline' mapping")
    pipeline_payload = dict(cast(dict[str, object], pipeline))

    preprocessing = pipeline_payload.get("preprocessing")
    if not isinstance(preprocessing, dict):
        raise ValueError("Config missing required 'pipeline.preprocessing' mapping")
    preprocessing_payload = dict(cast(dict[str, object], preprocessing))
    preprocessing_payload["native_volume_workers"] = int(worker_count)
    pipeline_payload["preprocessing"] = preprocessing_payload

    output = pipeline_payload.get("output")
    if not isinstance(output, dict):
        raise ValueError("Config missing required 'pipeline.output' mapping")
    output_payload = dict(cast(dict[str, object], output))
    output_payload["directory"] = str(production_output_root)
    pipeline_payload["output"] = output_payload

    payload["pipeline"] = pipeline_payload
    return payload


def _write_worker_config(
    *,
    base_config: Mapping[str, object],
    worker_count: int,
    config_dir: Path,
    sweep_output_dir: Path,
    production_root_base: Path,
) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(config_dir, field_name="worker config directory")
    _assert_path_within(config_dir, sweep_output_dir, field_name="worker config directory")
    production_output_root = production_root_base / _profile_label_for_worker(worker_count)
    payload = _worker_config_payload(
        base_config=base_config,
        worker_count=worker_count,
        production_output_root=production_output_root,
    )
    config_path = config_dir / f"native_volume_workers_{worker_count}.yaml"
    _reject_symlink_components(config_path, field_name="worker config path")
    _assert_path_within(config_path, sweep_output_dir, field_name="worker config path")
    _ = config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def _tiff_shape_dtype(path: Path) -> tuple[list[int], str]:
    import tifffile

    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        return [int(dimension) for dimension in series.shape], str(series.dtype)


def _read_tiff_array(path: Path) -> np.ndarray[Any, np.dtype[Any]]:
    import tifffile

    return cast(np.ndarray[Any, np.dtype[Any]], tifffile.imread(path))


def _clean_record_for_path(*, path: Path, repeat_root: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Clean TIFF listed by profile is missing: {path}")
    try:
        relative_path = path.relative_to(repeat_root)
    except ValueError as exc:
        raise ValueError(f"Clean TIFF is outside its repeat output root: {path} vs {repeat_root}") from exc
    shape, dtype = _tiff_shape_dtype(path)
    return {
        "relative_path": relative_path.as_posix(),
        "path": str(path),
        "shape": shape,
        "dtype": dtype,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _canonical_clean_relative_path(*, fov_id: int, round_id: int, channel_id: int) -> str:
    return CANONICAL_CLEAN_CONTRACT.format(
        fov_id=int(fov_id),
        round_id=int(round_id),
        channel_id=int(channel_id),
    )


def _expected_clean_keys(
    *,
    selected_fovs: Sequence[int],
    selected_channels_by_round: Mapping[str, Sequence[int]],
) -> set[tuple[int, str]]:
    expected: set[tuple[int, str]] = set()
    for fov_id in selected_fovs:
        for round_id_text, channel_ids in selected_channels_by_round.items():
            round_id = int(round_id_text)
            for channel_id in channel_ids:
                expected.add(
                    (
                        int(fov_id),
                        _canonical_clean_relative_path(
                            fov_id=int(fov_id),
                            round_id=round_id,
                            channel_id=int(channel_id),
                        ),
                    )
                )
    return expected


def _canonical_directory_status(*, repeat_root: Path, fov_id: int) -> dict[str, object]:
    fov_root = repeat_root / f"Position{fov_id}" / "output_pystar"
    children = {
        name: (fov_root / name).is_dir()
        for name in CANONICAL_OUTPUT_DIRS
        if name != "output_pystar"
    }
    return {
        "fov_id": int(fov_id),
        "repeat_root": str(repeat_root),
        "output_pystar_exists": fov_root.is_dir(),
        "required_children": children,
        "status": "pass" if fov_root.is_dir() and all(children.values()) else "fail",
    }


def _profile_fovs(profile_payload: Mapping[str, object]) -> Sequence[Mapping[str, object]]:
    fovs = profile_payload.get("fovs")
    if not isinstance(fovs, Sequence) or isinstance(fovs, (str, bytes, bytearray)):
        raise ValueError("Profile payload missing sequence field 'fovs'")
    return cast(Sequence[Mapping[str, object]], fovs)


def _selected_repeat(profile_fov: Mapping[str, object], repeat_index: int) -> Mapping[str, object]:
    repeats = profile_fov.get("repeats")
    if not isinstance(repeats, Sequence) or isinstance(repeats, (str, bytes, bytearray)):
        raise ValueError("Profile FOV payload missing sequence field 'repeats'")
    for raw_repeat in cast(Sequence[Mapping[str, object]], repeats):
        if int(cast(int, raw_repeat.get("repeat_index"))) == repeat_index:
            return raw_repeat
    raise ValueError(f"Profile FOV {profile_fov.get('fov_id')} has no repeat_index={repeat_index}")


def _clean_records_for_profile(
    *,
    label: str,
    profile_payload: Mapping[str, object],
    compare_repeat_index: int,
) -> tuple[dict[tuple[int, str], dict[str, object]], list[dict[str, object]]]:
    records: dict[tuple[int, str], dict[str, object]] = {}
    directory_statuses: list[dict[str, object]] = []
    for fov in _profile_fovs(profile_payload):
        fov_id = int(cast(int, fov["fov_id"]))
        repeat = _selected_repeat(fov, compare_repeat_index)
        repeat_root = Path(str(repeat["repeat_output_root"]))
        directory_statuses.append(_canonical_directory_status(repeat_root=repeat_root, fov_id=fov_id))
        output_files = repeat.get("output_files")
        if not isinstance(output_files, Sequence) or isinstance(output_files, (str, bytes, bytearray)):
            raise ValueError(f"Profile {label} FOV {fov_id} repeat {compare_repeat_index} has no output_files sequence")
        for raw_path in output_files:
            record = _clean_record_for_path(path=Path(str(raw_path)), repeat_root=repeat_root)
            key = (fov_id, str(record["relative_path"]))
            if key in records:
                raise ValueError(f"Duplicate clean TIFF relative path in profile {label}: {key}")
            record["label"] = label
            record["fov_id"] = fov_id
            record["repeat_index"] = compare_repeat_index
            records[key] = record
    return records, directory_statuses


def _array_equivalence_metrics_for_records(
    reference: Mapping[str, object],
    candidate: Mapping[str, object],
) -> tuple[bool | None, int | float | None, str | None]:
    try:
        reference_array = _read_tiff_array(Path(str(reference["path"])))
        candidate_array = _read_tiff_array(Path(str(candidate["path"])))
        if reference_array.shape != candidate_array.shape:
            return False, None, "shape mismatch prevents max_abs_diff calculation"
        if reference_array.size == 0 and candidate_array.size == 0:
            max_abs_diff: int | float = 0
        else:
            difference = np.abs(reference_array.astype(np.float64) - candidate_array.astype(np.float64))
            raw_max = float(np.max(difference))
            max_abs_diff = int(raw_max) if raw_max.is_integer() else raw_max
        return bool(np.array_equal(reference_array, candidate_array)), max_abs_diff, None
    except Exception as exc:  # pragma: no cover - exercised by real-data validation failures
        return None, None, f"{type(exc).__name__}: {exc}"


def _compare_clean_record_maps(
    *,
    reference_label: str,
    candidate_label: str,
    reference_records: Mapping[tuple[int, str], Mapping[str, object]],
    candidate_records: Mapping[tuple[int, str], Mapping[str, object]],
) -> dict[str, object]:
    reference_keys = set(reference_records)
    candidate_keys = set(candidate_records)
    missing = sorted(reference_keys - candidate_keys)
    extra = sorted(candidate_keys - reference_keys)
    file_rows: list[dict[str, object]] = []
    mismatch_count = 0
    shape_drift_count = 0
    dtype_drift_count = 0
    max_abs_diff_values: list[int | float] = []

    for key in sorted(reference_keys & candidate_keys):
        reference = reference_records[key]
        candidate = candidate_records[key]
        shape_equal = reference["shape"] == candidate["shape"]
        dtype_equal = reference["dtype"] == candidate["dtype"]
        size_equal = reference["size_bytes"] == candidate["size_bytes"]
        sha_equal = reference["sha256"] == candidate["sha256"]
        array_equal, max_abs_diff, array_error = _array_equivalence_metrics_for_records(reference, candidate)
        if max_abs_diff is not None:
            max_abs_diff_values.append(max_abs_diff)
        if not shape_equal:
            shape_drift_count += 1
        if not dtype_equal:
            dtype_drift_count += 1
        equivalent = bool(shape_equal and dtype_equal and array_equal is True and max_abs_diff == 0)
        if not equivalent:
            mismatch_count += 1
        file_rows.append(
            {
                "fov_id": key[0],
                "relative_path": key[1],
                "reference_path": reference["path"],
                "candidate_path": candidate["path"],
                "shape_equal": shape_equal,
                "dtype_equal": dtype_equal,
                "size_bytes_equal": size_equal,
                "sha256_equal": sha_equal,
                "array_equal": array_equal,
                "max_abs_diff": max_abs_diff,
                "array_compare_error": array_error,
                "shape_drift": not shape_equal,
                "dtype_drift": not dtype_equal,
                "reference_shape": reference["shape"],
                "candidate_shape": candidate["shape"],
                "reference_dtype": reference["dtype"],
                "candidate_dtype": candidate["dtype"],
                "reference_size_bytes": reference["size_bytes"],
                "candidate_size_bytes": candidate["size_bytes"],
                "reference_sha256": reference["sha256"],
                "candidate_sha256": candidate["sha256"],
                "status": "equivalent" if equivalent else "mismatch",
            }
        )

    missing_count = len(missing)
    extra_count = len(extra)
    max_abs_diff_overall: int | float | None = max(max_abs_diff_values) if max_abs_diff_values else None
    exact_equivalence = (
        missing_count == 0
        and extra_count == 0
        and mismatch_count == 0
        and shape_drift_count == 0
        and dtype_drift_count == 0
        and max_abs_diff_overall == 0
    )
    status = "equivalent" if exact_equivalence else "mismatch"
    return {
        "reference_label": reference_label,
        "candidate_label": candidate_label,
        "status": status,
        "exact_equivalence": exact_equivalence,
        "files_compared": len(reference_keys & candidate_keys),
        "missing_count": missing_count,
        "extra_count": extra_count,
        "missing_files": [f"FOV {fov_id}: {relative_path}" for fov_id, relative_path in missing],
        "extra_files": [f"FOV {fov_id}: {relative_path}" for fov_id, relative_path in extra],
        "mismatch_count": mismatch_count,
        "shape_drift_count": shape_drift_count,
        "dtype_drift_count": dtype_drift_count,
        "max_abs_diff": max_abs_diff_overall,
        "file_rows": file_rows,
    }


def _profile_timing_rows(*, label: str, worker_count: int | None, profile_payload: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for fov in _profile_fovs(profile_payload):
        summary = cast(Mapping[str, object], fov["summary"])
        elapsed = cast(Mapping[str, object], summary["run_elapsed_wall_ms"])
        by_phase = cast(Mapping[str, object], summary["by_phase"])
        volume_total = cast(Mapping[str, object], by_phase.get("volume_total", {}))
        calibration = cast(Mapping[str, object], by_phase.get("calibration_steps", {}))
        extraction = cast(Mapping[str, object], by_phase.get("extraction_steps", {}))
        rows.append(
            {
                "label": label,
                "worker_count": worker_count,
                "fov_id": int(cast(int, fov["fov_id"])),
                "repeat_count": elapsed.get("count"),
                "wall_total_ms": elapsed.get("total_duration_ms"),
                "wall_median_ms": elapsed.get("median_duration_ms"),
                "wall_mean_ms": elapsed.get("mean_duration_ms"),
                "volume_total_ms": volume_total.get("total_duration_ms"),
                "calibration_total_ms": calibration.get("total_duration_ms"),
                "extraction_total_ms": extraction.get("total_duration_ms"),
            }
        )
    return rows


def _profile_histogram_rows(*, label: str, worker_count: int | None, profile_payload: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for fov in _profile_fovs(profile_payload):
        summary = cast(Mapping[str, object], fov["summary"])
        histogram = cast(Mapping[str, object], summary["histogram_match_profile"])
        attribution = cast(Mapping[str, object], histogram["real_match_attribution"])
        real_duration = cast(Mapping[str, object], attribution["real_match_duration_ms"])
        noop_duration = cast(Mapping[str, object], attribution["no_reference_noop_duration_ms"])
        rows.append(
            {
                "label": label,
                "worker_count": worker_count,
                "fov_id": int(cast(int, fov["fov_id"])),
                "schema_name": attribution.get("schema_name"),
                "schema_version": attribution.get("schema_version"),
                "call_count": attribution.get("call_count"),
                "real_match_call_count": attribution.get("real_match_call_count"),
                "no_reference_noop_call_count": attribution.get("no_reference_noop_call_count"),
                "real_match_total_duration_ms": real_duration.get("total_duration_ms"),
                "real_match_median_duration_ms": real_duration.get("median_duration_ms"),
                "no_reference_noop_total_duration_ms": noop_duration.get("total_duration_ms"),
                "no_reference_noop_median_duration_ms": noop_duration.get("median_duration_ms"),
                "by_scope": attribution.get("by_scope"),
            }
        )
    return rows


def _profile_internal_equivalence(label: str, profile_payload: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for fov in _profile_fovs(profile_payload):
        equivalence = cast(Mapping[str, object], fov["clean_output_equivalence"])
        rows.append(
            {
                "label": label,
                "fov_id": int(cast(int, fov["fov_id"])),
                "status": equivalence.get("status"),
                "file_count": equivalence.get("file_count"),
                "mismatch_count": len(cast(Sequence[object], equivalence.get("mismatches", []))),
            }
        )
    return rows


def _profile_config_source(profile_payload: Mapping[str, object]) -> Mapping[str, object]:
    source = profile_payload.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("Profile payload missing mapping field 'source'")
    return cast(Mapping[str, object], source)


def _profile_output_dir(profile_payload: Mapping[str, object]) -> str | None:
    profile_configuration = profile_payload.get("profile_configuration")
    if not isinstance(profile_configuration, Mapping):
        return None
    output_dir = profile_configuration.get("output_dir")
    return None if output_dir is None else str(output_dir)


def _int_list_from_profile(value: object, *, field_name: str, allow_none: bool = False) -> list[int] | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"Profile field {field_name!r} must be a sequence of integers")
    values: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"Profile field {field_name!r} must contain integers; got {item!r}")
        values.append(int(item))
    return values


def _require_profile_mapping(payload: Mapping[str, object], key: str, *, label: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Profile {label} missing mapping field {key!r}")
    return cast(Mapping[str, object], value)


def _normalize_worker_config_delta(
    payload: Mapping[str, object],
    *,
    label: str,
) -> tuple[dict[str, object], object, object]:
    normalized = copy.deepcopy(dict(payload))
    pipeline = normalized.get("pipeline")
    if not isinstance(pipeline, dict):
        raise ValueError(f"Profile {label} source config missing 'pipeline' mapping")
    preprocessing = pipeline.get("preprocessing")
    if not isinstance(preprocessing, dict):
        raise ValueError(f"Profile {label} source config missing 'pipeline.preprocessing' mapping")
    output = pipeline.get("output")
    if not isinstance(output, dict):
        raise ValueError(f"Profile {label} source config missing 'pipeline.output' mapping")

    worker_count = preprocessing.pop("native_volume_workers", None)
    output_directory = output.pop("directory", None)
    return normalized, worker_count, output_directory


def _validate_worker_config_provenance(
    *,
    label: str,
    profile_config_payload: Mapping[str, object],
    base_config_payload: Mapping[str, object],
    expected_worker_count: int | None,
) -> dict[str, object]:
    normalized_profile, raw_worker_count, raw_output_directory = _normalize_worker_config_delta(
        profile_config_payload,
        label=label,
    )
    normalized_base, _, _ = _normalize_worker_config_delta(base_config_payload, label="base")
    if normalized_profile != normalized_base:
        raise ValueError(
            f"Profile {label} source config differs from the base config by more than "
            "pipeline.preprocessing.native_volume_workers and pipeline.output.directory"
        )
    if expected_worker_count is not None:
        if raw_worker_count is None and int(expected_worker_count) == STAGE32_POLICY_BASELINE_WORKER:
            pass
        elif isinstance(raw_worker_count, bool) or not isinstance(raw_worker_count, int):
            raise ValueError(
                f"Profile {label} source config must explicitly set integer "
                f"pipeline.preprocessing.native_volume_workers; got {raw_worker_count!r}"
            )
        elif int(raw_worker_count) != int(expected_worker_count):
            raise ValueError(
                f"Profile {label} source config worker count {raw_worker_count!r} "
                f"does not match expected {expected_worker_count}"
            )
    elif raw_worker_count is not None and (isinstance(raw_worker_count, bool) or not isinstance(raw_worker_count, int)):
        raise ValueError(
            f"Profile {label} source config worker count must be integer when present; got {raw_worker_count!r}"
        )
    if not isinstance(raw_output_directory, str) or not raw_output_directory:
        raise ValueError(f"Profile {label} source config must explicitly set pipeline.output.directory")
    return {
        "native_volume_workers": None if raw_worker_count is None else int(raw_worker_count),
        "output_directory": raw_output_directory,
        "only_worker_and_output_directory_changed": True,
    }


def _validate_loaded_profile_payload(
    *,
    label: str,
    profile_payload: Mapping[str, object],
    expected_worker_count: int | None,
    expected_fov_ids: Sequence[int],
    expected_target_rounds: Sequence[int] | None,
    expected_repeats: int,
    base_config_payload: Mapping[str, object],
    expected_config_path: Path | None,
) -> dict[str, object]:
    schema_name = profile_payload.get("schema_name")
    schema_version = profile_payload.get("schema_version")
    if schema_name != CALIBRATION_PROFILE_SCHEMA_NAME:
        raise ValueError(
            f"Profile {label} schema_name {schema_name!r} does not match "
            f"{CALIBRATION_PROFILE_SCHEMA_NAME!r}"
        )
    if schema_version != CALIBRATION_PROFILE_SCHEMA_VERSION:
        raise ValueError(
            f"Profile {label} schema_version {schema_version!r} does not match "
            f"{CALIBRATION_PROFILE_SCHEMA_VERSION!r}"
        )

    source = _profile_config_source(profile_payload)
    raw_config_path = source.get("config_path")
    if not isinstance(raw_config_path, str) or not raw_config_path:
        raise ValueError(f"Profile {label} source must record config_path")
    source_config_path = Path(raw_config_path).expanduser().resolve(strict=True)
    if expected_config_path is not None and source_config_path != expected_config_path.expanduser().resolve(strict=True):
        raise ValueError(
            f"Profile {label} source config_path {source_config_path} does not match "
            f"generated config path {expected_config_path}"
        )
    actual_config_sha = _sha256_file(source_config_path)
    recorded_config_sha = source.get("config_sha256")
    if recorded_config_sha != actual_config_sha:
        raise ValueError(
            f"Profile {label} source config_sha256 {recorded_config_sha!r} does not match "
            f"actual {actual_config_sha!r} for {source_config_path}"
        )

    profile_configuration = _require_profile_mapping(profile_payload, "profile_configuration", label=label)
    profile_fovs = _int_list_from_profile(profile_configuration.get("fov_ids"), field_name="profile_configuration.fov_ids")
    expected_fov_list = [int(value) for value in expected_fov_ids]
    if profile_fovs != expected_fov_list:
        raise ValueError(f"Profile {label} FOV surface {profile_fovs!r} does not match expected {expected_fov_list!r}")
    profile_rounds = _int_list_from_profile(
        profile_configuration.get("target_rounds"),
        field_name="profile_configuration.target_rounds",
        allow_none=True,
    )
    expected_round_list = None if expected_target_rounds is None else [int(value) for value in expected_target_rounds]
    if profile_rounds != expected_round_list:
        raise ValueError(
            f"Profile {label} target-round surface {profile_rounds!r} does not match expected {expected_round_list!r}"
        )
    if int(cast(int, profile_configuration.get("repeats"))) != int(expected_repeats):
        raise ValueError(
            f"Profile {label} repeats {profile_configuration.get('repeats')!r} does not match expected {expected_repeats}"
        )

    for fov in _profile_fovs(profile_payload):
        fov_id = int(cast(int, fov.get("fov_id")))
        if fov_id not in expected_fov_list:
            raise ValueError(f"Profile {label} contains unexpected FOV {fov_id}")
        fov_rounds = _int_list_from_profile(fov.get("target_rounds"), field_name="fovs[].target_rounds", allow_none=True)
        if fov_rounds != expected_round_list:
            raise ValueError(
                f"Profile {label} FOV {fov_id} target rounds {fov_rounds!r} do not match expected {expected_round_list!r}"
            )

    profile_config_payload = _read_yaml_mapping(source_config_path)
    worker_config_validation = _validate_worker_config_provenance(
        label=label,
        profile_config_payload=profile_config_payload,
        base_config_payload=base_config_payload,
        expected_worker_count=expected_worker_count,
    )

    return {
        "status": "pass",
        "schema_name": schema_name,
        "schema_version": schema_version,
        "source_config_path": str(source_config_path),
        "source_config_sha256": actual_config_sha,
        "surface_fov_ids": profile_fovs,
        "target_rounds": profile_rounds,
        "repeats": int(expected_repeats),
        "worker_config": worker_config_validation,
    }


def _run_or_load_worker_profile(
    *,
    worker_count: int,
    config_path: Path,
    profile_output_dir: Path,
    fov_ids: Sequence[int] | None,
    target_rounds: Sequence[int] | None,
    repeats: int,
    baseline_commit: str,
    validation_worktree: Path | None,
    existing_profile_json: Path | None,
) -> tuple[Path, Path | None, dict[str, object], str]:
    if existing_profile_json is not None:
        payload = _read_json_mapping(existing_profile_json)
        markdown_path = existing_profile_json.with_suffix(".md")
        return existing_profile_json, markdown_path if markdown_path.exists() else None, payload, "loaded_existing"

    json_path, markdown_path, payload = profile_native_preprocessing_calibration(
        config_path=config_path,
        output_dir=profile_output_dir,
        fov_ids=fov_ids,
        target_rounds=target_rounds,
        repeats=repeats,
        baseline_commit=baseline_commit,
        validation_worktree=validation_worktree,
    )
    _ = worker_count
    return json_path, markdown_path, payload, "run"


def _disk_usage_record(path: Path) -> dict[str, object]:
    resolved = path.expanduser().resolve(strict=False)
    try:
        usage = shutil.disk_usage(resolved if resolved.exists() else resolved.parent)
    except Exception as exc:  # pragma: no cover - platform/filesystem dependent
        return {"path": str(resolved), "status": "error", "error": f"{type(exc).__name__}: {exc}"}
    return {
        "path": str(resolved),
        "status": "available",
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_bytes": int(usage.free),
    }


def _system_resource_notes(paths: Mapping[str, Path] | None = None) -> dict[str, object]:
    mem_total_kib: int | None = None
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                parts = line.split()
                if len(parts) >= 2:
                    mem_total_kib = int(parts[1])
                break
    return {
        "cpu_count": os.cpu_count(),
        "mem_total_kib": mem_total_kib,
        "disk_usage": {
            label: _disk_usage_record(path)
            for label, path in ({} if paths is None else dict(paths)).items()
        },
    }


def _seq_channels_for_round(base_config: ExperimentConfig, round_id: int) -> list[int]:
    roles = cast(Mapping[int, str], base_config.dataset.channel_roles)
    round_structure = cast(Mapping[int, Sequence[int]], base_config.dataset.round_structure)
    return sorted(
        int(channel_id)
        for channel_id in round_structure.get(int(round_id), [])
        if roles.get(int(channel_id)) == "seq"
    )


def _normalized_sys_path_prefix(limit: int = 5) -> list[str]:
    entries: list[str] = []
    for entry in sys.path[:limit]:
        if entry == "":
            entries.append("")
            continue
        entries.append(str(Path(entry).expanduser().resolve(strict=False)))
    return entries


def _source_import_provenance(source_root: Path) -> dict[str, object]:
    source_root_resolved = source_root.expanduser().resolve(strict=True)
    pystar_module = sys.modules.get("pystar")
    module_file_raw = getattr(pystar_module, "__file__", None)
    module_file: str | None = None
    module_within_source_root = False
    if module_file_raw is not None:
        module_path = Path(str(module_file_raw)).expanduser().resolve(strict=False)
        module_file = str(module_path)
        try:
            _ = module_path.relative_to(source_root_resolved)
            module_within_source_root = True
        except ValueError:
            module_within_source_root = False

    sys_path_prefix = _normalized_sys_path_prefix()
    return {
        "python_executable": sys.executable,
        "pythonpath_env": os.environ.get("PYTHONPATH"),
        "pythonpath_effective_source_root": str(source_root_resolved),
        "pystar_import_path": module_file,
        "pystar_import_within_source_root": module_within_source_root,
        "source_root_first_on_sys_path": bool(sys_path_prefix and sys_path_prefix[0] == str(source_root_resolved)),
        "sys_path_prefix": sys_path_prefix,
    }


def _shell_command(args: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(arg)) for arg in args)


def _stage30a_replay_command(*, source_root: Path, argv: Sequence[str]) -> str:
    return _shell_command(["PYTHONPATH=" + str(source_root), "python", *argv])


def _build_profile_command(
    *,
    source_root: Path,
    config_path: Path,
    profile_output_dir: Path,
    fov_ids: Sequence[int] | None,
    target_rounds: Sequence[int] | None,
    repeats: int,
    baseline_commit: str,
    validation_worktree: Path | None,
) -> str:
    command = [
        "PYTHONPATH=" + str(source_root),
        "python",
        str(source_root / "scripts" / "profile_native_preprocessing_calibration.py"),
        "--config",
        str(config_path),
        "--output-dir",
        str(profile_output_dir),
        "--repeats",
        str(repeats),
        "--baseline-commit",
        baseline_commit,
    ]
    if fov_ids is not None:
        command.extend(["--fovs", ",".join(str(value) for value in fov_ids)])
    if target_rounds is not None:
        command.extend(["--rounds", ",".join(str(value) for value in target_rounds)])
    if validation_worktree is not None:
        command.extend(["--validation-worktree", str(validation_worktree)])
    return _shell_command(command)


def _determine_verdict(
    *,
    profile_records: Sequence[Mapping[str, object]],
    directory_contracts: Sequence[Mapping[str, object]],
    internal_equivalence: Sequence[Mapping[str, object]],
    clean_comparisons: Sequence[Mapping[str, object]],
    timing_rows: Sequence[Mapping[str, object]],
    histogram_rows: Sequence[Mapping[str, object]],
    skipped_workers: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    failed_profiles = [record for record in profile_records if record.get("status") != "completed"]
    failed_directories = [record for record in directory_contracts if record.get("status") != "pass"]
    failed_internal_equivalence = [record for record in internal_equivalence if record.get("status") != "equivalent"]
    mismatches = [comparison for comparison in clean_comparisons if comparison.get("status") != "equivalent"]
    histogram_schema_failures = [
        row
        for row in histogram_rows
        if row.get("schema_name") != HISTOGRAM_REAL_MATCH_ATTRIBUTION_SCHEMA_NAME
        or row.get("schema_version") != HISTOGRAM_REAL_MATCH_ATTRIBUTION_SCHEMA_VERSION
    ]
    if failed_profiles or failed_directories or failed_internal_equivalence or mismatches or histogram_schema_failures:
        return {
            "status": "fail",
            "regression": bool(mismatches),
            "recommended_worker_count": None,
            "reason": "One or more worker profiles, output directory contracts, internal repeat equivalence gates, histogram attribution schema gates, or clean TIFF comparisons failed.",
            "failed_profile_count": len(failed_profiles),
            "failed_directory_contract_count": len(failed_directories),
            "failed_internal_equivalence_count": len(failed_internal_equivalence),
            "clean_comparison_mismatch_count": len(mismatches),
            "histogram_schema_failure_count": len(histogram_schema_failures),
            "skipped_workers": list(skipped_workers),
        }
    if not clean_comparisons:
        return {
            "status": "needs_investigation",
            "regression": False,
            "recommended_worker_count": None,
            "reason": "No cross-run clean TIFF comparison was available; at least two completed profiles are required for Stage30a equivalence evidence.",
            "skipped_workers": list(skipped_workers),
        }

    wall_totals: dict[int, float] = {}
    for row in timing_rows:
        worker = row.get("worker_count")
        if worker is None:
            continue
        worker_int = int(cast(int, worker))
        value = row.get("wall_total_ms")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        wall_totals[worker_int] = wall_totals.get(worker_int, 0.0) + float(value)

    recommended_worker = None
    if wall_totals:
        recommended_worker = min(wall_totals, key=wall_totals.__getitem__)

    skipped_note = " Skipped worker counts were documented." if skipped_workers else ""
    return {
        "status": "pass",
        "regression": False,
        "recommended_worker_count": recommended_worker,
        "reason": "All completed worker profiles matched the selected clean TIFF reference." + skipped_note,
        "skipped_workers": list(skipped_workers),
    }


def _wall_totals_by_label(timing_rows: Sequence[Mapping[str, object]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for row in timing_rows:
        label = row.get("label")
        value = row.get("wall_total_ms")
        if not isinstance(label, str) or isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        totals[label] = totals.get(label, 0.0) + float(value)
    return totals


def _stage32_clean_equivalence_gate(clean_comparisons: Sequence[Mapping[str, object]]) -> dict[str, object]:
    missing_count = sum(int(cast(int, comparison.get("missing_count", 0))) for comparison in clean_comparisons)
    extra_count = sum(int(cast(int, comparison.get("extra_count", 0))) for comparison in clean_comparisons)
    mismatch_count = sum(int(cast(int, comparison.get("mismatch_count", 0))) for comparison in clean_comparisons)
    shape_drift_count = sum(int(cast(int, comparison.get("shape_drift_count", 0))) for comparison in clean_comparisons)
    dtype_drift_count = sum(int(cast(int, comparison.get("dtype_drift_count", 0))) for comparison in clean_comparisons)
    max_abs_values = [
        cast(int | float, comparison.get("max_abs_diff"))
        for comparison in clean_comparisons
        if isinstance(comparison.get("max_abs_diff"), (int, float)) and not isinstance(comparison.get("max_abs_diff"), bool)
    ]
    max_abs_diff = max(max_abs_values) if max_abs_values else None
    status = (
        "pass"
        if clean_comparisons
        and missing_count == 0
        and extra_count == 0
        and mismatch_count == 0
        and shape_drift_count == 0
        and dtype_drift_count == 0
        and max_abs_diff == 0
        else "not_compared" if not clean_comparisons else "fail"
    )
    return {
        "status": status,
        "comparison_count": len(clean_comparisons),
        "missing_count": missing_count,
        "extra_count": extra_count,
        "mismatch_count": mismatch_count,
        "shape_drift_count": shape_drift_count,
        "dtype_drift_count": dtype_drift_count,
        "max_abs_diff": max_abs_diff,
        "exact_equivalence_required": {
            "missing_count": 0,
            "extra_count": 0,
            "mismatch_count": 0,
            "shape_drift_count": 0,
            "dtype_drift_count": 0,
            "max_abs_diff": 0,
        },
    }


def _stage32_required_profile_labels(
    *,
    policy_surface: Mapping[str, object],
    available_labels: set[str],
) -> list[str]:
    labels: list[str] = []
    for worker in cast(Sequence[int], policy_surface["required_worker_counts"]):
        worker_count = int(worker)
        if worker_count == STAGE32_POLICY_BASELINE_WORKER and "baseline" in available_labels:
            label = "baseline"
        else:
            label = _profile_label_for_worker(worker_count)
        if label not in labels:
            labels.append(label)
    return labels


def _stage32_clean_surface_completeness_gate(
    *,
    clean_record_maps: Mapping[str, Mapping[tuple[int, str], Mapping[str, object]]],
    policy_surface: Mapping[str, object],
    expected_clean_keys: set[tuple[int, str]],
) -> dict[str, object]:
    required_labels = _stage32_required_profile_labels(
        policy_surface=policy_surface,
        available_labels=set(clean_record_maps),
    )
    if not expected_clean_keys:
        return {
            "status": "not_compared",
            "reason": "No configured clean TIFF surface could be derived from selected FOV/round/channel metadata.",
            "expected_clean_tiff_count": 0,
            "required_profile_labels": required_labels,
            "profile_rows": [],
        }

    profile_rows: list[dict[str, object]] = []
    failed = False
    evaluated = False
    for label in required_labels:
        records = clean_record_maps.get(label)
        if records is None:
            profile_rows.append(
                {
                    "label": label,
                    "status": "not_evaluated",
                    "reason": "Required profile did not complete or did not yield clean TIFF records.",
                    "expected_clean_tiff_count": len(expected_clean_keys),
                    "observed_clean_tiff_count": 0,
                    "missing_expected_clean_tiff_count": len(expected_clean_keys),
                    "unexpected_clean_tiff_count": 0,
                    "missing_expected_clean_tiffs": [
                        f"FOV {fov_id}: {relative_path}"
                        for fov_id, relative_path in sorted(expected_clean_keys)
                    ],
                    "unexpected_clean_tiffs": [],
                }
            )
            continue
        evaluated = True
        observed_keys = set(records)
        missing = sorted(expected_clean_keys - observed_keys)
        unexpected = sorted(observed_keys - expected_clean_keys)
        status = "pass" if not missing and not unexpected else "fail"
        failed = failed or status == "fail"
        profile_rows.append(
            {
                "label": label,
                "status": status,
                "expected_clean_tiff_count": len(expected_clean_keys),
                "observed_clean_tiff_count": len(observed_keys),
                "missing_expected_clean_tiff_count": len(missing),
                "unexpected_clean_tiff_count": len(unexpected),
                "missing_expected_clean_tiffs": [
                    f"FOV {fov_id}: {relative_path}"
                    for fov_id, relative_path in missing
                ],
                "unexpected_clean_tiffs": [
                    f"FOV {fov_id}: {relative_path}"
                    for fov_id, relative_path in unexpected
                ],
            }
        )

    if failed:
        status = "fail"
    elif evaluated:
        status = "pass"
    else:
        status = "not_evaluated"
    return {
        "status": status,
        "expected_clean_tiff_count": len(expected_clean_keys),
        "required_profile_labels": required_labels,
        "profile_rows": profile_rows,
        "missing_expected_clean_tiff_count": sum(
            int(cast(int, row.get("missing_expected_clean_tiff_count", 0)))
            for row in profile_rows
        ),
        "unexpected_clean_tiff_count": sum(
            int(cast(int, row.get("unexpected_clean_tiff_count", 0)))
            for row in profile_rows
        ),
    }


def _stage32_profile_source_consistency_gate(
    *,
    profile_payloads: Mapping[str, Mapping[str, object]],
    policy_surface: Mapping[str, object],
    validation_worktree: Path | None,
    source_root: Path,
) -> dict[str, object]:
    required_labels = _stage32_required_profile_labels(
        policy_surface=policy_surface,
        available_labels=set(profile_payloads),
    )
    expected_worktree = None if validation_worktree is None else str(validation_worktree.expanduser().resolve(strict=False))
    expected_source_root = str(source_root.expanduser().resolve(strict=False))
    source_rows: list[dict[str, object]] = []
    for label in required_labels:
        payload = profile_payloads.get(label)
        if payload is None:
            source_rows.append(
                {
                    "label": label,
                    "status": "not_evaluated",
                    "reason": "Required profile payload is unavailable.",
                    "candidate_commit": None,
                    "validation_worktree": None,
                }
            )
            continue
        source = _profile_config_source(payload)
        raw_worktree = source.get("validation_worktree")
        actual_worktree = None if raw_worktree is None else str(Path(str(raw_worktree)).expanduser().resolve(strict=False))
        raw_repo_root = source.get("repo_root")
        actual_repo_root = None if raw_repo_root is None else str(Path(str(raw_repo_root)).expanduser().resolve(strict=False))
        source_rows.append(
            {
                "label": label,
                "status": "evaluated",
                "repo_root": actual_repo_root,
                "candidate_commit": source.get("candidate_commit") or source.get("git_commit"),
                "validation_worktree": actual_worktree,
            }
        )

    evaluated_rows = [row for row in source_rows if row.get("status") == "evaluated"]
    missing_commit_labels = [str(row["label"]) for row in evaluated_rows if not row.get("candidate_commit")]
    missing_repo_root_labels = [str(row["label"]) for row in evaluated_rows if not row.get("repo_root")]
    commit_values = {str(row["candidate_commit"]) for row in evaluated_rows if row.get("candidate_commit")}
    repo_root_values = {str(row["repo_root"]) for row in evaluated_rows if row.get("repo_root")}
    worktree_values = {row.get("validation_worktree") for row in evaluated_rows}
    expected_worktree_mismatches = [
        str(row["label"])
        for row in evaluated_rows
        if expected_worktree is not None and row.get("validation_worktree") != expected_worktree
    ]
    expected_repo_root_mismatches = [
        str(row["label"])
        for row in evaluated_rows
        if row.get("repo_root") != expected_source_root
    ]
    worktree_mismatch = bool(expected_worktree_mismatches) or len(worktree_values) > 1
    commit_mismatch = len(commit_values) > 1
    repo_root_mismatch = bool(expected_repo_root_mismatches) or len(repo_root_values) > 1
    unavailable_labels = [str(row["label"]) for row in source_rows if row.get("status") != "evaluated"]
    status = (
        "fail"
        if missing_commit_labels or missing_repo_root_labels or commit_mismatch or repo_root_mismatch or worktree_mismatch
        else "pass"
        if evaluated_rows
        else "not_evaluated"
    )
    return {
        "status": status,
        "required_profile_labels": required_labels,
        "expected_source_root": expected_source_root,
        "expected_validation_worktree": expected_worktree,
        "candidate_commits": sorted(commit_values),
        "repo_roots": sorted(repo_root_values),
        "validation_worktrees": sorted(str(value) for value in worktree_values if value is not None),
        "missing_candidate_commit_labels": missing_commit_labels,
        "missing_repo_root_labels": missing_repo_root_labels,
        "candidate_commit_mismatch": commit_mismatch,
        "repo_root_mismatch": repo_root_mismatch,
        "repo_root_mismatch_labels": expected_repo_root_mismatches,
        "validation_worktree_mismatch": worktree_mismatch,
        "validation_worktree_mismatch_labels": expected_worktree_mismatches,
        "unavailable_profile_labels": unavailable_labels,
        "profile_sources": source_rows,
    }


def _stage32_surface_policy(
    *,
    surface_scope: str,
    limited_surface_reason: str,
    config_fovs: Sequence[int],
    selected_fovs: Sequence[int],
    config_rounds: Sequence[int],
    selected_rounds: Sequence[int],
    policy_candidate_workers: Sequence[int],
) -> dict[str, object]:
    if surface_scope not in STAGE32_SURFACE_SCOPES:
        raise ValueError(f"Unsupported Stage32 surface scope {surface_scope!r}; expected one of {STAGE32_SURFACE_SCOPES}")
    config_fov_set = {int(value) for value in config_fovs}
    selected_fov_set = {int(value) for value in selected_fovs}
    missing_fovs = sorted(config_fov_set - selected_fov_set)
    extra_fovs = sorted(selected_fov_set - config_fov_set)
    fov_surface_complete = not missing_fovs and not extra_fovs
    config_round_set = {int(value) for value in config_rounds}
    selected_round_set = {int(value) for value in selected_rounds}
    missing_rounds = sorted(config_round_set - selected_round_set)
    extra_rounds = sorted(selected_round_set - config_round_set)
    round_surface_complete = not missing_rounds and not extra_rounds
    declared_full = surface_scope == "full"
    effective_scope = "full" if declared_full and fov_surface_complete and round_surface_complete else "limited"
    reasons: list[str] = []
    if not declared_full:
        reasons.append(limited_surface_reason or "Stage32 run declared a limited validation surface.")
    if missing_fovs:
        reasons.append(f"Selected FOVs omit configured FOVs: {missing_fovs}.")
    if extra_fovs:
        reasons.append(f"Selected FOVs include FOVs outside the config surface: {extra_fovs}.")
    if missing_rounds:
        reasons.append(f"Selected rounds omit configured rounds: {missing_rounds}.")
    if extra_rounds:
        reasons.append(f"Selected rounds include rounds outside the config surface: {extra_rounds}.")
    if not reasons and effective_scope == "full":
        reasons.append("All configured FOVs and rounds were included.")
    limited_reason = None if effective_scope == "full" else "; ".join(reasons)
    return {
        "declared_surface_scope": surface_scope,
        "effective_surface_scope": effective_scope,
        "limited_surface_reason": limited_reason,
        "policy_candidate_workers": [int(worker) for worker in policy_candidate_workers],
        "baseline_worker_count": STAGE32_POLICY_BASELINE_WORKER,
        "required_worker_counts": sorted({STAGE32_POLICY_BASELINE_WORKER, *[int(worker) for worker in policy_candidate_workers]}),
        "config_fov_ids": [int(value) for value in config_fovs],
        "selected_fov_ids": [int(value) for value in selected_fovs],
        "selected_all_configured_fovs": fov_surface_complete,
        "fov_surface_complete": fov_surface_complete,
        "missing_configured_fov_ids": missing_fovs,
        "extra_selected_fov_ids": extra_fovs,
        "config_round_ids": [int(value) for value in config_rounds],
        "selected_round_ids": [int(value) for value in selected_rounds],
        "round_surface_complete": round_surface_complete,
        "missing_configured_round_ids": missing_rounds,
        "extra_selected_round_ids": extra_rounds,
        "reasons": reasons,
        "full_surface_policy_ready": effective_scope == "full",
    }


def _stage33_fov1_surface_policy(
    *,
    config_fovs: Sequence[int],
    selected_fovs: Sequence[int],
    config_rounds: Sequence[int],
    selected_rounds: Sequence[int],
    policy_candidate_workers: Sequence[int],
) -> dict[str, object]:
    policy_fov_ids = [int(value) for value in STAGE33_FOV1_POLICY_FOV_IDS]
    config_fov_set = {int(value) for value in config_fovs}
    selected_fov_list = [int(value) for value in selected_fovs]
    selected_fov_set = set(selected_fov_list)
    policy_fov_set = set(policy_fov_ids)
    duplicate_selected_fovs = sorted(
        {
            fov_id
            for fov_id in selected_fov_list
            if selected_fov_list.count(fov_id) > 1
        }
    )
    missing_policy_fovs = sorted(policy_fov_set - selected_fov_set)
    selected_non_policy_fovs = sorted(selected_fov_set - policy_fov_set)
    configured_non_policy_fovs = sorted(config_fov_set - policy_fov_set)
    missing_policy_fovs_in_config = sorted(policy_fov_set - config_fov_set)
    fov1_surface_complete = (
        selected_fov_list == policy_fov_ids
        and not duplicate_selected_fovs
        and not missing_policy_fovs
        and not selected_non_policy_fovs
    )

    config_round_set = {int(value) for value in config_rounds}
    selected_round_list = [int(value) for value in selected_rounds]
    selected_round_set = set(selected_round_list)
    duplicate_selected_rounds = sorted(
        {
            round_id
            for round_id in selected_round_list
            if selected_round_list.count(round_id) > 1
        }
    )
    missing_rounds = sorted(config_round_set - selected_round_set)
    extra_rounds = sorted(selected_round_set - config_round_set)
    round_surface_complete = not missing_rounds and not extra_rounds and not duplicate_selected_rounds
    fov1_policy_ready = fov1_surface_complete and round_surface_complete and not missing_policy_fovs_in_config
    fov_selection_invalid = bool(
        missing_policy_fovs_in_config
        or duplicate_selected_fovs
        or missing_policy_fovs
        or selected_non_policy_fovs
        or selected_fov_list != policy_fov_ids
    )
    round_selection_invalid = bool(extra_rounds or duplicate_selected_rounds)
    policy_surface_status = (
        "pass"
        if fov1_policy_ready
        else "fail"
        if fov_selection_invalid or round_selection_invalid
        else "inconclusive"
    )

    reasons: list[str] = []
    if missing_policy_fovs_in_config:
        reasons.append(f"Configured FOV surface does not include required Stage33 FOV1 policy FOVs: {missing_policy_fovs_in_config}.")
    if duplicate_selected_fovs:
        reasons.append(f"Selected FOVs repeat Stage33 FOV1 policy FOVs: {duplicate_selected_fovs}.")
    if missing_policy_fovs:
        reasons.append(f"Selected FOVs omit required Stage33 FOV1 policy FOVs: {missing_policy_fovs}.")
    if selected_non_policy_fovs:
        reasons.append(
            f"Selected FOVs include non-policy FOVs {selected_non_policy_fovs}; Stage33 can only recommend FOV1 readiness."
        )
    if missing_rounds:
        reasons.append(f"Selected rounds omit configured FOV1 policy rounds: {missing_rounds}.")
    if extra_rounds:
        reasons.append(f"Selected rounds include rounds outside the FOV1 policy config surface: {extra_rounds}.")
    if duplicate_selected_rounds:
        reasons.append(f"Selected rounds repeat FOV1 policy rounds: {duplicate_selected_rounds}.")
    if fov1_policy_ready:
        if configured_non_policy_fovs:
            reasons.append(
                "FOV1 policy surface included exactly FOV 1 and all configured rounds; "
                f"configured non-policy FOVs {configured_non_policy_fovs} were not validated and no multi-FOV readiness is claimed."
            )
        else:
            reasons.append(
                "FOV1 policy surface included exactly FOV 1 and all configured rounds; no multi-FOV readiness is claimed."
            )

    return {
        "policy_stage": "Stage33 FOV1",
        "policy_surface_status": policy_surface_status,
        "policy_surface_kind": "fov1_only",
        "declared_surface_scope": "fov1_only",
        "effective_surface_scope": "full" if fov1_policy_ready else "limited",
        "limited_surface_reason": None if fov1_policy_ready else "; ".join(reasons),
        "policy_candidate_workers": [int(worker) for worker in policy_candidate_workers],
        "baseline_worker_count": STAGE32_POLICY_BASELINE_WORKER,
        "required_worker_counts": sorted({STAGE32_POLICY_BASELINE_WORKER, *[int(worker) for worker in policy_candidate_workers]}),
        "policy_fov_ids": policy_fov_ids,
        "config_fov_ids": [int(value) for value in config_fovs],
        "configured_non_policy_fov_ids": configured_non_policy_fovs,
        "missing_policy_fov_ids_in_config": missing_policy_fovs_in_config,
        "selected_fov_ids": selected_fov_list,
        "fov_surface_complete": fov1_surface_complete,
        "fov1_policy_surface_complete": fov1_surface_complete,
        "duplicate_selected_fov_ids": duplicate_selected_fovs,
        "missing_policy_fov_ids": missing_policy_fovs,
        "extra_selected_fov_ids": selected_non_policy_fovs,
        "selected_non_policy_fov_ids": selected_non_policy_fovs,
        "config_round_ids": [int(value) for value in config_rounds],
        "selected_round_ids": selected_round_list,
        "round_surface_complete": round_surface_complete,
        "missing_configured_round_ids": missing_rounds,
        "extra_selected_round_ids": extra_rounds,
        "duplicate_selected_round_ids": duplicate_selected_rounds,
        "reasons": reasons,
        "full_surface_policy_ready": fov1_policy_ready,
        "fov1_policy_surface_ready": fov1_policy_ready,
        "multi_fov_policy_ready": False,
        "multi_fov_readiness_claimed": False,
        "multi_fov_readiness_disclaimer": "Stage33 validates only FOV1 native preprocessing; it must not be used as multi-FOV readiness evidence.",
    }


def _determine_stage32_policy_verdict(
    *,
    profile_records: Sequence[Mapping[str, object]],
    directory_contracts: Sequence[Mapping[str, object]],
    internal_equivalence: Sequence[Mapping[str, object]],
    timing_rows: Sequence[Mapping[str, object]],
    histogram_rows: Sequence[Mapping[str, object]],
    skipped_workers: Sequence[Mapping[str, object]],
    policy_surface: Mapping[str, object],
    clean_gate: Mapping[str, object],
    clean_surface_gate: Mapping[str, object],
    source_consistency_gate: Mapping[str, object],
    policy_label: str = "Stage32",
    pass_opt_in_recommendation: str | None = None,
    pass_reason: str | None = None,
    limited_opt_in_recommendation: str | None = None,
    limited_reason: str | None = None,
    limited_next_debugging_task: str | None = None,
    invalid_surface_opt_in_recommendation: str | None = None,
    invalid_surface_reason: str | None = None,
    invalid_surface_next_debugging_task: str | None = None,
) -> dict[str, object]:
    failed_profiles = [record for record in profile_records if record.get("status") != "completed"]
    failed_directories = [record for record in directory_contracts if record.get("status") != "pass"]
    failed_internal_equivalence = [record for record in internal_equivalence if record.get("status") != "equivalent"]
    histogram_schema_failures = [
        row
        for row in histogram_rows
        if row.get("schema_name") != HISTOGRAM_REAL_MATCH_ATTRIBUTION_SCHEMA_NAME
        or row.get("schema_version") != HISTOGRAM_REAL_MATCH_ATTRIBUTION_SCHEMA_VERSION
    ]
    completed_labels = {str(record["label"]) for record in profile_records if record.get("status") == "completed" and "label" in record}
    completed_workers = {
        int(cast(int, record["worker_count"]))
        for record in profile_records
        if record.get("status") == "completed" and record.get("worker_count") is not None
    }
    requested_workers = {
        int(cast(int, record["worker_count"]))
        for record in profile_records
        if record.get("worker_count") is not None
    }
    required_workers = [int(worker) for worker in cast(Sequence[int], policy_surface["required_worker_counts"])]
    candidate_workers = [int(worker) for worker in cast(Sequence[int], policy_surface["policy_candidate_workers"])]
    candidate_labels = [_profile_label_for_worker(worker) for worker in candidate_workers]
    baseline_label = "baseline" if "baseline" in completed_labels else _profile_label_for_worker(STAGE32_POLICY_BASELINE_WORKER)
    baseline_available = baseline_label in completed_labels
    missing_required_workers = [
        worker
        for worker in required_workers
        if (
            worker == STAGE32_POLICY_BASELINE_WORKER
            and not baseline_available
            and worker not in completed_workers
        )
        or (worker != STAGE32_POLICY_BASELINE_WORKER and worker not in completed_workers)
    ]
    requested_but_failed_required_workers = [worker for worker in missing_required_workers if worker in requested_workers]

    correctness_gate_failed = (
        failed_profiles
        or failed_directories
        or failed_internal_equivalence
        or histogram_schema_failures
        or clean_gate.get("status") == "fail"
        or clean_surface_gate.get("status") == "fail"
        or source_consistency_gate.get("status") == "fail"
    )
    if correctness_gate_failed:
        return {
            "status": "fail",
            "regression": clean_gate.get("status") == "fail" or clean_surface_gate.get("status") == "fail",
            "recommended_worker_count": None,
            "opt_in_recommendation": f"Do not recommend native_volume_workers=4; at least one {policy_label} correctness gate failed.",
            "reason": f"One or more {policy_label} worker-policy correctness gates failed before any speed recommendation.",
            "next_debugging_task": "Inspect fail_loud_errors, directory_contracts, internal_repeat_equivalence, histogram_attribution_summary, clean_tiff_equivalence, worker_policy.surface_completeness_gate, and worker_policy.source_consistency_gate for the first failing gate.",
            "failed_profile_count": len(failed_profiles),
            "failed_directory_contract_count": len(failed_directories),
            "failed_internal_equivalence_count": len(failed_internal_equivalence),
            "histogram_schema_failure_count": len(histogram_schema_failures),
            "clean_equivalence_gate": dict(clean_gate),
            "surface_completeness_gate": dict(clean_surface_gate),
            "source_consistency_gate": dict(source_consistency_gate),
            "missing_required_workers": missing_required_workers,
            "requested_but_failed_required_workers": requested_but_failed_required_workers,
            "skipped_workers": list(skipped_workers),
        }

    if policy_surface.get("policy_surface_status") == "fail":
        return {
            "status": "fail",
            "regression": False,
            "recommended_worker_count": None,
            "opt_in_recommendation": invalid_surface_opt_in_recommendation
            or "No opt-in worker policy recommendation; the requested policy surface is invalid.",
            "reason": invalid_surface_reason or f"{policy_label} policy surface selection is invalid.",
            "next_debugging_task": invalid_surface_next_debugging_task
            or f"Rerun {policy_label} with a valid policy surface selection.",
            "clean_equivalence_gate": dict(clean_gate),
            "surface_completeness_gate": dict(clean_surface_gate),
            "source_consistency_gate": dict(source_consistency_gate),
            "surface_reasons": list(cast(Sequence[object], policy_surface.get("reasons", []))),
            "missing_required_workers": missing_required_workers,
            "skipped_workers": list(skipped_workers),
        }

    if clean_gate.get("status") != "pass":
        return {
            "status": "inconclusive",
            "regression": False,
            "recommended_worker_count": None,
            "opt_in_recommendation": "No opt-in worker policy recommendation; serial/default vs workers4 clean TIFFs were not compared.",
            "reason": f"No {policy_label} cross-run clean TIFF comparison was available.",
            "next_debugging_task": "Provide completed serial/default and explicit workers4 profiles on the same surface.",
            "clean_equivalence_gate": dict(clean_gate),
            "surface_completeness_gate": dict(clean_surface_gate),
            "source_consistency_gate": dict(source_consistency_gate),
            "missing_required_workers": missing_required_workers,
            "skipped_workers": list(skipped_workers),
        }

    if clean_surface_gate.get("status") != "pass":
        return {
            "status": "inconclusive",
            "regression": False,
            "recommended_worker_count": None,
            "opt_in_recommendation": "No opt-in worker policy recommendation; expected clean TIFF surface completeness was not proven.",
            "reason": f"{policy_label} did not validate every expected clean TIFF on the selected FOV/round/channel surface.",
            "next_debugging_task": "Regenerate or load profiles whose output_files cover every expected clean_fov_{fov}_round_{round}_ch_{channel}.tif artifact.",
            "clean_equivalence_gate": dict(clean_gate),
            "surface_completeness_gate": dict(clean_surface_gate),
            "source_consistency_gate": dict(source_consistency_gate),
            "missing_required_workers": missing_required_workers,
            "skipped_workers": list(skipped_workers),
        }

    if source_consistency_gate.get("status") != "pass":
        return {
            "status": "inconclusive",
            "regression": False,
            "recommended_worker_count": None,
            "opt_in_recommendation": "No opt-in worker policy recommendation; source commit/worktree consistency was not proven.",
            "reason": f"{policy_label} did not prove serial/default and candidate profiles came from the same source commit and validation worktree.",
            "next_debugging_task": "Regenerate or load all required profiles from the same PyStar source commit and validation worktree.",
            "clean_equivalence_gate": dict(clean_gate),
            "surface_completeness_gate": dict(clean_surface_gate),
            "source_consistency_gate": dict(source_consistency_gate),
            "missing_required_workers": missing_required_workers,
            "skipped_workers": list(skipped_workers),
        }

    if missing_required_workers:
        status = "fail" if requested_but_failed_required_workers else "inconclusive"
        return {
            "status": status,
            "regression": False,
            "recommended_worker_count": None,
            "opt_in_recommendation": "No opt-in worker policy recommendation; required serial/default and workers4 profiles are incomplete.",
            "reason": f"{policy_label} policy mode requires completed serial/default and candidate worker profiles.",
            "next_debugging_task": "Run or load the missing required worker profiles before evaluating the policy gate.",
            "clean_equivalence_gate": dict(clean_gate),
            "surface_completeness_gate": dict(clean_surface_gate),
            "source_consistency_gate": dict(source_consistency_gate),
            "missing_required_workers": missing_required_workers,
            "requested_but_failed_required_workers": requested_but_failed_required_workers,
            "skipped_workers": list(skipped_workers),
        }

    if policy_surface.get("effective_surface_scope") != "full":
        return {
            "status": "inconclusive",
            "regression": False,
            "recommended_worker_count": None,
            "opt_in_recommendation": limited_opt_in_recommendation
            or "Exact equivalence held on the measured limited surface, but no full-surface opt-in policy recommendation is made.",
            "reason": limited_reason or f"{policy_label} was run on a limited surface; do not claim full-surface policy readiness.",
            "next_debugging_task": limited_next_debugging_task
            or f"Rerun {policy_label} with --surface-scope full and all configured FOVs/rounds.",
            "clean_equivalence_gate": dict(clean_gate),
            "surface_completeness_gate": dict(clean_surface_gate),
            "source_consistency_gate": dict(source_consistency_gate),
            "surface_reasons": list(cast(Sequence[object], policy_surface.get("reasons", []))),
            "missing_required_workers": missing_required_workers,
            "skipped_workers": list(skipped_workers),
        }

    wall_totals = _wall_totals_by_label(timing_rows)
    baseline_wall_total = wall_totals.get(baseline_label)
    candidate_wall_totals = {label: wall_totals.get(label) for label in candidate_labels}
    comparable_candidates = {
        int(label.rsplit("_", 1)[1]): value
        for label, value in candidate_wall_totals.items()
        if value is not None
    }
    if baseline_wall_total is None or not comparable_candidates:
        return {
            "status": "inconclusive",
            "regression": False,
            "recommended_worker_count": None,
            "opt_in_recommendation": "No opt-in worker policy recommendation; required timing rows are missing.",
            "reason": f"{policy_label} correctness gates passed but serial/default vs candidate wall-time comparison was incomplete.",
            "next_debugging_task": "Regenerate profiles with complete timing summaries for serial/default and workers4.",
            "clean_equivalence_gate": dict(clean_gate),
            "surface_completeness_gate": dict(clean_surface_gate),
            "source_consistency_gate": dict(source_consistency_gate),
            "baseline_label": baseline_label,
            "baseline_wall_total_ms": baseline_wall_total,
            "candidate_wall_totals_ms": candidate_wall_totals,
            "skipped_workers": list(skipped_workers),
        }

    best_worker = min(comparable_candidates, key=comparable_candidates.__getitem__)
    best_total = comparable_candidates[best_worker]
    if best_total >= baseline_wall_total:
        return {
            "status": "inconclusive",
            "regression": False,
            "recommended_worker_count": None,
            "opt_in_recommendation": "No opt-in worker policy recommendation; candidate workers did not improve measured wall time.",
            "reason": "Exact equivalence held, but the measured candidate wall time was not lower than the serial/default wall time.",
            "next_debugging_task": "Repeat the full-surface measurement or investigate resource contention before recommending worker parallelism.",
            "clean_equivalence_gate": dict(clean_gate),
            "surface_completeness_gate": dict(clean_surface_gate),
            "source_consistency_gate": dict(source_consistency_gate),
            "baseline_label": baseline_label,
            "baseline_wall_total_ms": baseline_wall_total,
            "candidate_wall_totals_ms": candidate_wall_totals,
            "skipped_workers": list(skipped_workers),
        }

    return {
        "status": "pass",
        "regression": False,
        "recommended_worker_count": best_worker,
        "opt_in_recommendation": (
            pass_opt_in_recommendation.format(best_worker=best_worker, policy_label=policy_label)
            if pass_opt_in_recommendation is not None
            else (
            f"Recommend native_volume_workers={best_worker} only as an opt-in setting under the validated Stage32 "
            "full-surface constraints; default/omitted native_volume_workers remains serial (1)."
            )
        ),
        "reason": pass_reason
        or f"{policy_label} full-surface correctness gates passed and the candidate worker setting improved measured preprocessing wall time.",
        "next_debugging_task": None,
        "clean_equivalence_gate": dict(clean_gate),
        "surface_completeness_gate": dict(clean_surface_gate),
        "source_consistency_gate": dict(source_consistency_gate),
        "baseline_label": baseline_label,
        "baseline_wall_total_ms": baseline_wall_total,
        "candidate_wall_totals_ms": candidate_wall_totals,
        "speedup_vs_baseline": baseline_wall_total / best_total if best_total > 0 else None,
        "skipped_workers": list(skipped_workers),
    }


def build_sweep_payload(
    *,
    config_path: Path,
    output_dir: Path,
    worker_counts: Sequence[int],
    expected_worker_counts: Sequence[int],
    fov_ids: Sequence[int] | None,
    target_rounds: Sequence[int] | None,
    repeats: int,
    compare_repeat_index: int,
    baseline_commit: str,
    validation_worktree: Path | None,
    production_root_base: Path,
    source_root: Path,
    existing_profiles: Mapping[str, Path],
    baseline_profile_json: Path | None,
    skip_reason: str,
    argv: Sequence[str],
    stage32_policy: bool = False,
    stage33_fov1_policy: bool = False,
    policy_candidate_workers: Sequence[int] = DEFAULT_STAGE32_POLICY_CANDIDATE_WORKERS,
    surface_scope: str = "limited",
    limited_surface_reason: str = "bounded validation surface; full configured round surface was not asserted",
) -> dict[str, object]:
    if stage32_policy and stage33_fov1_policy:
        raise ValueError("--stage32-policy and --stage33-fov1-policy are mutually exclusive")
    policy_enabled = stage32_policy or stage33_fov1_policy
    if repeats <= 0:
        raise ValueError(f"--repeats must be positive; got {repeats!r}")
    if compare_repeat_index < 0:
        raise ValueError(f"--compare-repeat-index must be non-negative; got {compare_repeat_index!r}")
    if compare_repeat_index >= repeats:
        raise ValueError(
            "--compare-repeat-index must reference one of the generated profile repeats; "
            f"got {compare_repeat_index!r} with --repeats={repeats!r}"
        )

    policy_candidate_workers = tuple(int(worker) for worker in policy_candidate_workers)
    if policy_enabled and not policy_candidate_workers:
        raise ValueError("Worker policy mode requires at least one candidate worker count")
    if policy_enabled and STAGE32_POLICY_BASELINE_WORKER in policy_candidate_workers:
        raise ValueError(
            "Policy candidate worker counts must exclude the serial/default baseline worker count "
            f"{STAGE32_POLICY_BASELINE_WORKER}"
        )
    if stage32_policy and surface_scope not in STAGE32_SURFACE_SCOPES:
        raise ValueError(f"--surface-scope must be one of {STAGE32_SURFACE_SCOPES}; got {surface_scope!r}")

    config_path = config_path.expanduser().resolve(strict=True)
    base_config_payload = _read_yaml_mapping(config_path)
    base_config = load_config(str(config_path))
    config_fovs = [int(value) for value in base_config.dataset.parsed_fovs]
    normalized_fovs = None if fov_ids is None else tuple(int(value) for value in fov_ids)
    if stage33_fov1_policy and normalized_fovs is None:
        normalized_fovs = tuple(int(value) for value in STAGE33_FOV1_POLICY_FOV_IDS)
    normalized_rounds = None if target_rounds is None else tuple(int(value) for value in target_rounds)
    expected_fov_ids = config_fovs[:1] if normalized_fovs is None else list(normalized_fovs)
    production_root_base = production_root_base.expanduser().resolve(strict=False)
    _reject_sweep_production_overlap(
        output_dir=output_dir,
        production_root_base=production_root_base,
        base_production_output_dir=Path(base_config.pipeline.output.directory),
    )
    output_dir = _prepare_sweep_output_dir(
        output_dir,
        stage32_policy=stage32_policy,
        stage33_fov1_policy=stage33_fov1_policy,
    )

    profile_records: list[dict[str, object]] = []
    profile_payloads: dict[str, dict[str, object]] = {}
    profile_commands: list[dict[str, object]] = []
    generated_worker_configs: list[dict[str, object]] = []
    config_dir = output_dir / "configs"
    profile_root = output_dir / "profiles"
    profile_root.mkdir(parents=True, exist_ok=True)

    if baseline_profile_json is not None:
        baseline_path = baseline_profile_json.expanduser().resolve(strict=True)
        try:
            baseline_payload = _read_json_mapping(baseline_path)
            baseline_validation = _validate_loaded_profile_payload(
                label="baseline",
                profile_payload=baseline_payload,
                expected_worker_count=STAGE32_POLICY_BASELINE_WORKER if policy_enabled else None,
                expected_fov_ids=expected_fov_ids,
                expected_target_rounds=normalized_rounds,
                expected_repeats=repeats,
                base_config_payload=base_config_payload,
                expected_config_path=None,
            )
            profile_payloads["baseline"] = baseline_payload
            source = _profile_config_source(baseline_payload)
            profile_records.append(
                {
                    "label": "baseline",
                    "worker_count": None,
                    "status": "completed",
                    "mode": "loaded_existing",
                    "profile_json_path": str(baseline_path),
                    "profile_markdown_path": str(baseline_path.with_suffix(".md")) if baseline_path.with_suffix(".md").exists() else None,
                    "profile_output_dir": _profile_output_dir(baseline_payload),
                    "config_path": baseline_validation["source_config_path"],
                    "config_sha256": baseline_validation["source_config_sha256"],
                    "profile_source_config_path": baseline_validation["source_config_path"],
                    "profile_source_config_sha256": baseline_validation["source_config_sha256"],
                    "equivalent_generated_config_path": None,
                    "equivalent_generated_config_sha256": None,
                    "candidate_commit_in_profile": source.get("candidate_commit") or source.get("git_commit"),
                    "profile_validation": baseline_validation,
                }
            )
        except Exception as exc:  # pragma: no cover - real-data failure evidence path
            profile_records.append(
                {
                    "label": "baseline",
                    "worker_count": None,
                    "status": "error",
                    "mode": "loaded_existing",
                    "profile_json_path": str(baseline_path),
                    "profile_markdown_path": str(baseline_path.with_suffix(".md")) if baseline_path.with_suffix(".md").exists() else None,
                    "profile_output_dir": None,
                    "config_path": None,
                    "config_sha256": None,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

    for worker_count in worker_counts:
        label = _profile_label_for_worker(int(worker_count))
        profile_output_dir = profile_root / label
        worker_config_path = _write_worker_config(
            base_config=base_config_payload,
            worker_count=int(worker_count),
            config_dir=config_dir,
            sweep_output_dir=output_dir,
            production_root_base=production_root_base,
        )
        generated_worker_configs.append(
            {
                "label": label,
                "worker_count": int(worker_count),
                "path": str(worker_config_path),
                "sha256": _sha256_file(worker_config_path),
                "payload_redacted": _redact_config_for_report(_read_yaml_mapping(worker_config_path)),
            }
        )
        existing_profile = existing_profiles.get(label)
        profile_commands.append(
            {
                "label": label,
                "worker_count": int(worker_count),
                "equivalent_profile_command": _build_profile_command(
                    source_root=source_root,
                    config_path=worker_config_path,
                    profile_output_dir=profile_output_dir,
                    fov_ids=normalized_fovs,
                    target_rounds=normalized_rounds,
                    repeats=repeats,
                    baseline_commit=baseline_commit,
                    validation_worktree=validation_worktree,
                ),
            }
        )
        try:
            json_path, markdown_path, payload, mode = _run_or_load_worker_profile(
                worker_count=int(worker_count),
                config_path=worker_config_path,
                profile_output_dir=profile_output_dir,
                fov_ids=normalized_fovs,
                target_rounds=normalized_rounds,
                repeats=repeats,
                baseline_commit=baseline_commit,
                validation_worktree=validation_worktree,
                existing_profile_json=existing_profile,
            )
            profile_validation = _validate_loaded_profile_payload(
                label=label,
                profile_payload=payload,
                expected_worker_count=int(worker_count),
                expected_fov_ids=expected_fov_ids,
                expected_target_rounds=normalized_rounds,
                expected_repeats=repeats,
                base_config_payload=base_config_payload,
                expected_config_path=worker_config_path if mode == "run" else None,
            )
            profile_payloads[label] = payload
            source = _profile_config_source(payload)
            profile_records.append(
                {
                    "label": label,
                    "worker_count": int(worker_count),
                    "status": "completed",
                    "mode": mode,
                    "profile_json_path": str(json_path),
                    "profile_markdown_path": None if markdown_path is None else str(markdown_path),
                    "profile_output_dir": _profile_output_dir(payload),
                    "config_path": profile_validation["source_config_path"],
                    "config_sha256": profile_validation["source_config_sha256"],
                    "profile_source_config_path": profile_validation["source_config_path"],
                    "profile_source_config_sha256": profile_validation["source_config_sha256"],
                    "equivalent_generated_config_path": str(worker_config_path),
                    "equivalent_generated_config_sha256": _sha256_file(worker_config_path),
                    "candidate_commit_in_profile": source.get("candidate_commit") or source.get("git_commit"),
                    "profile_validation": profile_validation,
                }
            )
        except Exception as exc:  # pragma: no cover - real-data failure evidence path
            profile_records.append(
                {
                    "label": label,
                    "worker_count": int(worker_count),
                    "status": "error",
                    "mode": "run" if existing_profile is None else "loaded_existing",
                    "profile_json_path": None if existing_profile is None else str(existing_profile),
                    "profile_markdown_path": None,
                    "profile_output_dir": str(profile_output_dir),
                    "config_path": str(worker_config_path),
                    "config_sha256": None,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

    clean_record_maps: dict[str, dict[tuple[int, str], dict[str, object]]] = {}
    directory_contracts: list[dict[str, object]] = []
    internal_equivalence: list[dict[str, object]] = []
    timing_rows: list[dict[str, object]] = []
    histogram_rows: list[dict[str, object]] = []

    for record in profile_records:
        if record.get("status") != "completed":
            continue
        label = str(record["label"])
        payload = profile_payloads[label]
        records, directories = _clean_records_for_profile(
            label=label,
            profile_payload=payload,
            compare_repeat_index=compare_repeat_index,
        )
        clean_record_maps[label] = records
        directory_contracts.extend(directories)
        worker_count_raw = record.get("worker_count")
        worker_count = None if worker_count_raw is None else int(cast(int, worker_count_raw))
        internal_equivalence.extend(_profile_internal_equivalence(label, payload))
        timing_rows.extend(_profile_timing_rows(label=label, worker_count=worker_count, profile_payload=payload))
        histogram_rows.extend(_profile_histogram_rows(label=label, worker_count=worker_count, profile_payload=payload))

    reference_label = "baseline" if "baseline" in clean_record_maps else _profile_label_for_worker(1)
    clean_comparisons: list[dict[str, object]] = []
    if reference_label in clean_record_maps:
        reference_records = clean_record_maps[reference_label]
        for label, candidate_records in sorted(clean_record_maps.items()):
            if label == reference_label:
                continue
            clean_comparisons.append(
                _compare_clean_record_maps(
                    reference_label=reference_label,
                    candidate_label=label,
                    reference_records=reference_records,
                    candidate_records=candidate_records,
                )
            )

    skipped_workers = [
        {"worker_count": int(worker), "reason": skip_reason}
        for worker in expected_worker_counts
        if int(worker) not in {int(value) for value in worker_counts}
    ]
    stage30a_verdict = _determine_verdict(
        profile_records=profile_records,
        directory_contracts=directory_contracts,
        internal_equivalence=internal_equivalence,
        clean_comparisons=clean_comparisons,
        timing_rows=timing_rows,
        histogram_rows=histogram_rows,
        skipped_workers=skipped_workers,
    )

    selected_fovs = expected_fov_ids
    selected_rounds = None if normalized_rounds is None else list(normalized_rounds)
    selected_round_ids = (
        tuple(int(round_id) for round_id in base_config.dataset.round_structure)
        if normalized_rounds is None
        else tuple(int(round_id) for round_id in normalized_rounds)
    )
    config_round_ids = tuple(int(round_id) for round_id in base_config.dataset.round_structure)
    selected_channels_by_round = {
        str(round_id): _seq_channels_for_round(base_config, int(round_id))
        for round_id in selected_round_ids
    }
    selected_channel_ids = sorted(
        {
            channel_id
            for channel_ids in selected_channels_by_round.values()
            for channel_id in channel_ids
        }
    )
    clean_gate = _stage32_clean_equivalence_gate(clean_comparisons)
    policy_surface: dict[str, object] | None = None
    clean_surface_gate: dict[str, object] | None = None
    source_consistency_gate: dict[str, object] | None = None
    if stage32_policy or stage33_fov1_policy:
        if stage33_fov1_policy:
            policy_surface = _stage33_fov1_surface_policy(
                config_fovs=config_fovs,
                selected_fovs=selected_fovs,
                config_rounds=config_round_ids,
                selected_rounds=selected_round_ids,
                policy_candidate_workers=policy_candidate_workers,
            )
            policy_label = "Stage33 FOV1"
            pass_opt_in_recommendation = (
                "Recommend native_volume_workers={best_worker} only as an opt-in setting under the validated "
                "FOV1 native preprocessing constraints; default/omitted native_volume_workers remains serial (1); "
                "no multi-FOV readiness is claimed."
            )
            pass_reason = (
                "Stage33 FOV1 correctness gates passed for FOV 1 across all configured rounds and the candidate "
                "worker setting improved measured preprocessing wall time."
            )
            limited_opt_in_recommendation = (
                "No FOV1 opt-in worker policy recommendation; the measured surface did not cover exactly FOV 1 "
                "and every configured round for the FOV1 policy surface."
            )
            limited_reason = (
                "Stage33 FOV1 did not validate the complete FOV1 policy surface; do not claim FOV1 readiness."
            )
            limited_next_debugging_task = (
                "Rerun Stage33 FOV1 with selected FOV 1 and all configured rounds, or use an isolated FOV1-only "
                "validation config whose configured rounds match the intended policy surface."
            )
            invalid_surface_opt_in_recommendation = (
                "Do not recommend native_volume_workers=4; Stage33 FOV1 policy mode only accepts selected FOV 1 "
                "and configured FOV1 policy rounds."
            )
            invalid_surface_reason = (
                "Stage33 FOV1 policy selection was invalid; FOV1 is the only policy surface and no multi-FOV "
                "or non-FOV1 readiness can be claimed."
            )
            invalid_surface_next_debugging_task = (
                "Rerun Stage33 FOV1 with --fovs 1 or omit --fovs so the helper selects FOV 1, and do not include "
                "round IDs outside the configured FOV1 policy surface."
            )
        else:
            policy_surface = _stage32_surface_policy(
                surface_scope=surface_scope,
                limited_surface_reason=limited_surface_reason,
                config_fovs=config_fovs,
                selected_fovs=selected_fovs,
                config_rounds=config_round_ids,
                selected_rounds=selected_round_ids,
                policy_candidate_workers=policy_candidate_workers,
            )
            policy_label = "Stage32"
            pass_opt_in_recommendation = None
            pass_reason = None
            limited_opt_in_recommendation = None
            limited_reason = None
            limited_next_debugging_task = None
            invalid_surface_opt_in_recommendation = None
            invalid_surface_reason = None
            invalid_surface_next_debugging_task = None
        clean_surface_gate = _stage32_clean_surface_completeness_gate(
            clean_record_maps=clean_record_maps,
            policy_surface=policy_surface,
            expected_clean_keys=_expected_clean_keys(
                selected_fovs=selected_fovs,
                selected_channels_by_round=selected_channels_by_round,
            ),
        )
        source_consistency_gate = _stage32_profile_source_consistency_gate(
            profile_payloads=profile_payloads,
            policy_surface=policy_surface,
            validation_worktree=validation_worktree,
            source_root=source_root,
        )
        verdict = _determine_stage32_policy_verdict(
            profile_records=profile_records,
            directory_contracts=directory_contracts,
            internal_equivalence=internal_equivalence,
            timing_rows=timing_rows,
            histogram_rows=histogram_rows,
            skipped_workers=skipped_workers,
            policy_surface=policy_surface,
            clean_gate=clean_gate,
            clean_surface_gate=clean_surface_gate,
            source_consistency_gate=source_consistency_gate,
            policy_label=policy_label,
            pass_opt_in_recommendation=pass_opt_in_recommendation,
            pass_reason=pass_reason,
            limited_opt_in_recommendation=limited_opt_in_recommendation,
            limited_reason=limited_reason,
            limited_next_debugging_task=limited_next_debugging_task,
            invalid_surface_opt_in_recommendation=invalid_surface_opt_in_recommendation,
            invalid_surface_reason=invalid_surface_reason,
            invalid_surface_next_debugging_task=invalid_surface_next_debugging_task,
        )
    else:
        verdict = stage30a_verdict
    contract = _sweep_contract(stage32_policy, stage33_fov1_policy=stage33_fov1_policy)
    return {
        "schema_name": contract["schema_name"],
        "schema_version": contract["schema_version"],
        "generated_at_utc": _utc_now_iso(),
        "source": {
            "source_root": str(source_root),
            "candidate_commit": _current_git_value(source_root, "rev-parse", "HEAD"),
            "candidate_subject": _current_git_value(source_root, "log", "-1", "--pretty=%s"),
            "candidate_branch": _current_git_value(source_root, "branch", "--show-current"),
            "candidate_dirty": _git_dirty(source_root),
            "candidate_status_short": _current_git_value(source_root, "status", "--short", "--branch"),
            "baseline_commit": baseline_commit,
            "baseline_profile_json": None if baseline_profile_json is None else str(baseline_profile_json),
            "validation_worktree": None if validation_worktree is None else str(validation_worktree),
            **_source_import_provenance(source_root),
            "pythonpath": os.environ.get("PYTHONPATH"),
        },
        "commands": {
            "stage30a_command": _shell_command(argv),
            "stage30a_replay_command": _stage30a_replay_command(source_root=source_root, argv=argv),
            "stage32_command": _shell_command(argv) if stage32_policy else None,
            "stage32_replay_command": _stage30a_replay_command(source_root=source_root, argv=argv) if stage32_policy else None,
            "stage33_command": _shell_command(argv) if stage33_fov1_policy else None,
            "stage33_replay_command": _stage30a_replay_command(source_root=source_root, argv=argv) if stage33_fov1_policy else None,
            "profile_invocations": profile_commands,
        },
        "resources": _system_resource_notes(
            {
                "source_root": source_root,
                "sweep_output_dir": output_dir,
                "production_root_base": production_root_base,
            }
        ),
        "validation_surface": {
            "base_config_path": str(config_path),
            "base_config_sha256": _sha256_file(config_path),
            "raw_data_path": str(base_config.dataset.raw_data_path),
            "filename_pattern": base_config.dataset.filename_pattern,
            "config_fov_ids": config_fovs,
            "selected_fov_ids": selected_fovs,
            "target_rounds": selected_rounds,
            "selected_channel_ids": selected_channel_ids,
            "selected_channel_ids_by_round": selected_channels_by_round,
            "round_structure": {str(key): list(value) for key, value in base_config.dataset.round_structure.items()},
            "channel_roles": {str(key): value for key, value in base_config.dataset.channel_roles.items()},
            "repeats": int(repeats),
            "compare_repeat_index": int(compare_repeat_index),
            "expected_worker_counts": [int(worker) for worker in expected_worker_counts],
            "requested_worker_counts": [int(worker) for worker in worker_counts],
            "skipped_workers": skipped_workers,
            "production_root_base": str(production_root_base),
            "stage32_surface_scope": surface_scope if stage32_policy else None,
            "stage32_limited_surface_reason": limited_surface_reason if stage32_policy else None,
            "stage33_fov1_policy_enabled": stage33_fov1_policy,
            "stage33_policy_fov_ids": list(STAGE33_FOV1_POLICY_FOV_IDS) if stage33_fov1_policy else None,
            "stage33_policy_surface_kind": "fov1_only" if stage33_fov1_policy else None,
        },
        "contracts": {
            "production_algorithm_changed_by_this_script": False,
            "canonical_clean_output_contract": CANONICAL_CLEAN_CONTRACT,
            "canonical_output_dirs": list(CANONICAL_OUTPUT_DIRS),
            "default_serial_behavior": "pipeline.preprocessing.native_volume_workers defaults to 1; this script writes explicit worker configs only for isolated validation runs.",
            "histogram_attribution_schema_name": HISTOGRAM_REAL_MATCH_ATTRIBUTION_SCHEMA_NAME,
            "histogram_attribution_schema_version": HISTOGRAM_REAL_MATCH_ATTRIBUTION_SCHEMA_VERSION,
            "stage32_policy_verdict_statuses": ["pass", "fail", "inconclusive"] if stage32_policy else None,
            "stage32_opt_in_candidate_workers": [int(worker) for worker in policy_candidate_workers] if stage32_policy else None,
            "stage33_fov1_policy_verdict_statuses": ["pass", "fail", "inconclusive"] if stage33_fov1_policy else None,
            "stage33_fov1_opt_in_candidate_workers": [int(worker) for worker in policy_candidate_workers] if stage33_fov1_policy else None,
            "stage33_multi_fov_readiness_claimed": False if stage33_fov1_policy else None,
        },
        "generated_worker_configs": generated_worker_configs,
        "profiles": profile_records,
        "directory_contracts": directory_contracts,
        "internal_repeat_equivalence": internal_equivalence,
        "clean_tiff_equivalence": {
            "reference_label": reference_label,
            "status": "equivalent" if clean_comparisons and all(item.get("status") == "equivalent" for item in clean_comparisons) else "not_compared" if not clean_comparisons else "mismatch",
            "stage32_exact_gate": clean_gate,
            "comparisons": clean_comparisons,
        },
        "timing_summary": timing_rows,
        "histogram_attribution_summary": histogram_rows,
        "downstream_metrics": {
            "status": "not_run",
            "reason": (
                "Stage33 FOV1 policy validation reuses preprocessing-only native calibration profiles for FOV1; Pearson/Spearman pseudobulk and matched spot statistics were not measured, so no downstream parity, full-pipeline speedup, or multi-FOV readiness claim is made."
                if stage33_fov1_policy
                else "Stage32 policy validation reuses the preprocessing-only native calibration profiles; Pearson/Spearman pseudobulk and matched spot statistics were not measured, so no downstream parity or full-pipeline speedup claim is made."
                if stage32_policy
                else "Stage30a helper runs preprocessing-only native calibration profiles; Pearson/Spearman pseudobulk and matched spot statistics were not measured."
            ),
        },
        "fail_loud_errors": [record for record in profile_records if record.get("status") == "error"],
        "worker_policy": {
            "enabled": True,
            "surface": policy_surface,
            "clean_equivalence_gate": clean_gate,
            "surface_completeness_gate": clean_surface_gate,
            "source_consistency_gate": source_consistency_gate,
            "stage30a_compatibility_verdict": stage30a_verdict,
            "default_serial_behavior_unchanged": True,
        } if policy_enabled else {"enabled": False},
        "verdict": verdict,
    }


def _format_optional(value: object) -> str:
    if value is None:
        return "not provided"
    return str(value)


def _format_float(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_sweep_markdown(payload: Mapping[str, object]) -> str:
    contract = _sweep_contract_for_payload(payload)
    label = str(contract["label"])
    source = cast(Mapping[str, object], payload["source"])
    surface = cast(Mapping[str, object], payload["validation_surface"])
    contracts = cast(Mapping[str, object], payload["contracts"])
    verdict = cast(Mapping[str, object], payload["verdict"])
    clean = cast(Mapping[str, object], payload["clean_tiff_equivalence"])
    resources = cast(Mapping[str, object], payload["resources"])
    downstream = cast(Mapping[str, object], payload["downstream_metrics"])
    worker_policy = cast(Mapping[str, object], payload.get("worker_policy", {"enabled": False}))
    commands = cast(Mapping[str, object], payload["commands"])

    lines = [
        f"# {label} Native Volume Worker Validation Report",
        "",
        f"Generated: `{payload['generated_at_utc']}`",
        "",
        "## Verdict",
        "",
        f"- Status: **{verdict.get('status')}**",
        f"- Regression flag: **{'yes' if verdict.get('regression') else 'no'}**",
        f"- Recommended worker count under tested constraints: `{_format_optional(verdict.get('recommended_worker_count'))}`",
        f"- Opt-in recommendation: {verdict.get('opt_in_recommendation', 'n/a')}",
        f"- Reason: {verdict.get('reason')}",
        "",
        "## Source Provenance",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Baseline commit | `{source.get('baseline_commit')}` |",
        f"| Baseline profile JSON | `{_format_optional(source.get('baseline_profile_json'))}` |",
        f"| Candidate source root | `{source.get('source_root')}` |",
        f"| Candidate commit | `{_format_optional(source.get('candidate_commit'))}` |",
        f"| Candidate subject | `{_format_optional(source.get('candidate_subject'))}` |",
        f"| Candidate branch | `{_format_optional(source.get('candidate_branch'))}` |",
        f"| Candidate dirty | `{_format_optional(source.get('candidate_dirty'))}` |",
        f"| Validation worktree | `{_format_optional(source.get('validation_worktree'))}` |",
        f"| PYTHONPATH env | `{_format_optional(source.get('pythonpath_env') or source.get('pythonpath'))}` |",
        f"| Source path used via PYTHONPATH/sys.path | `{_format_optional(source.get('pythonpath_effective_source_root') or source.get('source_root'))}` |",
        f"| Python executable | `{_format_optional(source.get('python_executable'))}` |",
        f"| pystar import path | `{_format_optional(source.get('pystar_import_path'))}` |",
        f"| pystar import within source root | `{source.get('pystar_import_within_source_root')}` |",
        f"| Source root first on sys.path | `{source.get('source_root_first_on_sys_path')}` |",
        f"| sys.path prefix | `{_format_optional(source.get('sys_path_prefix'))}` |",
        "",
        "Candidate status excerpt:",
        "",
        "```text",
        str(source.get("candidate_status_short") or "unavailable"),
        "```",
        "",
        "## Validation Surface",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Base config | `{surface.get('base_config_path')}` |",
        f"| Base config SHA256 | `{surface.get('base_config_sha256')}` |",
        f"| Raw data root | `{surface.get('raw_data_path')}` |",
        f"| Filename pattern | `{surface.get('filename_pattern')}` |",
        f"| Selected FOVs | `{surface.get('selected_fov_ids')}` |",
        f"| Target rounds | `{surface.get('target_rounds') or 'all configured rounds'}` |",
        f"| Selected channel IDs | `{surface.get('selected_channel_ids')}` |",
        f"| Selected channels by round | `{surface.get('selected_channel_ids_by_round')}` |",
        f"| Channel roles | `{surface.get('channel_roles')}` |",
        f"| Repeats | `{surface.get('repeats')}` |",
        f"| Compared repeat index | `{surface.get('compare_repeat_index')}` |",
        f"| Requested workers | `{surface.get('requested_worker_counts')}` |",
        f"| Expected workers | `{surface.get('expected_worker_counts')}` |",
        f"| Skipped workers | `{surface.get('skipped_workers')}` |",
        f"| Stage32 surface scope | `{surface.get('stage32_surface_scope')}` |",
        f"| Stage32 limited-surface reason | `{surface.get('stage32_limited_surface_reason')}` |",
        f"| Stage33 FOV1 policy enabled | `{surface.get('stage33_fov1_policy_enabled')}` |",
        f"| Stage33 policy FOV IDs | `{surface.get('stage33_policy_fov_ids')}` |",
        f"| Stage33 policy surface kind | `{surface.get('stage33_policy_surface_kind')}` |",
        f"| CPU count | `{resources.get('cpu_count')}` |",
        f"| MemTotal KiB | `{resources.get('mem_total_kib')}` |",
        f"| Disk usage | `{resources.get('disk_usage')}` |",
        "",
        "## Contract Guardrails",
        "",
        f"- Production algorithm changed by this script: `{contracts.get('production_algorithm_changed_by_this_script')}`",
        f"- Canonical clean TIFF contract: `{contracts.get('canonical_clean_output_contract')}`",
        f"- Canonical output dirs: `{contracts.get('canonical_output_dirs')}`",
        f"- Default serial/off behavior: {contracts.get('default_serial_behavior')}",
        f"- Histogram attribution schema: `{contracts.get('histogram_attribution_schema_name')}` v`{contracts.get('histogram_attribution_schema_version')}`",
        "",
        "## Commands",
        "",
        f"{label} command:",
        "",
        "```bash",
        str(commands.get("stage33_command") or commands.get("stage32_command") or commands.get("stage30a_command")),
        "```",
        "",
        "Replay command with explicit source path:",
        "",
        "```bash",
        str(commands.get("stage33_replay_command") or commands.get("stage32_replay_command") or commands.get("stage30a_replay_command")),
        "```",
        "",
        "Equivalent profile invocations generated by the sweep helper:",
        "",
    ]
    if worker_policy.get("enabled"):
        policy_surface = cast(Mapping[str, object], worker_policy.get("surface", {}))
        clean_gate = cast(Mapping[str, object], worker_policy.get("clean_equivalence_gate", {}))
        surface_gate = cast(Mapping[str, object], worker_policy.get("surface_completeness_gate", {}))
        source_gate = cast(Mapping[str, object], worker_policy.get("source_consistency_gate", {}))
        lines.extend(
            [
                f"## {label} Worker Policy Gate",
                "",
                f"- Declared surface scope: `{policy_surface.get('declared_surface_scope')}`",
                f"- Effective surface scope: `{policy_surface.get('effective_surface_scope')}`",
                f"- Full-surface policy ready: `{policy_surface.get('full_surface_policy_ready')}`",
                f"- Baseline worker count: `{policy_surface.get('baseline_worker_count')}`",
                f"- Candidate worker counts: `{policy_surface.get('policy_candidate_workers')}`",
                f"- Required worker counts: `{policy_surface.get('required_worker_counts')}`",
                f"- Clean exact-equivalence gate: `{clean_gate.get('status')}`",
                f"- Missing clean TIFF count: `{clean_gate.get('missing_count')}`",
                f"- Mismatch count: `{clean_gate.get('mismatch_count')}`",
                f"- Shape drift count: `{clean_gate.get('shape_drift_count')}`",
                f"- Dtype drift count: `{clean_gate.get('dtype_drift_count')}`",
                f"- Max absolute difference: `{clean_gate.get('max_abs_diff')}`",
                f"- Expected clean TIFF surface gate: `{surface_gate.get('status')}`",
                f"- Expected clean TIFF count: `{surface_gate.get('expected_clean_tiff_count')}`",
                f"- Missing expected clean TIFF count: `{surface_gate.get('missing_expected_clean_tiff_count')}`",
                f"- Unexpected clean TIFF count: `{surface_gate.get('unexpected_clean_tiff_count')}`",
                f"- Source consistency gate: `{source_gate.get('status')}`",
                "",
                "Policy surface notes:",
                "",
            ]
        )
        for reason in cast(Sequence[object], policy_surface.get("reasons", [])):
            lines.append(f"- {reason}")
        lines.append("")
    for command in cast(Sequence[Mapping[str, object]], commands["profile_invocations"]):
        lines.extend(
            [
                f"### {command.get('label')}",
                "",
                "```bash",
                str(command.get("equivalent_profile_command")),
                "```",
                "",
            ]
        )

    generated_configs = cast(Sequence[Mapping[str, object]], payload.get("generated_worker_configs", []))
    if generated_configs:
        lines.extend(
            [
                "## Generated Worker Configs",
                "",
                "| Label | Workers | Path | SHA256 | Output directory |",
                "| --- | ---: | --- | --- | --- |",
            ]
        )
        for config_record in generated_configs:
            config_payload = cast(Mapping[str, object], config_record.get("payload_redacted", {}))
            pipeline = cast(Mapping[str, object], config_payload.get("pipeline", {}))
            output = cast(Mapping[str, object], pipeline.get("output", {}))
            lines.append(
                "| {label} | {workers} | `{path}` | `{sha}` | `{output_dir}` |".format(
                    label=config_record.get("label"),
                    workers=config_record.get("worker_count"),
                    path=config_record.get("path"),
                    sha=config_record.get("sha256"),
                    output_dir=output.get("directory"),
                )
            )
        lines.append("")

    lines.extend(
        [
            "## Worker Profile Artifacts",
            "",
            "| Label | Workers | Status | Mode | Profile source config | Equivalent generated config | Profile JSON | Profile Markdown | Output root | Error |",
            "| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for record in cast(Sequence[Mapping[str, object]], payload["profiles"]):
        lines.append(
            "| {label} | {workers} | {status} | {mode} | `{source_config}` | `{generated_config}` | `{json}` | `{markdown}` | `{root}` | {error} |".format(
                label=record.get("label"),
                workers=_format_optional(record.get("worker_count")),
                status=record.get("status"),
                mode=record.get("mode"),
                source_config=_format_optional(record.get("profile_source_config_path") or record.get("config_path")),
                generated_config=_format_optional(record.get("equivalent_generated_config_path")),
                json=_format_optional(record.get("profile_json_path")),
                markdown=_format_optional(record.get("profile_markdown_path")),
                root=_format_optional(record.get("profile_output_dir")),
                error=_format_optional(record.get("error_message")) if record.get("status") == "error" else "",
            )
        )
    lines.append("")

    lines.extend(
        [
            "## Clean TIFF Equivalence",
            "",
            f"Reference label: `{clean.get('reference_label')}`",
            "",
            "| Comparison | Status | Files compared | Missing | Extra | Mismatches | Shape drift | Dtype drift | Max abs diff |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for comparison in cast(Sequence[Mapping[str, object]], clean.get("comparisons", [])):
        lines.append(
            "| {ref} vs {cand} | {status} | {files} | {missing} | {extra} | {mismatches} | {shape_drift} | {dtype_drift} | {max_abs_diff} |".format(
                ref=comparison.get("reference_label"),
                cand=comparison.get("candidate_label"),
                status=comparison.get("status"),
                files=comparison.get("files_compared"),
                missing=comparison.get("missing_count", len(cast(Sequence[object], comparison.get("missing_files", [])))),
                extra=comparison.get("extra_count", len(cast(Sequence[object], comparison.get("extra_files", [])))),
                mismatches=comparison.get("mismatch_count"),
                shape_drift=comparison.get("shape_drift_count"),
                dtype_drift=comparison.get("dtype_drift_count"),
                max_abs_diff=comparison.get("max_abs_diff"),
            )
        )
    if not cast(Sequence[object], clean.get("comparisons", [])):
        lines.append("| n/a | not_compared | 0 | 0 | 0 | 0 | 0 | 0 | n/a |")
    lines.append("")

    first_comparison = next(iter(cast(Sequence[Mapping[str, object]], clean.get("comparisons", []))), None)
    if first_comparison is not None:
        rows = cast(Sequence[Mapping[str, object]], first_comparison.get("file_rows", []))[:20]
        lines.extend(
            [
                "Representative clean TIFF rows from the first comparison:",
                "",
                "| Relative path | Shape equal | Dtype equal | Size equal | SHA equal | Array equal | Max abs diff | Status |",
                "| --- | --- | --- | --- | --- | --- | ---: | --- |",
            ]
        )
        for row in rows:
            lines.append(
                "| `{path}` | {shape} | {dtype} | {size} | {sha} | {array} | {max_abs_diff} | {status} |".format(
                    path=row.get("relative_path"),
                    shape=row.get("shape_equal"),
                    dtype=row.get("dtype_equal"),
                    size=row.get("size_bytes_equal"),
                    sha=row.get("sha256_equal"),
                    array=row.get("array_equal"),
                    max_abs_diff=row.get("max_abs_diff"),
                    status=row.get("status"),
                )
            )
        lines.append("")

    lines.extend(
        [
            "## Internal Repeat Equivalence",
            "",
            "| Label | FOV | Status | File count | Mismatch count |",
            "| --- | ---: | --- | ---: | ---: |",
        ]
    )
    for row in cast(Sequence[Mapping[str, object]], payload["internal_repeat_equivalence"]):
        lines.append(
            f"| {row.get('label')} | {row.get('fov_id')} | {row.get('status')} | {row.get('file_count')} | {row.get('mismatch_count')} |"
        )
    lines.append("")

    lines.extend(
        [
            "## Histogram Attribution Summary",
            "",
            "| Label | Workers | FOV | Calls | Real matches | No-ref no-ops | Real total ms | Real median ms | No-op total ms | No-op median ms |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in cast(Sequence[Mapping[str, object]], payload["histogram_attribution_summary"]):
        lines.append(
            "| {label} | {workers} | {fov} | {calls} | {real} | {noop} | {real_total} | {real_median} | {noop_total} | {noop_median} |".format(
                label=row.get("label"),
                workers=_format_optional(row.get("worker_count")),
                fov=row.get("fov_id"),
                calls=row.get("call_count"),
                real=row.get("real_match_call_count"),
                noop=row.get("no_reference_noop_call_count"),
                real_total=_format_float(row.get("real_match_total_duration_ms")),
                real_median=_format_float(row.get("real_match_median_duration_ms")),
                noop_total=_format_float(row.get("no_reference_noop_total_duration_ms")),
                noop_median=_format_float(row.get("no_reference_noop_median_duration_ms")),
            )
        )
    lines.append("")

    lines.extend(
        [
            "## Timing Summary",
            "",
            "Timing is interpreted only after clean-output equivalence gates pass. Wall time is the worker-selection metric for concurrent scheduling; summed per-volume timing can increase under parallel execution and is not by itself a regression.",
            "",
            "| Label | Workers | FOV | Repeats | Wall total ms | Wall median ms | Volume total ms | Calibration total ms | Extraction total ms |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in cast(Sequence[Mapping[str, object]], payload["timing_summary"]):
        lines.append(
            "| {label} | {workers} | {fov} | {repeats} | {wall_total} | {wall_median} | {volume_total} | {calibration_total} | {extraction_total} |".format(
                label=row.get("label"),
                workers=_format_optional(row.get("worker_count")),
                fov=row.get("fov_id"),
                repeats=row.get("repeat_count"),
                wall_total=_format_float(row.get("wall_total_ms")),
                wall_median=_format_float(row.get("wall_median_ms")),
                volume_total=_format_float(row.get("volume_total_ms")),
                calibration_total=_format_float(row.get("calibration_total_ms")),
                extraction_total=_format_float(row.get("extraction_total_ms")),
            )
        )
    lines.append("")

    lines.extend(
        [
            "## Downstream Metrics",
            "",
            f"- Status: `{downstream.get('status')}`",
            f"- Reason: {downstream.get('reason')}",
            "",
            "## Fail-Loud Errors",
            "",
        ]
    )
    errors = cast(Sequence[Mapping[str, object]], payload.get("fail_loud_errors", []))
    if not errors:
        lines.append("No worker profile errors were recorded by the Stage30a helper.")
    else:
        lines.extend(["| Label | Workers | Error type | Message |", "| --- | ---: | --- | --- |"])
        for error in errors:
            lines.append(
                f"| {error.get('label')} | {_format_optional(error.get('worker_count'))} | {error.get('error_type')} | {error.get('error_message')} |"
            )
    lines.append("")

    lines.extend(
        [
            "## Decision Rule",
            "",
            "If any clean output array differs, reject the speedup regardless of timing. If all compared arrays are exact-equivalent and one worker count improves wall time, treat that worker count as an opt-in validated setting for this measured surface only; this script does not change the default serial behavior.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_sweep_artifacts(payload: dict[str, object], output_dir: Path) -> tuple[Path, Path]:
    contract = _sweep_contract_for_payload(payload)
    stage32_policy = payload.get("schema_name") == STAGE32_POLICY_SCHEMA_NAME
    stage33_fov1_policy = payload.get("schema_name") == STAGE33_FOV1_POLICY_SCHEMA_NAME
    label = str(contract["label"])
    report_filenames = cast(Mapping[str, str], contract["report_filenames"])
    output_dir = _prepare_sweep_output_dir(
        output_dir,
        stage32_policy=stage32_policy,
        stage33_fov1_policy=stage33_fov1_policy,
    )
    json_path = output_dir / report_filenames["json"]
    markdown_path = output_dir / report_filenames["markdown"]
    temp_json_path = json_path.with_name(f"{json_path.name}.tmp")
    temp_markdown_path = markdown_path.with_name(f"{markdown_path.name}.tmp")
    _reject_symlink_components(json_path, field_name=f"{label} JSON report path")
    _reject_symlink_components(markdown_path, field_name=f"{label} Markdown report path")
    _reject_symlink_components(temp_json_path, field_name=f"temporary {label} JSON report path")
    _reject_symlink_components(temp_markdown_path, field_name=f"temporary {label} Markdown report path")
    _assert_path_within(json_path, output_dir, field_name=f"{label} JSON report path")
    _assert_path_within(markdown_path, output_dir, field_name=f"{label} Markdown report path")
    _assert_path_within(temp_json_path, output_dir, field_name=f"temporary {label} JSON report path")
    _assert_path_within(temp_markdown_path, output_dir, field_name=f"temporary {label} Markdown report path")
    write_backend_metadata(temp_json_path, payload)
    _ = temp_json_path.replace(json_path)
    _ = temp_markdown_path.write_text(render_sweep_markdown(payload), encoding="utf-8")
    _ = temp_markdown_path.replace(markdown_path)
    return json_path, markdown_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run or assemble Stage30a/Stage32/Stage33 real-data evidence for the native preprocessing "
            "volume-worker sweep and opt-in worker policy. Outputs are validation-only JSON/Markdown reports."
        )
    )
    _ = parser.add_argument("--config", required=True, type=Path, help="Base experiment YAML config shared by all worker runs.")
    _ = parser.add_argument("--output-dir", required=True, type=Path, help="Dedicated Stage30a/Stage32/Stage33 evidence output directory.")
    _ = parser.add_argument("--workers", default="1,2,3,4", help="Comma-separated worker counts to run or load. Defaults to 1,2,3,4.")
    _ = parser.add_argument("--expected-workers", default="1,2,3,4", help="Expected sweep values for skipped-worker documentation.")
    _ = parser.add_argument("--fovs", default=None, help="Optional comma-separated FOV ids. Defaults to the first configured FOV in the profile harness.")
    _ = parser.add_argument("--rounds", default=None, help="Optional comma-separated round ids. Defaults to all configured rounds.")
    _ = parser.add_argument("--repeats", type=int, default=2, help="Repeat count for each profile run.")
    _ = parser.add_argument("--compare-repeat-index", type=int, default=0, help="Repeat index used for cross-worker clean TIFF comparison.")
    _ = parser.add_argument("--baseline-commit", default=DEFAULT_BASELINE_COMMIT, help="Baseline commit hash to record in reports.")
    _ = parser.add_argument("--baseline-profile-json", type=Path, default=None, help="Optional existing baseline profile JSON to use as clean TIFF reference.")
    _ = parser.add_argument(
        "--existing-profile",
        action="append",
        default=[],
        help="Optional LABEL=PATH profile JSON. Numeric labels map to workers_N, e.g. 4=/path/profile.json.",
    )
    _ = parser.add_argument("--validation-worktree", type=Path, default=None, help="Validation worktree path recorded in the report.")
    _ = parser.add_argument("--production-root-base", type=Path, default=None, help="Base path for generated config pipeline.output.directory values. Defaults to a sibling of --output-dir.")
    _ = parser.add_argument("--source-root", type=Path, default=REPO_ROOT, help="PyStar source root to record and use in generated command strings.")
    _ = parser.add_argument("--skip-reason", default="not requested in this bounded run", help="Reason recorded for expected worker counts omitted from --workers.")
    _ = parser.add_argument(
        "--stage32-policy",
        action="store_true",
        help="Emit the Stage32 pass/fail/inconclusive policy payload and require serial/default vs candidate worker policy gates.",
    )
    _ = parser.add_argument(
        "--stage33-fov1-policy",
        action="store_true",
        help="Emit the Stage33 FOV1-only policy payload. This validates FOV1 readiness only and never claims multi-FOV readiness.",
    )
    _ = parser.add_argument(
        "--policy-candidate-workers",
        default="4",
        help="Comma-separated opt-in candidate worker counts for Stage32/Stage33 policy mode. Defaults to 4.",
    )
    _ = parser.add_argument(
        "--surface-scope",
        choices=STAGE32_SURFACE_SCOPES,
        default="limited",
        help="Stage32 surface declaration. Use 'full' only when all configured FOVs and rounds were measured.",
    )
    _ = parser.add_argument(
        "--limited-surface-reason",
        default="bounded validation surface; full configured round surface was not asserted",
        help="Reason recorded when Stage32 policy mode is run on a limited representative surface.",
    )
    _ = parser.add_argument(
        "--allow-non-pass-exit-zero",
        action="store_true",
        help="Write evidence and exit 0 even when the validation verdict is not pass.",
    )
    args = parser.parse_args()

    worker_counts = _parse_positive_int_list(cast(str, args.workers), field_name="worker count")
    expected_worker_counts = _parse_positive_int_list(cast(str, args.expected_workers), field_name="expected worker count")
    policy_candidate_workers = _parse_positive_int_list(cast(str, args.policy_candidate_workers), field_name="policy candidate worker count")
    fov_ids = _parse_nonnegative_int_list(cast(str | None, args.fovs), field_name="FOV id")
    target_rounds = _parse_nonnegative_int_list(cast(str | None, args.rounds), field_name="round id")
    output_dir = cast(Path, args.output_dir)
    production_root_base = cast(Path | None, args.production_root_base)
    if production_root_base is None:
        production_root_base = output_dir.with_name(f"{output_dir.name}_production_roots")

    payload = build_sweep_payload(
        config_path=cast(Path, args.config),
        output_dir=output_dir,
        worker_counts=worker_counts,
        expected_worker_counts=expected_worker_counts,
        fov_ids=fov_ids,
        target_rounds=target_rounds,
        repeats=cast(int, args.repeats),
        compare_repeat_index=cast(int, args.compare_repeat_index),
        baseline_commit=cast(str, args.baseline_commit),
        validation_worktree=cast(Path | None, args.validation_worktree),
        production_root_base=production_root_base,
        source_root=cast(Path, args.source_root).expanduser().resolve(strict=True),
        existing_profiles=_parse_existing_profile_specs(cast(Sequence[str], args.existing_profile)),
        baseline_profile_json=cast(Path | None, args.baseline_profile_json),
        skip_reason=cast(str, args.skip_reason),
        argv=sys.argv,
        stage32_policy=cast(bool, args.stage32_policy),
        stage33_fov1_policy=cast(bool, args.stage33_fov1_policy),
        policy_candidate_workers=policy_candidate_workers,
        surface_scope=cast(str, args.surface_scope),
        limited_surface_reason=cast(str, args.limited_surface_reason),
    )
    json_path, markdown_path = write_sweep_artifacts(payload, output_dir)
    verdict = cast(Mapping[str, object], payload["verdict"])
    label = str(_sweep_contract_for_payload(payload)["label"])
    print(f"{label} worker validation JSON: {json_path}")
    print(f"{label} worker validation Markdown: {markdown_path}")
    print(f"{label} verdict: {verdict.get('status')} (regression={verdict.get('regression')})")
    if verdict.get("status") != "pass" and not cast(bool, args.allow_non_pass_exit_zero):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
