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
        Path(str(module_file)).resolve().relative_to(source_root.resolve())
    except ValueError:
        for module_name in list(sys.modules):
            if module_name == "pystar" or module_name.startswith("pystar."):
                del sys.modules[module_name]


_prefer_source_root_for_pystar(REPO_ROOT)

from pystar.infrastructure import load_config
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


def _prepare_sweep_output_dir(output_dir: Path) -> Path:
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
            "--output-dir must be a dedicated Stage30a evidence directory, "
            f"not an unsafe root: {output_path}"
        )

    marker_path = output_path / STAGE30A_OUTPUT_MARKER
    if marker_path.exists():
        if marker_path.is_symlink() or not marker_path.is_file():
            raise ValueError(f"Stage30a output marker is invalid: {marker_path}")
        marker_text = marker_path.read_text(encoding="utf-8")
        expected = _sweep_output_marker_text()
        if marker_text != expected:
            raise ValueError(f"Stage30a output marker schema drifted: {marker_path}")
    else:
        entries = list(output_path.iterdir())
        if entries:
            raise ValueError(
                "--output-dir exists and is not an empty Stage30a sweep output directory. "
                f"Choose an empty/dedicated directory or one containing {STAGE30A_OUTPUT_MARKER}: {output_path}"
            )
        marker_path.write_text(_sweep_output_marker_text(), encoding="utf-8")
    return output_resolved


def _sweep_output_marker_text() -> str:
    return f"schema_name={STAGE30A_SWEEP_SCHEMA_NAME}\nschema_version={STAGE30A_SWEEP_SCHEMA_VERSION}\n"


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
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
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


def _arrays_equal_for_records(reference: Mapping[str, object], candidate: Mapping[str, object]) -> tuple[bool | None, str | None]:
    try:
        reference_array = _read_tiff_array(Path(str(reference["path"])))
        candidate_array = _read_tiff_array(Path(str(candidate["path"])))
        return bool(np.array_equal(reference_array, candidate_array)), None
    except Exception as exc:  # pragma: no cover - exercised by real-data validation failures
        return None, f"{type(exc).__name__}: {exc}"


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

    for key in sorted(reference_keys & candidate_keys):
        reference = reference_records[key]
        candidate = candidate_records[key]
        shape_equal = reference["shape"] == candidate["shape"]
        dtype_equal = reference["dtype"] == candidate["dtype"]
        size_equal = reference["size_bytes"] == candidate["size_bytes"]
        sha_equal = reference["sha256"] == candidate["sha256"]
        array_equal, array_error = _arrays_equal_for_records(reference, candidate)
        equivalent = bool(shape_equal and dtype_equal and array_equal is True)
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
                "array_compare_error": array_error,
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

    status = "equivalent" if not missing and not extra and mismatch_count == 0 else "mismatch"
    return {
        "reference_label": reference_label,
        "candidate_label": candidate_label,
        "status": status,
        "files_compared": len(reference_keys & candidate_keys),
        "missing_files": [f"FOV {fov_id}: {relative_path}" for fov_id, relative_path in missing],
        "extra_files": [f"FOV {fov_id}: {relative_path}" for fov_id, relative_path in extra],
        "mismatch_count": mismatch_count,
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
        if isinstance(raw_worker_count, bool) or not isinstance(raw_worker_count, int):
            raise ValueError(
                f"Profile {label} source config must explicitly set integer "
                f"pipeline.preprocessing.native_volume_workers; got {raw_worker_count!r}"
            )
        if int(raw_worker_count) != int(expected_worker_count):
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


def _system_resource_notes() -> dict[str, object]:
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
    }


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
            module_path.relative_to(source_root_resolved)
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
) -> dict[str, object]:
    if repeats <= 0:
        raise ValueError(f"--repeats must be positive; got {repeats!r}")
    if compare_repeat_index < 0:
        raise ValueError(f"--compare-repeat-index must be non-negative; got {compare_repeat_index!r}")
    if compare_repeat_index >= repeats:
        raise ValueError(
            "--compare-repeat-index must reference one of the generated profile repeats; "
            f"got {compare_repeat_index!r} with --repeats={repeats!r}"
        )

    output_dir = _prepare_sweep_output_dir(output_dir)
    config_path = config_path.expanduser().resolve(strict=True)
    base_config_payload = _read_yaml_mapping(config_path)
    base_config = load_config(str(config_path))
    config_fovs = [int(value) for value in base_config.dataset.parsed_fovs]
    normalized_fovs = None if fov_ids is None else tuple(int(value) for value in fov_ids)
    normalized_rounds = None if target_rounds is None else tuple(int(value) for value in target_rounds)
    expected_fov_ids = config_fovs[:1] if normalized_fovs is None else list(normalized_fovs)

    profile_records: list[dict[str, object]] = []
    profile_payloads: dict[str, dict[str, object]] = {}
    profile_commands: list[dict[str, object]] = []
    config_dir = output_dir / "configs"
    profile_root = output_dir / "profiles"
    profile_root.mkdir(parents=True, exist_ok=True)
    production_root_base = production_root_base.expanduser().resolve(strict=False)

    if baseline_profile_json is not None:
        baseline_path = baseline_profile_json.expanduser().resolve(strict=True)
        try:
            baseline_payload = _read_json_mapping(baseline_path)
            baseline_validation = _validate_loaded_profile_payload(
                label="baseline",
                profile_payload=baseline_payload,
                expected_worker_count=None,
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
    verdict = _determine_verdict(
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
    selected_channels_by_round = {
        str(round_id): [
            int(channel_id)
            for channel_id in base_config.dataset.round_structure.get(int(round_id), [])
        ]
        for round_id in selected_round_ids
    }
    selected_channel_ids = sorted(
        {
            channel_id
            for channel_ids in selected_channels_by_round.values()
            for channel_id in channel_ids
        }
    )
    return {
        "schema_name": STAGE30A_SWEEP_SCHEMA_NAME,
        "schema_version": STAGE30A_SWEEP_SCHEMA_VERSION,
        "generated_at_utc": _utc_now_iso(),
        "source": {
            "source_root": str(source_root),
            "candidate_commit": _current_git_value(source_root, "rev-parse", "HEAD"),
            "candidate_subject": _current_git_value(source_root, "log", "-1", "--pretty=%s"),
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
            "profile_invocations": profile_commands,
        },
        "resources": _system_resource_notes(),
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
        },
        "contracts": {
            "production_algorithm_changed_by_this_script": False,
            "canonical_clean_output_contract": CANONICAL_CLEAN_CONTRACT,
            "canonical_output_dirs": list(CANONICAL_OUTPUT_DIRS),
            "default_serial_behavior": "pipeline.preprocessing.native_volume_workers defaults to 1; this script writes explicit worker configs only for isolated validation runs.",
            "histogram_attribution_schema_name": HISTOGRAM_REAL_MATCH_ATTRIBUTION_SCHEMA_NAME,
            "histogram_attribution_schema_version": HISTOGRAM_REAL_MATCH_ATTRIBUTION_SCHEMA_VERSION,
        },
        "profiles": profile_records,
        "directory_contracts": directory_contracts,
        "internal_repeat_equivalence": internal_equivalence,
        "clean_tiff_equivalence": {
            "reference_label": reference_label,
            "status": "equivalent" if clean_comparisons and all(item.get("status") == "equivalent" for item in clean_comparisons) else "not_compared" if not clean_comparisons else "mismatch",
            "comparisons": clean_comparisons,
        },
        "timing_summary": timing_rows,
        "histogram_attribution_summary": histogram_rows,
        "downstream_metrics": {
            "status": "not_run",
            "reason": "Stage30a helper runs preprocessing-only native calibration profiles; Pearson/Spearman pseudobulk and matched spot statistics were not measured.",
        },
        "fail_loud_errors": [record for record in profile_records if record.get("status") == "error"],
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
    source = cast(Mapping[str, object], payload["source"])
    surface = cast(Mapping[str, object], payload["validation_surface"])
    contracts = cast(Mapping[str, object], payload["contracts"])
    verdict = cast(Mapping[str, object], payload["verdict"])
    clean = cast(Mapping[str, object], payload["clean_tiff_equivalence"])
    resources = cast(Mapping[str, object], payload["resources"])
    downstream = cast(Mapping[str, object], payload["downstream_metrics"])

    lines = [
        "# Stage30a Native Volume Worker Sweep Validation Report",
        "",
        f"Generated: `{payload['generated_at_utc']}`",
        "",
        "## Verdict",
        "",
        f"- Status: **{verdict.get('status')}**",
        f"- Regression flag: **{'yes' if verdict.get('regression') else 'no'}**",
        f"- Recommended worker count under tested constraints: `{_format_optional(verdict.get('recommended_worker_count'))}`",
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
        f"| CPU count | `{resources.get('cpu_count')}` |",
        f"| MemTotal KiB | `{resources.get('mem_total_kib')}` |",
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
        "Stage30a command:",
        "",
        "```bash",
        str(cast(Mapping[str, object], payload["commands"]).get("stage30a_command")),
        "```",
        "",
        "Replay command with explicit source path:",
        "",
        "```bash",
        str(cast(Mapping[str, object], payload["commands"]).get("stage30a_replay_command")),
        "```",
        "",
        "Equivalent profile invocations generated by the sweep helper:",
        "",
    ]
    for command in cast(Sequence[Mapping[str, object]], cast(Mapping[str, object], payload["commands"])["profile_invocations"]):
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
            "| Comparison | Status | Files compared | Missing | Extra | Mismatches |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for comparison in cast(Sequence[Mapping[str, object]], clean.get("comparisons", [])):
        lines.append(
            "| {ref} vs {cand} | {status} | {files} | {missing} | {extra} | {mismatches} |".format(
                ref=comparison.get("reference_label"),
                cand=comparison.get("candidate_label"),
                status=comparison.get("status"),
                files=comparison.get("files_compared"),
                missing=len(cast(Sequence[object], comparison.get("missing_files", []))),
                extra=len(cast(Sequence[object], comparison.get("extra_files", []))),
                mismatches=comparison.get("mismatch_count"),
            )
        )
    if not cast(Sequence[object], clean.get("comparisons", [])):
        lines.append("| n/a | not_compared | 0 | 0 | 0 | 0 |")
    lines.append("")

    first_comparison = next(iter(cast(Sequence[Mapping[str, object]], clean.get("comparisons", []))), None)
    if first_comparison is not None:
        rows = cast(Sequence[Mapping[str, object]], first_comparison.get("file_rows", []))[:20]
        lines.extend(
            [
                "Representative clean TIFF rows from the first comparison:",
                "",
                "| Relative path | Shape equal | Dtype equal | Size equal | SHA equal | Array equal | Status |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in rows:
            lines.append(
                "| `{path}` | {shape} | {dtype} | {size} | {sha} | {array} | {status} |".format(
                    path=row.get("relative_path"),
                    shape=row.get("shape_equal"),
                    dtype=row.get("dtype_equal"),
                    size=row.get("size_bytes_equal"),
                    sha=row.get("sha256_equal"),
                    array=row.get("array_equal"),
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
    output_dir = _prepare_sweep_output_dir(output_dir)
    json_path = output_dir / STAGE30A_REPORT_FILENAMES["json"]
    markdown_path = output_dir / STAGE30A_REPORT_FILENAMES["markdown"]
    temp_json_path = json_path.with_name(f"{json_path.name}.tmp")
    temp_markdown_path = markdown_path.with_name(f"{markdown_path.name}.tmp")
    _reject_symlink_components(json_path, field_name="Stage30a JSON report path")
    _reject_symlink_components(markdown_path, field_name="Stage30a Markdown report path")
    _reject_symlink_components(temp_json_path, field_name="temporary Stage30a JSON report path")
    _reject_symlink_components(temp_markdown_path, field_name="temporary Stage30a Markdown report path")
    _assert_path_within(json_path, output_dir, field_name="Stage30a JSON report path")
    _assert_path_within(markdown_path, output_dir, field_name="Stage30a Markdown report path")
    _assert_path_within(temp_json_path, output_dir, field_name="temporary Stage30a JSON report path")
    _assert_path_within(temp_markdown_path, output_dir, field_name="temporary Stage30a Markdown report path")
    write_backend_metadata(temp_json_path, payload)
    _ = temp_json_path.replace(json_path)
    temp_markdown_path.write_text(render_sweep_markdown(payload), encoding="utf-8")
    _ = temp_markdown_path.replace(markdown_path)
    return json_path, markdown_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run or assemble Stage30a real-data evidence for the native preprocessing "
            "volume-worker sweep. Outputs are validation-only JSON/Markdown reports."
        )
    )
    _ = parser.add_argument("--config", required=True, type=Path, help="Base experiment YAML config shared by all worker runs.")
    _ = parser.add_argument("--output-dir", required=True, type=Path, help="Dedicated Stage30a evidence output directory.")
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
        "--allow-non-pass-exit-zero",
        action="store_true",
        help="Write evidence and exit 0 even when the validation verdict is not pass.",
    )
    args = parser.parse_args()

    worker_counts = _parse_positive_int_list(cast(str, args.workers), field_name="worker count")
    expected_worker_counts = _parse_positive_int_list(cast(str, args.expected_workers), field_name="expected worker count")
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
    )
    json_path, markdown_path = write_sweep_artifacts(payload, output_dir)
    verdict = cast(Mapping[str, object], payload["verdict"])
    print(f"Stage30a worker sweep JSON: {json_path}")
    print(f"Stage30a worker sweep Markdown: {markdown_path}")
    print(f"Stage30a verdict: {verdict.get('status')} (regression={verdict.get('regression')})")
    if verdict.get("status") != "pass" and not cast(bool, args.allow_non_pass_exit_zero):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
