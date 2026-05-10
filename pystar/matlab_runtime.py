"""Shared helpers for explicit MATLAB runtime validation and provenance."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import tifffile


MATLAB_RUNTIME_MANIFEST_NAME = "runtime_manifest.json"
RuntimeManifest = Mapping[str, object]
RuntimeFileRecord = dict[str, object]


def repo_root() -> Path:
    """Return the repository root that owns repo-local MATLAB runtime bundles."""

    return Path(__file__).resolve().parents[1]


def trusted_matlab_runtime_root() -> Path:
    """Return the trusted repo-local root for MATLAB runtime bundles."""

    return (repo_root() / "matlab_runtime").resolve()


def sha256_file(path: Path) -> str:
    """Hash a runtime file using the canonical ``sha256:<hex>`` format."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def format_exception_message(prefix: str, exc: Exception) -> str:
    """Format a fail-loud exception message without hiding engine details."""

    detail = str(exc).strip()
    if detail:
        return f"{prefix}: {detail}"
    return f"{prefix} ({exc.__class__.__name__})"


def resolve_repo_runtime_path(
    configured_path: Path,
    *,
    config_label: str,
    trusted_root: Path,
) -> Path:
    """Resolve a configured runtime path and keep it inside a trusted root."""

    runtime_path = configured_path
    if not runtime_path.is_absolute():
        runtime_path = repo_root() / runtime_path

    resolved_runtime_path = runtime_path.resolve()
    resolved_trusted_root = trusted_root.resolve()
    try:
        _ = resolved_runtime_path.relative_to(resolved_trusted_root)
    except ValueError as exc:
        raise ValueError(
            f"{config_label} must resolve inside the repo-local '{resolved_trusted_root}' runtime root; got {resolved_runtime_path}"
        ) from exc
    return resolved_runtime_path


def load_runtime_manifest_json(
    runtime_dir: Path,
    *,
    manifest_label: str,
    missing_hint: str,
    manifest_name: str = MATLAB_RUNTIME_MANIFEST_NAME,
) -> dict[str, object]:
    """Load a runtime manifest JSON object with stage-specific wording."""

    manifest_path = (runtime_dir / manifest_name).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_label} is missing: {manifest_path}. {missing_hint}")

    resolved_runtime_dir = runtime_dir.resolve()
    try:
        _ = manifest_path.relative_to(resolved_runtime_dir)
    except ValueError as exc:
        raise ValueError(
            f"{manifest_label} resolves outside the runtime directory: {manifest_path}"
        ) from exc

    raw_manifest = cast(object, json.loads(manifest_path.read_text(encoding="utf-8")))
    if not isinstance(raw_manifest, dict):
        raise ValueError(f"{manifest_label} must be a JSON object: {manifest_path}")

    manifest_mapping = cast(Mapping[object, object], raw_manifest)
    return {str(key): value for key, value in manifest_mapping.items()}


def validate_runtime_manifest_file_buckets(
    runtime_manifest: RuntimeManifest,
    *,
    manifest_label: str,
) -> tuple[list[object], list[object]]:
    """Validate the common required/optional runtime manifest buckets."""

    required_files = runtime_manifest.get("required_files")
    optional_files = runtime_manifest.get("optional_files", [])
    if not isinstance(required_files, list) or not required_files:
        raise ValueError(f"{manifest_label} must declare a non-empty required_files list")
    if not isinstance(optional_files, list):
        raise ValueError(f"{manifest_label} optional_files must be a list")
    return cast(list[object], required_files), cast(list[object], optional_files)


def require_manifest_string(
    runtime_manifest: RuntimeManifest,
    *,
    key: str,
    manifest_label: str,
    allow_missing: bool = False,
) -> str | None:
    """Require a non-empty manifest string field with stable wording."""

    value = runtime_manifest.get(key)
    if value is None and allow_missing:
        return None
    if not isinstance(value, str) or not value.strip():
        if allow_missing:
            raise ValueError(f"{manifest_label} {key} must be a non-empty string when present")
        raise ValueError(f"{manifest_label} must declare a non-empty {key}")
    return value


def validate_runtime_manifest_file_entries(
    *,
    required_files: Sequence[object],
    optional_files: Sequence[object],
    manifest_label: str,
) -> None:
    """Validate per-file manifest entries shared by MATLAB stage runtimes."""

    for bucket_name, bucket, required_default in (
        ("required_files", required_files, True),
        ("optional_files", optional_files, False),
    ):
        for item in bucket:
            if not isinstance(item, Mapping):
                raise ValueError(f"{manifest_label} {bucket_name} entries must be JSON objects")

            entry = cast(Mapping[str, object], item)
            for key in ("name", "source_path", "role"):
                value = entry.get(key)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{manifest_label} entry in {bucket_name} is missing non-empty '{key}'")

            declared_name = cast(str, entry["name"])
            declared_path = Path(declared_name)
            if (
                declared_path.is_absolute()
                or ".." in declared_path.parts
                or declared_name in {".", ".."}
                or "/" in declared_name
                or "\\" in declared_name
            ):
                raise ValueError(
                    f"{manifest_label} entry in {bucket_name} declares unsafe runtime filename {declared_name!r}"
                )

            raw_required = entry.get("required", required_default)
            if not isinstance(raw_required, bool):
                raise ValueError(
                    f"{manifest_label} entry in {bucket_name} must declare boolean 'required' when present"
                )


def load_validated_runtime_manifest(
    runtime_dir: Path,
    *,
    manifest_label: str,
    missing_hint: str,
    package_name: str | None = None,
    manifest_name: str = MATLAB_RUNTIME_MANIFEST_NAME,
    required_string_fields: Sequence[str] = ("entrypoint",),
    optional_string_fields: Sequence[str] = (),
) -> dict[str, object]:
    """Load and validate the shared MATLAB runtime manifest contract."""

    manifest = load_runtime_manifest_json(
        runtime_dir,
        manifest_label=manifest_label,
        missing_hint=missing_hint,
        manifest_name=manifest_name,
    )
    required_files, optional_files = validate_runtime_manifest_file_buckets(
        manifest,
        manifest_label=manifest_label,
    )
    for key in required_string_fields:
        _ = require_manifest_string(
            manifest,
            key=key,
            manifest_label=manifest_label,
        )
    for key in optional_string_fields:
        _ = require_manifest_string(
            manifest,
            key=key,
            manifest_label=manifest_label,
            allow_missing=True,
        )
    if package_name is not None and manifest.get("package_name") != package_name:
        raise ValueError(
            f"{manifest_label} package_name mismatch: expected {package_name!r}, got {manifest.get('package_name')!r}"
        )
    validate_runtime_manifest_file_entries(
        required_files=required_files,
        optional_files=optional_files,
        manifest_label=manifest_label,
    )
    return manifest


def declared_runtime_filenames(runtime_manifest: RuntimeManifest) -> set[str]:
    """Return all declared runtime filenames across manifest buckets."""

    filenames: set[str] = set()
    for bucket_name in ("required_files", "optional_files"):
        bucket = runtime_manifest.get(bucket_name, [])
        if not isinstance(bucket, Sequence) or isinstance(bucket, (str, bytes)):
            continue
        for item in bucket:
            if not isinstance(item, Mapping):
                continue
            entry = cast(Mapping[str, object], item)
            name = entry.get("name")
            if isinstance(name, str):
                filenames.add(name)
    return filenames


def validate_manifest_declares_entrypoint_file(
    runtime_manifest: RuntimeManifest,
    entrypoint_name: str,
    *,
    manifest_label: str,
    expectation_label: str = "configured entrypoint file",
) -> None:
    """Require the manifest to declare the expected ``<entrypoint>.m`` file."""

    expected_entrypoint_file = f"{entrypoint_name}.m"
    if expected_entrypoint_file not in declared_runtime_filenames(runtime_manifest):
        raise ValueError(
            f"{manifest_label} must declare the {expectation_label}. Missing {expected_entrypoint_file!r} in runtime manifest"
        )


def validate_configured_entrypoint_contract(
    runtime_manifest: RuntimeManifest,
    configured_entrypoint: str,
    *,
    config_label: str,
    manifest_label: str,
) -> None:
    """Validate config-vs-manifest entrypoint alignment and declared filenames."""

    manifest_entrypoint = runtime_manifest.get("entrypoint")
    if configured_entrypoint != manifest_entrypoint:
        raise ValueError(
            f"{config_label} must match the repo-local {manifest_label}. Config entrypoint={configured_entrypoint!r}, manifest entrypoint={manifest_entrypoint!r}"
        )

    validate_manifest_declares_entrypoint_file(
        runtime_manifest,
        configured_entrypoint,
        manifest_label=manifest_label,
    )


def validate_runtime_step_metadata(
    steps: object,
    *,
    missing_steps_message: str,
    step_label: str,
) -> None:
    """Validate non-empty step metadata with non-negative ``duration_ms`` values."""

    if not isinstance(steps, list) or not steps:
        raise ValueError(missing_steps_message)
    for index, raw_step in enumerate(cast(list[object], steps)):
        if not isinstance(raw_step, Mapping):
            raise ValueError(f"{step_label} #{index} must be a mapping")
        step = cast(Mapping[str, object], raw_step)
        name = step.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{step_label} #{index} is missing a non-empty name")
        duration_ms = step.get("duration_ms")
        if not isinstance(duration_ms, (int, float)) or duration_ms < 0:
            raise ValueError(f"{step_label} '{name}' must report a non-negative duration_ms")


def resolve_staged_output_path(
    output_path_value: object,
    *,
    tmpdir_path: Path,
    missing_output_path_message: str,
    outside_tmpdir_prefix: str,
    missing_output_prefix: str,
) -> Path:
    """Resolve a MATLAB-produced temp output path and keep it inside ``tmpdir_path``."""

    if not isinstance(output_path_value, str) or not output_path_value.strip():
        raise ValueError(missing_output_path_message)
    output_path = Path(output_path_value)
    if not output_path.is_absolute():
        output_path = tmpdir_path / output_path
    output_path = output_path.resolve()
    resolved_tmpdir = tmpdir_path.resolve()
    if output_path.parent != resolved_tmpdir:
        raise ValueError(f"{outside_tmpdir_prefix}: {output_path}")
    if not output_path.exists():
        raise FileNotFoundError(f"{missing_output_prefix}: {output_path}")
    return output_path


def validate_expected_3d_staged_volume_shape(
    volume_shape_value: object,
    *,
    expected_shape_zyx: tuple[int, int, int],
    mismatch_prefix: str,
) -> None:
    """Validate a ``volume_shape_zyx`` metadata field against staged 3-D input."""

    if not isinstance(volume_shape_value, list):
        raise ValueError(
            f"{mismatch_prefix}: expected {list(expected_shape_zyx)}, got {volume_shape_value!r}"
        )
    try:
        normalized_shape = [int(value) for value in cast(list[Any], volume_shape_value)]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{mismatch_prefix}: expected {list(expected_shape_zyx)}, got {volume_shape_value!r}"
        ) from exc
    if normalized_shape != [int(value) for value in expected_shape_zyx]:
        raise ValueError(
            f"{mismatch_prefix}: expected {list(expected_shape_zyx)}, got {volume_shape_value!r}"
        )


def write_staged_3d_volume_tiff(volume_path: Path, volume: object, *, owner_label: str) -> None:
    """Write a staged 3-D TIFF stack with the shared MATLAB-provider contract."""

    staged_volume = np.asarray(volume)
    if staged_volume.ndim != 3:
        raise ValueError(f"{owner_label} expects a 3D staged volume, got ndim={staged_volume.ndim}")

    with tifffile.TiffWriter(volume_path) as writer:
        for plane_index in range(int(staged_volume.shape[0])):
            _ = writer.write(staged_volume[plane_index], photometric="minisblack", metadata=None)


def collect_runtime_file_records(
    runtime_manifest: RuntimeManifest,
    runtime_dir: Path,
    *,
    missing_required_prefix: str,
    missing_required_suffix: str,
) -> list[RuntimeFileRecord]:
    """Collect runtime-file provenance records with shared fail-loud checks."""

    records: list[RuntimeFileRecord] = []
    resolved_runtime_dir = runtime_dir.resolve()
    for bucket_name, required_default in (("required_files", True), ("optional_files", False)):
        bucket = runtime_manifest.get(bucket_name, [])
        if not isinstance(bucket, Sequence) or isinstance(bucket, (str, bytes)):
            continue
        for item in bucket:
            if not isinstance(item, Mapping):
                continue

            entry = cast(Mapping[str, object], item)
            declared_name = entry.get("name")
            role = entry.get("role")
            source_path = entry.get("source_path")
            if not isinstance(declared_name, str) or not isinstance(role, str) or not isinstance(source_path, str):
                continue

            candidate = (resolved_runtime_dir / Path(declared_name)).resolve()
            try:
                _ = candidate.relative_to(resolved_runtime_dir)
            except ValueError as exc:
                raise ValueError(
                    f"runtime manifest {bucket_name} entry resolves outside the runtime directory: {declared_name!r}"
                ) from exc

            is_required = bool(entry.get("required", required_default))
            is_used = is_required
            if is_used and not candidate.exists():
                raise FileNotFoundError(f"{missing_required_prefix}: {candidate}. {missing_required_suffix}")
            if candidate.exists() and not candidate.is_file():
                raise ValueError(f"Runtime manifest {bucket_name} entry must resolve to a file: {candidate}")

            record: RuntimeFileRecord = {
                "name": declared_name,
                "required": is_required,
                "used": is_used,
                "role": role,
                "source_path": source_path,
            }
            if candidate.exists():
                record["sha256"] = sha256_file(candidate)
            records.append(record)
    return records
