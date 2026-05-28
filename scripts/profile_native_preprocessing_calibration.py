from __future__ import annotations

import argparse
import hashlib
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, cast


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pystar.infrastructure import ExperimentConfig, PreprocessingStep, load_config
from pystar.io import ImageLoader, get_fov_output_structure
from pystar.preprocessing import (
    DataSanitizer,
    NATIVE_PREPROCESSING_TIMING_SCHEMA_NAME,
    NATIVE_PREPROCESSING_TIMING_SCHEMA_VERSION,
)
from pystar.serialization import write_backend_metadata


CALIBRATION_PROFILE_SCHEMA_NAME = "pystar_native_preprocessing_calibration_profile"
CALIBRATION_PROFILE_SCHEMA_VERSION = 1
CALIBRATION_PROFILE_FILENAMES = {
    "json": "native_preprocessing_calibration_profile.json",
    "markdown": "native_preprocessing_calibration_profile.md",
}
PROFILE_OUTPUT_MARKER = ".pystar_native_preprocessing_calibration_profile"
FOCUSED_CALIBRATION_METHODS = (
    "histogram_match",
    "min_max_normalize",
    "morpho_reconstruction_contrast",
)


def _profile_output_marker_text() -> str:
    return (
        f"schema_name={CALIBRATION_PROFILE_SCHEMA_NAME}\n"
        f"schema_version={CALIBRATION_PROFILE_SCHEMA_VERSION}\n"
    )


def _validate_profile_output_marker(marker_path: Path) -> None:
    if marker_path.is_symlink() or not marker_path.is_file():
        raise ValueError(f"Stage24 profile marker is invalid: {marker_path}")
    try:
        marker_text = marker_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Stage24 profile marker cannot be read: {marker_path}") from exc
    if marker_text != _profile_output_marker_text():
        raise ValueError(
            "Stage24 profile marker does not match the native preprocessing calibration profile schema: "
            f"{marker_path}"
        )


@dataclass(frozen=True)
class ProfileRunResult:
    repeat_index: int
    repeat_output_root: Path
    elapsed_wall_ms: float
    kernel_result: Mapping[str, Any]


def _duration_summary(values: Sequence[float]) -> dict[str, object]:
    if not values:
        return {
            "count": 0,
            "total_duration_ms": 0.0,
            "mean_duration_ms": None,
            "median_duration_ms": None,
            "min_duration_ms": None,
            "max_duration_ms": None,
            "stdev_duration_ms": None,
        }

    finite_values = [float(value) for value in values]
    for value in finite_values:
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Profile timings must be finite and non-negative; got {value!r}")

    total = round(sum(finite_values), 3)
    return {
        "count": len(finite_values),
        "total_duration_ms": total,
        "mean_duration_ms": round(total / len(finite_values), 3),
        "median_duration_ms": round(float(statistics.median(finite_values)), 3),
        "min_duration_ms": round(min(finite_values), 3),
        "max_duration_ms": round(max(finite_values), 3),
        "stdev_duration_ms": None
        if len(finite_values) < 2
        else round(float(statistics.stdev(finite_values)), 3),
    }


def _summary_float(summary: Mapping[str, object], key: str) -> float:
    value = summary.get(key)
    if value is None:
        return 0.0
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Duration summary field {key!r} must be numeric or null; got {value!r}")
    return float(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _current_git_commit(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            env={**os.environ, "GIT_MASTER": os.environ.get("GIT_MASTER", "1")},
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def parse_int_list(raw_value: str | None, *, field_name: str) -> tuple[int, ...] | None:
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


def _reject_symlink_components(path: Path, *, field_name: str) -> None:
    absolute_path = path.expanduser().absolute()
    parts = absolute_path.parts
    if not parts:
        return
    current = Path(parts[0])
    for part in parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{field_name} must not contain symlink component: {current}")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _assert_path_within(path: Path, parent: Path, *, field_name: str) -> None:
    parent_resolved = parent.resolve(strict=True)
    path_resolved = path.resolve(strict=False)
    if path_resolved == parent_resolved:
        return
    if not _is_relative_to(path_resolved, parent_resolved):
        raise ValueError(f"{field_name} must stay inside profile output directory: {path}")


def _reject_profile_output_production_overlap(config: ExperimentConfig, output_dir: Path) -> None:
    output_resolved = output_dir.expanduser().resolve(strict=False)
    production_output = Path(config.pipeline.output.directory).expanduser().resolve(strict=False)
    if output_resolved == production_output:
        overlaps = True
    else:
        overlaps = _is_relative_to(output_resolved, production_output) or _is_relative_to(
            production_output,
            output_resolved,
        )
    if overlaps:
        raise ValueError(
            "--output-dir must be a dedicated Stage24 profile artifact directory that does not overlap "
            f"the configured production output directory: {output_dir} vs {production_output}"
        )


def _prepare_profile_output_dir(output_dir: Path) -> Path:
    output_path = output_dir.expanduser()
    _reject_symlink_components(output_path, field_name="--output-dir")

    if output_path.exists():
        if not output_path.is_dir():
            raise ValueError(f"--output-dir must be a directory: {output_path}")
    else:
        output_path.mkdir(parents=True, exist_ok=False)

    _reject_symlink_components(output_path, field_name="--output-dir")
    output_resolved = output_path.resolve(strict=True)
    unsafe_roots = {Path("/").resolve(), REPO_ROOT.resolve(), Path.home().resolve()}
    if output_resolved in unsafe_roots:
        raise ValueError(
            "--output-dir must be a dedicated Stage24 profile artifact directory, "
            f"not an unsafe root: {output_path}"
        )

    marker_path = output_path / PROFILE_OUTPUT_MARKER
    if marker_path.exists():
        _validate_profile_output_marker(marker_path)
    else:
        entries = list(output_path.iterdir())
        if entries:
            raise ValueError(
                "--output-dir exists and is not a Stage24 profile output directory. "
                f"Choose an empty/dedicated directory or one containing {PROFILE_OUTPUT_MARKER}: {output_path}"
            )
        marker_path.write_text(_profile_output_marker_text(), encoding="utf-8")

    return output_path


def _safe_remove_repeat_root(repeat_root: Path, profile_output_dir: Path) -> None:
    marker_path = profile_output_dir / PROFILE_OUTPUT_MARKER
    if not marker_path.exists():
        raise ValueError(f"Refusing to delete repeat output without valid Stage24 marker: {marker_path}")
    _validate_profile_output_marker(marker_path)

    _assert_path_within(repeat_root, profile_output_dir, field_name="repeat output root")
    if not repeat_root.exists():
        return
    _reject_symlink_components(repeat_root, field_name="repeat output root")
    if repeat_root.is_symlink():
        raise ValueError(f"Refusing to delete symlinked repeat output root: {repeat_root}")
    if not repeat_root.is_dir():
        raise ValueError(f"Repeat output root must be a directory: {repeat_root}")
    shutil.rmtree(repeat_root)


def _tiff_shape_dtype(path: Path) -> tuple[list[int], str]:
    import tifffile

    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        return [int(dimension) for dimension in series.shape], str(series.dtype)


def _normalize_fov_ids(raw_fovs: Sequence[int] | None, config: ExperimentConfig) -> tuple[int, ...]:
    if raw_fovs is None:
        configured = tuple(int(fov_id) for fov_id in config.dataset.parsed_fovs)
        if not configured:
            raise ValueError("Config dataset.parsed_fovs is empty; provide --fovs explicitly")
        return (configured[0],)
    if not raw_fovs:
        raise ValueError("At least one FOV id is required")
    return tuple(int(fov_id) for fov_id in raw_fovs)


def _validate_native_sequence(sequence: Sequence[PreprocessingStep]) -> None:
    if not sequence:
        raise ValueError("Native preprocessing calibration profiling requires a non-empty preprocessing sequence")

    non_native = [step.method for step in sequence if step.provider != "native"]
    if non_native:
        raise ValueError(
            "Native preprocessing calibration profiling requires provider='native' for every preprocessing step; "
            f"non-native methods: {non_native}"
        )

    for step in sequence:
        if step.method == "histogram_match":
            scope = step.params.get("scope", "none")
            if scope not in {"inter_round", "intra_round", "none"}:
                raise ValueError(
                    "histogram_match profiling only supports params.scope values "
                    f"'inter_round', 'intra_round', or 'none'; got {scope!r}"
                )


def _writer_for_repeat(sanitizer: DataSanitizer, repeat_output_root: Path, fov_id: int):
    def write(img: Any, round_id: int, channel_id: int) -> Path:
        paths = get_fov_output_structure(repeat_output_root, fov_id)
        _reject_symlink_components(paths["cleaned"], field_name="clean output directory")
        _assert_path_within(paths["cleaned"], repeat_output_root, field_name="clean output directory")
        output_path = paths["cleaned"] / sanitizer._flat_clean_filename(fov_id, round_id, channel_id)
        _assert_path_within(output_path, repeat_output_root, field_name="clean output file")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Reuse the production clean-image writer for the TIFF contract while
        # directing output to the profiling repeat root instead of the configured
        # production output directory.
        import tifffile

        tifffile.imwrite(output_path, img, compression="zlib")
        return output_path

    return write


def _collect_output_fingerprints(output_files: Sequence[object]) -> dict[str, dict[str, object]]:
    fingerprints: dict[str, dict[str, object]] = {}
    for raw_path in output_files:
        path = Path(str(raw_path))
        if not path.exists():
            raise FileNotFoundError(f"Profiled preprocessing output is missing: {path}")
        relative_key = path.name
        if relative_key in fingerprints:
            raise ValueError(f"Duplicate clean output filename in profile repeat: {relative_key}")
        shape, dtype = _tiff_shape_dtype(path)
        fingerprints[relative_key] = {
            "path": str(path),
            "shape": shape,
            "dtype": dtype,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    return fingerprints


def _compare_repeat_outputs(run_results: Sequence[ProfileRunResult]) -> dict[str, object]:
    if not run_results:
        raise ValueError("Cannot compare clean outputs without profile runs")

    baseline_files = _collect_output_fingerprints(cast(Sequence[object], run_results[0].kernel_result["output_files"]))
    mismatches: list[dict[str, object]] = []
    repeats: list[dict[str, object]] = [
        {
            "repeat_index": run_results[0].repeat_index,
            "output_root": str(run_results[0].repeat_output_root),
            "file_count": len(baseline_files),
        }
    ]

    baseline_signature = {
        name: {"size_bytes": item["size_bytes"], "sha256": item["sha256"]}
        for name, item in baseline_files.items()
    }
    for run_result in run_results[1:]:
        current_files = _collect_output_fingerprints(cast(Sequence[object], run_result.kernel_result["output_files"]))
        current_signature = {
            name: {"size_bytes": item["size_bytes"], "sha256": item["sha256"]}
            for name, item in current_files.items()
        }
        repeats.append(
            {
                "repeat_index": run_result.repeat_index,
                "output_root": str(run_result.repeat_output_root),
                "file_count": len(current_files),
            }
        )
        if current_signature != baseline_signature:
            mismatches.append(
                {
                    "repeat_index": run_result.repeat_index,
                    "baseline_repeat_index": run_results[0].repeat_index,
                    "missing_files": sorted(set(baseline_signature) - set(current_signature)),
                    "extra_files": sorted(set(current_signature) - set(baseline_signature)),
                    "changed_files": sorted(
                        name
                        for name in set(baseline_signature) & set(current_signature)
                        if baseline_signature[name] != current_signature[name]
                    ),
                }
            )

    return {
        "status": "equivalent" if not mismatches else "mismatch",
        "baseline_repeat_index": run_results[0].repeat_index,
        "file_count": len(baseline_files),
        "baseline_files": baseline_files,
        "repeats": repeats,
        "mismatches": mismatches,
    }


def _method_durations(kernel_result: Mapping[str, Any]) -> dict[str, list[float]]:
    durations: dict[str, list[float]] = {}
    for volume in cast(Sequence[Mapping[str, Any]], kernel_result["preprocessing_timing"]["volumes"]):
        for phase_name in ("calibration_steps", "extraction_steps"):
            for step in cast(Sequence[Mapping[str, Any]], volume.get(phase_name, [])):
                method = str(step["method"])
                durations.setdefault(method, []).append(float(step["duration_ms"]))
    return durations


def _method_durations_for_phase(kernel_result: Mapping[str, Any], phase_name: str) -> dict[str, list[float]]:
    if phase_name not in {"calibration_steps", "extraction_steps"}:
        raise ValueError(f"Unsupported preprocessing phase for method timing: {phase_name!r}")
    durations: dict[str, list[float]] = {}
    for volume in cast(Sequence[Mapping[str, Any]], kernel_result["preprocessing_timing"]["volumes"]):
        for step in cast(Sequence[Mapping[str, Any]], volume.get(phase_name, [])):
            method = str(step["method"])
            durations.setdefault(method, []).append(float(step["duration_ms"]))
    return durations


def _phase_durations(kernel_result: Mapping[str, Any]) -> dict[str, list[float]]:
    durations = {
        "calibration_steps": [],
        "extraction_steps": [],
        "load": [],
        "clip_convert": [],
        "write": [],
        "volume_total": [],
    }
    for volume in cast(Sequence[Mapping[str, Any]], kernel_result["preprocessing_timing"]["volumes"]):
        durations["load"].append(float(volume["load_ms"]))
        durations["clip_convert"].append(float(volume["clip_convert_ms"]))
        durations["write"].append(float(volume["write_ms"]))
        durations["volume_total"].append(float(volume["total_ms"]))
        durations["calibration_steps"].append(
            round(
                sum(float(step["duration_ms"]) for step in cast(Sequence[Mapping[str, Any]], volume.get("calibration_steps", []))),
                3,
            )
        )
        durations["extraction_steps"].append(
            round(
                sum(float(step["duration_ms"]) for step in cast(Sequence[Mapping[str, Any]], volume.get("extraction_steps", []))),
                3,
            )
        )
    return durations


def _aggregate_runs(run_results: Sequence[ProfileRunResult]) -> dict[str, object]:
    method_durations: dict[str, list[float]] = {}
    calibration_method_durations: dict[str, list[float]] = {}
    extraction_method_durations: dict[str, list[float]] = {}
    phase_durations: dict[str, list[float]] = {}
    run_elapsed = [run.elapsed_wall_ms for run in run_results]

    for run in run_results:
        for method, durations in _method_durations(run.kernel_result).items():
            method_durations.setdefault(method, []).extend(durations)
        for method, durations in _method_durations_for_phase(run.kernel_result, "calibration_steps").items():
            calibration_method_durations.setdefault(method, []).extend(durations)
        for method, durations in _method_durations_for_phase(run.kernel_result, "extraction_steps").items():
            extraction_method_durations.setdefault(method, []).extend(durations)
        for phase, durations in _phase_durations(run.kernel_result).items():
            phase_durations.setdefault(phase, []).extend(durations)

    by_method = {method: _duration_summary(durations) for method, durations in sorted(method_durations.items())}
    by_calibration_method = {
        method: _duration_summary(durations)
        for method, durations in sorted(calibration_method_durations.items())
    }
    by_extraction_method = {
        method: _duration_summary(durations)
        for method, durations in sorted(extraction_method_durations.items())
    }
    by_phase = {phase: _duration_summary(durations) for phase, durations in sorted(phase_durations.items())}
    calibration_total = _summary_float(
        cast(Mapping[str, object], by_phase.get("calibration_steps", {})),
        "total_duration_ms",
    )

    focused = {}
    for method in FOCUSED_CALIBRATION_METHODS:
        all_summary = cast(Mapping[str, object], by_method.get(method, _duration_summary([])))
        calibration_summary = cast(Mapping[str, object], by_calibration_method.get(method, _duration_summary([])))
        extraction_summary = cast(Mapping[str, object], by_extraction_method.get(method, _duration_summary([])))
        calibration_method_total = _summary_float(calibration_summary, "total_duration_ms")
        focused[method] = {
            "all_phases": dict(all_summary),
            "calibration_phase": dict(calibration_summary),
            "extraction_phase": dict(extraction_summary),
            "percent_of_calibration_time": None
            if calibration_total <= 0.0 or calibration_method_total <= 0.0
            else round(calibration_method_total / calibration_total * 100.0, 3),
        }

    return {
        "run_elapsed_wall_ms": _duration_summary(run_elapsed),
        "by_phase": by_phase,
        "by_method": by_method,
        "by_calibration_method": by_calibration_method,
        "by_extraction_method": by_extraction_method,
        "focused_methods": focused,
    }


def _run_profile_repeats(
    *,
    config: ExperimentConfig,
    fov_id: int,
    target_rounds: Sequence[int] | None,
    repeats: int,
    output_dir: Path,
) -> list[ProfileRunResult]:
    if repeats <= 0:
        raise ValueError(f"--repeats must be positive; got {repeats!r}")
    _validate_native_sequence(config.pipeline.preprocessing.sequence)

    results: list[ProfileRunResult] = []
    for repeat_index in range(repeats):
        repeat_root = output_dir / f"fov_{fov_id}" / f"repeat_{repeat_index}"
        _safe_remove_repeat_root(repeat_root, output_dir)

        sanitizer = DataSanitizer(config)
        loader = ImageLoader(config)
        started = time.perf_counter()
        kernel_result = sanitizer._run_native_preprocessing_kernel(
            fov_id=fov_id,
            loader=loader,
            sequence=config.pipeline.preprocessing.sequence,
            target_rounds=None if target_rounds is None else list(target_rounds),
            output_writer=_writer_for_repeat(sanitizer, repeat_root, fov_id),
            print_progress=False,
        )
        elapsed_wall_ms = round((time.perf_counter() - started) * 1000.0, 3)
        results.append(
            ProfileRunResult(
                repeat_index=repeat_index,
                repeat_output_root=repeat_root,
                elapsed_wall_ms=elapsed_wall_ms,
                kernel_result=kernel_result,
            )
        )
    return results


def build_profile_payload(
    *,
    config: ExperimentConfig,
    config_path: Path,
    fov_ids: Sequence[int],
    target_rounds: Sequence[int] | None,
    repeats: int,
    output_dir: Path,
    baseline_commit: str | None = None,
    validation_worktree: Path | None = None,
) -> dict[str, object]:
    if not fov_ids:
        raise ValueError("At least one FOV id is required for calibration profiling")
    if repeats <= 0:
        raise ValueError(f"--repeats must be positive; got {repeats!r}")

    _validate_native_sequence(config.pipeline.preprocessing.sequence)
    _reject_profile_output_production_overlap(config, output_dir)
    output_dir = _prepare_profile_output_dir(output_dir)

    fov_payloads: list[dict[str, object]] = []
    for fov_id in fov_ids:
        run_results = _run_profile_repeats(
            config=config,
            fov_id=int(fov_id),
            target_rounds=target_rounds,
            repeats=repeats,
            output_dir=output_dir,
        )
        first_timing = cast(Mapping[str, Any], run_results[0].kernel_result["preprocessing_timing"])
        if first_timing["schema_name"] != NATIVE_PREPROCESSING_TIMING_SCHEMA_NAME:
            raise ValueError("Native preprocessing timing schema drifted while profiling calibration")
        if int(first_timing["schema_version"]) != NATIVE_PREPROCESSING_TIMING_SCHEMA_VERSION:
            raise ValueError("Native preprocessing timing schema version drifted while profiling calibration")

        repeats_payload = []
        for result in run_results:
            timing = cast(Mapping[str, Any], result.kernel_result["preprocessing_timing"])
            repeats_payload.append(
                {
                    "repeat_index": result.repeat_index,
                    "repeat_output_root": str(result.repeat_output_root),
                    "elapsed_wall_ms": result.elapsed_wall_ms,
                    "round_order": list(timing["round_order"]),
                    "volume_count": int(timing["volume_count"]),
                    "total_volume_ms": float(timing["total_volume_ms"]),
                    "output_files": list(result.kernel_result["output_files"]),
                    "timing": timing,
                }
            )

        fov_payloads.append(
            {
                "fov_id": int(fov_id),
                "round_order": list(first_timing["round_order"]),
                "target_rounds": None if target_rounds is None else [int(round_id) for round_id in target_rounds],
                "repeats": repeats_payload,
                "clean_output_equivalence": _compare_repeat_outputs(run_results),
                "summary": _aggregate_runs(run_results),
            }
        )

    candidate_commit = _current_git_commit(REPO_ROOT)
    return {
        "schema_name": CALIBRATION_PROFILE_SCHEMA_NAME,
        "schema_version": CALIBRATION_PROFILE_SCHEMA_VERSION,
        "source": {
            "repo_root": str(REPO_ROOT),
            "baseline_commit": baseline_commit,
            "candidate_commit": candidate_commit,
            "git_commit": candidate_commit,
            "config_path": str(config_path.resolve()),
            "config_sha256": getattr(config, "config_sha256", None),
            "validation_worktree": None if validation_worktree is None else str(validation_worktree),
        },
        "profile_configuration": {
            "output_dir": str(output_dir),
            "fov_ids": [int(fov_id) for fov_id in fov_ids],
            "target_rounds": None if target_rounds is None else [int(round_id) for round_id in target_rounds],
            "repeats": int(repeats),
            "preprocessing_sequence": [
                {
                    "index": index,
                    "method": step.method,
                    "provider": step.provider,
                    "params": dict(step.params),
                }
                for index, step in enumerate(config.pipeline.preprocessing.sequence)
            ],
        },
        "contracts": {
            "production_runtime_instrumentation_added": False,
            "canonical_clean_output_filename_contract": "clean_fov_{fov_id}_round_{round_id}_ch_{channel_id}.tif",
            "timing_source_schema_name": NATIVE_PREPROCESSING_TIMING_SCHEMA_NAME,
            "timing_source_schema_version": NATIVE_PREPROCESSING_TIMING_SCHEMA_VERSION,
            "speedup_claim": "none; profiling harness only reports measured timings",
        },
        "hotspot_call": {
            "status": "manual_required",
            "selected_hotspot": None,
            "rule": (
                "Use the measured calibration/extraction summaries plus clean-output equivalence "
                "to decide whether histogram_match, another calibration sub-step, extraction/morphology, "
                "or timing noise is the next real hotspot."
            ),
        },
        "fovs": fov_payloads,
    }


def render_profile_markdown(payload: Mapping[str, object]) -> str:
    source = cast(Mapping[str, object], payload["source"])
    profile_config = cast(Mapping[str, object], payload["profile_configuration"])
    contracts = cast(Mapping[str, object], payload["contracts"])
    lines = [
        "# Native Preprocessing Calibration Profile",
        "",
        "## Source",
        "",
        f"- Repository root: `{source.get('repo_root')}`",
        f"- Baseline commit: `{source.get('baseline_commit') or 'not provided'}`",
        f"- Candidate commit: `{source.get('candidate_commit') or source.get('git_commit') or 'unknown'}`",
        f"- Config path: `{source.get('config_path')}`",
        f"- Config SHA256: `{source.get('config_sha256') or 'unknown'}`",
        f"- Validation worktree: `{source.get('validation_worktree') or 'not provided'}`",
        "",
        "## Profile Configuration",
        "",
        f"- Output directory: `{profile_config.get('output_dir')}`",
        f"- FOV IDs: `{profile_config.get('fov_ids')}`",
        f"- Target rounds: `{profile_config.get('target_rounds') or 'all configured rounds'}`",
        f"- Repeats: `{profile_config.get('repeats')}`",
        "",
        "## Contract Guardrails",
        "",
        f"- Production runtime instrumentation added: `{contracts.get('production_runtime_instrumentation_added')}`",
        f"- Clean filename contract: `{contracts.get('canonical_clean_output_filename_contract')}`",
        f"- Speedup claim: {contracts.get('speedup_claim')}",
        "",
        "## Timing Summary",
        "",
    ]

    for fov in cast(Sequence[Mapping[str, object]], payload["fovs"]):
        equivalence = cast(Mapping[str, object], fov["clean_output_equivalence"])
        summary = cast(Mapping[str, object], fov["summary"])
        by_phase = cast(Mapping[str, object], summary["by_phase"])
        focused = cast(Mapping[str, object], summary["focused_methods"])
        lines.extend(
            [
                f"### FOV {fov['fov_id']}",
                "",
                f"- Round order: `{fov['round_order']}`",
                f"- Clean-output equivalence: `{equivalence['status']}` across `{equivalence['file_count']}` files",
                f"- Calibration phase: `{cast(Mapping[str, object], by_phase['calibration_steps']).get('total_duration_ms')}` ms total",
                f"- Extraction phase: `{cast(Mapping[str, object], by_phase['extraction_steps']).get('total_duration_ms')}` ms total",
                "",
                "| Method | Calibration count | Calibration total ms | Calibration mean ms | Extraction count | Extraction total ms | Extraction mean ms | % calibration |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for method in FOCUSED_CALIBRATION_METHODS:
            method_summary = cast(Mapping[str, object], focused[method])
            calibration_summary = cast(Mapping[str, object], method_summary["calibration_phase"])
            extraction_summary = cast(Mapping[str, object], method_summary["extraction_phase"])
            lines.append(
                "| {method} | {cal_count} | {cal_total} | {cal_mean} | {ext_count} | {ext_total} | {ext_mean} | {percent} |".format(
                    method=method,
                    cal_count=calibration_summary.get("count"),
                    cal_total=calibration_summary.get("total_duration_ms"),
                    cal_mean=calibration_summary.get("mean_duration_ms"),
                    ext_count=extraction_summary.get("count"),
                    ext_total=extraction_summary.get("total_duration_ms"),
                    ext_mean=extraction_summary.get("mean_duration_ms"),
                    percent=method_summary.get("percent_of_calibration_time"),
                )
            )
        lines.append("")

    lines.extend(
        [
            "## Interpretation Rule",
            "",
            "This harness reports timings only. Do not claim an end-to-end speedup unless a paired validation report compares baseline and candidate runs on the same real-data surface and shows equivalent clean outputs.",
            "",
            "## Real-Data Validation Artifact Expectations",
            "",
            "When running under `/media/zenglab/result/zhui/Leica_deconv_test_260106-worktrees/pystar-next`, keep the PyStar source fixed via `PYTHONPATH=/media/zenglab/result/zhui/PyStar` and write the JSON/Markdown artifacts to a validation-only directory. The validation report should record source commit, config path/hash, FOV/round/channel surface, clean TIFF paths, clean-output equivalence by shape/dtype/hash or array contents, calibration/histogram-match timings, total preprocessing timing, and any cache/warmup drift.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_profile_artifacts(payload: dict[str, object], output_dir: Path) -> tuple[Path, Path]:
    output_dir = _prepare_profile_output_dir(output_dir)
    json_path = output_dir / CALIBRATION_PROFILE_FILENAMES["json"]
    markdown_path = output_dir / CALIBRATION_PROFILE_FILENAMES["markdown"]
    temp_json_path = json_path.with_name(f"{json_path.name}.tmp")
    temp_markdown_path = markdown_path.with_name(f"{markdown_path.name}.tmp")
    _reject_symlink_components(json_path, field_name="profile JSON path")
    _reject_symlink_components(markdown_path, field_name="profile Markdown path")
    _reject_symlink_components(temp_json_path, field_name="temporary profile JSON path")
    _reject_symlink_components(temp_markdown_path, field_name="temporary profile Markdown path")
    _assert_path_within(json_path, output_dir, field_name="profile JSON path")
    _assert_path_within(markdown_path, output_dir, field_name="profile Markdown path")
    _assert_path_within(temp_json_path, output_dir, field_name="temporary profile JSON path")
    _assert_path_within(temp_markdown_path, output_dir, field_name="temporary profile Markdown path")
    write_backend_metadata(temp_json_path, payload)
    _ = temp_json_path.replace(json_path)
    _ = temp_markdown_path.write_text(render_profile_markdown(payload), encoding="utf-8")
    _ = temp_markdown_path.replace(markdown_path)
    return json_path, markdown_path


def profile_native_preprocessing_calibration(
    *,
    config_path: Path,
    output_dir: Path,
    fov_ids: Sequence[int] | None = None,
    target_rounds: Sequence[int] | None = None,
    repeats: int = 3,
    baseline_commit: str | None = None,
    validation_worktree: Path | None = None,
) -> tuple[Path, Path, dict[str, object]]:
    config = load_config(str(config_path))
    normalized_fovs = _normalize_fov_ids(fov_ids, config)
    payload = build_profile_payload(
        config=config,
        config_path=config_path,
        fov_ids=normalized_fovs,
        target_rounds=target_rounds,
        repeats=repeats,
        output_dir=output_dir,
        baseline_commit=baseline_commit,
        validation_worktree=validation_worktree,
    )
    json_path, markdown_path = write_profile_artifacts(payload, output_dir)
    return json_path, markdown_path, payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Profile native PyStar preprocessing calibration timing without adding production runtime overhead. "
            "The harness repeats the existing native kernel, records calibration/histogram-match timings, "
            "and verifies clean-output equivalence across repeats."
        )
    )
    _ = parser.add_argument("--config", required=True, type=Path, help="Experiment YAML config to profile.")
    _ = parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Validation/profiling artifact directory. Repeat clean outputs and reports are written here.",
    )
    _ = parser.add_argument(
        "--fovs",
        default=None,
        help="Optional comma-separated FOV ids. Defaults to the first configured FOV.",
    )
    _ = parser.add_argument(
        "--rounds",
        default=None,
        help="Optional comma-separated round ids for a focused surface. Defaults to all configured rounds.",
    )
    _ = parser.add_argument("--repeats", type=int, default=3, help="Repeat count for paired/interleaved timing.")
    _ = parser.add_argument(
        "--baseline-commit",
        default=None,
        help="Optional baseline commit/tag/hash to record for paired validation reports.",
    )
    _ = parser.add_argument(
        "--validation-worktree",
        type=Path,
        default=None,
        help="Optional real-data validation worktree path to record in the report.",
    )
    args = parser.parse_args()

    fov_ids = parse_int_list(cast(str | None, args.fovs), field_name="FOV id")
    target_rounds = parse_int_list(cast(str | None, args.rounds), field_name="round id")
    json_path, markdown_path, payload = profile_native_preprocessing_calibration(
        config_path=cast(Path, args.config),
        output_dir=cast(Path, args.output_dir),
        fov_ids=fov_ids,
        target_rounds=target_rounds,
        repeats=cast(int, args.repeats),
        baseline_commit=cast(str | None, args.baseline_commit),
        validation_worktree=cast(Path | None, args.validation_worktree),
    )
    print(f"Native preprocessing calibration profile JSON: {json_path}")
    print(f"Native preprocessing calibration profile Markdown: {markdown_path}")
    for fov in cast(Sequence[Mapping[str, object]], payload["fovs"]):
        equivalence = cast(Mapping[str, object], fov["clean_output_equivalence"])
        print(f"FOV {fov['fov_id']} clean-output equivalence: {equivalence['status']}")


if __name__ == "__main__":
    main()
