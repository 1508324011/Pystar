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
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np


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

from pystar import preprocessing as preprocessing_module
from pystar.infrastructure import ExperimentConfig, PreprocessingStep, load_config
from pystar.io import ImageLoader, get_fov_output_structure
from pystar.preprocessing import (
    DataSanitizer,
    NativeOutputWriterWithPlanner,
    NATIVE_PREPROCESSING_TIMING_SCHEMA_NAME,
    NATIVE_PREPROCESSING_TIMING_SCHEMA_VERSION,
)
from pystar.serialization import write_backend_metadata


CALIBRATION_PROFILE_SCHEMA_NAME = "pystar_native_preprocessing_calibration_profile"
CALIBRATION_PROFILE_SCHEMA_VERSION = 1
HISTOGRAM_MATCH_PROFILE_SCHEMA_NAME = "pystar_native_histogram_match_profile"
HISTOGRAM_MATCH_PROFILE_SCHEMA_VERSION = 1
HISTOGRAM_REAL_MATCH_ATTRIBUTION_SCHEMA_NAME = "pystar_native_histogram_real_match_attribution"
HISTOGRAM_REAL_MATCH_ATTRIBUTION_SCHEMA_VERSION = 1
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
    histogram_match_calls: tuple[Mapping[str, Any], ...] = ()


def _array_dtype(value: object | None) -> str | None:
    if value is None:
        return None
    return str(np.asarray(value).dtype)


def _array_shape(value: object | None) -> list[int] | None:
    if value is None:
        return None
    return [int(dimension) for dimension in np.asarray(value).shape]


def _histogram_reference_for_scope(
    params: Mapping[str, Any],
    context: Mapping[str, Any],
) -> tuple[str, object | None]:
    scope = str(params.get("scope", "none"))
    if scope == "inter_round":
        return scope, context.get("ref_round_image")
    if scope == "intra_round":
        return scope, context.get("ref_channel_image")
    return scope, None


@contextmanager
def _capture_histogram_match_calls() -> Iterator[list[dict[str, object]]]:
    """Capture histogram-match call details for the profiling harness only.

    The production native preprocessing provenance already records coarse
    per-step timings. Stage25 needs extra scope/reference/dtype/shape evidence
    without changing that production payload, so the profiling script wraps the
    registry entry only for the duration of a repeat run and restores it
    immediately afterwards.
    """

    original = preprocessing_module.PROCESSOR_MAP["histogram_match"]
    calls: list[dict[str, object]] = []

    def profiled_histogram_match(img: Any, params: Mapping[str, Any], ctx: Mapping[str, Any]) -> Any:
        scope, reference = _histogram_reference_for_scope(params, ctx)
        started = time.perf_counter()
        output = original(img, dict(params), dict(ctx))
        duration_ms = round((time.perf_counter() - started) * 1000.0, 3)

        has_reference = reference is not None
        calls.append(
            {
                "call_index": len(calls),
                "scope": scope,
                "has_reference": has_reference,
                "operation": "match_histograms" if has_reference else "no_reference_noop",
                "duration_ms": duration_ms,
                "input_dtype": _array_dtype(img),
                "input_shape": _array_shape(img),
                "reference_dtype": _array_dtype(reference),
                "reference_shape": _array_shape(reference),
                "output_dtype": _array_dtype(output),
                "output_shape": _array_shape(output),
                "output_is_input": output is img,
            }
        )
        return output

    preprocessing_module.PROCESSOR_MAP["histogram_match"] = profiled_histogram_match
    try:
        yield calls
    finally:
        preprocessing_module.PROCESSOR_MAP["histogram_match"] = original


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
    def output_path_for(round_id: int, channel_id: int) -> Path:
        paths = get_fov_output_structure(repeat_output_root, fov_id)
        output_path = paths["cleaned"] / sanitizer._flat_clean_filename(fov_id, round_id, channel_id)
        _assert_path_within(output_path, repeat_output_root, field_name="clean output file")
        return output_path

    def write(img: Any, round_id: int, channel_id: int) -> Path:
        output_path = output_path_for(round_id, channel_id)
        _reject_symlink_components(output_path.parent, field_name="clean output directory")
        _assert_path_within(output_path.parent, repeat_output_root, field_name="clean output directory")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Reuse the production clean-image writer for the TIFF contract while
        # directing output to the profiling repeat root instead of the configured
        # production output directory.
        import tifffile

        tifffile.imwrite(output_path, img, compression="zlib")
        return output_path

    return NativeOutputWriterWithPlanner(write=write, output_path_for=output_path_for)


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


def _unique_string_values(values: Sequence[object | None]) -> list[str]:
    return sorted({str(value) for value in values if value is not None})


def _unique_shapes(values: Sequence[object | None]) -> list[list[int]]:
    shapes: set[tuple[int, ...]] = set()
    for value in values:
        if value is None:
            continue
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            raise ValueError(f"Histogram profile shape field must be a sequence of integers or null; got {value!r}")
        shape_tuple = tuple(_coerce_shape_dimension(dimension) for dimension in value)
        shapes.add(shape_tuple)
    return [[int(dimension) for dimension in shape] for shape in sorted(shapes)]


def _coerce_shape_dimension(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"Histogram profile shape dimensions must be integers; got {value!r}")
    dimension = value.item() if isinstance(value, np.integer) else value
    if dimension < 0:
        raise ValueError(f"Histogram profile shape dimensions must be non-negative; got {value!r}")
    return dimension


def _histogram_call_duration(call: Mapping[str, Any]) -> float:
    value = call.get("duration_ms")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Histogram profile duration_ms must be numeric; got {value!r}")
    duration = float(value)
    if not math.isfinite(duration) or duration < 0:
        raise ValueError(f"Histogram profile duration_ms must be finite and non-negative; got {value!r}")
    return duration


def _histogram_real_match_attribution(calls: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    """Build the Stage29 real-match/no-op attribution block.

    Stage25 already recorded individual histogram-match calls in the dedicated
    calibration profile artifact. Stage29 keeps the same persisted profile
    schema compatible and adds a clearer attribution view so real reference-
    backed work is not averaged together with intentional no-reference no-ops.
    """

    real_match_calls = [call for call in calls if bool(call.get("has_reference"))]
    no_reference_noop_calls = [call for call in calls if not bool(call.get("has_reference"))]
    real_match_duration = _duration_summary([_histogram_call_duration(call) for call in real_match_calls])
    no_reference_noop_duration = _duration_summary(
        [_histogram_call_duration(call) for call in no_reference_noop_calls]
    )

    by_scope: dict[str, object] = {}
    for scope in sorted({str(call.get("scope", "none")) for call in calls}):
        scoped_calls = [call for call in calls if str(call.get("scope", "none")) == scope]
        scoped_real_matches = [call for call in scoped_calls if bool(call.get("has_reference"))]
        scoped_noops = [call for call in scoped_calls if not bool(call.get("has_reference"))]
        scoped_real_duration = _duration_summary(
            [_histogram_call_duration(call) for call in scoped_real_matches]
        )
        scoped_noop_duration = _duration_summary([_histogram_call_duration(call) for call in scoped_noops])
        by_scope[scope] = {
            "call_count": len(scoped_calls),
            "real_match_call_count": len(scoped_real_matches),
            "no_reference_noop_call_count": len(scoped_noops),
            "real_match_duration_ms": scoped_real_duration,
            "no_reference_noop_duration_ms": scoped_noop_duration,
            "real_match_total_duration_ms": scoped_real_duration["total_duration_ms"],
            "real_match_median_duration_ms": scoped_real_duration["median_duration_ms"],
            "no_reference_noop_total_duration_ms": scoped_noop_duration["total_duration_ms"],
            "no_reference_noop_median_duration_ms": scoped_noop_duration["median_duration_ms"],
            "input_dtypes": _unique_string_values([call.get("input_dtype") for call in scoped_calls]),
            "reference_dtypes": _unique_string_values(
                [call.get("reference_dtype") for call in scoped_real_matches]
            ),
            "output_dtypes": _unique_string_values([call.get("output_dtype") for call in scoped_calls]),
            "input_shapes": _unique_shapes([call.get("input_shape") for call in scoped_calls]),
            "reference_shapes": _unique_shapes(
                [call.get("reference_shape") for call in scoped_real_matches]
            ),
            "output_shapes": _unique_shapes([call.get("output_shape") for call in scoped_calls]),
        }

    return {
        "schema_name": HISTOGRAM_REAL_MATCH_ATTRIBUTION_SCHEMA_NAME,
        "schema_version": HISTOGRAM_REAL_MATCH_ATTRIBUTION_SCHEMA_VERSION,
        "source": "histogram_match_calls.has_reference",
        "call_count": len(calls),
        "real_match_call_count": len(real_match_calls),
        "no_reference_noop_call_count": len(no_reference_noop_calls),
        "real_match_duration_ms": real_match_duration,
        "no_reference_noop_duration_ms": no_reference_noop_duration,
        "real_match_total_duration_ms": real_match_duration["total_duration_ms"],
        "real_match_median_duration_ms": real_match_duration["median_duration_ms"],
        "no_reference_noop_total_duration_ms": no_reference_noop_duration["total_duration_ms"],
        "no_reference_noop_median_duration_ms": no_reference_noop_duration["median_duration_ms"],
        "classification": {
            "real_match": "has_reference is true; operation is reference-backed skimage.exposure.match_histograms",
            "no_reference_noop": "has_reference is false; op_histogram_match returned the input object unchanged",
        },
        "by_scope": by_scope,
    }


def _summarize_histogram_calls(calls: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    scopes = sorted({str(call.get("scope", "none")) for call in calls})
    by_scope: dict[str, object] = {}
    for scope in scopes:
        scoped_calls = [call for call in calls if str(call.get("scope", "none")) == scope]
        matched_calls = [call for call in scoped_calls if bool(call.get("has_reference"))]
        no_reference_calls = [call for call in scoped_calls if not bool(call.get("has_reference"))]
        matched_duration = _duration_summary([_histogram_call_duration(call) for call in matched_calls])
        no_reference_duration = _duration_summary([_histogram_call_duration(call) for call in no_reference_calls])
        by_scope[scope] = {
            "call_count": len(scoped_calls),
            "match_call_count": len(matched_calls),
            "no_reference_call_count": len(no_reference_calls),
            "real_match_call_count": len(matched_calls),
            "no_reference_noop_call_count": len(no_reference_calls),
            "duration_ms": _duration_summary([_histogram_call_duration(call) for call in scoped_calls]),
            "matched_duration_ms": matched_duration,
            "no_reference_duration_ms": no_reference_duration,
            "real_match_duration_ms": matched_duration,
            "no_reference_noop_duration_ms": no_reference_duration,
            "input_dtypes": _unique_string_values([call.get("input_dtype") for call in scoped_calls]),
            "reference_dtypes": _unique_string_values([call.get("reference_dtype") for call in matched_calls]),
            "output_dtypes": _unique_string_values([call.get("output_dtype") for call in scoped_calls]),
            "input_shapes": _unique_shapes([call.get("input_shape") for call in scoped_calls]),
            "reference_shapes": _unique_shapes([call.get("reference_shape") for call in matched_calls]),
            "output_shapes": _unique_shapes([call.get("output_shape") for call in scoped_calls]),
        }

    matched_all = [call for call in calls if bool(call.get("has_reference"))]
    no_reference_all = [call for call in calls if not bool(call.get("has_reference"))]
    matched_duration = _duration_summary([_histogram_call_duration(call) for call in matched_all])
    no_reference_duration = _duration_summary([_histogram_call_duration(call) for call in no_reference_all])
    return {
        "schema_name": HISTOGRAM_MATCH_PROFILE_SCHEMA_NAME,
        "schema_version": HISTOGRAM_MATCH_PROFILE_SCHEMA_VERSION,
        "call_count": len(calls),
        "match_call_count": len(matched_all),
        "no_reference_call_count": len(no_reference_all),
        "real_match_call_count": len(matched_all),
        "no_reference_noop_call_count": len(no_reference_all),
        "duration_ms": _duration_summary([_histogram_call_duration(call) for call in calls]),
        "matched_duration_ms": matched_duration,
        "no_reference_duration_ms": no_reference_duration,
        "real_match_duration_ms": matched_duration,
        "no_reference_noop_duration_ms": no_reference_duration,
        "real_match_attribution": _histogram_real_match_attribution(calls),
        "by_scope": by_scope,
    }


def _histogram_profile_for_runs(run_results: Sequence[ProfileRunResult]) -> dict[str, object]:
    all_calls: list[Mapping[str, Any]] = []
    repeat_summaries: list[dict[str, object]] = []
    for run in run_results:
        run_calls = [dict(call) for call in run.histogram_match_calls]
        all_calls.extend(run_calls)
        repeat_summary = _summarize_histogram_calls(run_calls)
        repeat_summary["repeat_index"] = int(run.repeat_index)
        repeat_summaries.append(repeat_summary)

    aggregate = _summarize_histogram_calls(all_calls)
    aggregate["repeats"] = repeat_summaries
    return aggregate


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
        "histogram_match_profile": _histogram_profile_for_runs(run_results),
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
        with _capture_histogram_match_calls() as histogram_match_calls:
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
                histogram_match_calls=tuple(dict(call) for call in histogram_match_calls),
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
            histogram_match_calls = [dict(call) for call in result.histogram_match_calls]
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
                    "histogram_match_calls": histogram_match_calls,
                    "histogram_match_profile": _summarize_histogram_calls(histogram_match_calls),
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
        "## Profile Schema",
        "",
        f"- Calibration profile schema: `{payload['schema_name']}` v`{payload['schema_version']}`",
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
        f"- Histogram profile schema: `{HISTOGRAM_MATCH_PROFILE_SCHEMA_NAME}` v`{HISTOGRAM_MATCH_PROFILE_SCHEMA_VERSION}`",
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
        histogram_profile = cast(Mapping[str, object], summary["histogram_match_profile"])
        histogram_by_scope = cast(Mapping[str, object], histogram_profile["by_scope"])
        lines.extend(
            [
                f"### FOV {fov['fov_id']}",
                "",
                f"- Round order: `{fov['round_order']}`",
                f"- Clean-output equivalence: `{equivalence['status']}` across `{equivalence['file_count']}` files",
                f"- Calibration phase: `{cast(Mapping[str, object], by_phase['calibration_steps']).get('total_duration_ms')}` ms total",
                f"- Extraction phase: `{cast(Mapping[str, object], by_phase['extraction_steps']).get('total_duration_ms')}` ms total",
                f"- Histogram-match calls: `{histogram_profile['call_count']}` total; `{histogram_profile['match_call_count']}` real matches; `{histogram_profile['no_reference_call_count']}` no-reference no-ops",
                f"- Histogram real-match total / median: `{cast(Mapping[str, object], histogram_profile['real_match_duration_ms']).get('total_duration_ms')}` / `{cast(Mapping[str, object], histogram_profile['real_match_duration_ms']).get('median_duration_ms')}` ms",
                f"- Histogram no-reference no-op total / median: `{cast(Mapping[str, object], histogram_profile['no_reference_noop_duration_ms']).get('total_duration_ms')}` / `{cast(Mapping[str, object], histogram_profile['no_reference_noop_duration_ms']).get('median_duration_ms')}` ms",
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
                "#### Histogram-match scope breakdown",
                "",
                "| Scope | Calls | Real matches | No-reference no-ops | Total ms | Real total ms | Real median ms | No-op total ms | No-op median ms | Input dtypes | Output dtypes | Input shapes |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
            ]
        )
        for scope, raw_scope_summary in sorted(histogram_by_scope.items()):
            scope_summary = cast(Mapping[str, object], raw_scope_summary)
            duration = cast(Mapping[str, object], scope_summary["duration_ms"])
            matched_duration = cast(Mapping[str, object], scope_summary["matched_duration_ms"])
            no_ref_duration = cast(Mapping[str, object], scope_summary["no_reference_noop_duration_ms"])
            lines.append(
                "| {scope} | {calls} | {matches} | {no_refs} | {total_ms} | {match_total_ms} | {match_median_ms} | {no_ref_total_ms} | {no_ref_median_ms} | `{input_dtypes}` | `{output_dtypes}` | `{input_shapes}` |".format(
                    scope=scope,
                    calls=scope_summary["call_count"],
                    matches=scope_summary["match_call_count"],
                    no_refs=scope_summary["no_reference_call_count"],
                    total_ms=duration.get("total_duration_ms"),
                    match_total_ms=matched_duration.get("total_duration_ms"),
                    match_median_ms=matched_duration.get("median_duration_ms"),
                    no_ref_total_ms=no_ref_duration.get("total_duration_ms"),
                    no_ref_median_ms=no_ref_duration.get("median_duration_ms"),
                    input_dtypes=scope_summary.get("input_dtypes"),
                    output_dtypes=scope_summary.get("output_dtypes"),
                    input_shapes=scope_summary.get("input_shapes"),
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
            "When running under `/media/zenglab/result/zhui/Leica_deconv_test_260106-worktrees/pystar-next`, keep the PyStar source fixed via `PYTHONPATH=/media/zenglab/result/zhui/PyStar` and write the JSON/Markdown artifacts to a validation-only directory. The validation report should record source commit, config path/hash, FOV/round/channel surface, JSON/Markdown profile artifact paths, clean TIFF paths, clean-output equivalence by shape/dtype/hash or array contents, histogram scope totals, reference/no-reference counts, calibration total, extraction total, volume total, wall time, and any cache/warmup drift.",
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
