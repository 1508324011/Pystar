from __future__ import annotations

import copy
import json
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
import pytest
import tifffile
import yaml

DEFAULT_CLEAN_TIFF_DTYPE = np.dtype("uint8")

from scripts.profile_native_preprocessing_calibration import (
    CALIBRATION_PROFILE_SCHEMA_NAME,
    CALIBRATION_PROFILE_SCHEMA_VERSION,
    HISTOGRAM_REAL_MATCH_ATTRIBUTION_SCHEMA_NAME,
    HISTOGRAM_REAL_MATCH_ATTRIBUTION_SCHEMA_VERSION,
)
from scripts.validate_native_volume_worker_sweep import (
    STAGE32_POLICY_SCHEMA_NAME,
    build_sweep_payload,
    write_sweep_artifacts,
)


def _base_config_payload(tmp_path: Path) -> dict[str, Any]:
    return {
        "dataset": {
            "raw_data_path": str(tmp_path / "raw"),
            "filename_pattern": "round{round}/Position{fov}/image_ch{ch}.tif",
            "pixel_size_xy_nm": 100.0,
            "pixel_size_z_nm": 300.0,
            "dimensions": {"z": 1, "height": 3, "width": 4},
            "io_chunk_size": {"z": 1, "y": 3, "x": 4},
            "fov_list": 1,
            "round_structure": {1: [0, 1], 2: [0, 1]},
            "channel_roles": {0: "seq", 1: "seq"},
        },
        "codebook": {
            "gene_list": str(tmp_path / "genes.csv"),
            "channel_base_index": 0,
            "encoding_tables": {"default": {"AA": 0, "CC": 1}},
            "topology": {
                "func": "none",
                "structure": [
                    {
                        "id": "seg1",
                        "rounds": [1, 2],
                        "csv_slice": [1, 2],
                        "encoding_table": "default",
                    }
                ],
                "physical_order": ["seg1"],
            },
        },
        "pipeline": {
            "scope_mode": "full_fov",
            "accelerator": "cpu",
            "preprocessing": {
                "sequence": [{"method": "none", "provider": "native", "params": {}}],
            },
            "registration": {"reference_round": 1, "source": {"method": "mip_all_channels", "mip_channels": [0, 1]}},
            "spot_finding": {"algorithm": "peak_local_max", "provider": "native"},
            "extraction": {"method": "box_sum", "provider": "native", "transform_application_mode": "image_warp"},
            "decoding": {},
            "output": {"directory": str(tmp_path / "production_out"), "save_qc_images": True},
            "qc": {"enable": True},
        },
    }


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _stage32_payload(
    *,
    config_path: Path,
    output_dir: Path,
    profile_paths: Mapping[str, Path],
    tmp_path: Path,
    target_rounds: Sequence[int] | None,
    validation_worktree: Path | None = None,
    worker_counts: Sequence[int] = (1, 4),
    existing_profiles: Mapping[str, Path] | None = None,
    baseline_profile_json: Path | None = None,
    surface_scope: str = "full",
    limited_surface_reason: str = "not limited",
    production_root_base: Path | None = None,
    policy_candidate_workers: Sequence[int] = (4,),
) -> dict[str, object]:
    round_args = [] if target_rounds is None else ["--rounds", ",".join(str(round_id) for round_id in target_rounds)]
    existing = profile_paths if existing_profiles is None else existing_profiles
    return build_sweep_payload(
        config_path=config_path,
        output_dir=output_dir,
        worker_counts=worker_counts,
        expected_worker_counts=(1, 4),
        fov_ids=None,
        target_rounds=target_rounds,
        repeats=1,
        compare_repeat_index=0,
        baseline_commit="baseline",
        validation_worktree=validation_worktree,
        production_root_base=production_root_base or (tmp_path / "generated_production_roots"),
        source_root=Path(__file__).resolve().parents[2],
        existing_profiles=existing,
        baseline_profile_json=baseline_profile_json,
        skip_reason="not skipped",
        argv=["validate_native_volume_worker_sweep.py", "--stage32-policy", *round_args],
        stage32_policy=True,
        policy_candidate_workers=policy_candidate_workers,
        surface_scope=surface_scope,
        limited_surface_reason=limited_surface_reason,
    )


def _worker_config_payload(base_payload: dict[str, Any], *, worker_count: int, output_dir: Path) -> dict[str, Any]:
    payload = copy.deepcopy(base_payload)
    preprocessing = cast(dict[str, Any], cast(dict[str, Any], payload["pipeline"])["preprocessing"])
    preprocessing["native_volume_workers"] = worker_count
    output = cast(dict[str, Any], cast(dict[str, Any], payload["pipeline"])["output"])
    output["directory"] = str(output_dir)
    return payload


def _clean_filename(fov_id: int, round_id: int, channel_id: int) -> str:
    return f"clean_fov_{fov_id}_round_{round_id}_ch_{channel_id}.tif"


def _ensure_canonical_dirs(repeat_root: Path, fov_id: int) -> Path:
    fov_root = repeat_root / f"Position{fov_id}" / "output_pystar"
    for name in ("transforms", "spots", "extraction", "decoded", "qc_reports", "clean_data"):
        (fov_root / name).mkdir(parents=True, exist_ok=True)
    return fov_root / "clean_data"


def _write_clean_tiffs(
    *,
    repeat_root: Path,
    fov_id: int,
    rounds: Sequence[int],
    channels: Sequence[int],
    value_offset: int = 0,
    dtype: np.dtype[np.generic] = DEFAULT_CLEAN_TIFF_DTYPE,
    omitted_outputs: set[tuple[int, int]] | None = None,
) -> list[str]:
    clean_dir = _ensure_canonical_dirs(repeat_root, fov_id)
    output_files: list[str] = []
    omitted_outputs = set() if omitted_outputs is None else omitted_outputs
    for round_id in rounds:
        for channel_id in channels:
            if (int(round_id), int(channel_id)) in omitted_outputs:
                continue
            base = np.arange(12, dtype=np.uint8).reshape(1, 3, 4) + round_id * 10 + channel_id
            array = (base + value_offset).astype(dtype)
            path = clean_dir / _clean_filename(fov_id, round_id, channel_id)
            _ = tifffile.imwrite(path, array)
            _ = output_files.append(str(path))
    return output_files


def _profile_payload(
    *,
    label: str,
    worker_count: int,
    source_config_path: Path,
    profile_output_dir: Path,
    fov_id: int,
    target_rounds: Sequence[int] | None,
    wall_total_ms: float,
    source_commit: str = "stage32-test-commit",
    validation_worktree: Path | None = None,
    value_offset: int = 0,
    omitted_outputs: set[tuple[int, int]] | None = None,
    clean_channels: Sequence[int] = (0, 1),
) -> dict[str, Any]:
    _ = label
    rounds = [1, 2] if target_rounds is None else list(target_rounds)
    repeat_root = profile_output_dir / "fov_1" / "repeat_0"
    output_files = _write_clean_tiffs(
        repeat_root=repeat_root,
        fov_id=fov_id,
        rounds=rounds,
        channels=clean_channels,
        value_offset=value_offset,
        omitted_outputs=omitted_outputs,
    )
    attribution = {
        "schema_name": HISTOGRAM_REAL_MATCH_ATTRIBUTION_SCHEMA_NAME,
        "schema_version": HISTOGRAM_REAL_MATCH_ATTRIBUTION_SCHEMA_VERSION,
        "call_count": 4,
        "real_match_call_count": 2,
        "no_reference_noop_call_count": 2,
        "real_match_duration_ms": {"total_duration_ms": 10.0, "median_duration_ms": 5.0},
        "no_reference_noop_duration_ms": {"total_duration_ms": 2.0, "median_duration_ms": 1.0},
        "by_scope": {},
    }
    payload: dict[str, Any] = {
        "schema_name": CALIBRATION_PROFILE_SCHEMA_NAME,
        "schema_version": CALIBRATION_PROFILE_SCHEMA_VERSION,
        "source": {
            "repo_root": str(Path(__file__).resolve().parents[2]),
            "config_path": str(source_config_path),
            "config_sha256": _sha256_for_test(source_config_path),
            "candidate_commit": source_commit,
            "git_commit": source_commit,
            "validation_worktree": None if validation_worktree is None else str(validation_worktree),
        },
        "profile_configuration": {
            "fov_ids": [fov_id],
            "target_rounds": None if target_rounds is None else list(target_rounds),
            "repeats": 1,
            "output_dir": str(profile_output_dir),
        },
        "fovs": [
            {
                "fov_id": fov_id,
                "target_rounds": None if target_rounds is None else list(target_rounds),
                "repeats": [
                    {
                        "repeat_index": 0,
                        "repeat_output_root": str(repeat_root),
                        "output_files": output_files,
                    }
                ],
                "clean_output_equivalence": {
                    "status": "equivalent",
                    "file_count": len(output_files),
                    "mismatches": [],
                },
                "summary": {
                    "run_elapsed_wall_ms": {
                        "count": 1,
                        "total_duration_ms": wall_total_ms,
                        "median_duration_ms": wall_total_ms,
                        "mean_duration_ms": wall_total_ms,
                    },
                    "by_phase": {
                        "volume_total": {"total_duration_ms": wall_total_ms},
                        "calibration_steps": {"total_duration_ms": wall_total_ms / 2.0},
                        "extraction_steps": {"total_duration_ms": wall_total_ms / 2.0},
                    },
                    "histogram_match_profile": {"real_match_attribution": attribution},
                },
            }
        ],
    }
    _ = worker_count
    return payload


def _sha256_for_test(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"sha256:{digest}"


def _mapping(value: object) -> Mapping[str, Any]:
    return cast(Mapping[str, Any], value)


def _sequence(value: object) -> Sequence[Any]:
    return cast(Sequence[Any], value)


def _write_existing_profiles(
    *,
    tmp_path: Path,
    base_config: dict[str, Any],
    target_rounds: Sequence[int] | None,
    validation_worktree: Path | None = None,
    workers4_value_offset: int = 0,
    omitted_outputs: set[tuple[int, int]] | None = None,
    clean_channels: Sequence[int] = (0, 1),
) -> tuple[Path, Path, dict[str, Path]]:
    config_path = tmp_path / "base.yaml"
    _write_yaml(config_path, base_config)
    profile_paths: dict[str, Path] = {}
    for worker_count, wall_total_ms, value_offset in ((1, 100.0, 0), (4, 40.0, workers4_value_offset)):
        label = f"workers_{worker_count}"
        source_config = tmp_path / "profile_source_configs" / f"{label}.yaml"
        _write_yaml(
            source_config,
            _worker_config_payload(base_config, worker_count=worker_count, output_dir=tmp_path / "profile_production" / label),
        )
        payload = _profile_payload(
            label=label,
            worker_count=worker_count,
            source_config_path=source_config,
            profile_output_dir=tmp_path / "existing_profiles" / label,
            fov_id=1,
            target_rounds=target_rounds,
            wall_total_ms=wall_total_ms,
            validation_worktree=validation_worktree,
            value_offset=value_offset,
            omitted_outputs=omitted_outputs,
            clean_channels=clean_channels,
        )
        profile_path = tmp_path / "profile_json" / f"{label}.json"
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        _ = profile_path.write_text(json.dumps(payload), encoding="utf-8")
        profile_paths[label] = profile_path
    return config_path, tmp_path / "stage32_out", profile_paths


def test_stage32_full_surface_pass_records_policy_and_exact_clean_gate(tmp_path: Path) -> None:
    validation_worktree = tmp_path / "pystar-next"
    base_config = _base_config_payload(tmp_path)
    base_config["api_token"] = "super-secret"
    config_path, output_dir, profile_paths = _write_existing_profiles(
        tmp_path=tmp_path,
        base_config=base_config,
        target_rounds=None,
        validation_worktree=validation_worktree,
    )

    payload = _stage32_payload(
        config_path=config_path,
        output_dir=output_dir,
        target_rounds=None,
        validation_worktree=validation_worktree,
        profile_paths=profile_paths,
        tmp_path=tmp_path,
        surface_scope="full",
    )

    assert payload["schema_name"] == STAGE32_POLICY_SCHEMA_NAME
    verdict = _mapping(payload["verdict"])
    worker_policy = _mapping(payload["worker_policy"])
    assert verdict["status"] == "pass"
    assert verdict["recommended_worker_count"] == 4
    clean_gate = _mapping(worker_policy["clean_equivalence_gate"])
    assert clean_gate == {
        "status": "pass",
        "comparison_count": 1,
        "missing_count": 0,
        "extra_count": 0,
        "mismatch_count": 0,
        "shape_drift_count": 0,
        "dtype_drift_count": 0,
        "max_abs_diff": 0,
        "exact_equivalence_required": {
            "missing_count": 0,
            "extra_count": 0,
            "mismatch_count": 0,
            "shape_drift_count": 0,
            "dtype_drift_count": 0,
            "max_abs_diff": 0,
        },
    }
    worker4_config = next(
        _mapping(record)
        for record in _sequence(payload["generated_worker_configs"])
        if _mapping(record)["worker_count"] == 4
    )
    assert "payload" not in worker4_config
    worker4_payload = _mapping(worker4_config["payload_redacted"])
    assert worker4_payload["api_token"] == "<redacted>"
    worker4_pipeline = _mapping(worker4_payload["pipeline"])
    worker4_preprocessing = _mapping(worker4_pipeline["preprocessing"])
    assert worker4_preprocessing["native_volume_workers"] == 4
    assert _mapping(payload["downstream_metrics"])["status"] == "not_run"

    json_path, markdown_path = write_sweep_artifacts(payload, output_dir)
    assert json_path.name == "stage32_native_volume_worker_policy_validation.json"
    assert markdown_path.name == "stage32_native_volume_worker_policy_validation.md"
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "Stage32 Worker Policy Gate" in markdown
    assert "native_volume_workers=4" in markdown


def test_stage32_limited_surface_is_inconclusive_even_when_clean_outputs_match(tmp_path: Path) -> None:
    config_path, output_dir, profile_paths = _write_existing_profiles(
        tmp_path=tmp_path,
        base_config=_base_config_payload(tmp_path),
        target_rounds=(1,),
    )

    payload = _stage32_payload(
        config_path=config_path,
        output_dir=output_dir,
        profile_paths=profile_paths,
        tmp_path=tmp_path,
        target_rounds=(1,),
        surface_scope="limited",
        limited_surface_reason="unit test intentionally measures one round",
    )

    worker_policy = _mapping(payload["worker_policy"])
    assert _mapping(worker_policy["clean_equivalence_gate"])["status"] == "pass"
    assert _mapping(worker_policy["surface"])["effective_surface_scope"] == "limited"
    verdict = _mapping(payload["verdict"])
    assert verdict["status"] == "inconclusive"
    assert "limited surface" in str(verdict["reason"])


def test_stage32_declared_full_multi_fov_subset_is_inconclusive(tmp_path: Path) -> None:
    base_config = _base_config_payload(tmp_path)
    dataset = cast(dict[str, Any], base_config["dataset"])
    dataset["fov_list"] = [1, 2]
    config_path, output_dir, profile_paths = _write_existing_profiles(
        tmp_path=tmp_path,
        base_config=base_config,
        target_rounds=None,
    )

    payload = _stage32_payload(
        config_path=config_path,
        output_dir=output_dir,
        profile_paths=profile_paths,
        tmp_path=tmp_path,
        target_rounds=None,
        surface_scope="full",
    )

    worker_policy = _mapping(payload["worker_policy"])
    surface = _mapping(worker_policy["surface"])
    assert surface["effective_surface_scope"] == "limited"
    assert surface["fov_surface_complete"] is False
    assert surface["missing_configured_fov_ids"] == [2]
    assert _mapping(worker_policy["surface_completeness_gate"])["status"] == "pass"
    assert _mapping(payload["verdict"])["status"] == "inconclusive"


def test_stage32_shared_clean_tiff_omission_fails_expected_surface_gate(tmp_path: Path) -> None:
    config_path, output_dir, profile_paths = _write_existing_profiles(
        tmp_path=tmp_path,
        base_config=_base_config_payload(tmp_path),
        target_rounds=None,
        omitted_outputs={(2, 1)},
    )

    payload = _stage32_payload(
        config_path=config_path,
        output_dir=output_dir,
        profile_paths=profile_paths,
        tmp_path=tmp_path,
        target_rounds=None,
        surface_scope="full",
    )

    worker_policy = _mapping(payload["worker_policy"])
    assert _mapping(worker_policy["clean_equivalence_gate"])["status"] == "pass"
    surface_gate = _mapping(worker_policy["surface_completeness_gate"])
    assert surface_gate["status"] == "fail"
    assert surface_gate["missing_expected_clean_tiff_count"] == 2
    first_profile_row = _mapping(_sequence(surface_gate["profile_rows"])[0])
    assert first_profile_row["missing_expected_clean_tiff_count"] == 1
    assert "clean_fov_1_round_2_ch_1.tif" in str(first_profile_row["missing_expected_clean_tiffs"])
    assert _mapping(payload["verdict"])["status"] == "fail"


def test_stage32_expected_surface_uses_only_seq_channels(tmp_path: Path) -> None:
    base_config = _base_config_payload(tmp_path)
    dataset = cast(dict[str, Any], base_config["dataset"])
    dataset["round_structure"] = {1: [0, 1, 2], 2: [0, 1, 2]}
    dataset["channel_roles"] = {0: "seq", 1: "seq", 2: "anchor"}
    config_path, output_dir, profile_paths = _write_existing_profiles(
        tmp_path=tmp_path,
        base_config=base_config,
        target_rounds=None,
        clean_channels=(0, 1),
    )

    payload = _stage32_payload(
        config_path=config_path,
        output_dir=output_dir,
        profile_paths=profile_paths,
        tmp_path=tmp_path,
        target_rounds=None,
        surface_scope="full",
    )

    surface = _mapping(payload["validation_surface"])
    assert surface["selected_channel_ids_by_round"] == {"1": [0, 1], "2": [0, 1]}
    surface_gate = _mapping(_mapping(payload["worker_policy"])["surface_completeness_gate"])
    assert surface_gate["status"] == "pass"
    assert surface_gate["expected_clean_tiff_count"] == 4
    assert "clean_fov_1_round_1_ch_2.tif" not in str(surface_gate)


def test_stage32_rejects_non_serial_loaded_baseline_profile(tmp_path: Path) -> None:
    config_path, output_dir, profile_paths = _write_existing_profiles(
        tmp_path=tmp_path,
        base_config=_base_config_payload(tmp_path),
        target_rounds=None,
    )

    payload = _stage32_payload(
        config_path=config_path,
        output_dir=output_dir,
        worker_counts=(4,),
        profile_paths=profile_paths,
        target_rounds=None,
        existing_profiles={"workers_4": profile_paths["workers_4"]},
        baseline_profile_json=profile_paths["workers_4"],
        tmp_path=tmp_path,
        surface_scope="full",
    )

    verdict = _mapping(payload["verdict"])
    assert verdict["status"] == "fail"
    fail_loud_errors = _sequence(payload["fail_loud_errors"])
    baseline_error = next(_mapping(record) for record in fail_loud_errors if _mapping(record)["label"] == "baseline")
    assert "does not match expected 1" in str(baseline_error["error_message"])


def test_stage32_value_drift_fails_exact_clean_gate(tmp_path: Path) -> None:
    config_path, output_dir, profile_paths = _write_existing_profiles(
        tmp_path=tmp_path,
        base_config=_base_config_payload(tmp_path),
        target_rounds=None,
        workers4_value_offset=3,
    )

    payload = _stage32_payload(
        config_path=config_path,
        output_dir=output_dir,
        profile_paths=profile_paths,
        tmp_path=tmp_path,
        target_rounds=None,
        surface_scope="full",
    )

    worker_policy = _mapping(payload["worker_policy"])
    clean_gate = _mapping(worker_policy["clean_equivalence_gate"])
    assert _mapping(payload["verdict"])["status"] == "fail"
    assert clean_gate["status"] == "fail"
    assert clean_gate["mismatch_count"] == 4
    assert clean_gate["shape_drift_count"] == 0
    assert clean_gate["dtype_drift_count"] == 0
    assert clean_gate["max_abs_diff"] == 3
    clean_tiff_equivalence = _mapping(payload["clean_tiff_equivalence"])
    first_comparison = _mapping(_sequence(clean_tiff_equivalence["comparisons"])[0])
    first_file_row = _mapping(_sequence(first_comparison["file_rows"])[0])
    assert first_file_row["max_abs_diff"] == 3


def test_stage32_rejects_candidate_worker_equal_to_serial_baseline(tmp_path: Path) -> None:
    config_path, output_dir, profile_paths = _write_existing_profiles(
        tmp_path=tmp_path,
        base_config=_base_config_payload(tmp_path),
        target_rounds=None,
    )

    with pytest.raises(ValueError, match="exclude the serial/default baseline"):
        _ = _stage32_payload(
            config_path=config_path,
            output_dir=output_dir,
            profile_paths=profile_paths,
            tmp_path=tmp_path,
            target_rounds=None,
            policy_candidate_workers=(1,),
        )


def test_stage32_rejects_generated_output_overlap_with_production_root(tmp_path: Path) -> None:
    base_config = _base_config_payload(tmp_path)
    config_path, output_dir, profile_paths = _write_existing_profiles(
        tmp_path=tmp_path,
        base_config=base_config,
        target_rounds=None,
    )
    production_output_dir = Path(str(_mapping(_mapping(base_config["pipeline"])["output"])["directory"]))

    with pytest.raises(ValueError, match="must not overlap the base config"):
        _ = _stage32_payload(
            config_path=config_path,
            output_dir=output_dir,
            profile_paths=profile_paths,
            tmp_path=tmp_path,
            target_rounds=None,
            production_root_base=production_output_dir,
        )
