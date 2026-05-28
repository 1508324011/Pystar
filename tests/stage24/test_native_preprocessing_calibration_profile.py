from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence, cast

import numpy as np
import pytest
import tifffile

from pystar.infrastructure import PreprocessingStep

from scripts.profile_native_preprocessing_calibration import (
    CALIBRATION_PROFILE_SCHEMA_NAME,
    CALIBRATION_PROFILE_SCHEMA_VERSION,
    build_profile_payload,
    parse_int_list,
    write_profile_artifacts,
)


def _step(method: str, params: dict[str, Any] | None = None) -> PreprocessingStep:
    return PreprocessingStep(method=method, provider="native", params={} if params is None else params)


def _config(tmp_path: Path, sequence: list[PreprocessingStep]) -> Any:
    dataset = SimpleNamespace(
        raw_data_path=tmp_path / "raw",
        filename_pattern="round{round}/Position{fov}/image_ch{ch}.tif",
        round_structure={1: [0, 1], 2: [0, 1]},
        channel_roles={0: "seq", 1: "seq"},
        dimensions={"z": 1, "height": 3, "width": 4},
        io_chunk_size={"z": 1, "y": 3, "x": 4},
        parsed_fovs=[1],
    )
    pipeline = SimpleNamespace(
        preprocessing=SimpleNamespace(sequence=sequence),
        output=SimpleNamespace(directory=str(tmp_path / "production_out")),
    )
    pipeline.preprocessing_providers_used = lambda: ["native"]
    pipeline.preprocessing_provider_mode = lambda: "native_only"
    return SimpleNamespace(
        dataset=dataset,
        pipeline=pipeline,
        config_sha256="sha256:test-config",
        config_source_path=tmp_path / "config.yaml",
    )


def _write_raw_fixture(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    for round_id in (1, 2):
        for channel_id in (0, 1):
            path = raw_root / f"round{round_id}" / "Position1" / f"image_ch{channel_id:02d}.tif"
            path.parent.mkdir(parents=True, exist_ok=True)
            base = round_id * 40 + channel_id * 7
            image = (np.arange(12, dtype=np.uint8).reshape(1, 3, 4) + base).astype(np.uint8)
            tifffile.imwrite(path, image)


def _first_fov(payload: dict[str, object]) -> dict[str, Any]:
    fovs = cast(Sequence[dict[str, Any]], payload["fovs"])
    return fovs[0]


def _mapping(value: object) -> dict[str, Any]:
    return cast(dict[str, Any], value)


def test_parse_int_list_deduplicates_and_fails_loudly() -> None:
    assert parse_int_list("1,2,1", field_name="FOV id") == (1, 2)
    assert parse_int_list(None, field_name="FOV id") is None

    with pytest.raises(ValueError, match="Invalid FOV id value"):
        _ = parse_int_list("1,nope", field_name="FOV id")

    with pytest.raises(ValueError, match="non-negative"):
        _ = parse_int_list("-1", field_name="FOV id")


def test_profile_payload_measures_calibration_and_preserves_clean_output_contract(tmp_path: Path) -> None:
    _write_raw_fixture(tmp_path)
    cfg = _config(
        tmp_path,
        [
            _step("min_max_normalize"),
            _step("histogram_match", {"scope": "inter_round"}),
            _step("histogram_match", {"scope": "intra_round"}),
            _step("morpho_reconstruction_contrast", {"radius": 1, "downsample_factor": 1.0}),
        ],
    )
    output_dir = tmp_path / "profile_artifacts"

    payload = build_profile_payload(
        config=cast(Any, cfg),
        config_path=tmp_path / "config.yaml",
        fov_ids=(1,),
        target_rounds=(1, 2),
        repeats=2,
        output_dir=output_dir,
        baseline_commit="baseline-test-commit",
        validation_worktree=tmp_path / "pystar-next",
    )

    assert payload["schema_name"] == CALIBRATION_PROFILE_SCHEMA_NAME
    assert payload["schema_version"] == CALIBRATION_PROFILE_SCHEMA_VERSION
    source = _mapping(payload["source"])
    assert source["baseline_commit"] == "baseline-test-commit"
    assert isinstance(source["candidate_commit"], (str, type(None)))
    assert payload["contracts"] == {
        "production_runtime_instrumentation_added": False,
        "canonical_clean_output_filename_contract": "clean_fov_{fov_id}_round_{round_id}_ch_{channel_id}.tif",
        "timing_source_schema_name": "pystar_native_preprocessing_timing",
        "timing_source_schema_version": 1,
        "speedup_claim": "none; profiling harness only reports measured timings",
    }
    assert _mapping(payload["hotspot_call"])["status"] == "manual_required"

    fov_payload = _first_fov(payload)
    assert fov_payload["round_order"] == [1, 2]
    assert fov_payload["target_rounds"] == [1, 2]
    assert fov_payload["clean_output_equivalence"]["status"] == "equivalent"
    assert fov_payload["clean_output_equivalence"]["file_count"] == 4
    first_baseline = next(iter(fov_payload["clean_output_equivalence"]["baseline_files"].values()))
    assert first_baseline["shape"] == [3, 4]
    assert first_baseline["dtype"] == "uint8"
    assert str(first_baseline["sha256"]).startswith("sha256:")
    assert fov_payload["summary"]["by_phase"]["calibration_steps"]["count"] == 8
    assert fov_payload["summary"]["focused_methods"]["histogram_match"]["calibration_phase"]["count"] == 16
    assert fov_payload["summary"]["focused_methods"]["morpho_reconstruction_contrast"]["calibration_phase"]["count"] == 0
    assert fov_payload["summary"]["focused_methods"]["morpho_reconstruction_contrast"]["extraction_phase"]["count"] == 8

    for repeat in fov_payload["repeats"]:
        assert repeat["timing"]["schema_name"] == "pystar_native_preprocessing_timing"
        output_files = [Path(path) for path in repeat["output_files"]]
        assert {path.name for path in output_files} == {
            "clean_fov_1_round_1_ch_0.tif",
            "clean_fov_1_round_1_ch_1.tif",
            "clean_fov_1_round_2_ch_0.tif",
            "clean_fov_1_round_2_ch_1.tif",
        }
        assert all("profile_artifacts" in str(path) for path in output_files)
        assert all(path.exists() for path in output_files)

    production_root = Path(cfg.pipeline.output.directory)
    assert not production_root.exists()


def test_profile_artifacts_are_written_as_json_and_markdown_without_speedup_claim(tmp_path: Path) -> None:
    _write_raw_fixture(tmp_path)
    cfg = _config(tmp_path, [_step("min_max_normalize"), _step("histogram_match", {"scope": "inter_round"})])
    output_dir = tmp_path / "profile_artifacts"
    payload = build_profile_payload(
        config=cast(Any, cfg),
        config_path=tmp_path / "config.yaml",
        fov_ids=(1,),
        target_rounds=(1,),
        repeats=1,
        output_dir=output_dir,
    )

    json_path, markdown_path = write_profile_artifacts(payload, output_dir)

    persisted = json.loads(json_path.read_text(encoding="utf-8"))
    assert persisted["schema_name"] == CALIBRATION_PROFILE_SCHEMA_NAME
    assert persisted["contracts"]["speedup_claim"] == "none; profiling harness only reports measured timings"
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "Do not claim an end-to-end speedup" in markdown
    assert "Baseline commit" in markdown
    assert "Candidate commit" in markdown
    assert "Extraction total ms" in markdown
    assert "Real-Data Validation Artifact Expectations" in markdown


def test_profile_artifact_writer_rejects_symlinked_temp_json(tmp_path: Path) -> None:
    output_dir = tmp_path / "profile_artifacts"
    output_dir.mkdir()
    marker = output_dir / ".pystar_native_preprocessing_calibration_profile"
    marker.write_text(
        "schema_name=pystar_native_preprocessing_calibration_profile\nschema_version=1\n",
        encoding="utf-8",
    )
    target = tmp_path / "outside.json"
    temp_json = output_dir / "native_preprocessing_calibration_profile.json.tmp"
    temp_json.symlink_to(target)

    with pytest.raises(ValueError, match="temporary profile JSON path"):
        _ = write_profile_artifacts(
            {
                "schema_name": CALIBRATION_PROFILE_SCHEMA_NAME,
                "schema_version": CALIBRATION_PROFILE_SCHEMA_VERSION,
                "source": {},
                "profile_configuration": {},
                "contracts": {},
                "fovs": [],
            },
            output_dir,
        )

    assert not target.exists()
    assert not (output_dir / "native_preprocessing_calibration_profile.json").exists()


def test_profile_artifact_writer_rejects_dangling_temp_symlink_inside_output_dir(tmp_path: Path) -> None:
    output_dir = tmp_path / "profile_artifacts"
    output_dir.mkdir()
    marker = output_dir / ".pystar_native_preprocessing_calibration_profile"
    marker.write_text(
        "schema_name=pystar_native_preprocessing_calibration_profile\nschema_version=1\n",
        encoding="utf-8",
    )
    temp_json = output_dir / "native_preprocessing_calibration_profile.json.tmp"
    temp_json.symlink_to(output_dir / "uncreated-target.json")

    with pytest.raises(ValueError, match="symlink component"):
        _ = write_profile_artifacts(
            {
                "schema_name": CALIBRATION_PROFILE_SCHEMA_NAME,
                "schema_version": CALIBRATION_PROFILE_SCHEMA_VERSION,
                "source": {},
                "profile_configuration": {},
                "contracts": {},
                "fovs": [],
            },
            output_dir,
        )

    assert not (output_dir / "uncreated-target.json").exists()
    assert temp_json.is_symlink()
    assert not (output_dir / "native_preprocessing_calibration_profile.json").exists()


def test_profile_payload_rejects_non_native_or_illegal_histogram_scope(tmp_path: Path) -> None:
    cfg_non_native = _config(
        tmp_path,
        [PreprocessingStep(method="min_max_normalize", provider="matlab", params={})],
    )
    with pytest.raises(ValueError, match="provider='native'"):
        _ = build_profile_payload(
            config=cast(Any, cfg_non_native),
            config_path=tmp_path / "config.yaml",
            fov_ids=(1,),
            target_rounds=None,
            repeats=1,
            output_dir=tmp_path / "profile_artifacts",
        )


def test_profile_output_dir_must_be_dedicated_before_repeat_deletion(tmp_path: Path) -> None:
    _write_raw_fixture(tmp_path)
    cfg = _config(tmp_path, [_step("min_max_normalize")])
    output_dir = tmp_path / "existing_non_profile"
    output_dir.mkdir()
    keep_file = output_dir / "do_not_delete.txt"
    keep_file.write_text("keep", encoding="utf-8")
    repeat_root = output_dir / "fov_1" / "repeat_0"
    repeat_root.mkdir(parents=True)
    sentinel = repeat_root / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="not a Stage24 profile output directory"):
        _ = build_profile_payload(
            config=cast(Any, cfg),
            config_path=tmp_path / "config.yaml",
            fov_ids=(1,),
            target_rounds=(1,),
            repeats=1,
            output_dir=output_dir,
        )

    assert keep_file.read_text(encoding="utf-8") == "keep"
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_profile_output_dir_rejects_stale_or_forged_marker(tmp_path: Path) -> None:
    _write_raw_fixture(tmp_path)
    cfg = _config(tmp_path, [_step("min_max_normalize")])
    output_dir = tmp_path / "profile_artifacts"
    output_dir.mkdir()
    marker = output_dir / ".pystar_native_preprocessing_calibration_profile"
    marker.write_text("schema_name=other\nschema_version=1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="marker does not match"):
        _ = build_profile_payload(
            config=cast(Any, cfg),
            config_path=tmp_path / "config.yaml",
            fov_ids=(1,),
            target_rounds=(1,),
            repeats=1,
            output_dir=output_dir,
        )


def test_profile_output_dir_must_not_overlap_production_output(tmp_path: Path) -> None:
    _write_raw_fixture(tmp_path)
    cfg = _config(tmp_path, [_step("min_max_normalize")])
    output_dir = Path(cfg.pipeline.output.directory) / "stage24_profile"

    with pytest.raises(ValueError, match="production output directory"):
        _ = build_profile_payload(
            config=cast(Any, cfg),
            config_path=tmp_path / "config.yaml",
            fov_ids=(1,),
            target_rounds=(1,),
            repeats=1,
            output_dir=output_dir,
        )

    assert not output_dir.exists()
    assert not Path(cfg.pipeline.output.directory).exists()


def test_profile_rejects_symlinked_output_dir(tmp_path: Path) -> None:
    _write_raw_fixture(tmp_path)
    cfg = _config(tmp_path, [_step("min_max_normalize")])
    target = tmp_path / "real_profile_dir"
    target.mkdir()
    symlink = tmp_path / "profile_link"
    symlink.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink component"):
        _ = build_profile_payload(
            config=cast(Any, cfg),
            config_path=tmp_path / "config.yaml",
            fov_ids=(1,),
            target_rounds=(1,),
            repeats=1,
            output_dir=symlink,
        )

    cfg_bad_scope = _config(tmp_path, [_step("histogram_match", {"scope": "bad_scope"})])
    with pytest.raises(ValueError, match="histogram_match profiling only supports"):
        _ = build_profile_payload(
            config=cast(Any, cfg_bad_scope),
            config_path=tmp_path / "config.yaml",
            fov_ids=(1,),
            target_rounds=None,
            repeats=1,
            output_dir=tmp_path / "profile_artifacts",
        )


def test_profile_cli_fails_loudly_for_invalid_config_before_runtime(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[2] / "scripts" / "profile_native_preprocessing_calibration.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            str(tmp_path / "missing.yaml"),
            "--output-dir",
            str(tmp_path / "profile_artifacts"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Config file not found" in result.stderr
