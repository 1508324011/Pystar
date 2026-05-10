from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from pystar.matlab_engine_bootstrap import (
    create_matlab_boundary_trace,
    finalize_matlab_boundary_trace,
    record_matlab_boundary_phase,
    summarize_matlab_boundary_traces,
)
from pystar.infrastructure import ExperimentConfig
from pystar.matlab_registration import (
    MATLABRegistrationBackend,
    load_matlab_registration_runtime_manifest,
)
from pystar.matlab_runtime import (
    collect_runtime_file_records,
    load_validated_runtime_manifest,
    resolve_staged_output_path,
    validate_configured_entrypoint_contract,
    validate_expected_3d_staged_volume_shape,
    validate_runtime_step_metadata,
    write_staged_3d_volume_tiff,
)


def _manifest_payload(
    *,
    package_name: str = "pystar_test_runtime",
    entrypoint: str = "run_stage",
    required_name: str = "run_stage.m",
    optional_name: str = "helper.m",
) -> dict[str, object]:
    return {
        "package_name": package_name,
        "entrypoint": entrypoint,
        "required_files": [
            {
                "name": required_name,
                "source_path": "repo-local entrypoint",
                "role": "python-facing test entrypoint",
                "required": True,
            }
        ],
        "optional_files": [
            {
                "name": optional_name,
                "source_path": "repo-local helper",
                "role": "optional test helper",
                "required": False,
            }
        ],
    }


def _write_manifest(runtime_dir: Path, payload: dict[str, object]) -> Path:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = runtime_dir / "runtime_manifest.json"
    _ = manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def _registration_config(*, entrypoint: str = "run_registration", local_entrypoint: str = "run_local_registration") -> ExperimentConfig:
    return cast(
        ExperimentConfig,
        cast(
            object,
            SimpleNamespace(
                providers=SimpleNamespace(
                    matlab=SimpleNamespace(
                        registration=SimpleNamespace(
                            runtime_path="unused",
                            entrypoint=entrypoint,
                            local_entrypoints={"demons_3d": local_entrypoint},
                        )
                    )
                )
            ),
        ),
    )


def test_load_validated_runtime_manifest_accepts_minimal_valid_payload(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtime"
    payload = _manifest_payload()
    _ = _write_manifest(runtime_dir, payload)

    manifest = load_validated_runtime_manifest(
        runtime_dir,
        manifest_label="Synthetic MATLAB runtime manifest",
        missing_hint="test fixture",
        package_name="pystar_test_runtime",
    )

    assert manifest["entrypoint"] == "run_stage"
    assert manifest["package_name"] == "pystar_test_runtime"


def test_load_validated_runtime_manifest_rejects_missing_required_files_bucket(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtime"
    payload = _manifest_payload()
    del payload["required_files"]
    _ = _write_manifest(runtime_dir, payload)

    with pytest.raises(ValueError, match=r"Synthetic MATLAB runtime manifest must declare a non-empty required_files list"):
        _ = load_validated_runtime_manifest(
            runtime_dir,
            manifest_label="Synthetic MATLAB runtime manifest",
            missing_hint="test fixture",
            package_name="pystar_test_runtime",
        )


def test_load_validated_runtime_manifest_rejects_package_name_mismatch(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtime"
    _ = _write_manifest(runtime_dir, _manifest_payload(package_name="wrong_package"))

    with pytest.raises(ValueError, match=r"Synthetic MATLAB runtime manifest package_name mismatch"):
        _ = load_validated_runtime_manifest(
            runtime_dir,
            manifest_label="Synthetic MATLAB runtime manifest",
            missing_hint="test fixture",
            package_name="pystar_test_runtime",
        )


def test_validate_configured_entrypoint_contract_rejects_entrypoint_mismatch() -> None:
    manifest = _manifest_payload(entrypoint="manifest_entry", required_name="manifest_entry.m")

    with pytest.raises(ValueError, match=r"providers\.matlab\.synthetic\.entrypoint must match the repo-local Synthetic MATLAB runtime manifest"):
        validate_configured_entrypoint_contract(
            manifest,
            "configured_entry",
            config_label="providers.matlab.synthetic.entrypoint",
            manifest_label="Synthetic MATLAB runtime manifest",
        )


def test_validate_configured_entrypoint_contract_rejects_undeclared_entrypoint_file() -> None:
    manifest = _manifest_payload(entrypoint="run_stage", required_name="other_file.m")

    with pytest.raises(ValueError, match=r"Synthetic MATLAB runtime manifest must declare the configured entrypoint file"):
        validate_configured_entrypoint_contract(
            manifest,
            "run_stage",
            config_label="providers.matlab.synthetic.entrypoint",
            manifest_label="Synthetic MATLAB runtime manifest",
        )


def test_load_matlab_registration_runtime_manifest_rejects_missing_local_entrypoint_file(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "registration_runtime"
    payload = _manifest_payload(
        entrypoint="run_registration",
        required_name="run_registration.m",
        optional_name="helper.m",
    )
    payload["local_entrypoint"] = "run_local_registration"
    _ = _write_manifest(runtime_dir, payload)

    with pytest.raises(
        ValueError,
        match=r"MATLAB registration runtime manifest must declare the configured local_entrypoint file",
    ):
        _ = load_matlab_registration_runtime_manifest(runtime_dir)


def test_matlab_registration_backend_rejects_local_entrypoint_config_manifest_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_dir = tmp_path / "registration_runtime"
    payload = _manifest_payload(
        entrypoint="run_registration",
        required_name="run_registration.m",
        optional_name="run_local_registration.m",
    )
    payload["local_entrypoint"] = "run_local_registration"
    _ = _write_manifest(runtime_dir, payload)

    monkeypatch.setattr(
        "pystar.matlab_registration.resolve_matlab_registration_runtime_path",
        lambda config: runtime_dir,
    )

    config = _registration_config(
        entrypoint="run_registration",
        local_entrypoint="configured_local_registration",
    )

    with pytest.raises(
        ValueError,
        match=r"providers\.matlab\.registration\.local_entrypoints\['demons_3d'\] must match the repo-local MATLAB runtime manifest",
    ):
        _ = MATLABRegistrationBackend(config)


def test_collect_runtime_file_records_requires_missing_required_files(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtime"
    manifest = _manifest_payload(required_name="required_entry.m", optional_name="optional_helper.m")
    _ = _write_manifest(runtime_dir, manifest)

    with pytest.raises(FileNotFoundError, match=r"Required synthetic runtime file is missing"):
        _ = collect_runtime_file_records(
            manifest,
            runtime_dir,
            missing_required_prefix="Required synthetic runtime file is missing",
            missing_required_suffix="synthetic runtime cannot proceed.",
        )


def test_collect_runtime_file_records_keeps_optional_missing_files_unused(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtime"
    manifest = _manifest_payload(required_name="required_entry.m", optional_name="optional_helper.m")
    _ = _write_manifest(runtime_dir, manifest)
    _ = (runtime_dir / "required_entry.m").write_text("function y = required_entry; y = 1; end\n", encoding="utf-8")

    records = collect_runtime_file_records(
        manifest,
        runtime_dir,
        missing_required_prefix="Required synthetic runtime file is missing",
        missing_required_suffix="synthetic runtime cannot proceed.",
    )

    assert len(records) == 2
    required_record = next(record for record in records if record["name"] == "required_entry.m")
    optional_record = next(record for record in records if record["name"] == "optional_helper.m")
    assert required_record["required"] is True
    assert required_record["used"] is True
    assert str(required_record["sha256"]).startswith("sha256:")
    assert optional_record["required"] is False
    assert optional_record["used"] is False
    assert "sha256" not in optional_record


def test_resolve_staged_output_path_rejects_escape_from_tmpdir(tmp_path: Path) -> None:
    tmpdir_path = tmp_path / "stage"
    tmpdir_path.mkdir()
    escaped_output = tmp_path / "escaped.csv"
    _ = escaped_output.write_text("synthetic\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"Synthetic MATLAB output must stay inside the staged temporary directory"):
        _ = resolve_staged_output_path(
            "../escaped.csv",
            tmpdir_path=tmpdir_path,
            missing_output_path_message="Synthetic MATLAB metadata must declare a non-empty output_path",
            outside_tmpdir_prefix="Synthetic MATLAB output must stay inside the staged temporary directory",
            missing_output_prefix="Synthetic MATLAB reported output that does not exist",
        )


def test_validate_expected_3d_staged_volume_shape_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match=r"Synthetic MATLAB metadata volume_shape_zyx mismatch"):
        validate_expected_3d_staged_volume_shape(
            [4, 8, 8],
            expected_shape_zyx=(3, 8, 8),
            mismatch_prefix="Synthetic MATLAB metadata volume_shape_zyx mismatch",
        )


def test_validate_runtime_step_metadata_rejects_negative_duration() -> None:
    with pytest.raises(ValueError, match=r"Synthetic MATLAB step 'load_stack' must report a non-negative duration_ms"):
        validate_runtime_step_metadata(
            [{"name": "load_stack", "duration_ms": -1.0}],
            missing_steps_message="Synthetic MATLAB metadata must declare a non-empty steps list",
            step_label="Synthetic MATLAB step",
        )


def test_write_staged_3d_volume_tiff_rejects_non_3d_volume(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"Synthetic MATLAB stage expects a 3D staged volume"):
        write_staged_3d_volume_tiff(
            tmp_path / "volume.tif",
            np.zeros((2, 3), dtype=np.float32),
            owner_label="Synthetic MATLAB stage",
        )


def test_boundary_helpers_preserve_non_negative_duration_contract() -> None:
    trace = create_matlab_boundary_trace(
        stage_name="synthetic_matlab_stage",
        runtime_dir=Path("/tmp/runtime"),
        entrypoint="synthetic_entry",
        session={},
        call_scope={"fov_id": 7},
    )
    record_matlab_boundary_phase(
        trace,
        phase_name="runtime_file_validation",
        duration_ms=12.5,
        seam_cost_key="runtime_file_validation_ms",
        details={"runtime_file_count": 2},
    )
    record_matlab_boundary_phase(
        trace,
        phase_name="matlab_call",
        duration_ms=34.0,
        seam_cost_key="matlab_call_ms",
    )
    finalized = finalize_matlab_boundary_trace(
        trace,
        session={},
        engine_reused_this_call=False,
    )
    summary = summarize_matlab_boundary_traces([finalized])

    assert finalized["total_duration_ms"] >= 0.0
    assert finalized["phase_timings_ms"]["runtime_file_validation"] == 12.5
    assert finalized["phase_timings_ms"]["matlab_call"] == 34.0
    assert summary["call_count"] == 1
    assert summary["total_duration_ms"] >= 0.0
    assert summary["aggregate_seam_costs_ms"]["runtime_file_validation_ms"] == 12.5
    assert summary["aggregate_seam_costs_ms"]["matlab_call_ms"] == 34.0
