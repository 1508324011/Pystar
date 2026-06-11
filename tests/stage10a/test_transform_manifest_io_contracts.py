from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from collections.abc import Callable
from typing import Protocol, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from pystar.infrastructure import ExperimentConfig
from pystar.io import (
    build_execution_envelope,
    build_release_contract,
    create_provenance,
    get_flow_3d_sidecar_filename,
    get_fov_output_structure,
    load_transform_manifest,
    materialize_round_transform_entry,
    save_transform_manifest,
)
from pystar.mining import SignalMiner


FOV_ID = 7
FloatArray = NDArray[np.float32]
ManifestPayload = dict[object, object]


class _SupportsItem(Protocol):
    def item(self) -> object: ...


def _full_fov_scope() -> dict[str, object]:
    return {
        "coverage_mode": "full_fov",
        "region_origin_zyx": [0, 0, 0],
        "region_shape_zyx": [2, 4, 4],
        "full_volume_shape_zyx": [2, 4, 4],
        "scope_note": "keep-me",
    }


def _settled_semantics() -> dict[str, object]:
    return {
        "representation": "residual",
        "composition": "sequential_global_then_local",
        "status": "settled",
        "recorded_at": "2026-05-09T12:00:00+00:00",
        "semantics_note": "keep-me",
    }


def _build_round_transform(
    round_id: int,
    *,
    flow_3d: FloatArray | dict[str, object] | None,
    is_reference_round: bool = False,
) -> dict[str, object]:
    return {
        "global_shift_3d": np.asarray([round_id, round_id + 1, round_id + 2], dtype=np.float32),
        "global_corr": 0.9 + (round_id * 0.001),
        "flow_2d": None,
        "flow_3d": flow_3d,
        "final_corr": 0.8 + (round_id * 0.001),
        "is_reference_round": is_reference_round,
        "_scope": _full_fov_scope(),
        "_semantics": _settled_semantics(),
        "backend_metadata": {"backend": "native"},
        "user_metadata": {"round_label": f"round-{round_id}"},
        "round_note": f"keep-round-{round_id}",
    }


def _build_transform_bundle(flow_3d: FloatArray) -> dict[object, object]:
    return {
        "manifest_note": {"keep": "yes"},
        1: _build_round_transform(1, flow_3d=None, is_reference_round=True),
        2: _build_round_transform(2, flow_3d=flow_3d, is_reference_round=False),
        "user_metadata": {"session": "synthetic-stage10a"},
    }


def _build_minimal_config(
    *,
    output_dir: Path,
    preprocessing_provider: str = "native",
    registration_global_provider: str = "native",
    registration_local_provider: str = "native",
    spot_finding_provider: str = "native",
    extraction_provider: str = "native",
    transform_application_mode: str = "image_warp",
    local_method: str = "demons_3d",
) -> ExperimentConfig:
    field_semantics = SimpleNamespace(
        representation="residual",
        composition="sequential_global_then_local",
        status="settled",
    )
    registration = SimpleNamespace(
        reference_round=1,
        field_semantics=field_semantics,
        global_provider=registration_global_provider,
        local_provider=registration_local_provider,
        enable_local=True,
        global_stage=SimpleNamespace(method="phase_corr_3d"),
        local_method=local_method,
    )
    preprocessing = SimpleNamespace(
        sequence=[SimpleNamespace(method="synthetic_clean", provider=preprocessing_provider)]
    )
    pipeline = SimpleNamespace(
        scope_mode="full_fov",
        accelerator="cpu",
        field_semantics=field_semantics,
        preprocessing=preprocessing,
        registration=registration,
        spot_finding=SimpleNamespace(provider=spot_finding_provider),
        extraction=SimpleNamespace(
            provider=extraction_provider,
            transform_application_mode=transform_application_mode,
        ),
        output=SimpleNamespace(directory=str(output_dir)),
    )
    pipeline.preprocessing_providers_used = lambda: sorted(
        {step.provider for step in pipeline.preprocessing.sequence}
    )
    pipeline.preprocessing_provider_mode = lambda: (
        "native_only"
        if set(pipeline.preprocessing_providers_used()) == {"native"}
        else "matlab_only"
        if set(pipeline.preprocessing_providers_used()) == {"matlab"}
        else "mixed"
    )
    pipeline.registration_provider_mode = lambda: (
        "native_only"
        if registration.global_provider == "native" and registration.local_provider in {None, "native"}
        else "matlab_only"
        if registration.global_provider == "matlab" and registration.local_provider in {None, "matlab"}
        else "mixed"
    )
    pipeline.uses_matlab_preprocessing = lambda: any(
        step.provider == "matlab" for step in pipeline.preprocessing.sequence
    )
    pipeline.uses_matlab_spot_finding = lambda: pipeline.spot_finding.provider == "matlab"
    pipeline.uses_matlab_extraction = lambda: pipeline.extraction.provider == "matlab"
    return cast(ExperimentConfig, cast(object, SimpleNamespace(pipeline=pipeline)))


def _build_valid_provenance(base_dir: Path, transforms: ManifestPayload) -> dict[str, object]:
    cfg = _build_minimal_config(output_dir=base_dir)
    release_contract = build_release_contract(cfg, transforms)
    stage_outcomes = cast(
        dict[str, dict[str, object]],
        {
            "round_summary": {
                1: {
                    "status": "completed",
                    "start_time": "2026-05-09T12:00:00+00:00",
                    "end_time": "2026-05-09T12:00:01+00:00",
                },
                2: {
                    "status": "completed",
                    "start_time": "2026-05-09T12:00:01+00:00",
                    "end_time": "2026-05-09T12:00:02+00:00",
                },
            }
        },
    )
    return create_provenance(
        pipeline_version="0.test",
        environment_hash="env-hash-stage10a",
        stage_outcomes=stage_outcomes,
        release_contract=release_contract,
        config_reference={
            "config_path": "synthetic_stage10a.yaml",
            "config_hash": "abc123",
            "key_parameters": {
                "scope_mode": "full_fov",
                "transform_application_mode": "image_warp",
            },
        },
        software_versions={"numpy": np.__version__},
        hardware_context={"cpu_count": 1, "memory_available_bytes": 1024**3},
        start_time="2026-05-09T12:00:00+00:00",
        end_time="2026-05-09T12:00:02+00:00",
        duration_seconds=2.0,
        execution_envelope=build_execution_envelope(cfg),
    )


def _load_persisted_manifest(manifest_path: Path) -> ManifestPayload:
    loaded = cast(_SupportsItem, np.load(manifest_path, allow_pickle=True))
    persisted = loaded.item()
    if not isinstance(persisted, dict):
        raise AssertionError(f"expected dict manifest payload, got {type(persisted)}")
    return cast(ManifestPayload, persisted)


def _rewrite_manifest(manifest_path: Path, payload: ManifestPayload) -> None:
    np.save(manifest_path, np.asarray(payload, dtype=object))


def _round_entry(manifest: ManifestPayload, round_id: int) -> dict[str, object]:
    return cast(dict[str, object], manifest[round_id])


def _descriptor(round_payload: dict[str, object]) -> dict[str, object]:
    return cast(dict[str, object], round_payload["flow_3d"])


def _provenance_payload(manifest: ManifestPayload) -> dict[str, object]:
    return cast(dict[str, object], manifest["_provenance"])


def _release_gate(contract: dict[str, object]) -> dict[str, object]:
    return cast(dict[str, object], contract["release_gate"])


def _requested_intent(contract: dict[str, object]) -> dict[str, object]:
    return cast(dict[str, object], contract["requested_intent"])


def _assert_image_warp_contract_accepted_by_signal_miner(contract: dict[str, object]) -> None:
    miner = cast(SignalMiner, SignalMiner.__new__(SignalMiner))
    miner._validate_image_warp_contract(
        FOV_ID,
        {"_provenance": {"release_contract": contract}},
    )


def test_release_contract_all_matlab_image_warp_can_be_valid_when_artifacts_validate(tmp_path: Path) -> None:
    flow_3d = np.arange(3 * 2 * 4 * 4, dtype=np.float32).reshape(3, 2, 4, 4)
    transforms = _build_transform_bundle(flow_3d)
    cfg = _build_minimal_config(
        output_dir=tmp_path,
        preprocessing_provider="matlab",
        registration_global_provider="matlab",
        registration_local_provider="matlab",
        spot_finding_provider="matlab",
        extraction_provider="matlab",
    )

    contract = cast(dict[str, object], build_release_contract(cfg, transforms))
    gate = _release_gate(contract)
    requested_intent = _requested_intent(contract)
    execution_envelope = cast(dict[str, object], requested_intent["execution_envelope"])
    registration_profile = cast(dict[str, object], requested_intent["registration_profile"])
    stage_contracts = cast(dict[str, dict[str, object]], requested_intent["matlab_stage_contracts"])

    assert gate["status"] == "valid"
    assert gate["reasons"] == []
    assert cast(dict[str, object], gate["gate0"])["passed"] is True
    assert execution_envelope["preprocessing_backend"] == "matlab_extracted"
    assert execution_envelope["registration_backend"] == "matlab_extracted"
    assert registration_profile["supports_image_warp_mainline"] is True
    assert registration_profile["declared_transform_capabilities"] == ["global_shift_3d", "flow_3d"]
    for stage_name in ("preprocessing", "registration", "spot_finding", "extraction"):
        assert stage_contracts[stage_name]["matlab_requested"] is True
        assert stage_contracts[stage_name]["current_support_status"] == "artifact_contract_supported"
        assert stage_contracts[stage_name]["promotion_blockers"] == []

    _assert_image_warp_contract_accepted_by_signal_miner(contract)


def test_release_contract_mixed_matlab_provider_image_warp_can_be_valid_when_artifacts_validate(tmp_path: Path) -> None:
    flow_3d = np.arange(3 * 2 * 4 * 4, dtype=np.float32).reshape(3, 2, 4, 4)
    transforms = _build_transform_bundle(flow_3d)
    cfg = _build_minimal_config(
        output_dir=tmp_path,
        registration_global_provider="matlab",
        registration_local_provider="native",
    )

    contract = cast(dict[str, object], build_release_contract(cfg, transforms))
    gate = _release_gate(contract)
    requested_intent = _requested_intent(contract)
    execution_envelope = cast(dict[str, object], requested_intent["execution_envelope"])
    stage_contracts = cast(dict[str, dict[str, object]], requested_intent["matlab_stage_contracts"])

    assert gate["status"] == "valid"
    assert gate["reasons"] == []
    assert execution_envelope["registration_backend"] == "provider_dispatch"
    assert stage_contracts["registration"]["matlab_requested"] is True
    assert stage_contracts["registration"]["current_support_status"] == "artifact_contract_supported"
    assert stage_contracts["registration"]["promotion_blockers"] == []

    _assert_image_warp_contract_accepted_by_signal_miner(contract)


def test_release_contract_rejects_unsupported_matlab_local_method_loudly(tmp_path: Path) -> None:
    flow_3d = np.arange(3 * 2 * 4 * 4, dtype=np.float32).reshape(3, 2, 4, 4)
    cfg = _build_minimal_config(
        output_dir=tmp_path,
        registration_global_provider="matlab",
        registration_local_provider="matlab",
        local_method="optical_flow",
    )

    with pytest.raises(ValueError, match="registration.local.provider='matlab'.*demons_3d"):
        _ = build_release_contract(cfg, _build_transform_bundle(flow_3d))


def test_release_contract_matlab_spot_finding_and_extraction_do_not_force_debug_only(tmp_path: Path) -> None:
    flow_3d = np.arange(3 * 2 * 4 * 4, dtype=np.float32).reshape(3, 2, 4, 4)
    cfg = _build_minimal_config(
        output_dir=tmp_path,
        spot_finding_provider="matlab",
        extraction_provider="matlab",
    )

    contract = cast(dict[str, object], build_release_contract(cfg, _build_transform_bundle(flow_3d)))
    gate = _release_gate(contract)
    requested_intent = _requested_intent(contract)
    stage_contracts = cast(dict[str, dict[str, object]], requested_intent["matlab_stage_contracts"])

    assert gate["status"] == "valid"
    assert gate["reasons"] == []
    assert requested_intent["application_intent"] == (
        "artifact_contract_image_warp_matlab_spot_finding_extraction_provider"
    )
    assert stage_contracts["registration"]["matlab_requested"] is False
    for stage_name in ("spot_finding", "extraction"):
        assert stage_contracts[stage_name]["matlab_requested"] is True
        assert stage_contracts[stage_name]["current_support_status"] == "artifact_contract_supported"
        assert stage_contracts[stage_name]["promotion_blockers"] == []

    _assert_image_warp_contract_accepted_by_signal_miner(contract)


def test_release_contract_invalid_matlab_image_warp_artifacts_remain_non_valid(tmp_path: Path) -> None:
    transforms = _build_transform_bundle(np.zeros((3, 2, 4, 4), dtype=np.float32))
    round_two = _round_entry(transforms, 2)
    round_two["flow_3d"] = None
    cfg = _build_minimal_config(
        output_dir=tmp_path,
        preprocessing_provider="matlab",
        registration_global_provider="matlab",
        registration_local_provider="matlab",
        spot_finding_provider="matlab",
        extraction_provider="matlab",
    )

    contract = cast(dict[str, object], build_release_contract(cfg, transforms))
    gate = _release_gate(contract)

    assert gate["status"] == "invalid"
    assert cast(dict[str, object], gate["gate0"])["passed"] is False
    assert "missing rounds: [2]" in cast(list[str], gate["reasons"])[0]
    with pytest.raises(ValueError, match="status='invalid'"):
        _assert_image_warp_contract_accepted_by_signal_miner(contract)


def test_release_contract_invalid_matlab_field_semantics_remain_non_valid(tmp_path: Path) -> None:
    transforms = _build_transform_bundle(np.zeros((3, 2, 4, 4), dtype=np.float32))
    round_two = _round_entry(transforms, 2)
    semantics = cast(dict[str, object], round_two["_semantics"])
    semantics["representation"] = "total"
    cfg = _build_minimal_config(
        output_dir=tmp_path,
        preprocessing_provider="matlab",
        registration_global_provider="matlab",
        registration_local_provider="matlab",
        spot_finding_provider="matlab",
        extraction_provider="matlab",
    )

    contract = cast(dict[str, object], build_release_contract(cfg, transforms))
    gate = _release_gate(contract)

    assert gate["status"] == "invalid"
    assert "Persisted round-level _semantics do not match" in cast(list[str], gate["reasons"])[0]
    with pytest.raises(ValueError, match="status='invalid'"):
        _assert_image_warp_contract_accepted_by_signal_miner(contract)


def test_release_contract_coordinate_mapping_stays_debug_only_and_signal_miner_rejects(tmp_path: Path) -> None:
    flow_3d = np.arange(3 * 2 * 4 * 4, dtype=np.float32).reshape(3, 2, 4, 4)
    cfg = _build_minimal_config(
        output_dir=tmp_path,
        transform_application_mode="coordinate_mapping",
    )

    contract = cast(dict[str, object], build_release_contract(cfg, _build_transform_bundle(flow_3d)))
    gate = _release_gate(contract)

    assert gate["status"] == "debug_only"
    assert cast(dict[str, object], gate["gate0"])["required"] is False
    with pytest.raises(ValueError, match="status='debug_only'"):
        _assert_image_warp_contract_accepted_by_signal_miner(contract)


def test_transform_manifest_public_io_preserves_legacy_shape_and_eager_lazy_flow_loading(tmp_path: Path) -> None:
    base_dir = tmp_path
    flow_3d = np.arange(3 * 2 * 4 * 4, dtype=np.float32).reshape(3, 2, 4, 4)
    transforms = _build_transform_bundle(flow_3d)
    provenance = _build_valid_provenance(base_dir, transforms)

    manifest_path = save_transform_manifest(base_dir, FOV_ID, transforms, provenance=provenance)

    paths = get_fov_output_structure(base_dir, FOV_ID)
    sidecar_name = get_flow_3d_sidecar_filename(FOV_ID, 2)
    sidecar_path = paths["transforms"] / sidecar_name
    summary_path = paths["qc"] / "provenance_summary.md"
    persisted = _load_persisted_manifest(manifest_path)

    assert sidecar_path.exists()
    assert summary_path.exists()
    assert summary_path.read_text(encoding="utf-8").startswith("# Provenance Summary")
    assert persisted["manifest_note"] == {"keep": "yes"}
    assert persisted["user_metadata"] == {"session": "synthetic-stage10a"}
    persisted_round_two = _round_entry(persisted, 2)
    assert _descriptor(persisted_round_two) == {
        "storage": "round_level_sidecar_npy",
        "path": sidecar_name,
        "shape": [3, 2, 4, 4],
        "dtype": "float32",
    }
    assert persisted_round_two["round_note"] == "keep-round-2"
    assert persisted["_contract"] == provenance["release_contract"]
    assert _provenance_payload(persisted)["release_contract"] == provenance["release_contract"]

    lazy_loaded = load_transform_manifest(base_dir, FOV_ID, load_provenance=True, hydrate_flow_3d=False)
    eager_loaded = load_transform_manifest(base_dir, FOV_ID, load_provenance=True, hydrate_flow_3d=True)
    lazy_round_two = _round_entry(cast(ManifestPayload, lazy_loaded), 2)
    eager_round_two = _round_entry(cast(ManifestPayload, eager_loaded), 2)
    materialized = materialize_round_transform_entry(
        base_dir,
        FOV_ID,
        2,
        lazy_round_two,
        hydrate_flow_3d=True,
    )

    assert _descriptor(lazy_round_two) == _descriptor(persisted_round_two)
    assert lazy_round_two["_scope"] == _full_fov_scope()
    assert lazy_round_two["_semantics"] == _settled_semantics()
    assert lazy_round_two["user_metadata"] == {"round_label": "round-2"}
    eager_flow_3d = eager_round_two["flow_3d"]
    materialized_flow_3d = cast(object, materialized["flow_3d"])
    assert isinstance(eager_flow_3d, np.memmap)
    assert eager_flow_3d.mode == "r"
    assert eager_flow_3d.flags.writeable is False
    assert isinstance(materialized_flow_3d, np.memmap)
    assert materialized_flow_3d.mode == "r"
    assert materialized_flow_3d.flags.writeable is False
    np.testing.assert_array_equal(cast(FloatArray, eager_round_two["flow_3d"]), flow_3d)
    np.testing.assert_array_equal(cast(FloatArray, materialized["flow_3d"]), flow_3d)
    assert materialized["round_note"] == "keep-round-2"
    assert eager_loaded["_contract"] == provenance["release_contract"]
    assert _provenance_payload(cast(ManifestPayload, eager_loaded))["release_contract"] == provenance["release_contract"]


def test_transform_manifest_hides_provenance_metadata_by_default(tmp_path: Path) -> None:
    base_dir = tmp_path
    flow_3d = np.arange(3 * 2 * 4 * 4, dtype=np.float32).reshape(3, 2, 4, 4)
    transforms = _build_transform_bundle(flow_3d)
    provenance = _build_valid_provenance(base_dir, transforms)

    _ = save_transform_manifest(base_dir, FOV_ID, transforms, provenance=provenance)

    loaded = cast(ManifestPayload, load_transform_manifest(base_dir, FOV_ID, hydrate_flow_3d=False))
    round_two = _round_entry(loaded, 2)

    assert "_provenance" not in loaded
    assert "_contract" not in loaded
    assert loaded["manifest_note"] == {"keep": "yes"}
    assert loaded["user_metadata"] == {"session": "synthetic-stage10a"}
    assert _descriptor(round_two)["path"] == get_flow_3d_sidecar_filename(FOV_ID, 2)


def test_transform_manifest_materialization_and_eager_load_fail_loudly_when_sidecar_is_missing(tmp_path: Path) -> None:
    base_dir = tmp_path
    flow_3d = np.arange(3 * 2 * 4 * 4, dtype=np.float32).reshape(3, 2, 4, 4)
    _ = save_transform_manifest(base_dir, FOV_ID, _build_transform_bundle(flow_3d))

    lazy_loaded = load_transform_manifest(base_dir, FOV_ID, hydrate_flow_3d=False)
    lazy_round_two = _round_entry(cast(ManifestPayload, lazy_loaded), 2)
    sidecar_path = get_fov_output_structure(base_dir, FOV_ID)["transforms"] / get_flow_3d_sidecar_filename(FOV_ID, 2)
    sidecar_path.unlink()

    with pytest.raises(FileNotFoundError, match="flow_3d sidecar referenced by transform manifest is missing"):
        _ = materialize_round_transform_entry(base_dir, FOV_ID, 2, lazy_round_two, hydrate_flow_3d=True)

    with pytest.raises(FileNotFoundError, match="flow_3d sidecar referenced by transform manifest is missing"):
        _ = load_transform_manifest(base_dir, FOV_ID, hydrate_flow_3d=True)


def _mutate_descriptor_path_escape(manifest: ManifestPayload) -> None:
    round_two = _round_entry(manifest, 2)
    descriptor = _descriptor(round_two)
    descriptor.update({"path": "../escape.npy"})


def _mutate_descriptor_nested_path(manifest: ManifestPayload) -> None:
    round_two = _round_entry(manifest, 2)
    descriptor = _descriptor(round_two)
    descriptor.update({"path": "nested/escape.npy"})


def _mutate_descriptor_storage(manifest: ManifestPayload) -> None:
    round_two = _round_entry(manifest, 2)
    descriptor = _descriptor(round_two)
    descriptor.update({"storage": "other_storage"})


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            _mutate_descriptor_path_escape,
            r"transform round 2\.flow_3d\.path must be a direct filename under transforms/",
        ),
        (
            _mutate_descriptor_nested_path,
            r"transform round 2\.flow_3d\.path must be a direct filename under transforms/",
        ),
        (
            _mutate_descriptor_storage,
            r"transform round 2\.flow_3d\.storage must be 'round_level_sidecar_npy'",
        ),
    ],
)
def test_transform_manifest_load_rejects_path_traversal_and_invalid_storage_descriptors(
    tmp_path: Path,
    mutate: Callable[[ManifestPayload], None],
    match: str,
) -> None:
    base_dir = tmp_path
    flow_3d = np.arange(3 * 2 * 4 * 4, dtype=np.float32).reshape(3, 2, 4, 4)
    manifest_path = save_transform_manifest(base_dir, FOV_ID, _build_transform_bundle(flow_3d))
    mutated = _load_persisted_manifest(manifest_path)
    mutate(mutated)
    _rewrite_manifest(manifest_path, mutated)

    with pytest.raises(ValueError, match=match):
        _ = load_transform_manifest(base_dir, FOV_ID, hydrate_flow_3d=False)


def _mutate_scope_coverage_mode(manifest: ManifestPayload) -> None:
    round_two = _round_entry(manifest, 2)
    scope = cast(dict[str, object], round_two["_scope"])
    scope["coverage_mode"] = "bogus"


def _mutate_semantics_representation(manifest: ManifestPayload) -> None:
    round_two = _round_entry(manifest, 2)
    semantics = cast(dict[str, object], round_two["_semantics"])
    semantics["representation"] = "bogus"


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            _mutate_scope_coverage_mode,
            r"transform round 2\._scope\.coverage_mode must be one of",
        ),
        (
            _mutate_semantics_representation,
            r"transform round 2\._semantics\.representation must be one of",
        ),
    ],
)
def test_transform_manifest_load_rejects_malformed_scope_and_semantics_via_public_facade(
    tmp_path: Path,
    mutate: Callable[[ManifestPayload], None],
    match: str,
) -> None:
    base_dir = tmp_path
    flow_3d = np.arange(3 * 2 * 4 * 4, dtype=np.float32).reshape(3, 2, 4, 4)
    manifest_path = save_transform_manifest(base_dir, FOV_ID, _build_transform_bundle(flow_3d))
    mutated = _load_persisted_manifest(manifest_path)
    mutate(mutated)
    _rewrite_manifest(manifest_path, mutated)

    with pytest.raises(ValueError, match=match):
        _ = load_transform_manifest(base_dir, FOV_ID, hydrate_flow_3d=False)


@pytest.mark.parametrize(
    ("descriptor_update", "match"),
    [
        ({"shape": [3, 9, 9, 9]}, r"flow_3d sidecar shape mismatch for round 2"),
        ({"dtype": "float64"}, r"flow_3d sidecar dtype mismatch for round 2"),
    ],
)
def test_transform_manifest_materialization_rejects_sidecar_shape_and_dtype_drift(
    tmp_path: Path,
    descriptor_update: dict[str, object],
    match: str,
) -> None:
    base_dir = tmp_path
    flow_3d = np.arange(3 * 2 * 4 * 4, dtype=np.float32).reshape(3, 2, 4, 4)
    manifest_path = save_transform_manifest(base_dir, FOV_ID, _build_transform_bundle(flow_3d))
    mutated = _load_persisted_manifest(manifest_path)
    round_two = _round_entry(mutated, 2)
    descriptor = _descriptor(round_two)
    descriptor.update(descriptor_update)
    _rewrite_manifest(manifest_path, mutated)

    with pytest.raises(ValueError, match=match):
        _ = load_transform_manifest(base_dir, FOV_ID, hydrate_flow_3d=True)


def test_transform_manifest_load_rejects_contract_and_provenance_drift(tmp_path: Path) -> None:
    base_dir = tmp_path
    flow_3d = np.arange(3 * 2 * 4 * 4, dtype=np.float32).reshape(3, 2, 4, 4)
    transforms = _build_transform_bundle(flow_3d)
    provenance = _build_valid_provenance(base_dir, transforms)
    manifest_path = save_transform_manifest(base_dir, FOV_ID, transforms, provenance=provenance)

    mutated = _load_persisted_manifest(manifest_path)
    original_contract = cast(dict[str, object], mutated["_contract"])
    contract = copy.deepcopy(original_contract)
    release_gate = cast(dict[str, object], contract["release_gate"])
    release_gate["status"] = "debug_only"
    mutated["_contract"] = contract
    _rewrite_manifest(manifest_path, mutated)

    with pytest.raises(ValueError, match=r"Transform manifest _contract must match _provenance\.release_contract"):
        _ = load_transform_manifest(base_dir, FOV_ID, load_provenance=True, hydrate_flow_3d=False)


def test_transform_manifest_load_rejects_provenance_without_matching_contract_when_requested(tmp_path: Path) -> None:
    base_dir = tmp_path
    flow_3d = np.arange(3 * 2 * 4 * 4, dtype=np.float32).reshape(3, 2, 4, 4)
    transforms = _build_transform_bundle(flow_3d)
    provenance = _build_valid_provenance(base_dir, transforms)
    manifest_path = save_transform_manifest(base_dir, FOV_ID, transforms, provenance=provenance)

    mutated = _load_persisted_manifest(manifest_path)
    del mutated["_contract"]
    _rewrite_manifest(manifest_path, mutated)

    with pytest.raises(ValueError, match=r"load_transform_manifest\(load_provenance=True\) requires both metadata fields"):
        _ = load_transform_manifest(base_dir, FOV_ID, load_provenance=True, hydrate_flow_3d=False)
