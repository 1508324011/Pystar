from __future__ import annotations

from pathlib import Path

import pytest

from pystar._stage_contracts import (
    StageSpec,
    get_ordered_stage_ids,
    get_ordered_stage_specs,
    get_stage_spec,
    validate_stage_ids,
    validate_stage_specs,
)


def test_stage_specs_are_immutable_and_deterministic() -> None:
    ordered_specs = get_ordered_stage_specs()

    assert [spec.stage_id for spec in ordered_specs] == [
        "preprocessing",
        "registration",
        "spot_finding",
        "signal_extraction",
        "decoding",
    ]
    assert [spec.order_index for spec in ordered_specs] == [1, 2, 3, 4, 5]
    assert [spec.display_label for spec in ordered_specs] == [
        "Preprocessing (Sanitization)",
        "Registration",
        "Spot Finding",
        "Signal Extraction",
        "Decoding",
    ]
    assert get_ordered_stage_ids() == (
        "preprocessing",
        "registration",
        "spot_finding",
        "signal_extraction",
        "decoding",
    )

    with pytest.raises(AttributeError):
        setattr(ordered_specs[0], "stage_id", "other")


def test_get_stage_spec_returns_canonical_spec_and_rejects_unknown_ids() -> None:
    registration = get_stage_spec("registration")

    assert registration == StageSpec(
        stage_id="registration",
        order_index=2,
        display_label="Registration",
        runner_calls=("ImageLoader.load_fov", "RegistrationEngine.register_fov"),
    )

    with pytest.raises(ValueError, match="unknown stage ID 'unknown_stage'"):
        _ = get_stage_spec("unknown_stage")


def test_validate_stage_ids_accepts_canonical_order_and_rejects_drift_duplicates_unknowns() -> None:
    canonical_stage_ids = get_ordered_stage_ids()
    assert validate_stage_ids(canonical_stage_ids) == canonical_stage_ids

    with pytest.raises(ValueError, match="unknown stage IDs.*bad_stage"):
        _ = validate_stage_ids(("preprocessing", "bad_stage"))

    with pytest.raises(ValueError, match="duplicate stage IDs.*registration"):
        _ = validate_stage_ids(
            (
                "preprocessing",
                "registration",
                "registration",
                "spot_finding",
                "signal_extraction",
                "decoding",
            )
        )

    with pytest.raises(ValueError, match="order drift detected"):
        _ = validate_stage_ids(
            (
                "preprocessing",
                "spot_finding",
                "registration",
                "signal_extraction",
                "decoding",
            )
        )


def test_validate_stage_specs_rejects_duplicate_ids_duplicate_order_indices_and_order_drift() -> None:
    canonical_specs = get_ordered_stage_specs()
    assert validate_stage_specs(canonical_specs) == canonical_specs

    with pytest.raises(ValueError, match="duplicate stage IDs.*registration"):
        _ = validate_stage_specs(
            (
                canonical_specs[0],
                canonical_specs[1],
                StageSpec(
                    stage_id="registration",
                    order_index=3,
                    display_label="Registration Duplicate",
                ),
                canonical_specs[3],
                canonical_specs[4],
            )
        )

    with pytest.raises(ValueError, match="duplicate order indices.*2"):
        _ = validate_stage_specs(
            (
                canonical_specs[0],
                canonical_specs[1],
                StageSpec(
                    stage_id="spot_finding",
                    order_index=2,
                    display_label="Spot Finding",
                ),
                canonical_specs[3],
                canonical_specs[4],
            )
        )

    with pytest.raises(ValueError, match=r"order drift detected for stage ID 'registration'.*order_index=2.*order_index=20"):
        _ = validate_stage_specs(
            (
                canonical_specs[0],
                StageSpec(
                    stage_id="registration",
                    order_index=20,
                    display_label="Registration",
                    runner_calls=("ImageLoader.load_fov", "RegistrationEngine.register_fov"),
                ),
                StageSpec(
                    stage_id="spot_finding",
                    order_index=30,
                    display_label="Spot Finding",
                ),
                StageSpec(
                    stage_id="signal_extraction",
                    order_index=40,
                    display_label="Signal Extraction",
                ),
                StageSpec(
                    stage_id="decoding",
                    order_index=50,
                    display_label="Decoding",
                ),
            )
        )

    with pytest.raises(ValueError, match="order drift detected"):
        _ = validate_stage_specs(
            (
                canonical_specs[0],
                StageSpec(
                    stage_id="spot_finding",
                    order_index=2,
                    display_label="Spot Finding",
                ),
                StageSpec(
                    stage_id="registration",
                    order_index=3,
                    display_label="Registration",
                    runner_calls=("ImageLoader.load_fov", "RegistrationEngine.register_fov"),
                ),
                canonical_specs[3],
                canonical_specs[4],
            )
        )

    with pytest.raises(ValueError, match="order drift detected"):
        _ = validate_stage_specs(
            (
                canonical_specs[1],
                canonical_specs[0],
                canonical_specs[2],
                canonical_specs[3],
                canonical_specs[4],
            )
        )


def test_batch_runner_uses_canonical_stage_labels_without_control_flow_rewrite() -> None:
    batch_script = Path(__file__).resolve().parents[2] / "scripts" / "batch_pystar.py"
    source = batch_script.read_text(encoding="utf-8")

    expected_fragments = [
        'log_stage_start(logger, "preprocessing")',
        "sanitize_fov(current_fov)",
        'record_stage_timing("preprocessing"',
        'log_stage_start(logger, "registration")',
        "data_xr = loader.load_fov(current_fov)",
        "reg_engine.register_fov(data_xr, current_fov)",
        'record_stage_timing("registration"',
        'log_stage_start(logger, "spot_finding")',
        "find_spots_in_fov(current_fov)",
        'record_stage_timing("spot_finding"',
        'log_stage_start(logger, "signal_extraction")',
        "mine_fov(current_fov)",
        'record_stage_timing("signal_extraction"',
        'log_stage_start(logger, "decoding")',
        "decode_fov(current_fov)",
        'record_stage_timing("decoding"',
        "write_performance_telemetry(",
    ]

    fragment_positions = [source.index(fragment) for fragment in expected_fragments]
    assert fragment_positions == sorted(fragment_positions)
