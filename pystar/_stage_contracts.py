"""Private immutable stage-order contract for the current PyStar runtime.

This module names the existing five-stage batch runner sequence without
changing execution semantics, userspace paths, public imports, or artifact
shapes. It is intentionally data-only: the batch runner remains the authority
for execution while this module owns stable stage IDs, labels, order, and
fail-loud validation helpers.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class StageSpec:
    """Immutable metadata for one existing runtime stage."""

    stage_id: str
    order_index: int
    display_label: str
    runner_calls: tuple[str, ...] = ()


_CANONICAL_STAGE_SPECS: tuple[StageSpec, ...] = (
    StageSpec(
        stage_id="preprocessing",
        order_index=1,
        display_label="Preprocessing (Sanitization)",
        runner_calls=("DataSanitizer.sanitize_fov",),
    ),
    StageSpec(
        stage_id="registration",
        order_index=2,
        display_label="Registration",
        runner_calls=("ImageLoader.load_fov", "RegistrationEngine.register_fov"),
    ),
    StageSpec(
        stage_id="spot_finding",
        order_index=3,
        display_label="Spot Finding",
        runner_calls=("SpotFinder.find_spots_in_fov",),
    ),
    StageSpec(
        stage_id="signal_extraction",
        order_index=4,
        display_label="Signal Extraction",
        runner_calls=("SignalMiner.mine_fov",),
    ),
    StageSpec(
        stage_id="decoding",
        order_index=5,
        display_label="Decoding",
        runner_calls=("Decoder.decode_fov",),
    ),
)

_CANONICAL_STAGE_SPEC_BY_ID = {spec.stage_id: spec for spec in _CANONICAL_STAGE_SPECS}
_CANONICAL_STAGE_IDS = tuple(spec.stage_id for spec in _CANONICAL_STAGE_SPECS)
_CANONICAL_STAGE_ORDER_BY_ID = {spec.stage_id: spec.order_index for spec in _CANONICAL_STAGE_SPECS}


def _dedupe_in_order(values: Iterable[str | int]) -> tuple[str | int, ...]:
    ordered: list[str | int] = []
    seen: set[str | int] = set()
    for value in values:
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return tuple(ordered)


def _duplicate_stage_ids(stage_specs: tuple[StageSpec, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for spec in stage_specs:
        if spec.stage_id in seen:
            duplicates.append(spec.stage_id)
        seen.add(spec.stage_id)
    return tuple(value for value in _dedupe_in_order(duplicates) if isinstance(value, str))


def _duplicate_order_indices(stage_specs: tuple[StageSpec, ...]) -> tuple[int, ...]:
    seen: set[int] = set()
    duplicates: list[int] = []
    for spec in stage_specs:
        if spec.order_index in seen:
            duplicates.append(spec.order_index)
        seen.add(spec.order_index)
    return tuple(value for value in _dedupe_in_order(duplicates) if isinstance(value, int))


def get_ordered_stage_specs() -> tuple[StageSpec, ...]:
    """Return the canonical five-stage runtime order."""

    return _CANONICAL_STAGE_SPECS


def get_ordered_stage_ids() -> tuple[str, ...]:
    """Return canonical stage IDs in execution order."""

    return _CANONICAL_STAGE_IDS


def get_stage_spec(stage_id: str) -> StageSpec:
    """Return one canonical stage spec or fail loudly for unknown IDs."""

    try:
        return _CANONICAL_STAGE_SPEC_BY_ID[stage_id]
    except KeyError as exc:
        raise ValueError(
            f"Stage order contract error: unknown stage ID {stage_id!r}. Expected one of {list(_CANONICAL_STAGE_IDS)}"
        ) from exc


def validate_stage_ids(stage_ids: Iterable[str]) -> tuple[str, ...]:
    """Validate an ordered stage-ID sequence against the canonical contract."""

    normalized = tuple(stage_ids)
    unknown_stage_ids = tuple(stage_id for stage_id in normalized if stage_id not in _CANONICAL_STAGE_SPEC_BY_ID)
    if unknown_stage_ids:
        raise ValueError(
            f"Stage order contract error: unknown stage IDs {list(_dedupe_in_order(unknown_stage_ids))}. Expected canonical stage IDs {list(_CANONICAL_STAGE_IDS)}"
        )

    seen: set[str] = set()
    duplicate_stage_ids: list[str] = []
    for stage_id in normalized:
        if stage_id in seen:
            duplicate_stage_ids.append(stage_id)
        seen.add(stage_id)
    if duplicate_stage_ids:
        raise ValueError(
            f"Stage order contract error: duplicate stage IDs {list(_dedupe_in_order(duplicate_stage_ids))}."
        )

    if normalized != _CANONICAL_STAGE_IDS:
        raise ValueError(
            f"Stage order contract error: order drift detected. Expected {list(_CANONICAL_STAGE_IDS)}, got {list(normalized)}"
        )
    return normalized


def validate_stage_specs(stage_specs: Iterable[StageSpec]) -> tuple[StageSpec, ...]:
    """Validate stage specs for duplicate IDs, duplicate order indices, and drift."""

    normalized = tuple(stage_specs)
    duplicate_stage_ids = _duplicate_stage_ids(normalized)
    if duplicate_stage_ids:
        raise ValueError(f"Stage order contract error: duplicate stage IDs {list(duplicate_stage_ids)}.")

    duplicate_order_indices = _duplicate_order_indices(normalized)
    if duplicate_order_indices:
        raise ValueError(
            f"Stage order contract error: duplicate order indices {list(duplicate_order_indices)}."
        )

    ordered_stage_ids: list[str] = []
    for spec in normalized:
        if spec.stage_id not in _CANONICAL_STAGE_SPEC_BY_ID:
            raise ValueError(
                f"Stage order contract error: unknown stage ID {spec.stage_id!r}. Expected one of {list(_CANONICAL_STAGE_IDS)}"
            )
        expected_order_index = _CANONICAL_STAGE_ORDER_BY_ID[spec.stage_id]
        if spec.order_index != expected_order_index:
            raise ValueError(
                f"Stage order contract error: order drift detected for stage ID {spec.stage_id!r}. Expected order_index={expected_order_index}, got order_index={spec.order_index}"
            )
        ordered_stage_ids.append(spec.stage_id)

    _ = validate_stage_ids(tuple(ordered_stage_ids))
    return normalized


_ = validate_stage_specs(_CANONICAL_STAGE_SPECS)
