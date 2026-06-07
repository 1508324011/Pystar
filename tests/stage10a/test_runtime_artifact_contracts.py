from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from pystar.runtime_artifacts import (
    FLOW_3D_SIDECAR_STORAGE,
    FieldSemantics,
    Flow3DSidecarDescriptor,
    ReleaseContract,
    ScopeMetadata,
    TransformEntry,
    TransformManifest,
)

FloatArray = NDArray[np.float32]


def _semantics_payload() -> dict[str, object]:
    return {
        "representation": "residual",
        "composition": "sequential_global_then_local",
        "status": "settled",
        "recorded_at": "2026-05-09T12:00:00+00:00",
        "legacy_semantics_note": "keep-me",
    }


def _tile_scope_payload() -> dict[str, object]:
    return {
        "coverage_mode": "tile_local",
        "region_origin_zyx": [2, 4, 6],
        "region_shape_zyx": [3, 8, 10],
        "full_volume_shape_zyx": [16, 64, 64],
        "tile_grid_shape_yx": [2, 3],
        "tile_index": 4,
        "legacy_scope_note": "keep-me",
    }


def _descriptor_payload() -> dict[str, object]:
    return {
        "storage": FLOW_3D_SIDECAR_STORAGE,
        "path": "transforms_fov_7_round_2_flow_3d.npy",
        "shape": [3, 2, 3, 4],
        "dtype": "float32",
        "legacy_descriptor_note": "keep-me",
    }


def _round_payload(*, shift_value: float, descriptor: dict[str, object] | None = None) -> dict[str, object]:
    flow_3d = _descriptor_payload() if descriptor is None else descriptor
    return {
        "global_shift_3d": [shift_value, shift_value + 1.0, shift_value + 2.0],
        "global_corr": 0.95,
        "flow_2d": np.ones((2, 3, 4), dtype=np.float32),
        "flow_3d": flow_3d,
        "final_corr": 0.91,
        "is_reference_round": False,
        "round_id": int(shift_value),
        "_semantics": _semantics_payload(),
        "_scope": _tile_scope_payload(),
        "backend_metadata": {"backend": "native"},
        "user_metadata": {"operator": "qa"},
        "legacy_round_note": {"keep": True},
    }


def test_transform_entry_round_trip_preserves_legacy_shape_and_extra_metadata() -> None:
    payload = _round_payload(shift_value=7.0)

    entry = TransformEntry.from_legacy(7, payload, field_name="transform round 7")
    dumped = entry.to_legacy()
    flow_3d = cast(object, entry.flow_3d)
    dumped_payload = cast(dict[object, object], dumped)
    dumped_round_ten_flow = cast(dict[str, object], dumped_payload["flow_3d"])

    assert entry.round_id == 7
    assert isinstance(flow_3d, Flow3DSidecarDescriptor)
    assert entry.user_metadata == {"operator": "qa"}
    assert entry.extra["legacy_round_note"] == {"keep": True}
    assert entry.field_semantics.extra["legacy_semantics_note"] == "keep-me"
    assert entry.scope is not None
    assert entry.scope.extra["legacy_scope_note"] == "keep-me"

    assert dumped_payload["legacy_round_note"] == {"keep": True}
    assert dumped_payload["user_metadata"] == {"operator": "qa"}
    assert dumped_round_ten_flow == _descriptor_payload()
    assert dumped_payload["_semantics"] == _semantics_payload()
    assert dumped_payload["_scope"] == _tile_scope_payload()
    np.testing.assert_array_equal(cast(FloatArray, dumped_payload["flow_2d"]), np.ones((2, 3, 4), dtype=np.float32))


def test_transform_manifest_round_trip_normalizes_round_key_order_and_preserves_metadata() -> None:
    payload = {
        "schema_version": 0,
        "manifest_note": {"keep": "yes"},
        "10": _round_payload(shift_value=10.0),
        2: _round_payload(shift_value=2.0),
        "user_metadata": {"session": "abc"},
    }

    manifest = TransformManifest.from_legacy(payload, fov_id=7)
    dumped = cast(dict[object, object], manifest.to_legacy(include_schema_version=True))
    round_keys: list[int] = []
    for raw_key in dumped:
        if isinstance(raw_key, int):
            round_keys.append(raw_key)

    assert [entry.round_id for entry in manifest.entries] == [2, 10]
    assert round_keys == [2, 10]
    assert dumped["schema_version"] == 0
    assert dumped["manifest_note"] == {"keep": "yes"}
    assert dumped["user_metadata"] == {"session": "abc"}
    round_two = cast(dict[str, object], dumped[2])
    assert round_two["legacy_round_note"] == {"keep": True}
    round_ten = cast(dict[str, object], dumped[10])
    round_ten_flow = cast(dict[str, object], round_ten["flow_3d"])
    assert round_ten_flow["path"] == "transforms_fov_7_round_2_flow_3d.npy"


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (
            lambda: FieldSemantics.from_legacy(
                {
                    "representation": "bogus",
                    "composition": "sequential_global_then_local",
                    "status": "settled",
                },
                field_name="transform._semantics",
            ),
            r"transform\._semantics\.representation",
        ),
        (
            lambda: ScopeMetadata.from_legacy(
                {
                    "coverage_mode": "tile_local",
                    "region_origin_zyx": [0, 0, 0],
                    "region_shape_zyx": [2, 2, 2],
                    "full_volume_shape_zyx": [8, 8, 8],
                    "tile_grid_shape_yx": [2, 2],
                    "tile_index": 5,
                },
                field_name="transform._scope",
            ),
            r"transform\._scope\.tile_index=5 exceeds tile grid capacity 4",
        ),
        (
            lambda: Flow3DSidecarDescriptor.from_legacy(
                {
                    "storage": FLOW_3D_SIDECAR_STORAGE,
                    "path": "../escape.npy",
                },
                field_name="transform.flow_3d",
            ),
            r"transform\.flow_3d\.path must be a direct filename under transforms/",
        ),
        (
            lambda: ReleaseContract.from_legacy(
                {
                    "requested_scope_mode": "full_fov",
                    "delivered_coverage": "full_fov",
                    "scope_valid": True,
                    "scope_status": "valid",
                    "release_gate": {"status": "bogus"},
                },
                field_name="release_contract",
            ),
            r"release_contract\.release_gate\.status must be one of",
        ),
    ],
)
def test_runtime_artifact_models_fail_loudly_on_malformed_payloads(
    factory: Callable[[], object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _ = factory()
