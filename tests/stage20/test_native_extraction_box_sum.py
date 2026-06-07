from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

from pystar.extraction_utils import (
    RoundExtractionTransformPlan,
    build_round_extraction_transform_plan,
    get_transform_scope,
    extract_box_sum_integer,
    extract_signal_volume,
    map_spot_coordinates,
)
from pystar.infrastructure import ExperimentConfig
from pystar.mining import SignalMiner
from pystar.runtime_artifacts import FieldSemantics, Flow3DSidecarDescriptor, ScopeMetadata
import pystar.mining as mining_module


FloatArray = npt.NDArray[np.float32]
ImageArray = npt.NDArray[np.generic]
EXPECTED_FIELD_SEMANTICS = {
    'representation': 'residual',
    'composition': 'sequential_global_then_local',
}


def _loop_oracle(img_vol: ImageArray, coords: FloatArray, box_size: tuple[int, int, int]) -> FloatArray:
    d, h, w = img_vol.shape
    bz, by, bx = box_size
    rz, ry, rx = bz // 2, by // 2, bx // 2

    intensities = np.zeros(len(coords), dtype=np.float32)
    coords_int = np.rint(coords).astype(np.int32)
    ic_z = coords_int[:, 0]
    ic_y = coords_int[:, 1]
    ic_x = coords_int[:, 2]

    for dz in range(-rz, rz + 1):
        for dy in range(-ry, ry + 1):
            for dx in range(-rx, rx + 1):
                cur_z = ic_z + dz
                cur_y = ic_y + dy
                cur_x = ic_x + dx
                valid_mask = (
                    (cur_z >= 0) & (cur_z < d)
                    & (cur_y >= 0) & (cur_y < h)
                    & (cur_x >= 0) & (cur_x < w)
                )
                if np.any(valid_mask):
                    intensities[valid_mask] += np.asarray(
                        img_vol[
                            cur_z[valid_mask],
                            cur_y[valid_mask],
                            cur_x[valid_mask],
                        ],
                        dtype=np.float32,
                    )

    return intensities


def _signal_miner(provider: str = 'native') -> SignalMiner:
    miner = SignalMiner.__new__(SignalMiner)
    field_semantics = SimpleNamespace(
        representation='residual',
        composition='sequential_global_then_local',
        status='settled',
        as_dict=lambda: {
            'representation': 'residual',
            'composition': 'sequential_global_then_local',
            'status': 'settled',
        },
    )
    miner.cfg = cast(
        ExperimentConfig,
        cast(
            object,
            SimpleNamespace(
                pipeline=SimpleNamespace(
                    extraction=SimpleNamespace(
                        provider=provider,
                        transform_application_mode='coordinate_mapping',
                    ),
                    field_semantics=field_semantics,
                )
            ),
        ),
    )
    return miner


def _transform_data(*, flow_3d: FloatArray | None = None) -> dict[str, object]:
    return {
        'global_shift_3d': np.zeros(3, dtype=np.float32),
        'flow_2d': None,
        'flow_3d': flow_3d,
        'is_reference_round': flow_3d is None,
        '_semantics': {
            'representation': 'residual',
            'composition': 'sequential_global_then_local',
            'status': 'settled',
        },
        '_scope': {
            'coverage_mode': 'full_fov',
            'region_origin_zyx': [0, 0, 0],
            'region_shape_zyx': [2, 4, 4],
            'full_volume_shape_zyx': [2, 4, 4],
        },
    }


def _flow_descriptor_payload() -> dict[str, object]:
    return {
        'storage': 'round_level_sidecar_npy',
        'path': 'transforms_fov_7_round_2_flow_3d.npy',
        'shape': [3, 2, 4, 4],
        'dtype': 'float32',
    }


def _tile_local_transform_data() -> dict[str, object]:
    payload = _transform_data()
    payload['_scope'] = {
        'coverage_mode': 'tile_local',
        'region_origin_zyx': [0, 0, 0],
        'region_shape_zyx': [1, 1, 1],
        'full_volume_shape_zyx': [2, 4, 4],
        'tile_grid_shape_yx': [1, 1],
        'tile_index': 1,
    }
    return payload


def test_round_extraction_transform_plan_exposes_models_and_preserves_legacy_payload() -> None:
    flow_3d = np.zeros((3, 2, 4, 4), dtype=np.float32)
    lazy_payload = _transform_data()
    lazy_payload['flow_3d'] = _flow_descriptor_payload()
    materialized_payload = _transform_data(flow_3d=flow_3d)

    plan = build_round_extraction_transform_plan(
        fov_id=7,
        round_id=2,
        transform_data=materialized_payload,
        source_transform_data=lazy_payload,
    )
    scope_payload = get_transform_scope(plan)
    legacy_payload = plan.legacy_transform_data()

    assert isinstance(plan, RoundExtractionTransformPlan)
    assert isinstance(plan.field_semantics, FieldSemantics)
    assert isinstance(plan.scope, ScopeMetadata)
    assert isinstance(plan.flow_descriptor, Flow3DSidecarDescriptor)
    assert plan.flow_descriptor.path == 'transforms_fov_7_round_2_flow_3d.npy'
    assert scope_payload == {
        'coverage_mode': 'full_fov',
        'region_origin_zyx': (0, 0, 0),
        'region_shape_zyx': (2, 4, 4),
        'full_volume_shape_zyx': (2, 4, 4),
    }
    np.testing.assert_array_equal(cast(FloatArray, legacy_payload['flow_3d']), flow_3d)


def test_extraction_helpers_accept_round_transform_plan_without_value_drift() -> None:
    img_vol = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
    ref_coords = np.asarray([[0.2, 1.8, 2.1], [1.0, 2.0, 3.0]], dtype=np.float32)
    transform_data = _transform_data(flow_3d=np.zeros((3, 2, 4, 4), dtype=np.float32))
    plan = build_round_extraction_transform_plan(
        fov_id=7,
        round_id=2,
        transform_data=transform_data,
    )

    dict_mapped = map_spot_coordinates(
        ref_coords,
        transform_data,
        expected_field_semantics=EXPECTED_FIELD_SEMANTICS,
    )
    plan_mapped = map_spot_coordinates(
        ref_coords,
        plan,
        expected_field_semantics=EXPECTED_FIELD_SEMANTICS,
    )
    dict_vals = extract_signal_volume(
        img_vol,
        ref_coords,
        transform_data,
        box_size=(1, 3, 3),
        transform_application_mode='image_warp',
        expected_field_semantics=EXPECTED_FIELD_SEMANTICS,
    )
    plan_vals = extract_signal_volume(
        img_vol,
        ref_coords,
        plan,
        box_size=(1, 3, 3),
        transform_application_mode='image_warp',
        expected_field_semantics=EXPECTED_FIELD_SEMANTICS,
    )

    np.testing.assert_array_equal(plan_mapped, dict_mapped)
    np.testing.assert_array_equal(plan_vals, dict_vals)


def test_signal_miner_round_transform_plan_uses_materialization_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    miner = _signal_miner(provider='native')
    setattr(cast(Any, miner.cfg.pipeline), 'output', SimpleNamespace(directory=str(tmp_path)))
    calls: list[dict[str, object]] = []

    def fake_materialize(
        base_dir: Path,
        fov_id: int,
        round_id: int,
        transform_data: dict[str, object],
        *,
        hydrate_flow_3d: bool = True,
    ) -> dict[str, object]:
        calls.append(
            {
                'base_dir': base_dir,
                'fov_id': fov_id,
                'round_id': round_id,
                'hydrate_flow_3d': hydrate_flow_3d,
            }
        )
        return dict(transform_data)

    monkeypatch.setattr(mining_module, 'materialize_round_transform_entry', fake_materialize)

    plan = miner._build_round_transform_plan(
        7,
        2,
        _transform_data(),
        hydrate_flow_3d=False,
    )

    assert isinstance(plan, RoundExtractionTransformPlan)
    assert calls == [
        {
            'base_dir': tmp_path,
            'fov_id': 7,
            'round_id': 2,
            'hydrate_flow_3d': False,
        }
    ]


def test_signal_miner_native_bridge_keeps_tile_scope_fail_loud_guardrail() -> None:
    miner = _signal_miner(provider='native')
    img_vol = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
    ref_coords = np.asarray([[1.0, 2.0, 2.0]], dtype=np.float32)
    plan = build_round_extraction_transform_plan(
        fov_id=7,
        round_id=2,
        transform_data=_tile_local_transform_data(),
    )

    with pytest.raises(ValueError, match='outside tile_local coverage'):
        _ = miner._extract_intensities_for_channel(
            img_vol=img_vol,
            ref_coords=ref_coords,
            transform_data=plan,
            box_size=(1, 3, 3),
            transform_application_mode='coordinate_mapping',
            fov_id=7,
            round_id=2,
            channel_id=0,
        )


@pytest.mark.parametrize(
    'box_size',
    [
        (1, 1, 1),
        (1, 3, 3),
        (2, 2, 2),
        (3, 5, 1),
        (4, 4, 4),
    ],
)
def test_extract_box_sum_integer_matches_loop_oracle_on_boundary_cases(box_size: tuple[int, int, int]) -> None:
    rng = np.random.default_rng(123)
    img_vol = (rng.normal(size=(4, 5, 6)) * 7.5).astype(np.float32)
    coords = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.2, 1.4, 1.6],
            [3.0, 4.0, 5.0],
            [-1.0, 2.0, 2.0],
            [3.8, 4.2, 5.9],
        ],
        dtype=np.float32,
    )

    got = extract_box_sum_integer(img_vol, coords, box_size=box_size)
    expected = _loop_oracle(img_vol, coords, box_size)

    assert got.dtype == np.float32
    assert got.shape == expected.shape == (5,)
    np.testing.assert_array_equal(got, expected)


def test_extract_box_sum_integer_handles_empty_coords_and_non_float_image_dtype() -> None:
    img_vol = np.arange(2 * 3 * 4, dtype=np.int16).reshape(2, 3, 4)
    coords = np.zeros((0, 3), dtype=np.float32)

    got = extract_box_sum_integer(cast(FloatArray, img_vol), coords, box_size=(1, 3, 3))

    assert got.dtype == np.float32
    assert got.shape == (0,)
    np.testing.assert_array_equal(got, np.zeros((0,), dtype=np.float32))


def test_extract_box_sum_integer_preserves_invalid_empty_coordinate_shape_error() -> None:
    img_vol = np.ones((2, 3, 4), dtype=np.float32)
    coords = np.asarray([], dtype=np.float32)

    with pytest.raises(IndexError):
        _ = extract_box_sum_integer(img_vol, coords, box_size=(1, 3, 3))


def test_extract_box_sum_integer_matches_loop_oracle_on_randomized_fixtures() -> None:
    rng = np.random.default_rng(321)
    for shape in [(1, 1, 1), (2, 3, 4), (5, 6, 7), (8, 12, 10)]:
        img_vol = rng.integers(-50, 50, size=shape, dtype=np.int16)
        for n_spots in [1, 3, 20, 100]:
            coords = rng.uniform(
                low=-2.0,
                high=np.asarray(shape, dtype=np.float32) + 2.0,
                size=(n_spots, 3),
            ).astype(np.float32)
            for box_size in [(1, 1, 1), (1, 3, 3), (2, 2, 2), (3, 5, 1)]:
                got = extract_box_sum_integer(cast(FloatArray, img_vol), coords, box_size=box_size)
                expected = _loop_oracle(img_vol, coords, box_size)
                np.testing.assert_array_equal(got, expected)


def test_extract_signal_volume_native_coordinate_mapping_uses_same_kernel() -> None:
    img_vol = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
    ref_coords = np.asarray([[0.2, 1.8, 2.1], [1.0, 2.0, 3.0]], dtype=np.float32)
    transform_data = _transform_data()

    mapped = map_spot_coordinates(
        ref_coords,
        transform_data,
        expected_field_semantics=EXPECTED_FIELD_SEMANTICS,
    )
    expected = extract_box_sum_integer(img_vol, mapped, box_size=(1, 3, 3))
    got = extract_signal_volume(
        img_vol,
        ref_coords,
        transform_data,
        box_size=(1, 3, 3),
        transform_application_mode='coordinate_mapping',
        expected_field_semantics=EXPECTED_FIELD_SEMANTICS,
    )

    np.testing.assert_array_equal(got, expected)


def test_extract_signal_volume_native_image_warp_uses_same_kernel() -> None:
    img_vol = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
    ref_coords = np.asarray([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]], dtype=np.float32)
    flow_3d = np.zeros((3, 2, 4, 4), dtype=np.float32)
    transform_data = _transform_data(flow_3d=flow_3d)

    got = extract_signal_volume(
        img_vol,
        ref_coords,
        transform_data,
        box_size=(1, 3, 3),
        transform_application_mode='image_warp',
        expected_field_semantics=EXPECTED_FIELD_SEMANTICS,
    )

    warped = extract_box_sum_integer(img_vol, ref_coords, box_size=(1, 3, 3))
    np.testing.assert_array_equal(got, warped)


def test_signal_miner_native_bridge_uses_optimized_kernel_for_coordinate_mapping() -> None:
    miner = _signal_miner(provider='native')
    img_vol = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
    ref_coords = np.asarray([[0.2, 1.8, 2.1], [1.0, 2.0, 3.0]], dtype=np.float32)
    transform_data = _transform_data()

    vals_obj, metadata = miner._extract_intensities_for_channel(
        img_vol=img_vol,
        ref_coords=ref_coords,
        transform_data=transform_data,
        box_size=(1, 3, 3),
        transform_application_mode='coordinate_mapping',
        fov_id=7,
        round_id=2,
        channel_id=0,
    )

    expected = extract_box_sum_integer(
        img_vol,
        map_spot_coordinates(
            ref_coords,
            transform_data,
            expected_field_semantics=miner._expected_field_semantics(),
        ),
        box_size=(1, 3, 3),
    )

    vals = cast(FloatArray, vals_obj)
    assert metadata is None
    np.testing.assert_array_equal(vals, expected)


def test_signal_miner_native_bridge_uses_optimized_kernel_for_image_warp() -> None:
    miner = _signal_miner(provider='native')
    miner.cfg.pipeline.extraction.transform_application_mode = 'image_warp'
    img_vol = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
    ref_coords = np.asarray([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]], dtype=np.float32)
    transform_data = _transform_data(flow_3d=np.zeros((3, 2, 4, 4), dtype=np.float32))

    vals_obj, metadata = miner._extract_intensities_for_channel(
        img_vol=img_vol,
        ref_coords=ref_coords,
        transform_data=transform_data,
        box_size=(1, 3, 3),
        transform_application_mode='image_warp',
        fov_id=7,
        round_id=2,
        channel_id=0,
    )

    expected = extract_box_sum_integer(img_vol, ref_coords, box_size=(1, 3, 3))

    vals = cast(FloatArray, vals_obj)
    assert metadata is None
    np.testing.assert_array_equal(vals, expected)
