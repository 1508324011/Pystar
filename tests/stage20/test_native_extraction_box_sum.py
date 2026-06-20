from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from pystar.extraction_utils import (
    RoundExtractionTransformPlan,
    _build_coordinate_mapping_sampling_plan,
    _build_image_warp_sampling_plan,
    build_round_extraction_transform_plan,
    get_transform_scope,
    extract_box_sum_integer,
    extract_signal_volume,
    map_spot_coordinates,
    warp_volume_to_reference,
)
from pystar.infrastructure import ExperimentConfig
from pystar.io import get_fov_output_structure
from pystar.mining import SignalMiner
from pystar.mining import _resolve_extraction_route
from pystar.runtime_artifacts import FieldSemantics, Flow3DSidecarDescriptor, ScopeMetadata
import pystar.extraction_utils as extraction_utils_module
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


def _signal_miner(
    provider: str = 'native',
    transform_application_mode: object = 'coordinate_mapping',
) -> SignalMiner:
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
                        transform_application_mode=transform_application_mode,
                    ),
                    field_semantics=field_semantics,
                )
            ),
        ),
    )
    return miner


def _route_config(provider: object, transform_application_mode: object) -> ExperimentConfig:
    return cast(
        ExperimentConfig,
        cast(
            object,
            SimpleNamespace(
                pipeline=SimpleNamespace(
                    extraction=SimpleNamespace(
                        provider=provider,
                        transform_application_mode=transform_application_mode,
                    )
                )
            ),
        ),
    )


def _transform_data(*, flow_3d: FloatArray | None = None, global_shift_3d: FloatArray | None = None) -> dict[str, object]:
    return {
        'global_shift_3d': np.zeros(3, dtype=np.float32) if global_shift_3d is None else np.asarray(global_shift_3d, dtype=np.float32),
        'flow_2d': None,
        'flow_3d': flow_3d,
        'is_reference_round': flow_3d is None and bool(np.allclose(
            np.zeros(3, dtype=np.float32) if global_shift_3d is None else np.asarray(global_shift_3d, dtype=np.float32),
            0.0,
        )),
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
        'tile_grid_shape_yx': [2, 2],
        'tile_index': 1,
    }
    return payload


@pytest.mark.parametrize(
    ('provider', 'mode', 'uses_native_sampling_plan', 'operation_label'),
    [
        ('native', 'coordinate_mapping', True, 'coordinate_mapping extraction'),
        ('native', 'image_warp', True, 'image_warp extraction'),
        ('matlab', 'coordinate_mapping', False, 'coordinate_mapping extraction'),
        ('matlab', 'image_warp', False, 'image_warp extraction'),
    ],
)
def test_resolve_extraction_route_accepts_supported_provider_mode_pairs(
    provider: str,
    mode: str,
    uses_native_sampling_plan: bool,
    operation_label: str,
) -> None:
    route = _resolve_extraction_route(_route_config(provider, mode))

    assert route.provider == provider
    assert route.transform_application_mode == mode
    assert route.uses_native_sampling_plan is uses_native_sampling_plan
    assert route.operation_label == operation_label


def test_resolve_extraction_route_rejects_unknown_provider_before_execution() -> None:
    with pytest.raises(ValueError, match="Unsupported extraction provider.*native.*matlab"):
        _ = _resolve_extraction_route(_route_config('custom', 'coordinate_mapping'))


def test_resolve_extraction_route_rejects_unknown_transform_application_mode_before_execution() -> None:
    with pytest.raises(ValueError, match="Unsupported transform application mode.*coordinate_mapping.*image_warp"):
        _ = _resolve_extraction_route(_route_config('native', 'sparse_sampling'))


def test_signal_miner_extract_channel_rejects_unsupported_provider_before_backend_dispatch() -> None:
    miner = _signal_miner(provider='custom')
    img_vol = np.ones((2, 4, 4), dtype=np.float32)
    ref_coords = np.asarray([[0.0, 1.0, 1.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="Unsupported extraction provider"):
        _ = miner._extract_intensities_for_channel(
            img_vol=img_vol,
            ref_coords=ref_coords,
            transform_data=_transform_data(),
            box_size=(1, 3, 3),
            transform_application_mode='coordinate_mapping',
            fov_id=7,
            round_id=2,
            channel_id=0,
        )


def test_signal_miner_extract_channel_rejects_unsupported_mode_before_backend_dispatch() -> None:
    miner = _signal_miner(
        provider='native',
        transform_application_mode='sparse_sampling',
    )
    img_vol = np.ones((2, 4, 4), dtype=np.float32)
    ref_coords = np.asarray([[0.0, 1.0, 1.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="Unsupported transform application mode"):
        _ = miner._extract_intensities_for_channel(
            img_vol=img_vol,
            ref_coords=ref_coords,
            transform_data=_transform_data(),
            box_size=(1, 3, 3),
            transform_application_mode='sparse_sampling',
            fov_id=7,
            round_id=2,
            channel_id=0,
        )


def _image_warp_oracle_values(
    img_vol: FloatArray,
    ref_coords: FloatArray,
    transform_data: dict[str, object] | RoundExtractionTransformPlan,
    box_size: tuple[int, int, int],
) -> FloatArray:
    warped = warp_volume_to_reference(
        img_vol,
        transform_data,
        expected_field_semantics=EXPECTED_FIELD_SEMANTICS,
    )
    return extract_box_sum_integer(warped, ref_coords, box_size=box_size)


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


def test_coordinate_mapping_sampling_plan_matches_existing_coordinate_mapping_oracle() -> None:
    img_vol = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
    ref_coords = np.asarray([[0.2, 1.8, 2.1], [1.0, 2.0, 3.0]], dtype=np.float32)
    transform_data = _transform_data(
        global_shift_3d=np.asarray([0.0, 0.25, -0.5], dtype=np.float32),
        flow_3d=np.zeros((3, *img_vol.shape), dtype=np.float32),
    )
    box_size = (1, 3, 3)

    plan = _build_coordinate_mapping_sampling_plan(
        img_shape=tuple(img_vol.shape),
        ref_coords=ref_coords,
        transform_data=transform_data,
        box_size=box_size,
        expected_field_semantics=EXPECTED_FIELD_SEMANTICS,
    )

    expected_coords = map_spot_coordinates(
        ref_coords,
        transform_data,
        expected_field_semantics=EXPECTED_FIELD_SEMANTICS,
    )
    expected = extract_box_sum_integer(img_vol, expected_coords, box_size=box_size)

    np.testing.assert_array_equal(plan.target_coords, expected_coords)
    np.testing.assert_array_equal(plan.sample(img_vol), expected)


def test_coordinate_mapping_sampling_plan_rejects_image_shape_mismatch() -> None:
    img_vol = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
    ref_coords = np.asarray([[0.2, 1.8, 2.1]], dtype=np.float32)
    plan = _build_coordinate_mapping_sampling_plan(
        img_shape=tuple(img_vol.shape),
        ref_coords=ref_coords,
        transform_data=_transform_data(),
        box_size=(1, 3, 3),
        expected_field_semantics=EXPECTED_FIELD_SEMANTICS,
    )

    with pytest.raises(ValueError, match='coordinate_mapping sampling plan image shape'):
        _ = plan.sample(np.ones((3, 4, 4), dtype=np.float32))


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


def test_image_warp_sampling_plan_reference_round_preserves_direct_box_sum() -> None:
    img_vol = (np.arange(3 * 5 * 6, dtype=np.float32).reshape(3, 5, 6) / np.float32(7.0))
    ref_coords = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.2, 2.6, 3.4],
            [2.0, 4.0, 5.0],
        ],
        dtype=np.float32,
    )
    transform_data = _transform_data()
    transform_data['_scope'] = {
        'coverage_mode': 'full_fov',
        'region_origin_zyx': [0, 0, 0],
        'region_shape_zyx': list(img_vol.shape),
        'full_volume_shape_zyx': list(img_vol.shape),
    }
    box_size = (1, 3, 3)

    plan = _build_image_warp_sampling_plan(
        img_shape=tuple(img_vol.shape),
        ref_coords=ref_coords,
        transform_data=transform_data,
        box_size=box_size,
        expected_field_semantics=EXPECTED_FIELD_SEMANTICS,
    )

    got = plan.sample(img_vol)
    expected = extract_box_sum_integer(img_vol, ref_coords, box_size=box_size)

    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, expected)


def test_image_warp_sampling_plan_matches_global_shift_oracle() -> None:
    img_vol = np.arange(4 * 6 * 7, dtype=np.float32).reshape(4, 6, 7)
    ref_coords = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [2.1, 4.2, 5.3],
            [3.0, 5.0, 6.0],
        ],
        dtype=np.float32,
    )
    transform_data = _transform_data(
        global_shift_3d=np.asarray([0.35, -0.45, 0.7], dtype=np.float32),
        flow_3d=np.zeros((3, *img_vol.shape), dtype=np.float32),
    )
    transform_data['_scope'] = {
        'coverage_mode': 'full_fov',
        'region_origin_zyx': [0, 0, 0],
        'region_shape_zyx': list(img_vol.shape),
        'full_volume_shape_zyx': list(img_vol.shape),
    }
    box_size = (2, 2, 2)

    plan = _build_image_warp_sampling_plan(
        img_shape=tuple(img_vol.shape),
        ref_coords=ref_coords,
        transform_data=transform_data,
        box_size=box_size,
        expected_field_semantics=EXPECTED_FIELD_SEMANTICS,
    )

    got = plan.sample(img_vol)
    expected = _image_warp_oracle_values(img_vol, ref_coords, transform_data, box_size)

    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, expected)


def test_image_warp_sampling_plan_matches_full_fov_flow_oracle_and_preserves_input() -> None:
    img_vol = np.linspace(-2.5, 3.5, num=4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    img_before = img_vol.copy()
    zz = np.linspace(-0.2, 0.2, num=4, dtype=np.float32)[:, None, None]
    yy = np.linspace(0.15, -0.15, num=5, dtype=np.float32)[None, :, None]
    xx = np.linspace(-0.1, 0.1, num=6, dtype=np.float32)[None, None, :]
    flow_3d = np.empty((3, *img_vol.shape), dtype=np.float32)
    flow_3d[0] = zz + np.zeros(img_vol.shape, dtype=np.float32)
    flow_3d[1] = yy + np.zeros(img_vol.shape, dtype=np.float32)
    flow_3d[2] = xx + np.zeros(img_vol.shape, dtype=np.float32)
    flow_before = flow_3d.copy()
    ref_coords = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.4, 3.2, 4.7],
            [3.0, 4.0, 5.0],
        ],
        dtype=np.float32,
    )
    transform_data = _transform_data(
        global_shift_3d=np.asarray([0.25, -0.25, 0.5], dtype=np.float32),
        flow_3d=flow_3d,
    )
    transform_data['_scope'] = {
        'coverage_mode': 'full_fov',
        'region_origin_zyx': [0, 0, 0],
        'region_shape_zyx': list(img_vol.shape),
        'full_volume_shape_zyx': list(img_vol.shape),
    }
    box_size = (3, 3, 1)

    plan = _build_image_warp_sampling_plan(
        img_shape=tuple(img_vol.shape),
        ref_coords=ref_coords,
        transform_data=transform_data,
        box_size=box_size,
        expected_field_semantics=EXPECTED_FIELD_SEMANTICS,
    )

    got = plan.sample(img_vol)
    expected = _image_warp_oracle_values(img_vol, ref_coords, transform_data, box_size)

    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, expected)
    np.testing.assert_array_equal(img_vol, img_before)
    np.testing.assert_array_equal(flow_3d, flow_before)


def test_image_warp_sampling_plan_preserves_map_coordinates_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    img_vol = np.linspace(0.0, 1.0, num=3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    flow_3d = np.zeros((3, *img_vol.shape), dtype=np.float32)
    flow_3d[0] = np.float32(0.1)
    flow_3d[1] = np.float32(-0.2)
    flow_3d[2] = np.float32(0.3)
    ref_coords = np.asarray([[0.0, 0.0, 0.0], [1.3, 2.2, 3.1]], dtype=np.float32)
    transform_data = _transform_data(
        global_shift_3d=np.asarray([0.25, -0.5, 0.75], dtype=np.float32),
        flow_3d=flow_3d,
    )
    transform_data['_scope'] = {
        'coverage_mode': 'full_fov',
        'region_origin_zyx': [0, 0, 0],
        'region_shape_zyx': list(img_vol.shape),
        'full_volume_shape_zyx': list(img_vol.shape),
    }
    calls: list[dict[str, object]] = []
    original_map_coordinates = cast(Any, extraction_utils_module.map_coordinates)

    def recording_map_coordinates(*args: object, **kwargs: object) -> object:
        calls.append(dict(kwargs))
        return original_map_coordinates(*args, **kwargs)

    monkeypatch.setattr(extraction_utils_module, 'map_coordinates', recording_map_coordinates)

    plan = _build_image_warp_sampling_plan(
        img_shape=tuple(img_vol.shape),
        ref_coords=ref_coords,
        transform_data=transform_data,
        box_size=(1, 3, 3),
        expected_field_semantics=EXPECTED_FIELD_SEMANTICS,
    )
    _ = plan.sample(img_vol)

    assert len(calls) == 2
    for call in calls:
        assert call == {
            'order': 1,
            'mode': 'constant',
            'cval': 0.0,
            'prefilter': False,
        }


def test_image_warp_sampling_plan_matches_tile_local_flow_oracle() -> None:
    img_vol = np.arange(4 * 6 * 7, dtype=np.float32).reshape(4, 6, 7)
    flow_3d = np.zeros((3, 2, 3, 4), dtype=np.float32)
    flow_3d[0] = np.float32(0.15)
    flow_3d[1] = np.linspace(-0.2, 0.2, num=3, dtype=np.float32)[None, :, None]
    flow_3d[2] = np.linspace(0.25, -0.25, num=4, dtype=np.float32)[None, None, :]
    ref_coords = np.asarray(
        [
            [1.0, 2.0, 2.0],
            [1.4, 3.2, 4.1],
            [2.0, 4.0, 5.0],
        ],
        dtype=np.float32,
    )
    transform_data = _transform_data(
        global_shift_3d=np.asarray([0.4, 0.1, -0.3], dtype=np.float32),
        flow_3d=flow_3d,
    )
    transform_data['_scope'] = {
        'coverage_mode': 'tile_local',
        'region_origin_zyx': [1, 2, 2],
        'region_shape_zyx': [2, 3, 4],
        'full_volume_shape_zyx': list(img_vol.shape),
        'tile_grid_shape_yx': [2, 2],
        'tile_index': 3,
    }
    box_size = (1, 3, 3)

    plan = _build_image_warp_sampling_plan(
        img_shape=tuple(img_vol.shape),
        ref_coords=ref_coords,
        transform_data=transform_data,
        box_size=box_size,
        expected_field_semantics=EXPECTED_FIELD_SEMANTICS,
    )

    got = plan.sample(img_vol)
    expected = _image_warp_oracle_values(img_vol, ref_coords, transform_data, box_size)

    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, expected)


def test_image_warp_sampling_plan_tile_local_scope_exclusion_fails_loudly() -> None:
    img_vol = np.arange(4 * 6 * 7, dtype=np.float32).reshape(4, 6, 7)
    transform_data = _transform_data(flow_3d=np.zeros((3, 2, 3, 4), dtype=np.float32))
    transform_data['_scope'] = {
        'coverage_mode': 'tile_local',
        'region_origin_zyx': [1, 2, 2],
        'region_shape_zyx': [2, 3, 4],
        'full_volume_shape_zyx': list(img_vol.shape),
        'tile_grid_shape_yx': [2, 2],
        'tile_index': 3,
    }
    ref_coords = np.asarray([[0.0, 0.0, 0.0], [1.0, 2.0, 2.0]], dtype=np.float32)

    with pytest.raises(ValueError, match='outside tile_local coverage'):
        _ = _build_image_warp_sampling_plan(
            img_shape=tuple(img_vol.shape),
            ref_coords=ref_coords,
            transform_data=transform_data,
            box_size=(1, 3, 3),
            expected_field_semantics=EXPECTED_FIELD_SEMANTICS,
        )


def test_image_warp_sampling_plan_rejects_unsupported_flow_2d() -> None:
    img_vol = np.ones((3, 4, 5), dtype=np.float32)
    transform_data = _transform_data(flow_3d=np.zeros((3, *img_vol.shape), dtype=np.float32))
    transform_data['flow_2d'] = np.zeros((2, img_vol.shape[1], img_vol.shape[2]), dtype=np.float32)
    transform_data['_scope'] = {
        'coverage_mode': 'full_fov',
        'region_origin_zyx': [0, 0, 0],
        'region_shape_zyx': list(img_vol.shape),
        'full_volume_shape_zyx': list(img_vol.shape),
    }

    with pytest.raises(ValueError, match='does not support 2D flow'):
        _ = _build_image_warp_sampling_plan(
            img_shape=tuple(img_vol.shape),
            ref_coords=np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32),
            transform_data=transform_data,
            expected_field_semantics=EXPECTED_FIELD_SEMANTICS,
        )


def test_image_warp_sampling_plan_rejects_missing_non_reference_flow_3d() -> None:
    img_vol = np.ones((3, 4, 5), dtype=np.float32)
    transform_data = _transform_data(
        global_shift_3d=np.asarray([0.25, 0.0, 0.0], dtype=np.float32),
        flow_3d=None,
    )
    transform_data['_scope'] = {
        'coverage_mode': 'full_fov',
        'region_origin_zyx': [0, 0, 0],
        'region_shape_zyx': list(img_vol.shape),
        'full_volume_shape_zyx': list(img_vol.shape),
    }

    with pytest.raises(ValueError, match='requires materialized flow_3d'):
        _ = _build_image_warp_sampling_plan(
            img_shape=tuple(img_vol.shape),
            ref_coords=np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32),
            transform_data=transform_data,
            expected_field_semantics=EXPECTED_FIELD_SEMANTICS,
        )


def test_image_warp_sampling_plan_rejects_unresolved_flow_3d_descriptor() -> None:
    img_vol = np.ones((3, 4, 5), dtype=np.float32)
    transform_data = _transform_data(
        global_shift_3d=np.asarray([0.25, 0.0, 0.0], dtype=np.float32),
        flow_3d=None,
    )
    transform_data['flow_3d'] = _flow_descriptor_payload()
    transform_data['_scope'] = {
        'coverage_mode': 'full_fov',
        'region_origin_zyx': [0, 0, 0],
        'region_shape_zyx': list(img_vol.shape),
        'full_volume_shape_zyx': list(img_vol.shape),
    }

    with pytest.raises(ValueError, match='unresolved manifest metadata'):
        _ = _build_image_warp_sampling_plan(
            img_shape=tuple(img_vol.shape),
            ref_coords=np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32),
            transform_data=transform_data,
            expected_field_semantics=EXPECTED_FIELD_SEMANTICS,
        )


def test_image_warp_sampling_plan_rejects_full_fov_flow_shape_mismatch() -> None:
    img_vol = np.ones((3, 4, 5), dtype=np.float32)
    transform_data = _transform_data(flow_3d=np.zeros((3, 2, 4, 5), dtype=np.float32))
    transform_data['_scope'] = {
        'coverage_mode': 'full_fov',
        'region_origin_zyx': [0, 0, 0],
        'region_shape_zyx': list(img_vol.shape),
        'full_volume_shape_zyx': list(img_vol.shape),
    }

    with pytest.raises(ValueError, match='does not match image volume'):
        _ = _build_image_warp_sampling_plan(
            img_shape=tuple(img_vol.shape),
            ref_coords=np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32),
            transform_data=transform_data,
            expected_field_semantics=EXPECTED_FIELD_SEMANTICS,
        )


def test_image_warp_sampling_plan_handles_empty_spot_matrix_like_oracle() -> None:
    img_vol = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    ref_coords = np.zeros((0, 3), dtype=np.float32)
    transform_data = _transform_data(flow_3d=np.zeros((3, *img_vol.shape), dtype=np.float32))
    transform_data['_scope'] = {
        'coverage_mode': 'full_fov',
        'region_origin_zyx': [0, 0, 0],
        'region_shape_zyx': list(img_vol.shape),
        'full_volume_shape_zyx': list(img_vol.shape),
    }

    plan = _build_image_warp_sampling_plan(
        img_shape=tuple(img_vol.shape),
        ref_coords=ref_coords,
        transform_data=transform_data,
        box_size=(1, 3, 3),
        expected_field_semantics=EXPECTED_FIELD_SEMANTICS,
    )

    got = plan.sample(img_vol)
    expected = _image_warp_oracle_values(img_vol, ref_coords, transform_data, (1, 3, 3))

    assert got.dtype == np.float32
    assert got.shape == (0,)
    np.testing.assert_array_equal(got, expected)


def test_image_warp_sampling_plan_preserves_invalid_empty_coordinate_shape_error() -> None:
    img_vol = np.ones((2, 3, 4), dtype=np.float32)
    coords = np.asarray([], dtype=np.float32)
    transform_data = _transform_data(flow_3d=np.zeros((3, *img_vol.shape), dtype=np.float32))
    transform_data['_scope'] = {
        'coverage_mode': 'full_fov',
        'region_origin_zyx': [0, 0, 0],
        'region_shape_zyx': list(img_vol.shape),
        'full_volume_shape_zyx': list(img_vol.shape),
    }

    with pytest.raises(IndexError):
        _ = _build_image_warp_sampling_plan(
            img_shape=tuple(img_vol.shape),
            ref_coords=coords,
            transform_data=transform_data,
            box_size=(1, 3, 3),
            expected_field_semantics=EXPECTED_FIELD_SEMANTICS,
        )


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


def test_signal_miner_mine_fov_reuses_coordinate_mapping_plan_per_round_image_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    miner = _signal_miner(provider='native')
    field_semantics = miner.cfg.pipeline.field_semantics
    miner.cfg = cast(
        ExperimentConfig,
        cast(
            object,
            SimpleNamespace(
                dataset=SimpleNamespace(
                    channel_roles={0: 'seq', 1: 'seq', 2: 'seq'},
                    round_structure={1: [0, 1]},
                ),
                pipeline=SimpleNamespace(
                    output=SimpleNamespace(directory=str(tmp_path)),
                    extraction=SimpleNamespace(
                        provider='native',
                        transform_application_mode='coordinate_mapping',
                        integration_box=[1, 3, 3],
                    ),
                    field_semantics=field_semantics,
                    qc_images_enabled=lambda: False,
                ),
            ),
        ),
    )
    paths = get_fov_output_structure(tmp_path, 7)
    spots_df = pd.DataFrame(
        {
            'z': [0.0, 1.0],
            'y': [1.0, 2.0],
            'x': [2.0, 3.0],
            'intensity': [10.0, 20.0],
        }
    )
    spots_df.to_csv(paths['spots'] / 'spots_fov_7.csv', index=False)

    img_ch0 = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
    img_ch1 = img_ch0 + np.float32(100.0)
    loaded_images = {(1, 0): img_ch0, (1, 1): img_ch1}
    miner.loader = cast(
        Any,
        SimpleNamespace(load_clean_image=lambda fov_id, round_id, channel_id: loaded_images[(round_id, channel_id)].copy()),
    )

    transform_data = _transform_data()
    miner._load_transforms = lambda fov_id: {1: dict(transform_data)}  # type: ignore[method-assign]

    def fake_materialize_round_transform(
        fov_id: int,
        round_id: int,
        transform_data: Mapping[str, Any],
        *,
        hydrate_flow_3d: bool = True,
    ) -> dict[str, Any]:
        return dict(transform_data)

    miner._materialize_round_transform = fake_materialize_round_transform  # type: ignore[method-assign]
    miner._validate_scope_contract = (  # type: ignore[method-assign]
        lambda fov_id, transforms: {'delivered_coverage': 'full_fov'}
    )

    map_call_count = 0
    original_map_spot_coordinates = extraction_utils_module.map_spot_coordinates

    def counting_map_spot_coordinates(
        ref_coords: FloatArray,
        transform_data: Any,
        expected_field_semantics: Mapping[str, str] | None = None,
    ) -> FloatArray:
        nonlocal map_call_count
        map_call_count += 1
        return original_map_spot_coordinates(
            ref_coords,
            transform_data,
            expected_field_semantics=expected_field_semantics,
        )

    monkeypatch.setattr(extraction_utils_module, 'map_spot_coordinates', counting_map_spot_coordinates)

    miner.mine_fov(7)

    ref_coords = cast(FloatArray, spots_df[['z', 'y', 'x']].to_numpy(dtype=np.float32))
    expected_coords = original_map_spot_coordinates(
        ref_coords,
        transform_data,
        expected_field_semantics=EXPECTED_FIELD_SEMANTICS,
    )
    expected = np.zeros((2, 1, 3), dtype=np.float32)
    expected[:, 0, 0] = extract_box_sum_integer(img_ch0, expected_coords, box_size=(1, 3, 3))
    expected[:, 0, 1] = extract_box_sum_integer(img_ch1, expected_coords, box_size=(1, 3, 3))
    matrix_path = paths['extraction'] / 'intensity_matrix_fov_7.npy'
    staged_matrix_path = matrix_path.with_name(f"{matrix_path.name}.tmp")
    assert matrix_path.exists()
    assert not staged_matrix_path.exists()
    got = np.load(matrix_path, allow_pickle=False)
    got_mmap = np.load(matrix_path, allow_pickle=False, mmap_mode='r')

    assert map_call_count == 1
    assert isinstance(got_mmap, np.memmap)
    assert got_mmap.mode == 'r'
    assert got.dtype == np.float32
    assert got_mmap.dtype == np.float32
    assert got.shape == expected.shape
    assert got_mmap.shape == expected.shape
    np.testing.assert_array_equal(got[:, :, 2], np.zeros((2, 1), dtype=np.float32))
    np.testing.assert_array_equal(got, expected)
    np.testing.assert_array_equal(got_mmap, expected)


def test_signal_miner_mine_fov_writes_memmap_npy_and_zero_fills_missing_channels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    miner = _signal_miner(provider='native')
    field_semantics = miner.cfg.pipeline.field_semantics
    miner.cfg = cast(
        ExperimentConfig,
        cast(
            object,
            SimpleNamespace(
                dataset=SimpleNamespace(
                    channel_roles={0: 'seq', 1: 'seq'},
                    round_structure={1: [0], 2: [1]},
                ),
                pipeline=SimpleNamespace(
                    output=SimpleNamespace(directory=str(tmp_path)),
                    extraction=SimpleNamespace(
                        provider='native',
                        transform_application_mode='coordinate_mapping',
                        integration_box=[1, 3, 3],
                    ),
                    field_semantics=field_semantics,
                    qc_images_enabled=lambda: False,
                ),
            ),
        ),
    )
    paths = get_fov_output_structure(tmp_path, 7)
    spots_df = pd.DataFrame(
        {
            'z': [0.0, 1.0],
            'y': [1.0, 2.0],
            'x': [2.0, 3.0],
            'intensity': [10.0, 20.0],
        }
    )
    spots_df.to_csv(paths['spots'] / 'spots_fov_7.csv', index=False)

    img_ch0 = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
    img_ch1 = img_ch0 + np.float32(100.0)
    loaded_images = {(1, 0): img_ch0, (2, 1): img_ch1}
    miner.loader = cast(
        Any,
        SimpleNamespace(load_clean_image=lambda fov_id, round_id, channel_id: loaded_images[(round_id, channel_id)].copy()),
    )

    transform_data = _transform_data()
    miner._load_transforms = lambda fov_id: {1: dict(transform_data), 2: dict(transform_data)}  # type: ignore[method-assign]

    def fake_materialize_round_transform(
        fov_id: int,
        round_id: int,
        transform_data: Mapping[str, Any],
        *,
        hydrate_flow_3d: bool = True,
    ) -> dict[str, Any]:
        return dict(transform_data)

    miner._materialize_round_transform = fake_materialize_round_transform  # type: ignore[method-assign]
    miner._validate_scope_contract = (  # type: ignore[method-assign]
        lambda fov_id, transforms: {'delivered_coverage': 'full_fov'}
    )

    open_memmap_calls: list[dict[str, object]] = []
    original_open_memmap = mining_module.np.lib.format.open_memmap

    def recording_open_memmap(filename: str | Path, *args: object, **kwargs: object) -> object:
        open_memmap_calls.append(
            {
                'filename': Path(filename),
                'mode': kwargs.get('mode'),
                'dtype': kwargs.get('dtype'),
                'shape': kwargs.get('shape'),
            }
        )
        return original_open_memmap(filename, *args, **kwargs)

    monkeypatch.setattr(mining_module.np.lib.format, 'open_memmap', recording_open_memmap)
    monkeypatch.setattr(
        mining_module.np,
        'save',
        lambda *args, **kwargs: pytest.fail('mine_fov must not call np.save for intensity matrix output'),
    )

    miner.mine_fov(7)

    matrix_path = paths['extraction'] / 'intensity_matrix_fov_7.npy'
    staged_matrix_path = matrix_path.with_name(f"{matrix_path.name}.tmp")
    write_memmap_calls = [call for call in open_memmap_calls if call['mode'] == 'w+']
    assert write_memmap_calls == [
        {'filename': staged_matrix_path, 'mode': 'w+', 'dtype': np.float32, 'shape': (2, 2, 2)}
    ]
    readonly_memmap_calls = [call for call in open_memmap_calls if call['mode'] == 'r']
    assert any(call['filename'] == matrix_path for call in readonly_memmap_calls)

    ref_coords = cast(FloatArray, spots_df[['z', 'y', 'x']].to_numpy(dtype=np.float32))
    expected_coords = map_spot_coordinates(
        ref_coords,
        transform_data,
        expected_field_semantics=EXPECTED_FIELD_SEMANTICS,
    )
    expected = np.zeros((2, 2, 2), dtype=np.float32)
    expected[:, 0, 0] = extract_box_sum_integer(img_ch0, expected_coords, box_size=(1, 3, 3))
    expected[:, 1, 1] = extract_box_sum_integer(img_ch1, expected_coords, box_size=(1, 3, 3))

    assert matrix_path.exists()
    assert not staged_matrix_path.exists()
    got = np.load(matrix_path, allow_pickle=False)
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, expected)

    mmap_loaded = np.load(matrix_path, allow_pickle=False, mmap_mode='r')
    assert isinstance(mmap_loaded, np.memmap)
    assert mmap_loaded.mode == 'r'
    assert mmap_loaded.dtype == np.float32
    np.testing.assert_array_equal(mmap_loaded, expected)


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


def test_signal_miner_image_warp_hot_path_profile_records_flow_and_cache_details(
    tmp_path: Path,
) -> None:
    miner = _signal_miner(provider='native', transform_application_mode='image_warp')
    field_semantics = miner.cfg.pipeline.field_semantics
    miner.cfg = cast(
        ExperimentConfig,
        cast(
            object,
            SimpleNamespace(
                dataset=SimpleNamespace(
                    channel_roles={0: 'seq', 1: 'seq'},
                    round_structure={1: [0, 1]},
                ),
                pipeline=SimpleNamespace(
                    scope_mode='full_fov',
                    output=SimpleNamespace(directory=str(tmp_path)),
                    extraction=SimpleNamespace(
                        provider='native',
                        transform_application_mode='image_warp',
                        integration_box=[1, 3, 3],
                        profile_hot_path=True,
                    ),
                    field_semantics=field_semantics,
                    qc_images_enabled=lambda: False,
                ),
            ),
        ),
    )
    paths = get_fov_output_structure(tmp_path, 7)
    spots_df = pd.DataFrame(
        {
            'z': [0.0, 1.0],
            'y': [1.0, 2.0],
            'x': [2.0, 3.0],
            'intensity': [10.0, 20.0],
        }
    )
    spots_df.to_csv(paths['spots'] / 'spots_fov_7.csv', index=False)

    img_ch0 = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
    img_ch1 = img_ch0 + np.float32(100.0)
    loaded_images = {(1, 0): img_ch0, (1, 1): img_ch1}
    miner.loader = cast(
        Any,
        SimpleNamespace(load_clean_image=lambda fov_id, round_id, channel_id: loaded_images[(round_id, channel_id)].copy()),
    )

    flow_3d = np.zeros((3, 2, 4, 4), dtype=np.float32)
    transform_data = _transform_data(flow_3d=flow_3d)
    miner._load_transforms = lambda fov_id: {1: dict(transform_data)}  # type: ignore[method-assign]
    miner._materialize_round_transform = (  # type: ignore[method-assign]
        lambda fov_id, round_id, transform_data, *, hydrate_flow_3d=True: dict(transform_data)
    )
    miner._validate_scope_contract = (  # type: ignore[method-assign]
        lambda fov_id, transforms: {'delivered_coverage': 'full_fov'}
    )
    miner._validate_image_warp_contract = lambda fov_id, transforms: None  # type: ignore[method-assign]

    miner.mine_fov(7)

    profile_path = paths['qc'] / 'extraction_hot_path_profile_fov_7.json'
    assert profile_path.exists()
    profile = json.loads(profile_path.read_text(encoding='utf-8'))
    prep_events = [
        event for event in profile['events']
        if event['bucket'] == 'coordinate_or_warp_preparation'
    ]
    assert len(prep_events) >= 2
    miss_event = next(event for event in prep_events if event['details'].get('cache_hit') is False)
    hit_event = next(event for event in prep_events if event['details'].get('cache_hit') is True)
    assert miss_event['details']['sampling_plan'] == 'image_warp'
    assert miss_event['details']['transform_application_mode'] == 'image_warp'
    assert miss_event['details']['flow_3d']['shape'] == [3, 2, 4, 4]
    assert miss_event['details']['spot_count'] == 2
    assert hit_event['details']['sampling_plan'] == 'image_warp'
    assert hit_event['details']['cache_hit'] is True

    got = np.load(paths['extraction'] / 'intensity_matrix_fov_7.npy', allow_pickle=False)
    expected = np.zeros((2, 1, 2), dtype=np.float32)
    ref_coords = cast(FloatArray, spots_df[['z', 'y', 'x']].to_numpy(dtype=np.float32))
    expected[:, 0, 0] = extract_box_sum_integer(img_ch0, ref_coords, box_size=(1, 3, 3))
    expected[:, 0, 1] = extract_box_sum_integer(img_ch1, ref_coords, box_size=(1, 3, 3))
    np.testing.assert_array_equal(got, expected)
