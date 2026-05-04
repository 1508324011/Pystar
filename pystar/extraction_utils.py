"""Coordinate replay and signal extraction helpers.

All public functions in this module use NumPy image coordinates in `z, y, x`
order. Spot coordinates are expressed in the reference-round frame unless a
function name explicitly says it maps them into a moving-round image. Transform
payloads are the dictionaries persisted by `RegistrationEngine`; their
`_semantics` and `_scope` metadata are treated as part of the runtime contract,
not optional decoration.
"""

from collections.abc import Mapping
from typing import cast

import numpy as np
import numpy.typing as npt
from scipy.ndimage import map_coordinates

from .runtime_artifacts import FieldSemantics, ScopeMetadata


FloatArray = npt.NDArray[np.float32]
TransformData = Mapping[str, object] | None


def _unknown_field_semantics() -> dict[str, str]:
    return _field_semantics_public_payload(FieldSemantics.unknown())


def _field_semantics_public_payload(semantics: FieldSemantics) -> dict[str, str]:
    return {
        "representation": semantics.representation,
        "composition": semantics.composition,
        "status": semantics.status,
    }


def _normalize_field_semantics_payload(
    payload: object,
    *,
    field_name: str,
) -> dict[str, str]:
    semantics = FieldSemantics.from_legacy(payload, field_name=field_name)
    return _field_semantics_public_payload(semantics)


def validate_field_semantics(
    transform_data: TransformData,
    expected_field_semantics: Mapping[str, str] | None,
) -> dict[str, object]:
    """Compare persisted transform semantics with the caller's expectation.

    Parameters
    ----------
    transform_data:
        Per-round transform dictionary loaded from the transform manifest. The
        `_semantics` member, when present, declares field representation and
        composition.
    expected_field_semantics:
        The semantics requested by the pipeline config. `None` means the caller
        only wants the actual payload normalized, not enforced.

    Returns
    -------
    dict
        A small validation report containing `valid`, normalized `expected`,
        normalized `actual`, and the list of mismatched semantic axes. Only
        `representation` and `composition` are enforced because `status` is
        provenance, not geometry.
    """
    actual_payload = None if transform_data is None else transform_data.get("_semantics")
    actual = _normalize_field_semantics_payload(
        actual_payload,
        field_name="transform _semantics",
    )

    if expected_field_semantics is None:
        return {
            "valid": True,
            "expected": _unknown_field_semantics(),
            "actual": actual,
            "mismatches": [],
        }

    expected = _normalize_field_semantics_payload(
        expected_field_semantics,
        field_name="expected_field_semantics",
    )
    mismatches = [
        key
        for key in ("representation", "composition")
        if expected.get(key) != actual.get(key)
    ]
    return {
        "valid": len(mismatches) == 0,
        "expected": expected,
        "actual": actual,
        "mismatches": mismatches,
    }


def _raise_if_field_semantics_mismatch(
    transform_data: TransformData,
    expected_field_semantics: Mapping[str, str] | None,
    *,
    operation: str,
) -> None:
    if transform_data is None or expected_field_semantics is None:
        return

    validation = validate_field_semantics(transform_data, expected_field_semantics)
    if validation["valid"]:
        return

    expected = validation["expected"]
    actual = validation["actual"]
    if not isinstance(expected, Mapping) or not isinstance(actual, Mapping):
        raise ValueError(f"Field semantics mismatch for {operation}: malformed validation payload")
    raise ValueError(
        "Field semantics mismatch for {operation}: expected representation={expected_rep!r}, "
        "composition={expected_comp!r}; got representation={actual_rep!r}, composition={actual_comp!r}, "
        "status={actual_status!r}.".format(
            operation=operation,
            expected_rep=expected.get("representation"),
            expected_comp=expected.get("composition"),
            actual_rep=actual.get("representation"),
            actual_comp=actual.get("composition"),
            actual_status=actual.get("status"),
        )
    )


def _require_materialized_flow(flow: object, field_name: str) -> None:
    if flow is None or isinstance(flow, np.ndarray):
        return
    if isinstance(flow, Mapping):
        raise ValueError(
            f"{field_name} contains unresolved manifest metadata. "
            f"Load transforms via the persisted transform loader before replaying extraction."
        )
    raise ValueError(f"Unsupported {field_name} payload type: {type(flow)}")


def _map_coordinates_float32(
    input_array: npt.ArrayLike,
    coordinates: list[FloatArray],
    *,
    order: int,
    mode: str,
    cval: float | None = None,
    prefilter: bool | None = None,
) -> FloatArray:
    if cval is None and prefilter is None:
        mapped = map_coordinates(input_array, coordinates, order=order, mode=mode)
    elif cval is None:
        assert prefilter is not None
        mapped = map_coordinates(
            input_array,
            coordinates,
            order=order,
            mode=mode,
            prefilter=prefilter,
        )
    elif prefilter is None:
        mapped = map_coordinates(
            input_array,
            coordinates,
            order=order,
            mode=mode,
            cval=cval,
        )
    else:
        mapped = map_coordinates(
            input_array,
            coordinates,
            order=order,
            mode=mode,
            cval=cval,
            prefilter=prefilter,
        )
    return cast(FloatArray, np.asarray(mapped, dtype=np.float32))


def _sparse_coordinate_bases_3d(shape: tuple[int, int, int]) -> tuple[FloatArray, FloatArray, FloatArray]:
    nz, ny, nx = shape
    return (
        np.arange(nz, dtype=np.float32)[:, None, None],
        np.arange(ny, dtype=np.float32)[None, :, None],
        np.arange(nx, dtype=np.float32)[None, None, :],
    )


def _broadcast_coordinate_views_3d(
    shape: tuple[int, int, int],
    z_coords: npt.ArrayLike,
    y_coords: npt.ArrayLike,
    x_coords: npt.ArrayLike,
) -> list[FloatArray]:
    return [
        cast(FloatArray, np.broadcast_to(np.asarray(z_coords, dtype=np.float32), shape)),
        cast(FloatArray, np.broadcast_to(np.asarray(y_coords, dtype=np.float32), shape)),
        cast(FloatArray, np.broadcast_to(np.asarray(x_coords, dtype=np.float32), shape)),
    ]


def _shape_3d(shape: tuple[int, ...], *, field_name: str) -> tuple[int, int, int]:
    if len(shape) != 3:
        raise ValueError(f"{field_name} must be 3D, got shape {shape}")
    return (int(shape[0]), int(shape[1]), int(shape[2]))


def get_transform_scope(transform_data: TransformData) -> dict[str, object] | None:
    """Return normalized spatial coverage metadata for a transform.

    `None` means the transform does not declare a scope and is treated like a
    full-FOV artifact by older callers. A returned scope has `coverage_mode`
    (`full_fov` or `tile_local`), `region_origin_zyx`, `region_shape_zyx`, and
    `full_volume_shape_zyx`. Tile-local scopes also contain `tile_grid_shape_yx`
    and `tile_index` so extraction can reject coordinates outside the delivered
    deformation field instead of extrapolating silently.
    """
    if transform_data is None:
        return None

    payload = transform_data.get("_scope")
    if payload is None:
        return None
    scope = ScopeMetadata.from_legacy(payload, field_name="transform _scope")
    normalized: dict[str, object] = {
        "coverage_mode": scope.coverage_mode,
        "region_origin_zyx": scope.region_origin_zyx,
        "region_shape_zyx": scope.region_shape_zyx,
        "full_volume_shape_zyx": scope.full_volume_shape_zyx,
    }
    if scope.coverage_mode == "tile_local":
        normalized["tile_grid_shape_yx"] = scope.tile_grid_shape_yx
        normalized["tile_index"] = scope.tile_index
    return normalized


def coords_within_transform_scope(
    ref_coords: FloatArray,
    transform_data: TransformData,
) -> npt.NDArray[np.bool_]:
    """Mask reference-frame coordinates that lie inside a transform scope.

    Coordinates are compared against half-open intervals in `z, y, x` order:
    `[origin, origin + shape)`. Full-FOV transforms accept every coordinate.
    Tile-local transforms only accept spots inside the persisted tile region.
    """
    scope = get_transform_scope(transform_data)
    if scope is None or scope["coverage_mode"] == "full_fov":
        return np.ones(len(ref_coords), dtype=bool)

    origin = scope["region_origin_zyx"]
    shape = scope["region_shape_zyx"]
    if not isinstance(origin, tuple) or not isinstance(shape, tuple):
        raise ValueError("transform _scope is malformed: missing region_origin_zyx/region_shape_zyx tuples")

    z0, y0, x0 = origin
    dz, dy, dx = shape
    return (
        (ref_coords[:, 0] >= z0)
        & (ref_coords[:, 0] < z0 + dz)
        & (ref_coords[:, 1] >= y0)
        & (ref_coords[:, 1] < y0 + dy)
        & (ref_coords[:, 2] >= x0)
        & (ref_coords[:, 2] < x0 + dx)
    )


def _require_coords_within_scope(
    ref_coords: FloatArray,
    transform_data: TransformData,
    *,
    operation: str,
) -> dict[str, object] | None:
    scope = get_transform_scope(transform_data)
    if scope is None or scope["coverage_mode"] == "full_fov":
        return scope

    in_scope = coords_within_transform_scope(ref_coords, transform_data)
    if np.all(in_scope):
        return scope

    outside_count = int((~in_scope).sum())
    raise ValueError(
        f"{operation} received {outside_count} reference coordinates outside tile_local coverage. "
        "Filter coordinates by persisted scope metadata before replaying extraction."
    )


def _is_reference_round_transform(transform_data: Mapping[str, object], global_shift: FloatArray) -> bool:
    marker = transform_data.get('is_reference_round')
    if isinstance(marker, bool):
        return marker
    return bool(np.allclose(global_shift, 0.0) and transform_data.get('flow_2d') is None and transform_data.get('flow_3d') is None)


def map_spot_coordinates(
    ref_coords: FloatArray,
    transform_data: TransformData,
    expected_field_semantics: Mapping[str, str] | None = None,
) -> FloatArray:
    """Map reference-frame spot coordinates into a moving-round image.

    This is the legacy/diagnostic `coordinate_mapping` replay path. Input and
    output arrays have shape `(N, 3)` in `z, y, x` order. The first operation is
    `mapped -= global_shift_3d`, because a spot observed at reference position
    `p_ref` is sampled from moving image position `p_ref - shift`. Optional
    `flow_2d` or `flow_3d` residual fields are then sampled at the mapped
    coordinate and added component-wise.

    The function fails loudly when field semantics disagree with the pipeline
    contract, when a transform still contains unresolved sidecar descriptors,
    or when tile-local scope metadata does not cover every requested spot.
    """
    if transform_data is None:
        return ref_coords.astype(np.float32)

    _raise_if_field_semantics_mismatch(
        transform_data,
        expected_field_semantics,
        operation='coordinate_mapping replay',
    )
    scope = _require_coords_within_scope(
        ref_coords,
        transform_data,
        operation='coordinate_mapping replay',
    )

    mapped = ref_coords.copy().astype(np.float32)
    global_shift = np.asarray(transform_data.get('global_shift_3d', np.zeros(3, dtype=np.float32)), dtype=np.float32)
    mapped -= global_shift

    flow_2d = transform_data.get('flow_2d')
    flow_3d = transform_data.get('flow_3d')
    _require_materialized_flow(flow_2d, 'flow_2d')
    _require_materialized_flow(flow_3d, 'flow_3d')
    flow = flow_2d if flow_2d is not None else flow_3d

    if flow is not None:
        if isinstance(flow, np.ndarray) and flow.ndim == 3:
            if scope is not None and scope.get('coverage_mode') == 'tile_local':
                origin = scope['region_origin_zyx']
                region_shape = scope['region_shape_zyx']
                if not isinstance(origin, tuple) or not isinstance(region_shape, tuple):
                    raise ValueError('transform _scope is malformed for tile_local coordinate_mapping replay')
                _, y0, x0 = origin
                _, h, w = region_shape
                if flow.shape[1:] != (h, w):
                    raise ValueError(
                        f"tile_local flow_2d shape {flow.shape[1:]} does not match persisted scope region {(h, w)}"
                    )
                sample_y = np.clip(mapped[:, 1] - y0, 0, h - 1)
                sample_x = np.clip(mapped[:, 2] - x0, 0, w - 1)
            else:
                h, w = flow.shape[1], flow.shape[2]
                sample_y = np.clip(mapped[:, 1], 0, h - 1)
                sample_x = np.clip(mapped[:, 2], 0, w - 1)
            dy = _map_coordinates_float32(
                flow[0],
                [sample_y, sample_x],
                order=1,
                mode='nearest',
            )
            dx = _map_coordinates_float32(
                flow[1],
                [sample_y, sample_x],
                order=1,
                mode='nearest',
            )
            mapped[:, 1] += dy
            mapped[:, 2] += dx
        elif isinstance(flow, np.ndarray) and flow.ndim == 4:
            if scope is not None and scope.get('coverage_mode') == 'tile_local':
                origin = scope['region_origin_zyx']
                region_shape = scope['region_shape_zyx']
                if not isinstance(origin, tuple) or not isinstance(region_shape, tuple):
                    raise ValueError('transform _scope is malformed for tile_local coordinate_mapping replay')
                z0, y0, x0 = origin
                d, h, w = region_shape
                if flow.shape[1:] != (d, h, w):
                    raise ValueError(
                        f"tile_local flow_3d shape {flow.shape[1:]} does not match persisted scope region {(d, h, w)}"
                    )
                sample_z = np.clip(mapped[:, 0] - z0, 0, d - 1)
                sample_y = np.clip(mapped[:, 1] - y0, 0, h - 1)
                sample_x = np.clip(mapped[:, 2] - x0, 0, w - 1)
            else:
                d, h, w = flow.shape[1:]
                sample_z = np.clip(mapped[:, 0], 0, d - 1)
                sample_y = np.clip(mapped[:, 1], 0, h - 1)
                sample_x = np.clip(mapped[:, 2], 0, w - 1)
            dz = _map_coordinates_float32(
                flow[0],
                [sample_z, sample_y, sample_x],
                order=1,
                mode='nearest',
            )
            dy = _map_coordinates_float32(
                flow[1],
                [sample_z, sample_y, sample_x],
                order=1,
                mode='nearest',
            )
            dx = _map_coordinates_float32(
                flow[2],
                [sample_z, sample_y, sample_x],
                order=1,
                mode='nearest',
            )
            mapped[:, 0] += dz
            mapped[:, 1] += dy
            mapped[:, 2] += dx

    return np.asarray(mapped, dtype=np.float32)


def extract_box_sum_integer(img_vol: FloatArray, coords: FloatArray, box_size: tuple[int, int, int] = (1, 3, 3)) -> FloatArray:
    """Sum a centered integer box around each coordinate.

    Coordinates are rounded to the nearest voxel before summation. `box_size` is
    `(z, y, x)` and uses integer half-widths (`size // 2`), so `(1, 3, 3)` sums
    one z-plane and a 3x3 XY patch. Samples outside the image bounds contribute
    zero; they are not renormalized by the number of in-bounds voxels. This
    matches the fail-simple box-sum semantics used by the pipeline miners.
    """
    d, h, w = img_vol.shape
    bz, by, bx = box_size
    rz, ry, rx = bz // 2, by // 2, bx // 2

    n_spots = len(coords)
    intensities = np.zeros(n_spots, dtype=np.float32)
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
                    (cur_z >= 0) & (cur_z < d) &
                    (cur_y >= 0) & (cur_y < h) &
                    (cur_x >= 0) & (cur_x < w)
                )
                if np.any(valid_mask):
                    val = img_vol[cur_z[valid_mask], cur_y[valid_mask], cur_x[valid_mask]]
                    intensities[valid_mask] += val

    return np.asarray(intensities, dtype=np.float32)


def warp_volume_to_reference(
    img_vol: FloatArray,
    transform_data: TransformData,
    expected_field_semantics: Mapping[str, str] | None = None,
) -> FloatArray:
    """Warp one moving-round volume into reference-round coordinates.

    The returned volume has the same `z, y, x` shape as `img_vol`, but its
    pixels are sampled as if the moving round had been registered to the
    reference round. The implementation uses inverse sampling: reference grid
    coordinates are shifted by `-global_shift_3d`, then optional 3D residual
    fields are applied before interpolation. Non-reference rounds in
    `image_warp` mode require a materialized `flow_3d`; 2D flow is rejected
    because it cannot define a full volumetric warp.

    Tile-local transforms are allowed only when `_scope` precisely describes
    the covered subvolume. In that mode only the scoped region is locally
    deformed and stitched back into the full warped volume.
    """
    data: dict[str, object] = {} if transform_data is None else dict(transform_data)
    _raise_if_field_semantics_mismatch(
        data,
        expected_field_semantics,
        operation='image_warp extraction',
    )

    global_shift = np.asarray(data.get('global_shift_3d', np.zeros(3, dtype=np.float32)), dtype=np.float32)
    flow_3d = data.get('flow_3d')
    flow_2d = data.get('flow_2d')
    _require_materialized_flow(flow_2d, 'flow_2d')
    _require_materialized_flow(flow_3d, 'flow_3d')
    if flow_2d is not None:
        raise ValueError('image_warp mode does not support 2D flow yet')
    if flow_3d is None and not _is_reference_round_transform(data, global_shift):
        raise ValueError(
            'image_warp mode requires materialized flow_3d for non-reference rounds; '
            'coordinate_mapping is the legacy diagnostic path for non-3D transforms'
        )

    img_shape = _shape_3d(img_vol.shape, field_name='img_vol')
    z_coords, y_coords, x_coords = _sparse_coordinate_bases_3d(img_shape)
    warped = _map_coordinates_float32(
        img_vol,
        _broadcast_coordinate_views_3d(
            img_shape,
            z_coords - global_shift[0],
            y_coords - global_shift[1],
            x_coords - global_shift[2],
        ),
        order=1,
        mode='constant',
        cval=0.0,
        prefilter=False,
    )

    if not isinstance(flow_3d, np.ndarray):
        return warped

    flow_arr = np.asarray(flow_3d, dtype=np.float32)
    scope = get_transform_scope(data)
    if scope is not None and scope.get('coverage_mode') == 'tile_local':
        origin = scope['region_origin_zyx']
        region_shape = scope['region_shape_zyx']
        if not isinstance(origin, tuple) or not isinstance(region_shape, tuple):
            raise ValueError('transform _scope is malformed for tile_local image_warp extraction')
        z0, y0, x0 = origin
        dz, dy, dx = region_shape
        if flow_arr.shape[1:] != (dz, dy, dx):
            raise ValueError(
                f"tile_local flow_3d shape {flow_arr.shape[1:]} does not match persisted scope region {(dz, dy, dx)}"
            )
        z1, y1, x1 = z0 + dz, y0 + dy, x0 + dx
        warped_region = warped[z0:z1, y0:y1, x0:x1]
        warped_region_shape = _shape_3d(warped_region.shape, field_name='tile_local warped region')
        local_z, local_y, local_x = _sparse_coordinate_bases_3d(warped_region_shape)
        warped_local = _map_coordinates_float32(
            warped_region,
            _broadcast_coordinate_views_3d(
                warped_region_shape,
                local_z + flow_arr[0],
                local_y + flow_arr[1],
                local_x + flow_arr[2],
            ),
            order=1,
            mode='constant',
            cval=0.0,
            prefilter=False,
        )
        warped = warped.copy()
        warped[z0:z1, y0:y1, x0:x1] = warped_local
        return warped

    if flow_arr.shape[1:] != warped.shape:
        raise ValueError(
            f"image_warp flow_3d shape {flow_arr.shape[1:]} does not match image volume {warped.shape}; "
            'persist explicit _scope metadata for tile_local artifacts'
        )

    warped_shape = _shape_3d(warped.shape, field_name='warped image volume')
    z_coords, y_coords, x_coords = _sparse_coordinate_bases_3d(warped_shape)
    return _map_coordinates_float32(
        warped,
        _broadcast_coordinate_views_3d(
            warped_shape,
            z_coords + flow_arr[0],
            y_coords + flow_arr[1],
            x_coords + flow_arr[2],
        ),
        order=1,
        mode='constant',
        cval=0.0,
        prefilter=False,
    )


def extract_signal_volume(
    img_vol: FloatArray,
    ref_coords: FloatArray,
    transform_data: TransformData,
    box_size: tuple[int, int, int] = (1, 3, 3),
    transform_application_mode: str = 'coordinate_mapping',
    expected_field_semantics: Mapping[str, str] | None = None,
) -> FloatArray:
    """Extract one intensity vector for one round/channel volume.

    `ref_coords` are always reference-round `z, y, x` spot coordinates. In
    `coordinate_mapping` mode they are first mapped into the moving image and
    summed there. In `image_warp` mode the whole moving volume is registered to
    reference space first, then the original reference coordinates are summed.
    Both paths share the same box-sum implementation so differences between
    them isolate transform-application semantics rather than integration logic.
    """
    data: dict[str, object] = {} if transform_data is None else dict(transform_data)
    scope = _require_coords_within_scope(
        ref_coords,
        data,
        operation='image_warp extraction' if transform_application_mode == 'image_warp' else 'coordinate_mapping extraction',
    )

    if transform_application_mode == 'coordinate_mapping':
        target_coords = map_spot_coordinates(
            ref_coords,
            data,
            expected_field_semantics=expected_field_semantics,
        )
        return extract_box_sum_integer(img_vol, target_coords, box_size)

    if transform_application_mode == 'image_warp':
        warped = warp_volume_to_reference(
            img_vol,
            data,
            expected_field_semantics=expected_field_semantics,
        )
        return extract_box_sum_integer(warped, ref_coords, box_size)

    raise ValueError(f'Unsupported transform application mode: {transform_application_mode}')
