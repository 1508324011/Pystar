from collections.abc import Mapping
from typing import cast

import numpy as np
import numpy.typing as npt
from scipy.ndimage import map_coordinates


FloatArray = npt.NDArray[np.float32]
TransformData = Mapping[str, object] | None
FIELD_SEMANTICS_REPRESENTATIONS = {"residual", "total", "unknown"}
FIELD_SEMANTICS_COMPOSITIONS = {"sequential_global_then_local", "independent", "unknown"}
FIELD_SEMANTICS_STATUSES = {"settled", "provisional", "unknown"}
SCOPE_COVERAGE_MODES = {"full_fov", "tile_local"}


def _unknown_field_semantics() -> dict[str, str]:
    return {
        "representation": "unknown",
        "composition": "unknown",
        "status": "unknown",
    }


def _normalize_field_semantics_payload(
    payload: object,
    *,
    field_name: str,
) -> dict[str, str]:
    if payload is None:
        return _unknown_field_semantics()
    if not isinstance(payload, Mapping):
        raise ValueError(f"{field_name} must be a mapping")

    representation = payload.get("representation", "unknown")
    composition = payload.get("composition", "unknown")
    status = payload.get("status", "unknown")

    if representation not in FIELD_SEMANTICS_REPRESENTATIONS:
        raise ValueError(
            f"{field_name}.representation must be one of {sorted(FIELD_SEMANTICS_REPRESENTATIONS)}, got {representation!r}"
        )
    if composition not in FIELD_SEMANTICS_COMPOSITIONS:
        raise ValueError(
            f"{field_name}.composition must be one of {sorted(FIELD_SEMANTICS_COMPOSITIONS)}, got {composition!r}"
        )
    if status not in FIELD_SEMANTICS_STATUSES:
        raise ValueError(
            f"{field_name}.status must be one of {sorted(FIELD_SEMANTICS_STATUSES)}, got {status!r}"
        )

    return {
        "representation": str(representation),
        "composition": str(composition),
        "status": str(status),
    }


def validate_field_semantics(
    transform_data: TransformData,
    expected_field_semantics: Mapping[str, str] | None,
) -> dict[str, object]:
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


def _coerce_int_tuple(
    value: object,
    *,
    field_name: str,
    expected_length: int,
) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list/tuple of {expected_length} integers")
    if len(value) != expected_length:
        raise ValueError(f"{field_name} must contain {expected_length} integers")

    coerced: list[int] = []
    for item in value:
        if not isinstance(item, (int, np.integer)):
            raise ValueError(f"{field_name} entries must be integers, got {item!r}")
        coerced.append(int(item))
    return tuple(coerced)


def get_transform_scope(transform_data: TransformData) -> dict[str, object] | None:
    if transform_data is None:
        return None

    payload = transform_data.get("_scope")
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise ValueError("transform _scope must be a mapping")

    coverage_mode = payload.get("coverage_mode")
    if coverage_mode not in SCOPE_COVERAGE_MODES:
        raise ValueError(
            f"transform _scope.coverage_mode must be one of {sorted(SCOPE_COVERAGE_MODES)}, got {coverage_mode!r}"
        )

    region_origin_zyx = _coerce_int_tuple(
        payload.get("region_origin_zyx"),
        field_name="transform _scope.region_origin_zyx",
        expected_length=3,
    )
    region_shape_zyx = _coerce_int_tuple(
        payload.get("region_shape_zyx"),
        field_name="transform _scope.region_shape_zyx",
        expected_length=3,
    )
    full_volume_shape_zyx = _coerce_int_tuple(
        payload.get("full_volume_shape_zyx"),
        field_name="transform _scope.full_volume_shape_zyx",
        expected_length=3,
    )

    if any(value < 0 for value in region_origin_zyx):
        raise ValueError("transform _scope.region_origin_zyx must contain non-negative integers")
    if any(value <= 0 for value in region_shape_zyx):
        raise ValueError("transform _scope.region_shape_zyx must contain positive integers")
    if any(value <= 0 for value in full_volume_shape_zyx):
        raise ValueError("transform _scope.full_volume_shape_zyx must contain positive integers")

    for origin, size, full_size, axis_name in zip(
        region_origin_zyx,
        region_shape_zyx,
        full_volume_shape_zyx,
        ("z", "y", "x"),
    ):
        if origin + size > full_size:
            raise ValueError(
                f"transform _scope {axis_name}-axis region exceeds full volume bounds: "
                f"origin={origin}, size={size}, full={full_size}"
            )

    normalized: dict[str, object] = {
        "coverage_mode": str(coverage_mode),
        "region_origin_zyx": region_origin_zyx,
        "region_shape_zyx": region_shape_zyx,
        "full_volume_shape_zyx": full_volume_shape_zyx,
    }

    tile_grid_shape_yx = payload.get("tile_grid_shape_yx")
    tile_index = payload.get("tile_index")
    if coverage_mode == "tile_local":
        tile_grid_shape = _coerce_int_tuple(
            tile_grid_shape_yx,
            field_name="transform _scope.tile_grid_shape_yx",
            expected_length=2,
        )
        if any(value <= 0 for value in tile_grid_shape):
            raise ValueError("transform _scope.tile_grid_shape_yx must contain positive integers")
        if not isinstance(tile_index, (int, np.integer)) or int(tile_index) <= 0:
            raise ValueError("transform _scope.tile_index must be a positive integer for tile_local coverage")
        tile_index_int = int(tile_index)
        max_tiles = int(tile_grid_shape[0] * tile_grid_shape[1])
        if tile_index_int > max_tiles:
            raise ValueError(
                f"transform _scope.tile_index={tile_index_int} exceeds tile grid capacity {max_tiles}"
            )
        normalized["tile_grid_shape_yx"] = tile_grid_shape
        normalized["tile_index"] = tile_index_int

    return normalized


def coords_within_transform_scope(
    ref_coords: FloatArray,
    transform_data: TransformData,
) -> npt.NDArray[np.bool_]:
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

    z_coords, y_coords, x_coords = np.indices(img_vol.shape, dtype=np.float32)
    warped = _map_coordinates_float32(
        img_vol,
        [
            z_coords - global_shift[0],
            y_coords - global_shift[1],
            x_coords - global_shift[2],
        ],
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
        local_z, local_y, local_x = np.indices(warped_region.shape, dtype=np.float32)
        warped_local = _map_coordinates_float32(
            warped_region,
            [
                local_z + flow_arr[0],
                local_y + flow_arr[1],
                local_x + flow_arr[2],
            ],
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

    z_coords, y_coords, x_coords = np.indices(warped.shape, dtype=np.float32)
    return _map_coordinates_float32(
        warped,
        [
            z_coords + flow_arr[0],
            y_coords + flow_arr[1],
            x_coords + flow_arr[2],
        ],
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
