from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray


ArrayLike = NDArray[Any]


def split_extent_evenly(length: int, parts: int) -> list[tuple[int, int]]:
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    if parts <= 0:
        raise ValueError(f"parts must be positive, got {parts}")
    if parts > length:
        raise ValueError(
            f"parts must not exceed length for discrete tiling, got parts={parts}, length={length}"
        )

    boundaries = np.linspace(0, length, parts + 1, dtype=int)
    return [(int(boundaries[idx]), int(boundaries[idx + 1])) for idx in range(parts)]


def resolve_grid_shape_yx(
    full_shape_zyx: Sequence[int],
    *,
    tile_size: int,
    sqrt_pieces: int | None = None,
    tile_grid_shape_yx: Sequence[int] | None = None,
) -> tuple[tuple[int, int], str]:
    if len(full_shape_zyx) != 3:
        raise ValueError(f"full_shape_zyx must contain 3 integers, got {full_shape_zyx!r}")

    _, y_dim, x_dim = (int(value) for value in full_shape_zyx)
    if y_dim <= 0 or x_dim <= 0:
        raise ValueError(f"full_shape_zyx must be positive, got {full_shape_zyx!r}")
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")

    if tile_grid_shape_yx is not None:
        if len(tile_grid_shape_yx) != 2:
            raise ValueError("tile_grid_shape_yx must contain exactly two integers [y_tiles, x_tiles]")
        grid_shape = (int(tile_grid_shape_yx[0]), int(tile_grid_shape_yx[1]))
        if grid_shape[0] <= 0 or grid_shape[1] <= 0:
            raise ValueError(f"tile_grid_shape_yx must be positive, got {tile_grid_shape_yx!r}")
        if grid_shape[0] > y_dim or grid_shape[1] > x_dim:
            raise ValueError(
                "tile_grid_shape_yx must not exceed spatial dimensions, "
                f"got grid_shape_yx={grid_shape!r}, full_shape_zyx={full_shape_zyx!r}"
            )
        return grid_shape, "tile_grid_shape_yx"

    if sqrt_pieces is not None:
        sqrt_value = int(sqrt_pieces)
        if sqrt_value <= 0:
            raise ValueError(f"sqrt_pieces must be positive when provided, got {sqrt_pieces}")
        if sqrt_value > y_dim or sqrt_value > x_dim:
            raise ValueError(
                "sqrt_pieces must not exceed spatial dimensions, "
                f"got sqrt_pieces={sqrt_value}, full_shape_zyx={full_shape_zyx!r}"
            )
        return (sqrt_value, sqrt_value), "sqrt_pieces"

    y_tiles = max(1, int(ceil(y_dim / tile_size)))
    x_tiles = max(1, int(ceil(x_dim / tile_size)))
    return (y_tiles, x_tiles), "tile_size"


@dataclass(frozen=True)
class TileSpec:
    tile_index: int
    grid_position_yx: tuple[int, int]
    grid_shape_yx: tuple[int, int]
    region_origin_zyx: tuple[int, int, int]
    region_shape_zyx: tuple[int, int, int]
    write_origin_zyx: tuple[int, int, int]
    write_shape_zyx: tuple[int, int, int]
    write_offset_zyx: tuple[int, int, int]
    full_volume_shape_zyx: tuple[int, int, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "tile_index": self.tile_index,
            "grid_position_yx": [int(self.grid_position_yx[0]), int(self.grid_position_yx[1])],
            "grid_shape_yx": [int(self.grid_shape_yx[0]), int(self.grid_shape_yx[1])],
            "region_origin_zyx": [int(value) for value in self.region_origin_zyx],
            "region_shape_zyx": [int(value) for value in self.region_shape_zyx],
            "write_origin_zyx": [int(value) for value in self.write_origin_zyx],
            "write_shape_zyx": [int(value) for value in self.write_shape_zyx],
            "write_offset_zyx": [int(value) for value in self.write_offset_zyx],
            "full_volume_shape_zyx": [int(value) for value in self.full_volume_shape_zyx],
        }


@dataclass(frozen=True)
class TileLayout:
    full_volume_shape_zyx: tuple[int, int, int]
    grid_shape_yx: tuple[int, int]
    overlap_yx: tuple[int, int]
    grid_source: str
    tiles: tuple[TileSpec, ...]

    @property
    def tile_count(self) -> int:
        return len(self.tiles)

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "grid_shape_yx": [int(self.grid_shape_yx[0]), int(self.grid_shape_yx[1])],
            "overlap_yx": [int(self.overlap_yx[0]), int(self.overlap_yx[1])],
            "grid_source": self.grid_source,
            "tile_count": self.tile_count,
            "tiles": [tile.as_dict() for tile in self.tiles],
        }


def build_yx_tile_layout(
    full_shape_zyx: Sequence[int],
    *,
    grid_shape_yx: Sequence[int],
    overlap_yx: Sequence[int] = (0, 0),
    grid_source: str = "manual",
    index_order: str = "row_major_yx",
) -> TileLayout:
    if len(full_shape_zyx) != 3:
        raise ValueError(f"full_shape_zyx must contain 3 integers, got {full_shape_zyx!r}")
    if len(grid_shape_yx) != 2:
        raise ValueError(f"grid_shape_yx must contain 2 integers, got {grid_shape_yx!r}")
    if len(overlap_yx) != 2:
        raise ValueError(f"overlap_yx must contain 2 integers, got {overlap_yx!r}")

    z_dim, y_dim, x_dim = (int(value) for value in full_shape_zyx)
    y_tiles, x_tiles = (int(value) for value in grid_shape_yx)
    overlap_y, overlap_x = (int(value) for value in overlap_yx)
    if z_dim <= 0 or y_dim <= 0 or x_dim <= 0:
        raise ValueError(f"full_shape_zyx must be positive, got {full_shape_zyx!r}")
    if y_tiles <= 0 or x_tiles <= 0:
        raise ValueError(f"grid_shape_yx must be positive, got {grid_shape_yx!r}")
    if overlap_y < 0 or overlap_x < 0:
        raise ValueError(f"overlap_yx must be non-negative, got {overlap_yx!r}")
    if index_order not in {"row_major_yx", "column_major_xy"}:
        raise ValueError(
            "index_order must be either 'row_major_yx' or 'column_major_xy', "
            f"got {index_order!r}"
        )

    y_ranges = split_extent_evenly(y_dim, y_tiles)
    x_ranges = split_extent_evenly(x_dim, x_tiles)

    grid_sequence: list[tuple[int, tuple[int, int], int, tuple[int, int]]] = []
    if index_order == "row_major_yx":
        for grid_y, y_range in enumerate(y_ranges):
            for grid_x, x_range in enumerate(x_ranges):
                grid_sequence.append((grid_y, y_range, grid_x, x_range))
    else:
        for grid_x, x_range in enumerate(x_ranges):
            for grid_y, y_range in enumerate(y_ranges):
                grid_sequence.append((grid_y, y_range, grid_x, x_range))

    tiles: list[TileSpec] = []
    tile_index = 1
    for grid_y, (core_y0, core_y1), grid_x, (core_x0, core_x1) in grid_sequence:
        region_y0 = max(0, core_y0 - overlap_y)
        region_y1 = min(y_dim, core_y1 + overlap_y)
        region_x0 = max(0, core_x0 - overlap_x)
        region_x1 = min(x_dim, core_x1 + overlap_x)

        if core_y1 <= core_y0 or core_x1 <= core_x0:
            raise ValueError(
                "Tiling produced an empty write region for tile "
                f"({grid_y}, {grid_x}) in full_shape_zyx={full_shape_zyx!r} "
                f"with grid_shape_yx={grid_shape_yx!r}"
            )
        if region_y1 <= region_y0 or region_x1 <= region_x0:
            raise ValueError(
                "Tiling produced an empty extraction region for tile "
                f"({grid_y}, {grid_x}) in full_shape_zyx={full_shape_zyx!r} "
                f"with grid_shape_yx={grid_shape_yx!r}"
            )

        tiles.append(
            TileSpec(
                tile_index=tile_index,
                grid_position_yx=(grid_y, grid_x),
                grid_shape_yx=(y_tiles, x_tiles),
                region_origin_zyx=(0, region_y0, region_x0),
                region_shape_zyx=(z_dim, region_y1 - region_y0, region_x1 - region_x0),
                write_origin_zyx=(0, core_y0, core_x0),
                write_shape_zyx=(z_dim, core_y1 - core_y0, core_x1 - core_x0),
                write_offset_zyx=(0, core_y0 - region_y0, core_x0 - region_x0),
                full_volume_shape_zyx=(z_dim, y_dim, x_dim),
            )
        )
        tile_index += 1

    return TileLayout(
        full_volume_shape_zyx=(z_dim, y_dim, x_dim),
        grid_shape_yx=(y_tiles, x_tiles),
        overlap_yx=(overlap_y, overlap_x),
        grid_source=grid_source,
        tiles=tuple(tiles),
    )


def build_matlab_subtile_layout(
    full_shape_zyx: Sequence[int],
    *,
    sqrt_pieces: int,
    overlap_yx: Sequence[int] | None = None,
    grid_source: str = "matlab_subtile",
) -> TileLayout:
    if len(full_shape_zyx) != 3:
        raise ValueError(f"full_shape_zyx must contain 3 integers, got {full_shape_zyx!r}")

    z_dim, y_dim, x_dim = (int(value) for value in full_shape_zyx)
    sqrt_value = int(sqrt_pieces)
    if sqrt_value <= 0:
        raise ValueError(f"sqrt_pieces must be positive, got {sqrt_pieces}")
    if sqrt_value > y_dim or sqrt_value > x_dim:
        raise ValueError(
            "sqrt_pieces must not exceed spatial dimensions for MATLAB subtile layout, "
            f"got sqrt_pieces={sqrt_value}, full_shape_zyx={full_shape_zyx!r}"
        )

    core_tile_y = y_dim // sqrt_value
    core_tile_x = x_dim // sqrt_value
    if core_tile_y <= 0 or core_tile_x <= 0:
        raise ValueError(
            "MATLAB subtile layout requires non-empty per-axis core tiles, "
            f"got full_shape_zyx={full_shape_zyx!r}, sqrt_pieces={sqrt_value}"
        )

    if overlap_yx is None:
        overlap_y = int(core_tile_y * 0.1)
        overlap_x = int(core_tile_x * 0.1)
    else:
        if len(overlap_yx) != 2:
            raise ValueError(f"overlap_yx must contain 2 integers, got {overlap_yx!r}")
        overlap_y, overlap_x = (int(value) for value in overlap_yx)
        if overlap_y < 0 or overlap_x < 0:
            raise ValueError(f"overlap_yx must be non-negative, got {overlap_yx!r}")

    tiles: list[TileSpec] = []
    tile_index = 1
    for grid_x in range(sqrt_value):
        for grid_y in range(sqrt_value):
            core_y0 = grid_y * core_tile_y
            core_x0 = grid_x * core_tile_x
            core_y1 = y_dim if grid_y == sqrt_value - 1 else (grid_y + 1) * core_tile_y
            core_x1 = x_dim if grid_x == sqrt_value - 1 else (grid_x + 1) * core_tile_x

            region_y0 = core_y0 if grid_y == 0 else max(0, core_y0 - overlap_y)
            region_x0 = core_x0 if grid_x == 0 else max(0, core_x0 - overlap_x)
            region_y1 = y_dim if grid_y == sqrt_value - 1 else min(y_dim, core_y1 + overlap_y)
            region_x1 = x_dim if grid_x == sqrt_value - 1 else min(x_dim, core_x1 + overlap_x)

            tiles.append(
                TileSpec(
                    tile_index=tile_index,
                    grid_position_yx=(grid_y, grid_x),
                    grid_shape_yx=(sqrt_value, sqrt_value),
                    region_origin_zyx=(0, region_y0, region_x0),
                    region_shape_zyx=(z_dim, region_y1 - region_y0, region_x1 - region_x0),
                    write_origin_zyx=(0, core_y0, core_x0),
                    write_shape_zyx=(z_dim, core_y1 - core_y0, core_x1 - core_x0),
                    write_offset_zyx=(0, core_y0 - region_y0, core_x0 - region_x0),
                    full_volume_shape_zyx=(z_dim, y_dim, x_dim),
                )
            )
            tile_index += 1

    return TileLayout(
        full_volume_shape_zyx=(z_dim, y_dim, x_dim),
        grid_shape_yx=(sqrt_value, sqrt_value),
        overlap_yx=(overlap_y, overlap_x),
        grid_source=grid_source,
        tiles=tuple(tiles),
    )


def extract_tile(array: ArrayLike, tile: TileSpec) -> ArrayLike:
    if tuple(array.shape[-3:]) != tile.full_volume_shape_zyx:
        raise ValueError(
            "Tile extraction shape mismatch: "
            f"expected trailing spatial shape {tile.full_volume_shape_zyx}, got {array.shape[-3:]}"
        )

    z0, y0, x0 = tile.region_origin_zyx
    dz, dy, dx = tile.region_shape_zyx
    z1, y1, x1 = z0 + dz, y0 + dy, x0 + dx
    slicer = (..., slice(z0, z1), slice(y0, y1), slice(x0, x1))
    return np.asarray(array[slicer])


def extract_tile_write_window(array: ArrayLike, tile: TileSpec) -> ArrayLike:
    if tuple(array.shape[-3:]) != tile.full_volume_shape_zyx:
        raise ValueError(
            "Tile write-window extraction shape mismatch: "
            f"expected trailing spatial shape {tile.full_volume_shape_zyx}, got {array.shape[-3:]}"
        )

    z0, y0, x0 = tile.write_origin_zyx
    dz, dy, dx = tile.write_shape_zyx
    z1, y1, x1 = z0 + dz, y0 + dy, x0 + dx
    slicer = (..., slice(z0, z1), slice(y0, y1), slice(x0, x1))
    return np.asarray(array[slicer])


def stitch_tiles(
    tile_outputs: Iterable[tuple[TileSpec, ArrayLike]],
    *,
    full_shape_zyx: Sequence[int],
) -> ArrayLike:
    outputs = list(tile_outputs)
    if not outputs:
        raise ValueError("tile_outputs must not be empty")

    normalized_full_shape = tuple(int(value) for value in full_shape_zyx)
    _, first_array = outputs[0]
    prefix_shape = tuple(first_array.shape[:-3])
    stitched_shape = prefix_shape + normalized_full_shape
    stitched = np.zeros(stitched_shape, dtype=np.asarray(first_array).dtype)
    coverage = np.zeros(normalized_full_shape, dtype=np.uint16)

    for tile, tile_array in outputs:
        arr = np.asarray(tile_array)
        if tuple(arr.shape[:-3]) != prefix_shape:
            raise ValueError(
                f"Tile output prefix shape mismatch: expected {prefix_shape}, got {arr.shape[:-3]}"
            )
        if tuple(arr.shape[-3:]) != tile.region_shape_zyx:
            raise ValueError(
                "Tile output spatial shape mismatch: "
                f"expected {tile.region_shape_zyx}, got {arr.shape[-3:]} for tile {tile.tile_index}"
            )
        if tile.full_volume_shape_zyx != normalized_full_shape:
            raise ValueError(
                "Tile full-volume shape mismatch during stitch: "
                f"expected {normalized_full_shape}, got {tile.full_volume_shape_zyx}"
            )

        src_z0, src_y0, src_x0 = tile.write_offset_zyx
        src_dz, src_dy, src_dx = tile.write_shape_zyx
        dst_z0, dst_y0, dst_x0 = tile.write_origin_zyx
        dst_dz, dst_dy, dst_dx = tile.write_shape_zyx

        src_slicer = (..., slice(src_z0, src_z0 + src_dz), slice(src_y0, src_y0 + src_dy), slice(src_x0, src_x0 + src_dx))
        dst_slicer = (..., slice(dst_z0, dst_z0 + dst_dz), slice(dst_y0, dst_y0 + dst_dy), slice(dst_x0, dst_x0 + dst_dx))
        stitched[dst_slicer] = arr[src_slicer]
        coverage[dst_z0:dst_z0 + dst_dz, dst_y0:dst_y0 + dst_dy, dst_x0:dst_x0 + dst_dx] += 1

    if np.any(coverage == 0):
        raise ValueError("Tile stitching left uncovered voxels in the target full volume")
    if np.any(coverage > 1):
        raise ValueError("Tile stitching detected overlapping write coverage in the target full volume")

    return stitched
