from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import dask.array as da
import numpy as np
import pytest
import tifffile
from numpy.typing import NDArray

from pystar.infrastructure import ExperimentConfig
from pystar._io_image_loader import ImageLoader as PrivateImageLoader
from pystar.io import ImageLoader


def _build_config(
    *,
    raw_dir: Path,
    output_dir: Path,
    round_structure: dict[int, list[int]] | None = None,
    channel_roles: dict[int, str] | None = None,
) -> ExperimentConfig:
    dataset = SimpleNamespace(
        raw_data_path=raw_dir,
        filename_pattern="round{round:03d}/Position{fov}/*_ch{ch}.tif",
        dimensions={"z": 2, "height": 3, "width": 4},
        io_chunk_size={"z": 1, "y": 2, "x": 2},
        round_structure=round_structure if round_structure is not None else {1: [2]},
        channel_roles=channel_roles if channel_roles is not None else {2: "sequencing"},
        pixel_size_xy_nm=100.0,
        pixel_size_z_nm=500.0,
    )
    pipeline = SimpleNamespace(output=SimpleNamespace(directory=str(output_dir)))
    return cast(ExperimentConfig, cast(object, SimpleNamespace(dataset=dataset, pipeline=pipeline)))


def _touch_raw_tiff(raw_dir: Path, *, fov: int, round_id: int, channel_label: str, name: str = "img") -> Path:
    raw_path = raw_dir / f"round{round_id:03d}" / f"Position{fov}" / f"{name}_ch{channel_label}.tif"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(b"synthetic-path-only")
    return raw_path


def _write_raw_tiff(
    raw_dir: Path,
    *,
    fov: int,
    round_id: int,
    channel_label: str,
    data: NDArray[Any],
) -> Path:
    raw_path = raw_dir / f"round{round_id:03d}" / f"Position{fov}" / f"img_ch{channel_label}.tif"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(raw_path, data)
    return raw_path


def test_public_facade_reexports_private_image_loader_without_private_module_importing_facade() -> None:
    assert ImageLoader is PrivateImageLoader

    source_path = Path(__file__).resolve().parents[2] / "pystar" / "_io_image_loader.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name != "pystar.io" for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert not (node.level == 1 and node.module == "io")
            assert node.module != "pystar.io"


def test_get_path_prefers_zero_padded_channel_and_falls_back_to_plain(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "output"
    cfg = _build_config(raw_dir=raw_dir, output_dir=output_dir)
    loader = ImageLoader(cfg)

    padded_path = _touch_raw_tiff(raw_dir, fov=5, round_id=1, channel_label="02", name="padded")
    _ = _touch_raw_tiff(raw_dir, fov=5, round_id=1, channel_label="2", name="plain")

    assert loader._get_path(5, 1, 2) == padded_path

    padded_path.unlink()
    plain_path = raw_dir / "round001" / "Position5" / "plain_ch2.tif"
    assert loader._get_path(5, 1, 2) == plain_path


def test_get_path_fails_loudly_for_missing_and_ambiguous_raw_inputs(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "output"
    cfg = _build_config(raw_dir=raw_dir, output_dir=output_dir)
    loader = ImageLoader(cfg)

    with pytest.raises(FileNotFoundError, match="Data missing"):
        _ = loader._get_path(5, 1, 2)

    _ = _touch_raw_tiff(raw_dir, fov=5, round_id=1, channel_label="02", name="first")
    _ = _touch_raw_tiff(raw_dir, fov=5, round_id=1, channel_label="02", name="second")

    with pytest.raises(ValueError, match="Ambiguous pattern"):
        _ = loader._get_path(5, 1, 2)


def test_clean_image_path_and_missing_clean_image_error_are_canonical(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "output"
    cfg = _build_config(raw_dir=raw_dir, output_dir=output_dir)
    loader = ImageLoader(cfg)

    clean_path = loader.get_clean_path(7, 3, 2)

    assert clean_path == (
        output_dir
        / "Position7"
        / "output_pystar"
        / "clean_data"
        / "clean_fov_7_round_3_ch_2.tif"
    )
    with pytest.raises(FileNotFoundError, match="Clean image not found"):
        _ = loader.load_clean_image(7, 3, 2)


def test_load_fov_preserves_lazy_xarray_contract_and_zero_pads_missing_channels(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "output"
    cfg = _build_config(
        raw_dir=raw_dir,
        output_dir=output_dir,
        round_structure={2: [1], 1: [2]},
        channel_roles={3: "aux", 1: "anchor", 2: "sequencing"},
    )
    loader = ImageLoader(cfg)
    fov_id = 5
    round_one_channel_two = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    round_two_channel_one = np.full((2, 3, 4), 7, dtype=np.uint8)
    _write_raw_tiff(raw_dir, fov=fov_id, round_id=1, channel_label="02", data=round_one_channel_two)
    _write_raw_tiff(raw_dir, fov=fov_id, round_id=2, channel_label="01", data=round_two_channel_one)

    data = loader.load_fov(fov_id)

    assert data.dims == ("round", "channel", "z", "y", "x")
    assert data.name == "fov_5"
    assert data.sizes == {"round": 2, "channel": 3, "z": 2, "y": 3, "x": 4}
    assert data.coords["round"].values.tolist() == [1, 2]
    assert data.coords["channel"].values.tolist() == [1, 2, 3]
    np.testing.assert_array_equal(data.coords["z"].values, np.asarray([0.0, 500.0]))
    np.testing.assert_array_equal(data.coords["y"].values, np.asarray([0.0, 100.0, 200.0]))
    np.testing.assert_array_equal(data.coords["x"].values, np.asarray([0.0, 100.0, 200.0, 300.0]))
    assert data.attrs["fov_id"] == fov_id
    assert data.attrs["valid_channels_map"] == {2: [1], 1: [2]}
    assert data.attrs["channel_roles"] == {3: "aux", 1: "anchor", 2: "sequencing"}
    assert isinstance(data.data, da.Array)
    assert isinstance(data.sel(round=1, channel=2).data, da.Array)
    assert data.dtype == np.uint8

    np.testing.assert_array_equal(data.sel(round=1, channel=2).values, round_one_channel_two)
    np.testing.assert_array_equal(data.sel(round=2, channel=1).values, round_two_channel_one)
    np.testing.assert_array_equal(
        data.sel(round=1, channel=1).values,
        np.zeros((2, 3, 4), dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        data.sel(round=2, channel=3).values,
        np.zeros((2, 3, 4), dtype=np.uint8),
    )
