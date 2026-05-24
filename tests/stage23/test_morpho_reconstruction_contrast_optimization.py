from __future__ import annotations

import numpy as np
import pytest

from pystar import preprocessing as preprocessing_module
from pystar.preprocessing import op_morpho_reconstruction_contrast


def test_morpho_reconstruction_contrast_preserves_contract_for_2d() -> None:
    rng = np.random.default_rng(23)
    image = rng.random((48, 64), dtype=np.float32)
    image_before = image.copy()

    optimized = op_morpho_reconstruction_contrast(
        image,
        {"radius": 4, "downsample_factor": 0.5},
        {},
    )

    assert optimized.dtype == np.float32
    assert optimized.shape == image.shape
    assert np.isfinite(optimized).all()
    assert float(optimized.min()) >= 0.0
    assert float(optimized.max()) <= 1.0
    np.testing.assert_array_equal(image, image_before)


def test_morpho_reconstruction_contrast_3d_matches_per_slice_2d_application() -> None:
    rng = np.random.default_rng(24)
    image = rng.random((3, 40, 52), dtype=np.float32)
    params = {"radius": 3, "downsample_factor": 0.5}
    image_before = image.copy()

    optimized_stack = op_morpho_reconstruction_contrast(image, params, {})
    per_slice = np.stack(
        [
            op_morpho_reconstruction_contrast(slice_2d, params, {})
            for slice_2d in image
        ]
    )

    assert optimized_stack.dtype == np.float32
    assert optimized_stack.shape == image.shape
    np.testing.assert_array_equal(optimized_stack, per_slice)
    np.testing.assert_array_equal(image, image_before)


def test_morphology_disk_cache_reuses_same_radius_argument() -> None:
    preprocessing_module._morphology_disk.cache_clear()

    disk_radius_3_first = preprocessing_module._morphology_disk(3)
    disk_radius_4 = preprocessing_module._morphology_disk(4)
    disk_radius_3_second = preprocessing_module._morphology_disk(3)
    cache_info = preprocessing_module._morphology_disk.cache_info()

    assert disk_radius_3_first is disk_radius_3_second
    assert disk_radius_3_first is not disk_radius_4
    assert disk_radius_3_first.shape != disk_radius_4.shape
    assert cache_info.hits == 1
    assert cache_info.misses == 2


def test_morpho_reconstruction_contrast_fails_loudly_for_invalid_ndim() -> None:
    image = np.ones((2, 3, 4, 5), dtype=np.float32)

    with pytest.raises(ValueError, match="expects a 2D image or 3D stack"):
        op_morpho_reconstruction_contrast(image, {"radius": 3, "downsample_factor": 0.5}, {})


def test_morpho_reconstruction_contrast_fails_loudly_for_empty_stack() -> None:
    image = np.ones((0, 16, 16), dtype=np.float32)

    with pytest.raises(ValueError, match="expects a non-empty 3D stack"):
        op_morpho_reconstruction_contrast(image, {"radius": 3, "downsample_factor": 0.5}, {})


def test_morpho_reconstruction_contrast_fails_loudly_for_invalid_downsample() -> None:
    image = np.ones((16, 16), dtype=np.float32)

    with pytest.raises(ValueError, match="expects downsample_factor > 0"):
        op_morpho_reconstruction_contrast(image, {"radius": 3, "downsample_factor": 0.0}, {})


def test_morpho_reconstruction_contrast_handles_tiny_images_without_zero_sized_resizes() -> None:
    image = np.array([[0.0, 0.25], [0.5, 1.0]], dtype=np.float32)
    image_before = image.copy()

    optimized = op_morpho_reconstruction_contrast(image, {"radius": 3, "downsample_factor": 0.25}, {})

    assert optimized.dtype == np.float32
    assert optimized.shape == image.shape
    assert np.isfinite(optimized).all()
    assert float(optimized.min()) >= 0.0
    assert float(optimized.max()) <= 1.0
    np.testing.assert_array_equal(image, image_before)
