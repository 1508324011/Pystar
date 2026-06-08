from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from pystar.spot_finding import (
    SpotFinder,
    _build_spot_finding_route,
    _resolve_spot_finding_route,
)


FloatArray = NDArray[np.float32]
ImageArray = NDArray[np.generic]


def _route_config(provider: object, algorithm: object) -> Any:
    return SimpleNamespace(
        pipeline=SimpleNamespace(
            spot_finding=SimpleNamespace(provider=provider, algorithm=algorithm)
        )
    )


@pytest.mark.parametrize(
    ("provider", "algorithm", "handler_name", "final_algorithm", "is_matlab"),
    [
        ("native", "spotiflow", "_run_spotiflow", "spotiflow", False),
        ("native", "blob_dog", "_run_blob_dog", "blob_dog", False),
        ("native", "peak_local_max", "_run_peak_local_max", "peak_local_max", False),
        ("matlab", "peak_local_max", "_run_matlab_spot_finding", "matlab_peak_local_max", True),
    ],
)
def test_resolve_spot_finding_route_accepts_supported_provider_algorithm_pairs(
    provider: str,
    algorithm: str,
    handler_name: str,
    final_algorithm: str,
    is_matlab: bool,
) -> None:
    route = _resolve_spot_finding_route(_route_config(provider, algorithm))

    assert route.provider == provider
    assert route.algorithm == algorithm
    assert route.handler_name == handler_name
    assert route.final_algorithm == final_algorithm
    assert route.is_matlab is is_matlab
    assert route.is_native is (not is_matlab)


def test_build_spot_finding_route_rejects_unknown_provider_before_dispatch() -> None:
    with pytest.raises(ValueError, match="Unsupported spot_finding provider.*native.*matlab"):
        _ = _build_spot_finding_route(provider_value="custom", algorithm_value="peak_local_max")


def test_build_spot_finding_route_rejects_unknown_algorithm_before_dispatch() -> None:
    with pytest.raises(ValueError, match="Unsupported spot_finding algorithm.*spotiflow.*blob_dog.*peak_local_max"):
        _ = _build_spot_finding_route(provider_value="native", algorithm_value="max3d")


@pytest.mark.parametrize(
    ("provider_value", "algorithm_value", "expected_message"),
    [
        (None, "peak_local_max", "spot_finding provider must be a string"),
        ("native", None, "spot_finding algorithm must be a string"),
    ],
)
def test_build_spot_finding_route_rejects_non_string_values_before_dispatch(
    provider_value: object,
    algorithm_value: object,
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        _ = _build_spot_finding_route(provider_value=provider_value, algorithm_value=algorithm_value)


def test_build_spot_finding_route_rejects_unsupported_matlab_algorithm_pair() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported spot_finding route.*provider='matlab'.*algorithm='blob_dog'.*matlab.*peak_local_max",
    ):
        _ = _build_spot_finding_route(provider_value="matlab", algorithm_value="blob_dog")


def test_spot_finder_route_dispatch_uses_handler_name_for_native_and_matlab_paths() -> None:
    finder = SpotFinder.__new__(SpotFinder)
    calls: list[dict[str, object]] = []

    def fake_native(vol_3d: ImageArray) -> pd.DataFrame:
        calls.append({"handler": "native", "shape": tuple(vol_3d.shape)})
        return pd.DataFrame({"z": [0.0], "y": [1.0], "x": [2.0], "intensity": [3.0]})

    def fake_matlab(
        vol_3d: ImageArray,
        *,
        fov_id: int,
        round_id: int,
        channel_id: int,
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        calls.append(
            {
                "handler": "matlab",
                "shape": tuple(vol_3d.shape),
                "fov_id": fov_id,
                "round_id": round_id,
                "channel_id": channel_id,
            }
        )
        spots = pd.DataFrame({"z": [0.0], "y": [1.0], "x": [2.0], "intensity": [3.0]})
        return spots, {"provider": "matlab"}

    finder._run_peak_local_max = fake_native  # type: ignore[method-assign]
    finder._run_matlab_spot_finding = fake_matlab  # type: ignore[method-assign]
    volume = np.ones((2, 3, 4), dtype=np.float32)

    native_route = _build_spot_finding_route(provider_value="native", algorithm_value="peak_local_max")
    native_df, native_metadata = finder._run_spot_finding_route(
        native_route,
        volume,
        fov_id=7,
        round_id=1,
        channel_id=0,
    )
    matlab_route = _build_spot_finding_route(provider_value="matlab", algorithm_value="peak_local_max")
    matlab_df, matlab_metadata = finder._run_spot_finding_route(
        matlab_route,
        volume,
        fov_id=7,
        round_id=1,
        channel_id=0,
    )

    assert list(native_df.columns) == ["z", "y", "x", "intensity"]
    assert native_metadata is None
    assert list(matlab_df.columns) == ["z", "y", "x", "intensity"]
    assert matlab_metadata == {"provider": "matlab"}
    assert calls == [
        {"handler": "native", "shape": (2, 3, 4)},
        {"handler": "matlab", "shape": (2, 3, 4), "fov_id": 7, "round_id": 1, "channel_id": 0},
    ]


def test_spot_finder_route_dispatch_rejects_missing_handler_without_fallback() -> None:
    finder = SpotFinder.__new__(SpotFinder)
    route = replace(
        _build_spot_finding_route(provider_value="native", algorithm_value="peak_local_max"),
        handler_name="_missing_peak_local_max_handler",
    )
    volume = np.ones((2, 3, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="missing handler.*_missing_peak_local_max_handler.*provider='native'.*algorithm='peak_local_max'"):
        _ = finder._run_spot_finding_route(
            route,
            volume,
            fov_id=7,
            round_id=1,
            channel_id=0,
        )
