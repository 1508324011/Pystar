from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pystar.matlab_registration import MATLABRegistrationBackend


FLOW_VARIABLE = "displacement_field_yxz"


def _backend() -> MATLABRegistrationBackend:
    return MATLABRegistrationBackend.__new__(MATLABRegistrationBackend)


def _metadata(raw_flow: Any) -> dict[str, object]:
    return {
        "flow_variable": FLOW_VARIABLE,
        "flow_shape_yxz_component": [int(value) for value in raw_flow.shape],
    }


def _expected_flow_zyx(raw_flow: Any) -> Any:
    dx_zyx = np.transpose(np.asarray(raw_flow[..., 0], dtype=np.float32), (2, 0, 1))
    dy_zyx = np.transpose(np.asarray(raw_flow[..., 1], dtype=np.float32), (2, 0, 1))
    dz_zyx = np.transpose(np.asarray(raw_flow[..., 2], dtype=np.float32), (2, 0, 1))
    return np.stack([dz_zyx, dy_zyx, dx_zyx], axis=0).astype(np.float32, copy=False)


def test_load_local_flow_zyx_uses_variable_names_and_preserves_valid_conversion(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _backend()
    flow_output_path = Path("/tmp/local-flow.mat")
    raw_flow = np.arange(2 * 3 * 4 * 3, dtype=np.float64).reshape(2, 3, 4, 3)
    recorded: dict[str, object] = {}

    def fake_loadmat(path: Path, *, variable_names: list[str]) -> dict[str, object]:
        recorded["path"] = path
        recorded["variable_names"] = variable_names
        return {FLOW_VARIABLE: raw_flow}

    monkeypatch.setattr("pystar.matlab_registration.loadmat", fake_loadmat)

    result = backend._load_local_flow_zyx(flow_output_path, _metadata(raw_flow), round_id=7)

    assert recorded["path"] == flow_output_path
    assert recorded["variable_names"] == [FLOW_VARIABLE]
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, _expected_flow_zyx(raw_flow))


def test_load_local_flow_zyx_still_fails_loudly_when_requested_variable_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _backend()
    flow_output_path = Path("/tmp/local-flow.mat")
    raw_flow = np.zeros((2, 3, 4, 3), dtype=np.float32)
    recorded: dict[str, object] = {}

    def fake_loadmat(path: Path, *, variable_names: list[str]) -> dict[str, object]:
        recorded["path"] = path
        recorded["variable_names"] = variable_names
        return {"__header__": b"header", "__version__": "1.0", "__globals__": []}

    monkeypatch.setattr("pystar.matlab_registration.loadmat", fake_loadmat)

    with pytest.raises(
        ValueError,
        match=r"MATLAB local-registration output is missing variable 'displacement_field_yxz': /tmp/local-flow\.mat",
    ):
        _ = backend._load_local_flow_zyx(flow_output_path, _metadata(raw_flow), round_id=7)

    assert recorded["path"] == flow_output_path
    assert recorded["variable_names"] == [FLOW_VARIABLE]


@pytest.mark.parametrize(
    ("raw_flow", "expected_message"),
    [
        (
            np.zeros((2, 3, 4), dtype=np.float32),
            r"MATLAB local-registration raw flow for round 7 must be 4D \[Y, X, Z, 3\], got \(2, 3, 4\)",
        ),
        (
            np.zeros((2, 3, 4, 2), dtype=np.float32),
            r"MATLAB local-registration raw flow for round 7 must have 3 components, got 2",
        ),
        (
            np.zeros((2, 3, 5, 3), dtype=np.float32),
            r"MATLAB local-registration raw flow shape mismatch for round 7: metadata=\[2, 3, 4, 3\], actual=\[2, 3, 5, 3\]",
        ),
    ],
)
def test_load_local_flow_zyx_rejects_bad_shapes(
    monkeypatch: pytest.MonkeyPatch,
    raw_flow: Any,
    expected_message: str,
) -> None:
    backend = _backend()
    flow_output_path = Path("/tmp/local-flow.mat")

    def fake_loadmat(path: Path, *, variable_names: list[str]) -> dict[str, object]:
        assert path == flow_output_path
        assert variable_names == [FLOW_VARIABLE]
        return {FLOW_VARIABLE: raw_flow}

    monkeypatch.setattr("pystar.matlab_registration.loadmat", fake_loadmat)

    metadata = {
        "flow_variable": FLOW_VARIABLE,
        "flow_shape_yxz_component": [2, 3, 4, 3],
    }

    with pytest.raises(ValueError, match=expected_message):
        _ = backend._load_local_flow_zyx(flow_output_path, metadata, round_id=7)
