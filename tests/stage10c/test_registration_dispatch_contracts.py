from __future__ import annotations

from types import MethodType, SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from pystar.registration import RegistrationEngine, _RoundRegistrationStrategy


FloatArray = NDArray[np.float32]


def _make_engine() -> RegistrationEngine:
    guards = SimpleNamespace(
        skip_if_global_corr_below=0.37,
        reject_if_correlation_worse=True,
    )
    registration = SimpleNamespace(
        global_provider="native",
        local_provider="native",
        local_method="demons_3d",
        enable_local=True,
        downsample_factor=1,
        global_max_shift=200,
        guards=guards,
        reference_round=1,
        demons_3d=SimpleNamespace(use_tiling=False),
        optical_flow=SimpleNamespace(),
        bspline=SimpleNamespace(),
    )
    pipeline = SimpleNamespace(registration=registration)
    cfg = cast(Any, SimpleNamespace(pipeline=pipeline))
    return RegistrationEngine(cfg)


def _scope_descriptor() -> dict[str, object]:
    return {
        "coverage_mode": "full_fov",
        "region_origin_zyx": [0, 0, 0],
        "region_shape_zyx": [2, 4, 4],
        "full_volume_shape_zyx": [2, 4, 4],
    }


def _small_volume() -> FloatArray:
    return np.ones((2, 4, 4), dtype=np.float32)


def _small_mip() -> FloatArray:
    return np.ones((4, 4), dtype=np.float32)


def test_round_strategy_selects_supported_global_and_local_combinations_explicitly() -> None:
    engine = _make_engine()

    native_demons = engine._build_round_registration_strategy(
        global_provider="native",
        local_provider="native",
        local_method="demons_3d",
    )
    native_flow = engine._build_round_registration_strategy(
        global_provider="native",
        local_provider="native",
        local_method="optical_flow",
    )
    native_bspline = engine._build_round_registration_strategy(
        global_provider="native",
        local_provider="native",
        local_method="bspline",
    )
    matlab_demons = engine._build_round_registration_strategy(
        global_provider="matlab",
        local_provider="matlab",
        local_method="demons_3d",
    )

    assert native_demons.global_handler_name == "_run_global_registration_native"
    assert native_demons.local_handler_name == "_run_local_native_demons_3d"
    assert native_demons.local_default_mode == "native_local_registration"

    assert native_flow.local_handler_name == "_run_local_native_optical_flow"
    assert native_bspline.local_handler_name == "_run_local_native_bspline"

    assert matlab_demons.global_handler_name == "_run_global_registration_matlab"
    assert matlab_demons.local_handler_name == "_run_local_matlab_demons_3d"
    assert matlab_demons.local_default_mode == "experimental_local_kernel_swap"


def test_round_strategy_rejects_unsupported_local_provider_method_combinations_loudly() -> None:
    engine = _make_engine()

    with pytest.raises(ValueError, match="Unsupported registration.local.method for provider 'native': 'bad_method'"):
        _ = engine._build_round_registration_strategy(
            local_provider="native",
            local_method="bad_method",
        )

    with pytest.raises(
        ValueError,
        match=r"registration\.local\.provider='matlab' currently supports only local_method='demons_3d'",
    ):
        _ = engine._build_round_registration_strategy(
            local_provider="matlab",
            local_method="optical_flow",
        )

    with pytest.raises(ValueError, match="Unsupported registration.local.provider: 'custom'"):
        _ = engine._build_round_registration_strategy(
            local_provider="custom",
            local_method="demons_3d",
        )


def test_round_strategy_rejects_unsupported_global_provider_loudly() -> None:
    engine = _make_engine()

    with pytest.raises(ValueError, match="Unsupported registration.global.provider: 'custom'"):
        _ = engine._build_round_registration_strategy(global_provider="custom")


def test_register_round_delegates_to_canonical_orchestrated_path_with_explicit_strategy() -> None:
    engine = _make_engine()
    captured: dict[str, Any] = {}

    def fake_orchestrated(self: RegistrationEngine, **kwargs: Any) -> tuple[dict[str, Any], FloatArray, dict[str, Any]]:
        captured.update(kwargs)
        return ({"ok": True}, np.zeros((4, 4), dtype=np.float32), {"delegated": True})

    engine._register_round_orchestrated = MethodType(fake_orchestrated, engine)  # type: ignore[method-assign]

    round_transform, final_img_qc, backend_metadata = engine._register_round(
        fov_id=7,
        round_id=2,
        ref_round=1,
        ref_scope_3d=_small_volume(),
        ref_mip_clean=_small_mip(),
        mov_scope_3d=_small_volume(),
        scope_descriptor=cast(dict[str, Any], _scope_descriptor()),
    )

    strategy = cast(_RoundRegistrationStrategy, captured["strategy"])
    assert captured["fov_id"] == 7
    assert captured["round_id"] == 2
    assert strategy.global_provider == "native"
    assert strategy.local_provider == "native"
    assert strategy.local_method == "demons_3d"
    assert strategy.local_skip_if_global_corr_below == pytest.approx(0.37)
    assert round_transform == {"ok": True}
    np.testing.assert_array_equal(final_img_qc, np.zeros((4, 4), dtype=np.float32))
    assert backend_metadata == {"delegated": True}


def test_legacy_native_helper_delegates_to_canonical_round_authority() -> None:
    engine = _make_engine()
    captured: dict[str, Any] = {}

    def fake_orchestrated(self: RegistrationEngine, **kwargs: Any) -> tuple[dict[str, Any], FloatArray, dict[str, Any]]:
        captured.update(kwargs)
        return ({"legacy": "native"}, np.ones((4, 4), dtype=np.float32), {"path": "canonical"})

    engine._register_round_orchestrated = MethodType(fake_orchestrated, engine)  # type: ignore[method-assign]

    result = engine._register_round_native(
        ref_scope_3d=_small_volume(),
        ref_mip_clean=_small_mip(),
        mov_scope_3d=_small_volume(),
    )

    strategy = cast(_RoundRegistrationStrategy, captured["strategy"])
    assert strategy.global_provider == "native"
    assert strategy.local_provider == "native"
    assert captured["scope_descriptor"]["coverage_mode"] == "full_fov"
    assert result[0] == {"legacy": "native"}
    assert result[2] == {"path": "canonical"}


def test_legacy_matlab_helper_delegates_to_canonical_round_authority() -> None:
    engine = _make_engine()
    captured: dict[str, Any] = {}

    def fake_orchestrated(self: RegistrationEngine, **kwargs: Any) -> tuple[dict[str, Any], FloatArray, dict[str, Any]]:
        captured.update(kwargs)
        return ({"legacy": "matlab"}, np.full((4, 4), 2.0, dtype=np.float32), {"path": "canonical"})

    engine._register_round_orchestrated = MethodType(fake_orchestrated, engine)  # type: ignore[method-assign]

    result = engine._register_round_matlab_extracted(
        fov_id=11,
        round_id=3,
        ref_round=1,
        ref_scope_3d=_small_volume(),
        ref_mip_clean=_small_mip(),
        mov_scope_3d=_small_volume(),
        scope_descriptor=cast(dict[str, Any], _scope_descriptor()),
    )

    strategy = cast(_RoundRegistrationStrategy, captured["strategy"])
    assert strategy.global_provider == "matlab"
    assert strategy.local_provider == "matlab"
    assert captured["fov_id"] == 11
    assert captured["round_id"] == 3
    assert result[0] == {"legacy": "matlab"}
    assert result[2] == {"path": "canonical"}


def test_skip_low_global_correlation_uses_configured_threshold_not_hard_coded_value() -> None:
    engine = _make_engine()
    captured: dict[str, Any] = {}

    def fake_global(self: RegistrationEngine, **kwargs: Any) -> tuple[FloatArray, float, None]:
        return np.zeros(3, dtype=np.float32), 0.95, None

    def fake_post_global(self: RegistrationEngine, **kwargs: Any) -> tuple[FloatArray, FloatArray, FloatArray, float]:
        return (
            np.ones((4, 4), dtype=np.float32),
            np.zeros(2, dtype=np.float32),
            np.ones((4, 4), dtype=np.float32),
            0.30,
        )

    def fake_attach(
        backend_metadata: Any,
        *,
        provider: str,
        local_method: str,
        status: str,
        corr_after_global: float,
        reject_if_worse: bool | None = None,
        default_mode: str | None = None,
        corr_after_local: float | None = None,
    ) -> dict[str, Any]:
        captured.update(
            {
                "provider": provider,
                "local_method": local_method,
                "status": status,
                "corr_after_global": corr_after_global,
                "default_mode": default_mode,
            }
        )
        return {"captured": True}

    engine._run_global_registration = MethodType(fake_global, engine)  # type: ignore[method-assign]
    engine._compute_post_global_registration_state = MethodType(fake_post_global, engine)  # type: ignore[method-assign]

    from pystar import registration as registration_module

    original_attach = registration_module._attach_local_flow_metadata
    registration_module._attach_local_flow_metadata = fake_attach
    try:
        round_transform, final_img_qc, backend_metadata = engine._register_round_orchestrated(
            fov_id=1,
            round_id=2,
            ref_round=1,
            ref_scope_3d=_small_volume(),
            ref_mip_clean=_small_mip(),
            mov_scope_3d=_small_volume(),
            scope_descriptor=cast(dict[str, Any], _scope_descriptor()),
            strategy=engine._build_round_registration_strategy(
                local_skip_if_global_corr_below=0.31,
            ),
        )
    finally:
        registration_module._attach_local_flow_metadata = original_attach

    assert captured["status"] == "skipped_low_global_corr"
    assert captured["corr_after_global"] == pytest.approx(0.30)
    assert backend_metadata == {"captured": True}
    assert round_transform["flow_2d"] is None
    assert round_transform["flow_3d"] is None
    np.testing.assert_array_equal(final_img_qc, np.ones((4, 4), dtype=np.float32))


def test_orchestrated_round_uses_explicit_strategy_handler_names_without_runtime_lookup() -> None:
    engine = _make_engine()

    def fail_global_lookup(self: RegistrationEngine, provider: str) -> str:
        raise AssertionError(f"unexpected global provider lookup for {provider!r}")

    def fail_local_lookup(self: RegistrationEngine, *, local_provider: str, local_method: str) -> str:
        raise AssertionError(
            f"unexpected local provider lookup for {(local_provider, local_method)!r}"
        )

    def fake_global_native(self: RegistrationEngine, **kwargs: Any) -> tuple[FloatArray, float, dict[str, Any]]:
        return np.zeros(3, dtype=np.float32), 0.95, {"backend": "synthetic-global"}

    def fake_post_global(self: RegistrationEngine, **kwargs: Any) -> tuple[FloatArray, FloatArray, FloatArray, float]:
        return (
            np.ones((4, 4), dtype=np.float32),
            np.zeros(2, dtype=np.float32),
            np.full((4, 4), 2.0, dtype=np.float32),
            0.91,
        )

    def fake_local_native(self: RegistrationEngine, context: Any) -> Any:
        return SimpleNamespace(
            flow_2d=None,
            flow_3d=np.zeros((3, 2, 4, 4), dtype=np.float32),
            final_corr=0.93,
            final_img_qc=np.full((4, 4), 3.0, dtype=np.float32),
            backend_metadata={"backend": "synthetic-local"},
        )

    engine._resolve_global_registration_handler_name = MethodType(fail_global_lookup, engine)  # type: ignore[method-assign]
    engine._resolve_local_registration_handler_name = MethodType(fail_local_lookup, engine)  # type: ignore[method-assign]
    engine._run_global_registration_native = MethodType(fake_global_native, engine)  # type: ignore[method-assign]
    engine._compute_post_global_registration_state = MethodType(fake_post_global, engine)  # type: ignore[method-assign]
    engine._run_local_native_demons_3d = MethodType(fake_local_native, engine)  # type: ignore[method-assign]

    strategy = _RoundRegistrationStrategy(
        global_provider="native",
        local_provider="native",
        local_method="demons_3d",
        enable_local=True,
        local_skip_if_global_corr_below=0.20,
        global_handler_name="_run_global_registration_native",
        local_handler_name="_run_local_native_demons_3d",
        local_default_mode="native_local_registration",
    )

    round_transform, final_img_qc, backend_metadata = engine._register_round_orchestrated(
        fov_id=1,
        round_id=2,
        ref_round=1,
        ref_scope_3d=_small_volume(),
        ref_mip_clean=_small_mip(),
        mov_scope_3d=_small_volume(),
        scope_descriptor=cast(dict[str, Any], _scope_descriptor()),
        strategy=strategy,
    )

    assert round_transform["global_corr"] == pytest.approx(0.95)
    assert round_transform["final_corr"] == pytest.approx(0.93)
    assert round_transform["flow_2d"] is None
    assert cast(FloatArray, round_transform["flow_3d"]).shape == (3, 2, 4, 4)
    np.testing.assert_array_equal(final_img_qc, np.full((4, 4), 3.0, dtype=np.float32))
    assert backend_metadata == {"backend": "synthetic-local"}


def test_stubbed_successful_round_preserves_transform_contract_keys() -> None:
    engine = _make_engine()

    def fake_global(self: RegistrationEngine, **kwargs: Any) -> tuple[FloatArray, float, dict[str, Any]]:
        return np.asarray([1.0, 2.0, 3.0], dtype=np.float32), 0.91, {"backend": "synthetic"}

    def fake_post_global(self: RegistrationEngine, **kwargs: Any) -> tuple[FloatArray, FloatArray, FloatArray, float]:
        return (
            np.ones((4, 4), dtype=np.float32),
            np.asarray([2.0, 3.0], dtype=np.float32),
            np.full((4, 4), 5.0, dtype=np.float32),
            0.88,
        )

    def fake_dispatch(self: RegistrationEngine, context: Any) -> Any:
        return SimpleNamespace(
            flow_2d=None,
            flow_3d=np.zeros((3, 2, 4, 4), dtype=np.float32),
            final_corr=0.92,
            final_img_qc=np.full((4, 4), 9.0, dtype=np.float32),
            backend_metadata={"backend": "synthetic-local"},
        )

    engine._run_global_registration = MethodType(fake_global, engine)  # type: ignore[method-assign]
    engine._compute_post_global_registration_state = MethodType(fake_post_global, engine)  # type: ignore[method-assign]
    engine._dispatch_local_registration = MethodType(fake_dispatch, engine)  # type: ignore[method-assign]

    round_transform, final_img_qc, backend_metadata = engine._register_round_orchestrated(
        fov_id=1,
        round_id=2,
        ref_round=1,
        ref_scope_3d=_small_volume(),
        ref_mip_clean=_small_mip(),
        mov_scope_3d=_small_volume(),
        scope_descriptor=cast(dict[str, Any], _scope_descriptor()),
    )

    assert set(round_transform.keys()) == {
        "global_shift_3d",
        "global_corr",
        "flow_2d",
        "flow_3d",
        "final_corr",
        "is_reference_round",
    }
    assert round_transform["global_corr"] == pytest.approx(0.91)
    assert round_transform["final_corr"] == pytest.approx(0.92)
    assert round_transform["is_reference_round"] is False
    assert round_transform["flow_2d"] is None
    assert cast(FloatArray, round_transform["flow_3d"]).shape == (3, 2, 4, 4)
    np.testing.assert_array_equal(cast(FloatArray, round_transform["global_shift_3d"]), np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
    np.testing.assert_array_equal(final_img_qc, np.full((4, 4), 9.0, dtype=np.float32))
    assert backend_metadata == {"backend": "synthetic-local"}
