from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from pystar.infrastructure import MatlabProviderConfig, MatlabSharedSessionConfig
from pystar.matlab_engine_bootstrap import (
    MATLAB_SHARED_SESSION_NAME_MAX_LENGTH,
    MATLABSessionCapsule,
    MatlabSharedSessionOwner,
    resolve_matlab_shared_session_name,
    should_use_shared_matlab_session,
)


class FakeMatlabEngine:
    _pystar_fake_engine = True

    def __init__(self, *, sentinel: dict[str, Any] | None = None, expose_entrypoint: bool = True) -> None:
        self.added_paths: list[str] = []
        self.eval_calls: list[tuple[str, int]] = []
        self.quit_count = 0
        self.shared_names: list[str] = []
        self._sentinel = None if sentinel is None else dict(sentinel)
        if expose_entrypoint:
            self.run_stage = lambda *args, **kwargs: "ok"

    def addpath(self, path: str, *, nargout: int = 0) -> None:
        assert nargout == 0
        self.added_paths.append(path)

    def eval(self, command: str, *, nargout: int = 0) -> str | None:
        self.eval_calls.append((command, nargout))
        if "shareEngine" in command:
            marker = "shareEngine('"
            start = command.index(marker) + len(marker)
            end = command.index("')", start)
            self.shared_names.append(command[start:end])
        return None

    def version(self, *, nargout: int = 1) -> str:
        assert nargout == 1
        return "R2024a"

    def quit(self) -> None:
        self.quit_count += 1

    def _pystar_get_sentinel(self) -> dict[str, Any] | None:
        return None if self._sentinel is None else dict(self._sentinel)

    def _pystar_set_sentinel(self, sentinel: dict[str, Any]) -> None:
        self._sentinel = dict(sentinel)


class FakeMatlabEngineModule:
    def __init__(self, *, names: tuple[str, ...] = (), start_engine: FakeMatlabEngine | None = None) -> None:
        self.names = tuple(names)
        self.start_engine = start_engine or FakeMatlabEngine()
        self.connected_engines: dict[str, FakeMatlabEngine] = {}
        self.find_calls = 0
        self.connect_calls: list[Any] = []
        self.start_calls = 0

    def find_matlab(self) -> tuple[str, ...]:
        self.find_calls += 1
        return self.names

    def connect_matlab(self, name: str | None = None) -> FakeMatlabEngine:
        self.connect_calls.append(name)
        if name is None:
            raise AssertionError("shared-session MVP must not call unnamed connect_matlab()")
        try:
            return self.connected_engines[name]
        except KeyError as exc:
            raise RuntimeError(f"No fake MATLAB session named {name}") from exc

    def start_matlab(self) -> FakeMatlabEngine:
        self.start_calls += 1
        return self.start_engine


def _module_loader(module: FakeMatlabEngineModule):
    def load(_consumer: str):
        return module, {
            "consumer": "test",
            "configured_environment": {"configured": True},
            "configure_environment_ms": 1.0,
            "engine_module_import_ms": 2.0,
            "factory_resolution_ms": 3.0,
        }

    return load


def _fake_config(
    *,
    enabled: bool = True,
    name: str | None = None,
    lifetime: str = "run",
    uses_matlab: bool = True,
    config_source_path: Path | None = Path("/tmp/experiment_config.yaml"),
    config_sha256: str | None = "sha256:0123456789abcdef",
) -> Any:
    shared_session = MatlabSharedSessionConfig(
        enabled=enabled,
        name=name,
        lifetime=cast(Any, lifetime),
        health_check_timeout_s=30.0,
    )
    return SimpleNamespace(
        config_source_path=config_source_path,
        config_sha256=config_sha256,
        providers=SimpleNamespace(matlab=SimpleNamespace(shared_session=shared_session)),
        pipeline=SimpleNamespace(
            uses_matlab_preprocessing=lambda: uses_matlab,
            uses_matlab_registration=lambda: False,
            uses_matlab_spot_finding=lambda: False,
            uses_matlab_extraction=lambda: False,
        ),
    )


def test_shared_session_config_defaults_are_disabled_and_provider_parses_block() -> None:
    provider = MatlabProviderConfig()

    assert provider.shared_session.enabled is False
    assert provider.shared_session.name is None
    assert provider.shared_session.lifetime == "run"
    assert provider.shared_session.health_check_timeout_s == 30.0

    parsed = MatlabProviderConfig(
        shared_session=MatlabSharedSessionConfig(enabled=True, name="pystar_manual", lifetime="fov")
    )
    assert parsed.shared_session.enabled is True
    assert parsed.shared_session.name == "pystar_manual"
    assert parsed.shared_session.lifetime == "fov"


def test_shared_session_config_rejects_invalid_name_and_timeout() -> None:
    with pytest.raises(ValueError, match="providers.matlab.shared_session.name"):
        _ = MatlabSharedSessionConfig(enabled=True, name="bad-name")

    with pytest.raises(ValueError):
        _ = MatlabSharedSessionConfig(enabled=True, health_check_timeout_s=0)


def test_generated_session_name_uses_slurm_worker_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "7")
    config = _fake_config(config_source_path=Path("/tmp/my config.yaml"))

    identity = resolve_matlab_shared_session_name(config)

    assert identity["name"] == "pystar_my_config_01234567_slurm_12345_7"
    assert identity["name_source"] == "generated"
    assert identity["run_id_source"] == "slurm"


def test_generated_session_name_is_matlab_namelengthmax_safe_and_keeps_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "12345678901234567890")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "98765432109876543210")
    config = _fake_config(
        config_source_path=Path(
            "/tmp/experiment_config_stage12a_matlab_lifecycle_shared_position1_validation.yaml"
        ),
        config_sha256="sha256:48e987a8b29c7635d88fedcba0000000",
    )

    identity = resolve_matlab_shared_session_name(config)

    assert len(identity["name"]) <= MATLAB_SHARED_SESSION_NAME_MAX_LENGTH
    assert identity["name"].startswith("pystar_")
    assert "48e987a8" in identity["name"]
    assert identity["name_source"] == "generated"
    assert identity["run_id_source"] == "slurm"
    assert resolve_matlab_shared_session_name(config)["name"] == identity["name"]

    other_identity = resolve_matlab_shared_session_name(
        _fake_config(
            config_source_path=config.config_source_path,
            config_sha256="sha256:58e987a8b29c7635d88fedcba0000000",
        )
    )
    assert other_identity["name"] != identity["name"]


def test_configured_session_name_rejects_real_matlab_namelengthmax_overflow() -> None:
    too_long = "p" + "a" * MATLAB_SHARED_SESSION_NAME_MAX_LENGTH

    with pytest.raises(ValueError, match="providers.matlab.shared_session.name.*63"):
        _ = MatlabSharedSessionConfig(enabled=True, name=too_long)


def test_generated_session_name_falls_back_to_pid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
    monkeypatch.setattr("pystar.matlab_engine_bootstrap.os.getpid", lambda: 4242)
    config = _fake_config(config_sha256=None)

    identity = resolve_matlab_shared_session_name(config)

    assert identity["name"] == "pystar_experiment_config_nohash_pid_4242"
    assert identity["run_id_source"] == "pid"


def test_should_use_shared_matlab_session_requires_opt_in_and_matlab_provider() -> None:
    assert should_use_shared_matlab_session(_fake_config(enabled=True, uses_matlab=True)) is True
    assert should_use_shared_matlab_session(_fake_config(enabled=False, uses_matlab=True)) is False
    assert should_use_shared_matlab_session(_fake_config(enabled=True, uses_matlab=False)) is False


def test_public_stage_constructors_keep_default_no_owner_api() -> None:
    from pystar.mining import SignalMiner
    from pystar.preprocessing import DataSanitizer
    from pystar.registration import RegistrationEngine
    from pystar.spot_finding import SpotFinder

    for stage_class in (DataSanitizer, RegistrationEngine, SpotFinder, SignalMiner):
        signature = inspect.signature(stage_class)
        parameter = signature.parameters["matlab_session_owner"]
        assert parameter.default is None


def test_batch_runner_passes_one_owner_to_matlab_capable_stages() -> None:
    batch_script = Path(__file__).resolve().parents[2] / "scripts" / "batch_pystar.py"
    source = batch_script.read_text(encoding="utf-8")

    expected_fragments = [
        "matlab_session_owner_cm = nullcontext(None)",
        "if should_use_shared_matlab_session(cfg):",
        "MatlabSharedSessionOwner.from_config(cfg, fov_id=current_fov)",
        "with matlab_session_owner_cm as matlab_session_owner:",
        "DataSanitizer(cfg, matlab_session_owner=matlab_session_owner)",
        "RegistrationEngine(cfg, matlab_session_owner=matlab_session_owner)",
        "SpotFinder(cfg, matlab_session_owner=matlab_session_owner)",
        "SignalMiner(cfg, matlab_session_owner=matlab_session_owner)",
        "Decoder(cfg)",
    ]

    fragment_positions = [source.index(fragment) for fragment in expected_fragments]
    assert fragment_positions == sorted(fragment_positions)


def test_owner_starts_and_shares_named_session_when_exact_name_absent(tmp_path: Path) -> None:
    engine = FakeMatlabEngine()
    module = FakeMatlabEngineModule(names=(), start_engine=engine)
    owner = MatlabSharedSessionOwner.from_config(
        _fake_config(name="pystar_test_start"),
        fov_id=3,
        engine_module_loader=_module_loader(module),
    )

    acquired, record = owner.ensure_engine(
        consumer="test consumer",
        runtime_dir=tmp_path,
        entrypoint="run_stage",
        startup_failure_prefix="startup failed",
        addpath_failure_prefix="addpath failed",
    )

    assert acquired is engine
    assert module.find_calls == 1
    assert module.connect_calls == []
    assert module.start_calls == 1
    assert engine.shared_names == ["pystar_test_start"]
    assert engine.added_paths == [str(tmp_path)]
    assert record["engine_acquire_mode"] == "cold_start"
    assert record["shared_session_mode"] == "started_owned"
    assert record["health_check_status"] == "passed"
    assert engine._pystar_get_sentinel() is not None


def test_owner_connects_to_exact_existing_name_without_unnamed_attach(tmp_path: Path) -> None:
    engine = FakeMatlabEngine()
    module = FakeMatlabEngineModule(names=("pystar_existing",))
    module.connected_engines["pystar_existing"] = engine
    owner = MatlabSharedSessionOwner.from_config(
        _fake_config(name="pystar_existing"),
        fov_id=3,
        engine_module_loader=_module_loader(module),
    )

    acquired, record = owner.ensure_engine(
        consumer="test consumer",
        runtime_dir=tmp_path,
        entrypoint="run_stage",
        startup_failure_prefix="startup failed",
        addpath_failure_prefix="addpath failed",
    )

    assert acquired is engine
    assert module.connect_calls == ["pystar_existing"]
    assert module.start_calls == 0
    assert record["engine_acquire_mode"] == "connect_existing"
    assert record["shared_session_mode"] == "attached_existing"
    assert record["claimed_existing_without_sentinel"] is True


def test_existing_session_sentinel_mismatch_fails_without_quitting_borrowed_engine(tmp_path: Path) -> None:
    engine = FakeMatlabEngine(
        sentinel={
            "sentinel_schema_version": "1.0",
            "session_name": "pystar_existing",
            "pystar_source_root": "/other/source/root",
            "matlab_runtime_root": "/other/runtime/root",
        }
    )
    module = FakeMatlabEngineModule(names=("pystar_existing",))
    module.connected_engines["pystar_existing"] = engine
    owner = MatlabSharedSessionOwner.from_config(
        _fake_config(name="pystar_existing"),
        fov_id=3,
        engine_module_loader=_module_loader(module),
    )

    with pytest.raises(RuntimeError, match="sentinel identity mismatch"):
        _ = owner.ensure_engine(
            consumer="test consumer",
            runtime_dir=tmp_path,
            entrypoint="run_stage",
            startup_failure_prefix="startup failed",
            addpath_failure_prefix="addpath failed",
        )

    assert engine.quit_count == 0
    assert module.start_calls == 0


def test_sentinel_appdata_eval_uses_value_returning_matlab_commands() -> None:
    class EvalOnlyEngine:
        def __init__(self) -> None:
            self.sentinel: dict[str, Any] | None = None

        def eval(self, command: str, *, nargout: int = 0) -> Any:
            if command.startswith("isappdata"):
                assert nargout == 1
                return self.sentinel is not None
            if command.startswith("jsonencode"):
                assert nargout == 1
                import json

                return json.dumps(self.sentinel)
            if command.startswith("setappdata"):
                assert nargout == 0
                self.sentinel = {
                    "sentinel_schema_version": "1.0",
                    "session_name": "pystar_eval",
                    "pystar_source_root": "/source",
                    "matlab_runtime_root": "/runtime",
                }
                return None
            raise AssertionError(f"unexpected MATLAB eval command: {command}")

    from pystar.matlab_engine_bootstrap import _read_pystar_sentinel, _write_pystar_sentinel

    engine = EvalOnlyEngine()

    assert _read_pystar_sentinel(engine) is None
    _write_pystar_sentinel(
        engine,
        {
            "sentinel_schema_version": "1.0",
            "session_name": "pystar_eval",
            "pystar_source_root": "/source",
            "matlab_runtime_root": "/runtime",
        },
    )

    assert _read_pystar_sentinel(engine) == {
        "sentinel_schema_version": "1.0",
        "session_name": "pystar_eval",
        "pystar_source_root": "/source",
        "matlab_runtime_root": "/runtime",
    }


def test_real_engine_missing_fake_sentinel_proxy_falls_back_to_appdata_eval() -> None:
    class MissingMatlabFunctionProxy:
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("Unrecognized function or variable '_pystar_get_sentinel'.")

    class RealEngineLike:
        def __init__(self) -> None:
            self.sentinel = {
                "sentinel_schema_version": "1.0",
                "session_name": "pystar_real",
                "pystar_source_root": "/source",
                "matlab_runtime_root": "/runtime",
            }
            self.proxy_requested = False
            self.set_proxy_requested = False

        def __getattr__(self, name: str) -> Any:
            if name == "_pystar_get_sentinel":
                self.proxy_requested = True
                return MissingMatlabFunctionProxy()
            if name == "_pystar_set_sentinel":
                self.set_proxy_requested = True
                return MissingMatlabFunctionProxy()
            raise AttributeError(name)

        def eval(self, command: str, *, nargout: int = 0) -> Any:
            if command.startswith("isappdata"):
                assert nargout == 1
                return True
            if command.startswith("jsonencode"):
                assert nargout == 1
                import json

                return json.dumps(self.sentinel)
            if command.startswith("setappdata"):
                assert nargout == 0
                self.sentinel = {
                    "sentinel_schema_version": "1.0",
                    "session_name": "pystar_real_written",
                    "pystar_source_root": "/source",
                    "matlab_runtime_root": "/runtime",
                }
                return None
            raise AssertionError(f"unexpected MATLAB eval command: {command}")

    from pystar.matlab_engine_bootstrap import _read_pystar_sentinel, _write_pystar_sentinel

    engine = RealEngineLike()

    assert _read_pystar_sentinel(engine) == engine.sentinel
    assert engine.proxy_requested is False
    _write_pystar_sentinel(
        engine,
        {
            "sentinel_schema_version": "1.0",
            "session_name": "pystar_real_written",
            "pystar_source_root": "/source",
            "matlab_runtime_root": "/runtime",
        },
    )
    assert engine.sentinel["session_name"] == "pystar_real_written"
    assert engine.set_proxy_requested is False


def test_owned_health_check_failure_cleans_up_new_engine(tmp_path: Path) -> None:
    engine = FakeMatlabEngine(expose_entrypoint=False)
    module = FakeMatlabEngineModule(names=(), start_engine=engine)
    owner = MatlabSharedSessionOwner.from_config(
        _fake_config(name="pystar_bad_start"),
        fov_id=3,
        engine_module_loader=_module_loader(module),
    )

    with pytest.raises(RuntimeError, match="entrypoint 'run_stage'"):
        _ = owner.ensure_engine(
            consumer="test consumer",
            runtime_dir=tmp_path,
            entrypoint="run_stage",
            startup_failure_prefix="startup failed",
            addpath_failure_prefix="addpath failed",
        )

    assert engine.quit_count == 1


def test_borrowed_capsule_validates_runtime_files_before_shared_health_check(tmp_path: Path) -> None:
    engine = FakeMatlabEngine(expose_entrypoint=False)
    module = FakeMatlabEngineModule(names=(), start_engine=engine)
    owner = MatlabSharedSessionOwner.from_config(
        _fake_config(name="pystar_runtime_files_first"),
        fov_id=5,
        engine_module_loader=_module_loader(module),
    )

    def missing_runtime_file_validator() -> list[dict[str, Any]]:
        raise FileNotFoundError("Required MATLAB runtime file is missing: fake_entrypoint.m")

    capsule = MATLABSessionCapsule(
        consumer="stage provider='matlab'",
        runtime_dir=tmp_path,
        entrypoint="run_stage",
        startup_failure_prefix="startup failed",
        addpath_failure_prefix="addpath failed",
        session_owner=owner,
        runtime_file_validator=missing_runtime_file_validator,
    )

    with pytest.raises(FileNotFoundError, match="fake_entrypoint.m"):
        capsule.ensure_engine()

    assert module.start_calls == 0
    assert engine.quit_count == 0


def test_borrowed_capsule_close_does_not_quit_shared_engine(tmp_path: Path) -> None:
    engine = FakeMatlabEngine()
    module = FakeMatlabEngineModule(names=(), start_engine=engine)
    owner = MatlabSharedSessionOwner.from_config(
        _fake_config(name="pystar_capsule"),
        fov_id=5,
        engine_module_loader=_module_loader(module),
    )
    capsule = MATLABSessionCapsule(
        consumer="stage provider='matlab'",
        runtime_dir=tmp_path,
        entrypoint="run_stage",
        startup_failure_prefix="startup failed",
        addpath_failure_prefix="addpath failed",
        session_owner=owner,
    )

    assert capsule.ensure_engine() is engine
    assert capsule.resolve_callable("run_stage")() == "ok"
    capsule.close()

    assert engine.quit_count == 0
    assert owner.engine is engine
    snapshot = capsule.summarize_session_lifecycle()
    assert snapshot is not None
    session = snapshot["sessions"][0]
    assert session["shared_session"]["shared_session_name"] == "pystar_capsule"
    assert session["shared_session"]["shared_session_mode"] == "started_owned"

    owner.close()
    assert engine.quit_count == 1
