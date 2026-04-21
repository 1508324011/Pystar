from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, Mapping, Optional

import numpy as np
import pandas as pd
import tifffile

from .infrastructure import ExperimentConfig
from .matlab_engine_bootstrap import close_matlab_engine_best_effort, load_matlab_engine_module


MATLAB_EXTRACTION_RUNTIME_MANIFEST_NAME = "runtime_manifest.json"
MATLAB_EXTRACTION_PACKAGE_NAME = "pystar_extraction"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _trusted_matlab_runtime_root() -> Path:
    return (_repo_root() / "matlab_runtime").resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _format_exception_message(prefix: str, exc: Exception) -> str:
    detail = str(exc).strip()
    if detail:
        return f"{prefix}: {detail}"
    return f"{prefix} ({exc.__class__.__name__})"


def resolve_matlab_extraction_runtime_path(config: ExperimentConfig) -> Path:
    runtime_path = config.providers.matlab.extraction.runtime_path
    if not runtime_path.is_absolute():
        runtime_path = _repo_root() / runtime_path
    resolved_runtime_path = runtime_path.resolve()
    trusted_root = _trusted_matlab_runtime_root()
    try:
        resolved_runtime_path.relative_to(trusted_root)
    except ValueError as exc:
        raise ValueError(
            "providers.matlab.extraction.runtime_path must resolve inside the repo-local "
            f"'{trusted_root}' runtime root; got {resolved_runtime_path}"
        ) from exc
    return resolved_runtime_path


def load_matlab_extraction_runtime_manifest(runtime_dir: Path) -> Dict[str, Any]:
    manifest_path = runtime_dir / MATLAB_EXTRACTION_RUNTIME_MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"MATLAB extraction runtime manifest is missing: {manifest_path}. "
            "Expected repo-local manifest for MATLAB extraction provider."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"MATLAB extraction runtime manifest must be a JSON object: {manifest_path}")

    required_files = manifest.get("required_files")
    optional_files = manifest.get("optional_files", [])
    entrypoint = manifest.get("entrypoint")
    package_name = manifest.get("package_name")
    if not isinstance(required_files, list) or not required_files:
        raise ValueError("MATLAB extraction runtime manifest must declare a non-empty required_files list")
    if not isinstance(optional_files, list):
        raise ValueError("MATLAB extraction runtime manifest optional_files must be a list")
    if not isinstance(entrypoint, str) or not entrypoint.strip():
        raise ValueError("MATLAB extraction runtime manifest must declare a non-empty entrypoint")
    if package_name != MATLAB_EXTRACTION_PACKAGE_NAME:
        raise ValueError(
            "MATLAB extraction runtime manifest package_name mismatch: "
            f"expected {MATLAB_EXTRACTION_PACKAGE_NAME!r}, got {package_name!r}"
        )

    for bucket_name, bucket in (("required_files", required_files), ("optional_files", optional_files)):
        for item in bucket:
            if not isinstance(item, dict):
                raise ValueError(f"MATLAB extraction runtime manifest {bucket_name} entries must be JSON objects")
            for key in ("name", "source_path", "role"):
                value = item.get(key)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"MATLAB extraction runtime manifest entry in {bucket_name} is missing non-empty '{key}'"
                    )

    return manifest


def _validate_runtime_entrypoint_contract(runtime_manifest: Mapping[str, Any], configured_entrypoint: str) -> None:
    manifest_entrypoint = runtime_manifest.get("entrypoint")
    if configured_entrypoint != manifest_entrypoint:
        raise ValueError(
            "providers.matlab.extraction.entrypoint must match the repo-local MATLAB runtime manifest. "
            f"Config entrypoint={configured_entrypoint!r}, manifest entrypoint={manifest_entrypoint!r}"
        )

    declared_filenames = {
        item["name"]
        for bucket_name in ("required_files", "optional_files")
        for item in runtime_manifest.get(bucket_name, [])
        if isinstance(item, Mapping) and isinstance(item.get("name"), str)
    }
    expected_entrypoint_file = f"{configured_entrypoint}.m"
    if expected_entrypoint_file not in declared_filenames:
        raise ValueError(
            "MATLAB extraction runtime manifest must declare the configured entrypoint file. "
            f"Missing {expected_entrypoint_file!r} in runtime manifest"
        )


def build_matlab_extraction_plan(
    config: ExperimentConfig,
    *,
    fov_id: int,
    round_id: int,
    channel_id: int,
    n_spots: int,
    volume_shape_zyx: tuple[int, int, int],
    box_size: tuple[int, int, int],
    transform_application_mode: str,
) -> Dict[str, Any]:
    matlab_cfg = config.providers.matlab.extraction
    return {
        "fov_id": int(fov_id),
        "round_id": int(round_id),
        "channel_id": int(channel_id),
        "method": config.pipeline.extraction.method,
        "transform_application_mode": transform_application_mode,
        "n_spots": int(n_spots),
        "box_size_zyx": [int(value) for value in box_size],
        "volume_shape_zyx": [int(value) for value in volume_shape_zyx],
        "volume_transfer_mode": matlab_cfg.volume_transfer_mode,
        "coords_transfer_mode": matlab_cfg.coords_transfer_mode,
        "input_volume_dtype": matlab_cfg.input_volume_dtype,
    }


def _load_matlab_engine_factory() -> Callable[[], Any]:
    matlab_engine = load_matlab_engine_module(
        consumer="extraction.provider='matlab'",
    )

    start_matlab = getattr(matlab_engine, "start_matlab", None)
    if start_matlab is None or not callable(start_matlab):
        raise RuntimeError("Imported 'matlab.engine' module does not expose callable start_matlab()")
    return start_matlab


def _write_staged_volume_tiff(volume_path: Path, volume: Any) -> None:
    staged_volume = np.asarray(volume)
    if staged_volume.ndim != 3:
        raise ValueError(f"MATLAB extraction expects a 3D staged volume, got ndim={staged_volume.ndim}")

    with tifffile.TiffWriter(volume_path) as writer:
        for plane in staged_volume:
            writer.write(np.asarray(plane), photometric="minisblack", metadata=None)


class MATLABExtractionBackend:
    def __init__(
        self,
        config: ExperimentConfig,
        *,
        engine_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.config = config
        self.engine_factory = engine_factory
        self.runtime_dir = resolve_matlab_extraction_runtime_path(config)
        self.runtime_manifest = load_matlab_extraction_runtime_manifest(self.runtime_dir)
        self.entrypoint = config.providers.matlab.extraction.entrypoint
        _validate_runtime_entrypoint_contract(self.runtime_manifest, self.entrypoint)
        self._engine: Any = None

    def close(self) -> None:
        if self._engine is None:
            return
        try:
            close_matlab_engine_best_effort(
                self._engine,
                consumer="extraction.provider='matlab'",
            )
        finally:
            self._engine = None

    def _ensure_engine(self) -> Any:
        if self._engine is not None:
            return self._engine

        factory = self.engine_factory or _load_matlab_engine_factory()
        try:
            engine = factory()
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                _format_exception_message(
                    "Failed to start MATLAB Engine for extraction.provider='matlab'",
                    exc,
                )
            ) from exc

        try:
            engine.addpath(str(self.runtime_dir), nargout=0)
        except Exception as exc:  # pragma: no cover
            try:
                engine.quit()
            except Exception:
                pass
            raise RuntimeError(
                _format_exception_message(
                    f"Failed to add MATLAB extraction runtime path: {self.runtime_dir}",
                    exc,
                )
            ) from exc

        self._engine = engine
        return engine

    def _resolve_callable(self) -> Any:
        engine = self._ensure_engine()
        try:
            return getattr(engine, self.entrypoint)
        except AttributeError as exc:
            raise RuntimeError(
                f"MATLAB runtime path {self.runtime_dir} does not expose entrypoint '{self.entrypoint}'"
            ) from exc

    def _collect_runtime_file_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for bucket_name, required_default in (("required_files", True), ("optional_files", False)):
            for item in self.runtime_manifest.get(bucket_name, []):
                file_path = self.runtime_dir / item["name"]
                is_required = bool(item.get("required", required_default))
                is_used = is_required
                if is_used and not file_path.exists():
                    raise FileNotFoundError(
                        f"Required MATLAB extraction runtime file is missing: {file_path}. "
                        "MATLAB extraction provider cannot proceed."
                    )

                record = {
                    "name": item["name"],
                    "required": is_required,
                    "used": is_used,
                    "role": item["role"],
                    "source_path": item["source_path"],
                }
                if file_path.exists():
                    record["sha256"] = _sha256_file(file_path)
                records.append(record)
        return records

    def _normalize_input_volume(self, volume: Any) -> Any:
        if volume.ndim != 3:
            raise ValueError(f"MATLAB extraction expects a 3D volume, got ndim={volume.ndim}")

        input_dtype = np.dtype(self.config.providers.matlab.extraction.input_volume_dtype)
        arr = np.asarray(volume)
        if input_dtype == np.float32:
            return np.asarray(arr, dtype=np.float32)
        if input_dtype == np.uint8:
            if arr.dtype == np.uint8:
                return arr
            if np.issubdtype(arr.dtype, np.floating):
                max_val = float(np.max(arr)) if arr.size else 0.0
                if max_val <= 1.0:
                    arr = np.clip(arr, 0.0, 1.0) * 255.0
                else:
                    arr = np.clip(arr, 0.0, 255.0)
                return np.rint(arr).astype(np.uint8)
            return np.clip(arr, 0, 255).astype(np.uint8)

        raise ValueError(
            f"Unsupported providers.matlab.extraction.input_volume_dtype={self.config.providers.matlab.extraction.input_volume_dtype!r}"
        )

    def _validate_response_metadata(
        self,
        metadata: Mapping[str, Any],
        *,
        tmpdir_path: Path,
        round_id: int,
        channel_id: int,
        expected_count: int,
        expected_shape_zyx: tuple[int, int, int],
    ) -> Path:
        metadata_round_id = metadata.get("round_id")
        if not isinstance(metadata_round_id, (int, float)) or int(metadata_round_id) != round_id:
            raise ValueError(
                f"MATLAB extraction metadata round_id mismatch: expected {round_id}, got {metadata_round_id!r}"
            )

        metadata_channel_id = metadata.get("channel_id")
        if not isinstance(metadata_channel_id, (int, float)) or int(metadata_channel_id) != channel_id:
            raise ValueError(
                f"MATLAB extraction metadata channel_id mismatch: expected {channel_id}, got {metadata_channel_id!r}"
            )

        metadata_count = metadata.get("n_spots")
        if not isinstance(metadata_count, (int, float)) or int(metadata_count) != expected_count:
            raise ValueError(
                f"MATLAB extraction metadata n_spots mismatch: expected {expected_count}, got {metadata_count!r}"
            )

        volume_shape = metadata.get("volume_shape_zyx")
        if not isinstance(volume_shape, list) or [int(v) for v in volume_shape] != [int(v) for v in expected_shape_zyx]:
            raise ValueError(
                "MATLAB extraction metadata volume_shape_zyx mismatch: "
                f"expected {list(expected_shape_zyx)}, got {volume_shape!r}"
            )

        output_path_value = metadata.get("output_path")
        if not isinstance(output_path_value, str) or not output_path_value.strip():
            raise ValueError("MATLAB extraction metadata must declare a non-empty output_path")
        output_path = Path(output_path_value)
        if not output_path.is_absolute():
            output_path = tmpdir_path / output_path
        output_path = output_path.resolve()
        if output_path.parent != tmpdir_path.resolve():
            raise ValueError(
                "MATLAB extraction output must stay inside the staged temporary directory: "
                f"{output_path}"
            )
        if not output_path.exists():
            raise FileNotFoundError(
                f"MATLAB extraction reported output that does not exist: {output_path}"
            )

        steps = metadata.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ValueError("MATLAB extraction metadata must declare a non-empty steps list")
        for index, step in enumerate(steps):
            if not isinstance(step, Mapping):
                raise ValueError(f"MATLAB extraction step #{index} must be a mapping")
            name = step.get("name")
            duration_ms = step.get("duration_ms")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"MATLAB extraction step #{index} is missing a non-empty name")
            if not isinstance(duration_ms, (int, float)) or duration_ms < 0:
                raise ValueError(
                    f"MATLAB extraction step '{name}' must report a non-negative duration_ms"
                )

        return output_path

    def extract_intensities(
        self,
        volume: Any,
        coords_zyx: Any,
        *,
        fov_id: int,
        round_id: int,
        channel_id: int,
        box_size: tuple[int, int, int],
        transform_application_mode: str,
    ) -> Dict[str, Any]:
        volume_for_matlab = self._normalize_input_volume(volume)
        coords = np.asarray(coords_zyx, dtype=np.float32)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"MATLAB extraction expects coords shaped (N, 3), got {coords.shape}")

        plan = build_matlab_extraction_plan(
            self.config,
            fov_id=fov_id,
            round_id=round_id,
            channel_id=channel_id,
            n_spots=int(len(coords)),
            volume_shape_zyx=(int(volume_for_matlab.shape[0]), int(volume_for_matlab.shape[1]), int(volume_for_matlab.shape[2])),
            box_size=box_size,
            transform_application_mode=transform_application_mode,
        )
        matlab_callable = self._resolve_callable()

        with TemporaryDirectory(prefix=f"pystar_matlab_extraction_fov{fov_id}_round{round_id}_ch{channel_id}_") as tmpdir:
            tmpdir_path = Path(tmpdir)
            volume_path = tmpdir_path / f"extraction_input_fov_{fov_id}_round_{round_id}_ch_{channel_id}.tif"
            coords_path = tmpdir_path / f"coords_fov_{fov_id}_round_{round_id}_ch_{channel_id}.csv"
            _write_staged_volume_tiff(volume_path, volume_for_matlab)
            pd.DataFrame(
                {
                    "spot_index": np.arange(len(coords), dtype=np.int64),
                    "z": coords[:, 0],
                    "y": coords[:, 1],
                    "x": coords[:, 2],
                }
            ).to_csv(coords_path, index=False)

            try:
                metadata_json = matlab_callable(
                    str(volume_path),
                    str(coords_path),
                    json.dumps(plan, sort_keys=True),
                    nargout=1,
                )
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    _format_exception_message(
                        f"MATLAB extraction entrypoint '{self.entrypoint}' failed for FOV {fov_id} round {round_id} channel {channel_id}",
                        exc,
                    )
                ) from exc

            if not isinstance(metadata_json, str):
                raise ValueError(
                    f"MATLAB extraction entrypoint '{self.entrypoint}' must return a JSON string metadata payload"
                )

            try:
                metadata = json.loads(metadata_json)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    _format_exception_message(
                        f"MATLAB extraction entrypoint '{self.entrypoint}' returned invalid JSON metadata",
                        exc,
                    )
                ) from exc
            if not isinstance(metadata, dict):
                raise ValueError("MATLAB extraction metadata payload must decode to a JSON object")

            output_path = self._validate_response_metadata(
                metadata,
                tmpdir_path=tmpdir_path.resolve(),
                round_id=round_id,
                channel_id=channel_id,
                expected_count=int(len(coords)),
                expected_shape_zyx=(int(volume_for_matlab.shape[0]), int(volume_for_matlab.shape[1]), int(volume_for_matlab.shape[2])),
            )
            output_df = pd.read_csv(output_path)
            if "spot_index" not in output_df.columns or "intensity" not in output_df.columns:
                raise ValueError("MATLAB extraction output must contain 'spot_index' and 'intensity' columns")
            spot_index = np.asarray(pd.to_numeric(output_df["spot_index"], errors="raise"), dtype=np.int64)
            expected_index = np.arange(len(coords), dtype=np.int64)
            if not np.array_equal(spot_index, expected_index):
                raise ValueError(
                    "MATLAB extraction output spot_index ordering mismatch: expected sequential indices "
                    f"0..{len(coords) - 1}"
                )
            intensities = np.asarray(pd.to_numeric(output_df["intensity"], errors="raise"), dtype=np.float32)
            if len(intensities) != len(coords):
                raise ValueError(
                    f"MATLAB extraction output length mismatch: expected {len(coords)}, got {len(intensities)}"
                )

        return {
            "intensities": intensities,
            "backend_metadata": {
                "provider": "matlab",
                "runtime_path": str(self.runtime_dir),
                "runtime_manifest": str(self.runtime_dir / MATLAB_EXTRACTION_RUNTIME_MANIFEST_NAME),
                "entrypoint": self.entrypoint,
                "runtime_files": self._collect_runtime_file_records(),
                "matlab_metadata": metadata,
                "normalized_result": {
                    "spot_count": int(len(intensities)),
                    "columns": list(output_df.columns),
                    "dtype": str(intensities.dtype),
                },
            },
        }
