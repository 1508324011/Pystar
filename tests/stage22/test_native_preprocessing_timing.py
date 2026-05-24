from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import yaml

from pystar import preprocessing as preprocessing_module
from pystar.infrastructure import PreprocessingStep
from pystar.io import get_fov_output_structure
from pystar.preprocessing import (
    NATIVE_PREPROCESSING_TIMING_SCHEMA_NAME,
    NATIVE_PREPROCESSING_TIMING_SCHEMA_VERSION,
    DataSanitizer,
)


class _FakeDelayedVolume:
    def __init__(self, array: np.ndarray) -> None:
        self._array = array

    def compute(self) -> np.ndarray:
        return self._array.copy()


class _FakeLoader:
    def __init__(self, volumes: dict[tuple[int, int], np.ndarray], root: Path) -> None:
        self.volumes = volumes
        self.root = root

    def _get_path(self, fov: int, round_id: int, channel_id: int) -> Path:
        _ = fov
        return self.root / f"round{round_id}" / f"ch{channel_id}.tif"

    def _lazy_load_tiff(self, path: Path) -> _FakeDelayedVolume:
        round_id = int(path.parent.name.replace("round", ""))
        channel_id = int(path.stem.replace("ch", ""))
        return _FakeDelayedVolume(self.volumes[(round_id, channel_id)])


def _step(method: str, params: dict[str, Any] | None = None) -> PreprocessingStep:
    return PreprocessingStep(method=method, provider="native", params={} if params is None else params)


def _config(tmp_path: Path, sequence: list[PreprocessingStep]) -> Any:
    dataset = SimpleNamespace(
        raw_data_path=tmp_path / "raw",
        filename_pattern="round{round}/Position{fov}/*_ch{ch}.tif",
        round_structure={1: [0, 1], 2: [0, 1]},
        channel_roles={0: "seq", 1: "seq"},
        dimensions={"z": 1, "height": 2, "width": 2},
        io_chunk_size={"z": 1, "y": 2, "x": 2},
    )
    pipeline = SimpleNamespace(
        preprocessing=SimpleNamespace(sequence=sequence),
        output=SimpleNamespace(directory=str(tmp_path / "out")),
    )
    pipeline.preprocessing_providers_used = lambda: ["native"]
    pipeline.preprocessing_provider_mode = lambda: "native_only"
    return SimpleNamespace(dataset=dataset, pipeline=pipeline)


def _volumes() -> dict[tuple[int, int], np.ndarray]:
    return {
        (1, 0): np.full((1, 2, 2), 10, dtype=np.uint8),
        (1, 1): np.full((1, 2, 2), 20, dtype=np.uint8),
        (2, 0): np.full((1, 2, 2), 30, dtype=np.uint8),
        (2, 1): np.full((1, 2, 2), 40, dtype=np.uint8),
    }


def test_native_only_provenance_adds_timing_without_changing_clean_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _config(
        tmp_path,
        [
            _step("none"),
            _step("min_max_normalize"),
            _step("morpho_reconstruction_contrast", {"radius": 1, "downsample_factor": 1.0}),
        ],
    )
    sanitizer = DataSanitizer(cast(Any, cfg))
    sanitizer.loader = cast(Any, _FakeLoader(_volumes(), tmp_path / "raw"))

    def fake_morphology(img: Any, params: dict[str, Any], ctx: dict[str, Any]) -> Any:
        _ = (params, ctx)
        return img

    monkeypatch.setitem(preprocessing_module.PROCESSOR_MAP, "morpho_reconstruction_contrast", fake_morphology)

    provenance = sanitizer.sanitize_fov(1)

    assert provenance["version"] == "1.0"
    assert provenance["backend"] == "native_pystar"
    assert provenance["provider"] == "native"
    assert provenance["pipeline_split"] == {
        "calibration_steps": ["none", "min_max_normalize"],
        "extraction_steps": ["morpho_reconstruction_contrast"],
    }
    assert provenance["input_contract"]["rounds_processed"] == [1, 2]

    paths = get_fov_output_structure(Path(cfg.pipeline.output.directory), 1)
    expected_outputs = [
        paths["cleaned"] / "clean_fov_1_round_1_ch_0.tif",
        paths["cleaned"] / "clean_fov_1_round_1_ch_1.tif",
        paths["cleaned"] / "clean_fov_1_round_2_ch_0.tif",
        paths["cleaned"] / "clean_fov_1_round_2_ch_1.tif",
    ]
    assert [Path(path) for path in provenance["output_files"]] == expected_outputs
    assert all(path.exists() for path in expected_outputs)

    timing = provenance["preprocessing_timing"]
    assert timing["schema_name"] == NATIVE_PREPROCESSING_TIMING_SCHEMA_NAME
    assert timing["schema_version"] == NATIVE_PREPROCESSING_TIMING_SCHEMA_VERSION
    assert timing["round_order"] == [1, 2]
    assert timing["volume_count"] == 4
    assert isinstance(timing["total_volume_ms"], float)
    assert [
        (volume["round_id"], volume["channel_id"])
        for volume in timing["volumes"]
    ] == [(1, 0), (1, 1), (2, 0), (2, 1)]
    first_volume = timing["volumes"][0]
    assert [step["method"] for step in first_volume["calibration_steps"]] == ["none", "min_max_normalize"]
    assert [step["method"] for step in first_volume["extraction_steps"]] == ["morpho_reconstruction_contrast"]
    for units_key in ("load_ms", "clip_convert_ms", "write_ms", "total_ms"):
        assert isinstance(first_volume[units_key], float)
        assert first_volume[units_key] >= 0.0
    assert timing["summary"]["by_method"]["none"]["count"] == 4
    assert timing["summary"]["by_method"]["morpho_reconstruction_contrast"]["count"] == 4
    assert timing["summary"]["by_phase"]["write"]["count"] == 4

    persisted = yaml.safe_load((paths["qc"] / "preprocessing_provenance.yaml").read_text(encoding="utf-8"))
    assert persisted["preprocessing_timing"]["schema_name"] == NATIVE_PREPROCESSING_TIMING_SCHEMA_NAME


def test_native_kernel_preserves_inter_and_intra_round_reference_context(tmp_path: Path) -> None:
    cfg = _config(tmp_path, [_step("none")])
    sanitizer = DataSanitizer(cast(Any, cfg))
    sanitizer.loader = cast(Any, _FakeLoader(_volumes(), tmp_path / "raw"))

    observed_contexts: list[tuple[int, int, bool, bool]] = []

    def writer(img: Any, round_id: int, channel_id: int) -> Path:
        _ = img
        path = tmp_path / "stage" / f"r{round_id}_c{channel_id}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"ok")
        return path

    sequence = [
        _step("none"),
        _step("histogram_match", {"scope": "inter_round"}),
        _step("histogram_match", {"scope": "intra_round"}),
    ]

    original_run_pipeline_with_timing = sanitizer._run_pipeline_with_timing

    def spy_run_pipeline_with_timing(
        img_vol: Any,
        pipeline_seq: list[PreprocessingStep],
        context: dict[str, Any],
    ) -> tuple[Any, list[dict[str, Any]]]:
        # The native loop calls calibration before extraction. This sequence has
        # no extraction steps because _split_sequence remains unchanged.
        if not pipeline_seq:
            return original_run_pipeline_with_timing(img_vol, pipeline_seq, context)
        assert pipeline_seq == sequence
        value = int(np.asarray(img_vol)[0, 0, 0])
        round_id, channel_id = {
            10: (1, 0),
            20: (1, 1),
            30: (2, 0),
            40: (2, 1),
        }[value]
        observed_contexts.append(
            (
                round_id,
                channel_id,
                context["ref_round_image"] is not None,
                context["ref_channel_image"] is not None,
            )
        )
        return original_run_pipeline_with_timing(img_vol, pipeline_seq, context)

    sanitizer._run_pipeline_with_timing = cast(Any, spy_run_pipeline_with_timing)

    result = sanitizer._run_native_preprocessing_kernel(
        fov_id=1,
        loader=sanitizer.loader,
        sequence=sequence,
        target_rounds=None,
        output_writer=writer,
    )

    assert result["round_order"] == [1, 2]
    assert observed_contexts == [
        (1, 0, False, False),
        (1, 1, False, True),
        (2, 0, True, False),
        (2, 1, True, True),
    ]


def test_provider_dispatch_native_segment_uses_same_timing_shape_and_stage_paths(tmp_path: Path) -> None:
    cfg = _config(tmp_path, [_step("none")])
    sanitizer = DataSanitizer(cast(Any, cfg))
    sanitizer._make_loader = cast(Any, lambda _root, _pattern: _FakeLoader(_volumes(), tmp_path / "raw"))
    sanitizer._run_pipeline_with_timing = cast(
        Any,
        lambda img_vol, pipeline_seq, context: (img_vol, [
            {"index": index, "method": step.method, "provider": step.provider, "duration_ms": 0.0}
            for index, step in enumerate(pipeline_seq)
        ]),
    )

    record = sanitizer._run_native_sequence_segment(
        1,
        [_step("none"), _step("morpho_reconstruction_contrast", {"radius": 1, "downsample_factor": 1.0})],
        input_root=tmp_path / "input_stage",
        input_filename_pattern="round{round}/Position{fov}/*_ch{ch}.tif",
        output_root=tmp_path / "next_stage",
        segment_index=3,
    )

    assert record["provider"] == "native"
    assert record["segment_index"] == 3
    assert record["pipeline_split"] == {
        "calibration_steps": ["none"],
        "extraction_steps": ["morpho_reconstruction_contrast"],
    }
    assert record["preprocessing_timing"]["segment_index"] == 3
    assert record["preprocessing_timing"]["schema_name"] == NATIVE_PREPROCESSING_TIMING_SCHEMA_NAME
    assert record["preprocessing_timing"]["volume_count"] == 4
    assert all("next_stage" in path for path in record["output_files"])
    assert all(Path(path).exists() for path in record["output_files"])


def test_empty_preprocessing_sequence_remains_unchanged(tmp_path: Path) -> None:
    cfg = _config(tmp_path, [])
    sanitizer = DataSanitizer(cast(Any, cfg))

    provenance = sanitizer._native_sanitize_fov(1)

    assert provenance["output_files"] == []
    assert provenance["pipeline_split"] == {"calibration_steps": [], "extraction_steps": []}
    assert "preprocessing_timing" not in provenance
