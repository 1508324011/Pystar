from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace, MethodType
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from pystar._artifact_schemas import (
    build_intensity_matrix_metadata_payload,
    build_intensity_matrix_spec,
    build_spot_row_lineage,
    empty_spot_table,
    intensity_matrix_metadata_path,
    validate_decoded_table,
    validate_spot_table,
)
from pystar.decoding import Decoder
from pystar.infrastructure import ExperimentConfig
from pystar.io import get_fov_output_structure
from pystar.matlab_extraction import MATLABExtractionBackend
from pystar.serialization import write_backend_metadata


FOV_ID = 7


def build_minimal_decoder_config(*, gene_list: Path, output_dir: Path) -> ExperimentConfig:
    topology = SimpleNamespace(
        func="none",
        structure=[
            SimpleNamespace(
                id="seq",
                rounds=[1, 2],
                csv_slice=[1, 2],
                anchor_base=None,
                encoding_table="dinucleotide",
            )
        ],
        physical_order=["seq"],
    )
    return cast(
        ExperimentConfig,
        cast(
            object,
            SimpleNamespace(
                codebook=SimpleNamespace(
                    gene_list=gene_list,
                    topology=topology,
                    encoding_tables={"dinucleotide": {"AA": 1}},
                    channel_base_index=1,
                ),
                dataset=SimpleNamespace(
                    channel_roles={0: "seq"},
                    round_structure={1: [0], 2: [0]},
                ),
                pipeline=SimpleNamespace(
                    output=SimpleNamespace(directory=str(output_dir)),
                    decoding=SimpleNamespace(gating_mode="pattern_first"),
                ),
            ),
        ),
    )


def build_decoder(*, output_dir: Path, gene_list: Path) -> Decoder:
    return Decoder(build_minimal_decoder_config(gene_list=gene_list, output_dir=output_dir))


def build_matlab_extraction_config() -> ExperimentConfig:
    return cast(
        ExperimentConfig,
        cast(
            object,
            SimpleNamespace(
                providers=SimpleNamespace(
                    matlab=SimpleNamespace(
                        extraction=SimpleNamespace(
                            input_volume_dtype="float32",
                            volume_transfer_mode="tiff",
                            coords_transfer_mode="csv",
                        )
                    )
                ),
                pipeline=SimpleNamespace(
                    extraction=SimpleNamespace(method="box_sum")
                ),
            ),
        ),
    )


def bare_decoder() -> Decoder:
    return Decoder.__new__(Decoder)


def make_spot_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "z": [1.0, 4.0, 7.0],
            "y": [2.0, 5.0, 8.0],
            "x": [3.0, 6.0, 9.0],
            "intensity": [10.0, 20.0, 30.0],
            "channel": [0, 0, 0],
            "fov": [FOV_ID, FOV_ID, FOV_ID],
            "algo": ["peak_local_max", "peak_local_max", "peak_local_max"],
        }
    )


def write_spot_matrix_family(
    tmp_path: Path,
    *,
    spots_df: pd.DataFrame,
    metadata_lineage_df: pd.DataFrame | None = None,
    include_lineage: bool = True,
    matrix: npt.NDArray[np.float32] | None = None,
) -> tuple[dict[str, Path], Path, Path, Any, npt.NDArray[np.float32], pd.DataFrame]:
    paths = get_fov_output_structure(tmp_path, FOV_ID)
    spots_path = paths["spots"] / f"spots_fov_{FOV_ID}.csv"
    matrix_path = paths["extraction"] / f"intensity_matrix_fov_{FOV_ID}.npy"
    metadata_path = intensity_matrix_metadata_path(matrix_path)

    validated_spots = validate_spot_table(
        spots_df,
        fov_id=FOV_ID,
        path=spots_path,
        context="stage11a spot write",
    )
    validated_spots.to_csv(spots_path, index=False)

    spec = build_intensity_matrix_spec(fov_id=FOV_ID, n_spots=len(validated_spots), rounds=[1, 2], channels=[0])
    if matrix is None:
        matrix = np.arange(np.prod(spec.expected_shape), dtype=np.float32).reshape(spec.expected_shape)
    np.save(matrix_path, matrix)

    payload_kwargs: dict[str, Any] = {}
    if include_lineage:
        lineage_source = validated_spots if metadata_lineage_df is None else validate_spot_table(
            metadata_lineage_df,
            fov_id=FOV_ID,
            path=None,
            context="stage11a lineage source",
        )
        payload_kwargs["spot_row_lineage"] = build_spot_row_lineage(
            lineage_source,
            fov_id=FOV_ID,
            path=None,
            context="stage11a lineage build",
        )
    payload = build_intensity_matrix_metadata_payload(spec, **payload_kwargs)
    write_backend_metadata(metadata_path, payload)

    return paths, spots_path, matrix_path, spec, matrix, validated_spots


def load_validated_spots(spots_path: Path) -> pd.DataFrame:
    return validate_spot_table(
        pd.read_csv(spots_path),
        fov_id=FOV_ID,
        path=spots_path,
        context="stage11a spot read",
    )


def test_decoder_matrix_loader_accepts_matching_spot_row_lineage(tmp_path: Path) -> None:
    decoder = bare_decoder()
    _paths, spots_path, matrix_path, spec, matrix, _spots = write_spot_matrix_family(
        tmp_path,
        spots_df=make_spot_table(),
    )

    loaded = decoder._load_validated_intensity_matrix(
        fov_id=FOV_ID,
        matrix_path=matrix_path,
        matrix_spec=spec,
        spots_df=load_validated_spots(spots_path),
        spots_path=spots_path,
    )

    np.testing.assert_array_equal(loaded, matrix)


def test_decoder_matrix_loader_rejects_reordered_spot_rows_with_same_shape(tmp_path: Path) -> None:
    decoder = bare_decoder()
    original = make_spot_table()
    reordered = original.iloc[[2, 0, 1]].reset_index(drop=True)
    _paths, spots_path, matrix_path, spec, _matrix, _spots = write_spot_matrix_family(
        tmp_path,
        spots_df=reordered,
        metadata_lineage_df=original,
    )

    with pytest.raises(ValueError, match=r"spot_row_lineage fingerprint mismatch"):
        decoder._load_validated_intensity_matrix(
            fov_id=FOV_ID,
            matrix_path=matrix_path,
            matrix_spec=spec,
            spots_df=load_validated_spots(spots_path),
            spots_path=spots_path,
        )


def test_decoder_matrix_loader_rejects_same_shape_metadata_from_different_spot_table(tmp_path: Path) -> None:
    decoder = bare_decoder()
    current = make_spot_table()
    foreign = current.copy()
    foreign["z"] = [101.0, 102.0, 103.0]
    foreign["intensity"] = [41.0, 42.0, 43.0]

    _paths, spots_path, matrix_path, spec, _matrix, _spots = write_spot_matrix_family(
        tmp_path,
        spots_df=current,
        metadata_lineage_df=foreign,
    )

    with pytest.raises(ValueError, match=r"spot_row_lineage fingerprint mismatch"):
        decoder._load_validated_intensity_matrix(
            fov_id=FOV_ID,
            matrix_path=matrix_path,
            matrix_spec=spec,
            spots_df=load_validated_spots(spots_path),
            spots_path=spots_path,
        )


def test_decoder_matrix_loader_uses_explicit_legacy_compatibility_when_lineage_absent(tmp_path: Path) -> None:
    decoder = bare_decoder()
    _paths, spots_path, matrix_path, spec, matrix, _spots = write_spot_matrix_family(
        tmp_path,
        spots_df=make_spot_table(),
        include_lineage=False,
    )

    loaded = decoder._load_validated_intensity_matrix(
        fov_id=FOV_ID,
        matrix_path=matrix_path,
        matrix_spec=spec,
        spots_df=load_validated_spots(spots_path),
        spots_path=spots_path,
    )

    np.testing.assert_array_equal(loaded, matrix)


def test_decode_fov_empty_artifacts_remain_canonical_with_row_lineage_sidecar(tmp_path: Path) -> None:
    gene_list = tmp_path / "genes.csv"
    gene_list.write_text("GeneA,AA\n", encoding="utf-8")
    decoder = build_decoder(output_dir=tmp_path, gene_list=gene_list)

    paths = get_fov_output_structure(tmp_path, FOV_ID)
    spots_path = paths["spots"] / f"spots_fov_{FOV_ID}.csv"
    matrix_path = paths["extraction"] / f"intensity_matrix_fov_{FOV_ID}.npy"
    metadata_path = intensity_matrix_metadata_path(matrix_path)

    empty_spots = empty_spot_table(extra_columns=("channel", "fov", "algo"))
    empty_spots.to_csv(spots_path, index=False)
    spec = build_intensity_matrix_spec(fov_id=FOV_ID, n_spots=0, rounds=[1, 2], channels=[0])
    np.save(matrix_path, np.zeros(spec.expected_shape, dtype=np.float32))
    payload = build_intensity_matrix_metadata_payload(
        spec,
        spot_row_lineage=build_spot_row_lineage(
            empty_spots,
            fov_id=FOV_ID,
            path=spots_path,
            context="stage11a empty lineage build",
        ),
    )
    write_backend_metadata(metadata_path, payload)

    decoded = decoder.decode_fov(FOV_ID)

    assert len(decoded) == 0
    for suffix in ("", "_goodreads", "_pre_pattern_check"):
        decoded_path = paths["decoded"] / f"decoded_fov_{FOV_ID}{suffix}.csv"
        assert decoded_path.exists()
        reloaded = pd.read_csv(decoded_path)
        assert len(validate_decoded_table(reloaded, fov_id=FOV_ID, path=decoded_path, context="stage11a empty decode read")) == 0


def test_matlab_extraction_output_still_rejects_non_sequential_spot_index(tmp_path: Path) -> None:
    backend = MATLABExtractionBackend.__new__(MATLABExtractionBackend)
    backend.config = build_matlab_extraction_config()
    backend.runtime_dir = tmp_path
    backend.runtime_manifest = {}
    backend.entrypoint = "synthetic_entry"
    backend._session_capsule = cast(Any, SimpleNamespace(
        session_lifecycle={},
        summarize_session_lifecycle=lambda: None,
    ))
    backend._resolve_runtime_file_records = MethodType(lambda self: ([], {}), backend)
    backend._consume_last_engine_acquire = MethodType(lambda self: {}, backend)

    def fake_callable(volume_path: str, coords_path: str, plan_json: str, nargout: int = 1) -> str:
        _ = plan_json
        _ = nargout
        coords_df = pd.read_csv(coords_path)
        output_path = Path(coords_path).parent / "synthetic_output.csv"
        reversed_index = coords_df["spot_index"].iloc[::-1].to_numpy(dtype=np.int64)
        pd.DataFrame(
            {
                "spot_index": reversed_index,
                "intensity": np.linspace(1.0, 2.0, len(reversed_index), dtype=np.float32),
            }
        ).to_csv(output_path, index=False)
        metadata = {
            "round_id": 1,
            "channel_id": 0,
            "n_spots": int(len(coords_df)),
            "volume_shape_zyx": [2, 3, 4],
            "output_path": str(output_path),
            "steps": [{"name": "write_output", "duration_ms": 0.0}],
        }
        return json.dumps(metadata, sort_keys=True)

    backend._resolve_callable = MethodType(lambda self: fake_callable, backend)

    volume = np.zeros((2, 3, 4), dtype=np.float32)
    coords = np.asarray([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]], dtype=np.float32)

    with pytest.raises(ValueError, match=r"spot_index ordering mismatch"):
        backend.extract_intensities(
            volume,
            coords,
            fov_id=FOV_ID,
            round_id=1,
            channel_id=0,
            box_size=(1, 1, 1),
            transform_application_mode="coordinate_mapping",
        )
