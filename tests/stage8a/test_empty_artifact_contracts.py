import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pandas as pd
import pytest

from pystar._artifact_schemas import (
    DecodedTableSchema,
    SpotTableSchema,
    build_intensity_matrix_metadata_payload,
    build_intensity_matrix_spec,
    empty_decoded_table,
    empty_spot_table,
    intensity_matrix_metadata_path,
    validate_decoded_table,
    validate_intensity_matrix,
    validate_intensity_matrix_consumer_contract,
    validate_intensity_matrix_metadata_payload,
    validate_spot_table,
)
from pystar.decoding import Decoder
from pystar.infrastructure import ExperimentConfig
from pystar.io import get_fov_output_structure
from pystar.mining import SignalMiner
from pystar.serialization import write_backend_metadata


FOV_ID = 7


def build_minimal_signal_miner(*, output_dir: Path, qc_enabled: bool = True) -> SignalMiner:
    miner = SignalMiner.__new__(SignalMiner)
    miner.cfg = cast(
        ExperimentConfig,
        cast(
            object,
            SimpleNamespace(
                pipeline=SimpleNamespace(
                    output=SimpleNamespace(directory=str(output_dir)),
                    qc_images_enabled=lambda: qc_enabled,
                )
            ),
        ),
    )
    return miner


def build_minimal_decoder(*, output_dir: Path, gene_list: Path) -> Decoder:
    decoder = Decoder.__new__(Decoder)
    decoder.cfg = cast(
        ExperimentConfig,
        cast(
            object,
            SimpleNamespace(
                codebook=SimpleNamespace(
                    gene_list=gene_list,
                    topology=SimpleNamespace(
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
                    ),
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
    decoder.output_dir = output_dir
    return decoder


def test_empty_spot_table_round_trips_and_validates(tmp_path: Path) -> None:
    path = tmp_path / "spots_fov_7.csv"
    spot_table = empty_spot_table(extra_columns=("channel", "fov", "algo"))

    validated = validate_spot_table(
        spot_table,
        fov_id=FOV_ID,
        path=path,
        context="stage8a empty spot write",
    )
    validated.to_csv(path, index=False)
    reloaded = pd.read_csv(path)

    assert list(SpotTableSchema().required_columns) == ["z", "y", "x", "intensity"]
    assert all(column in reloaded.columns for column in SpotTableSchema().required_columns)
    assert len(
        validate_spot_table(
            reloaded,
            fov_id=FOV_ID,
            path=path,
            context="stage8a empty spot read",
        )
    ) == 0


def test_spot_table_rejects_missing_columns_and_fov_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "spots_fov_7.csv"

    with pytest.raises(ValueError, match="FOV 7.*missing required columns.*z.*Expected schema"):
        validate_spot_table(
            pd.DataFrame({"y": [1], "x": [2], "intensity": [3]}),
            fov_id=FOV_ID,
            path=path,
            context="stage8a malformed spot read",
        )

    with pytest.raises(ValueError, match="FOV 7.*fov.*requested fov_id=7.*Expected schema"):
        validate_spot_table(
            pd.DataFrame({"z": [0], "y": [1], "x": [2], "intensity": [3], "fov": [8]}),
            fov_id=FOV_ID,
            path=path,
            context="stage8a malformed spot read",
        )


def test_empty_intensity_matrix_and_sidecar_contract_round_trip(tmp_path: Path) -> None:
    matrix_path = tmp_path / "intensity_matrix_fov_7.npy"
    metadata_path = intensity_matrix_metadata_path(matrix_path)
    spec = build_intensity_matrix_spec(fov_id=FOV_ID, n_spots=0, rounds=[1, 2, 3], channels=[0, 2])
    matrix = np.zeros(spec.expected_shape, dtype=np.float32)

    validated_matrix = validate_intensity_matrix(
        matrix,
        spec,
        path=matrix_path,
        context="stage8a empty intensity save",
    )
    np.save(matrix_path, validated_matrix)
    write_backend_metadata(metadata_path, build_intensity_matrix_metadata_payload(spec))
    metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    persisted_spec = validate_intensity_matrix_metadata_payload(
        metadata_payload,
        fov_id=FOV_ID,
        path=metadata_path,
        context="stage8a empty intensity metadata read",
    )
    validate_intensity_matrix_consumer_contract(
        persisted_spec,
        spec,
        path=metadata_path,
        context="stage8a empty intensity consumer",
        matrix_path=matrix_path,
    )

    assert tuple(np.load(matrix_path, allow_pickle=False).shape) == (0, 3, 2)


@pytest.mark.parametrize(
    "matrix_factory, match",
    [
        (lambda: np.zeros((0, 2), dtype=np.float32), "rank mismatch"),
        (lambda: np.zeros((1, 3, 2), dtype=np.float32), "axis 0 mismatch"),
        (lambda: np.zeros((0, 2, 2), dtype=np.float32), "axis 1 mismatch"),
        (lambda: np.zeros((0, 3, 1), dtype=np.float32), "axis 2 mismatch"),
        (lambda: np.zeros((0, 3, 2), dtype=np.float64), "dtype float64 is not float32"),
        (lambda: np.asarray([[[]]], dtype=object), "dtype object is not numeric"),
    ],
)
def test_invalid_empty_intensity_matrices_fail_loudly(tmp_path: Path, matrix_factory, match: str) -> None:
    spec = build_intensity_matrix_spec(fov_id=FOV_ID, n_spots=0, rounds=[1, 2, 3], channels=[0, 2])

    with pytest.raises(ValueError, match=f"FOV 7.*{match}.*Expected schema"):
        validate_intensity_matrix(
            matrix_factory(),
            spec,
            path=tmp_path / "intensity_matrix_fov_7.npy",
            context="stage8a invalid intensity read",
        )


def test_invalid_intensity_metadata_fails_loudly(tmp_path: Path) -> None:
    spec = build_intensity_matrix_spec(fov_id=FOV_ID, n_spots=0, rounds=[1, 2], channels=[0])
    metadata_path = tmp_path / "intensity_matrix_fov_7_metadata.json"

    negative_payload = build_intensity_matrix_metadata_payload(spec)
    negative_payload["matrix_shape"] = [0, -2, 1]
    with pytest.raises(ValueError, match="FOV 7.*negative dimensions.*Expected schema"):
        validate_intensity_matrix_metadata_payload(
            negative_payload,
            fov_id=FOV_ID,
            path=metadata_path,
            context="stage8a invalid metadata read",
        )

    negative_spots_payload = build_intensity_matrix_metadata_payload(spec)
    negative_spots_payload["n_spots"] = -1
    negative_spots_payload["matrix_shape"] = [-1, 2, 1]
    with pytest.raises(ValueError, match="FOV 7.*n_spots.*non-negative.*Expected schema"):
        validate_intensity_matrix_metadata_payload(
            negative_spots_payload,
            fov_id=FOV_ID,
            path=metadata_path,
            context="stage8a invalid metadata read",
        )

    sidecar_spec = build_intensity_matrix_spec(fov_id=FOV_ID, n_spots=1, rounds=[1, 2], channels=[0])
    with pytest.raises(ValueError, match="FOV 7.*n_spots=1.*consumer expects n_spots=0.*Expected schema"):
        validate_intensity_matrix_consumer_contract(
            sidecar_spec,
            spec,
            path=metadata_path,
            context="stage8a invalid metadata consumer",
            matrix_path=tmp_path / "intensity_matrix_fov_7.npy",
        )

    wrong_order_spec = build_intensity_matrix_spec(fov_id=FOV_ID, n_spots=0, rounds=[2, 1], channels=[0])
    with pytest.raises(ValueError, match="FOV 7.*round_order mismatch.*Expected schema"):
        validate_intensity_matrix_consumer_contract(
            wrong_order_spec,
            spec,
            path=metadata_path,
            context="stage8a invalid metadata consumer",
            matrix_path=tmp_path / "intensity_matrix_fov_7.npy",
        )


def test_empty_decoded_table_contract_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "decoded_fov_7.csv"
    decoded_table = empty_decoded_table(extra_columns=("channel", "fov", "algo", "pattern_valid", "in_codebook", "gating_mode"))

    validated = validate_decoded_table(
        decoded_table,
        fov_id=FOV_ID,
        path=path,
        context="stage8a empty decoded write",
    )
    validated.to_csv(path, index=False)
    reloaded = pd.read_csv(path)

    assert all(column in reloaded.columns for column in DecodedTableSchema().required_columns)
    assert len(
        validate_decoded_table(
            reloaded,
            fov_id=FOV_ID,
            path=path,
            context="stage8a empty decoded read",
        )
    ) == 0


def test_decoded_artifact_writer_preserves_empty_schema(tmp_path: Path) -> None:
    gene_list = tmp_path / "genes.csv"
    gene_list.write_text("GeneA,AA\n", encoding="utf-8")
    decoder = build_minimal_decoder(output_dir=tmp_path, gene_list=gene_list)

    output_path = tmp_path / "decoded_fov_7_goodreads.csv"
    written = decoder._write_decoded_artifact(
        empty_decoded_table(extra_columns=decoder._decoded_artifact_extra_columns()),
        fov_id=FOV_ID,
        path=output_path,
        context="stage8a empty decoded writer",
    )

    reloaded = pd.read_csv(output_path)
    assert len(written) == 0
    assert all(column in reloaded.columns for column in DecodedTableSchema().required_columns)


def test_decoder_no_spots_writes_three_canonical_decoded_artifacts(tmp_path: Path) -> None:
    gene_list = tmp_path / "genes.csv"
    gene_list.write_text("GeneA,AA\n", encoding="utf-8")
    decoder = build_minimal_decoder(output_dir=tmp_path, gene_list=gene_list)
    paths = get_fov_output_structure(tmp_path, FOV_ID)
    spots_path = paths["spots"] / f"spots_fov_{FOV_ID}.csv"
    matrix_path = paths["extraction"] / f"intensity_matrix_fov_{FOV_ID}.npy"
    spec = build_intensity_matrix_spec(fov_id=FOV_ID, n_spots=0, rounds=[1, 2], channels=[0])

    empty_spot_table(extra_columns=("channel", "fov", "algo")).to_csv(spots_path, index=False)
    np.save(matrix_path, np.zeros(spec.expected_shape, dtype=np.float32))
    write_backend_metadata(intensity_matrix_metadata_path(matrix_path), build_intensity_matrix_metadata_payload(spec))

    decoded = decoder.decode_fov(FOV_ID)

    assert len(decoded) == 0
    for suffix in ("", "_goodreads", "_pre_pattern_check"):
        decoded_path = paths["decoded"] / f"decoded_fov_{FOV_ID}{suffix}.csv"
        assert decoded_path.exists()
        reloaded = pd.read_csv(decoded_path)
        assert all(column in reloaded.columns for column in DecodedTableSchema().required_columns)
        assert len(validate_decoded_table(reloaded, fov_id=FOV_ID, path=decoded_path, context="stage8a no-spots read")) == 0


def test_decoder_all_filtered_keeps_pre_pattern_diagnostics(tmp_path: Path) -> None:
    gene_list = tmp_path / "genes.csv"
    gene_list.write_text("GeneA,AA\n", encoding="utf-8")
    decoder = build_minimal_decoder(output_dir=tmp_path, gene_list=gene_list)
    decoder.gene_map = {"00": "GeneA"}

    def always_fail_pattern(barcode: str) -> bool:
        _ = barcode
        return False

    decoder._validate_end_bases = always_fail_pattern
    paths = get_fov_output_structure(tmp_path, FOV_ID)
    spots_path = paths["spots"] / f"spots_fov_{FOV_ID}.csv"
    matrix_path = paths["extraction"] / f"intensity_matrix_fov_{FOV_ID}.npy"
    spots = validate_spot_table(
        pd.DataFrame({"z": [1.0], "y": [2.0], "x": [3.0], "intensity": [4.0], "fov": [FOV_ID]}),
        fov_id=FOV_ID,
        path=spots_path,
        context="stage8a all-filtered spot write",
    )
    spec = build_intensity_matrix_spec(fov_id=FOV_ID, n_spots=1, rounds=[1, 2], channels=[0])

    spots.to_csv(spots_path, index=False)
    np.save(matrix_path, np.ones(spec.expected_shape, dtype=np.float32))
    write_backend_metadata(intensity_matrix_metadata_path(matrix_path), build_intensity_matrix_metadata_payload(spec))

    decoded = decoder.decode_fov(FOV_ID)

    assert len(decoded) == 0
    main = pd.read_csv(paths["decoded"] / f"decoded_fov_{FOV_ID}.csv")
    goodreads = pd.read_csv(paths["decoded"] / f"decoded_fov_{FOV_ID}_goodreads.csv")
    pre_pattern = pd.read_csv(paths["decoded"] / f"decoded_fov_{FOV_ID}_pre_pattern_check.csv")
    assert len(main) == 0
    assert len(goodreads) == 0
    assert len(pre_pattern) == 1
    assert bool(pre_pattern.loc[0, "in_codebook"])
    assert not bool(pre_pattern.loc[0, "pattern_valid"])
    validate_decoded_table(pre_pattern, fov_id=FOV_ID, path=None, context="stage8a all-filtered pre-pattern read")


def test_decoder_rejects_runtime_empty_gene_map_before_stop_iteration(tmp_path: Path) -> None:
    gene_list = tmp_path / "genes.csv"
    gene_list.write_text("GeneA,AA\n", encoding="utf-8")
    decoder = build_minimal_decoder(output_dir=tmp_path, gene_list=gene_list)
    decoder.gene_map = {}
    paths = get_fov_output_structure(tmp_path, FOV_ID)
    spots_path = paths["spots"] / f"spots_fov_{FOV_ID}.csv"
    matrix_path = paths["extraction"] / f"intensity_matrix_fov_{FOV_ID}.npy"
    spec = build_intensity_matrix_spec(fov_id=FOV_ID, n_spots=1, rounds=[1, 2], channels=[0])

    pd.DataFrame({"z": [1.0], "y": [2.0], "x": [3.0], "intensity": [4.0]}).to_csv(spots_path, index=False)
    np.save(matrix_path, np.ones(spec.expected_shape, dtype=np.float32))
    write_backend_metadata(intensity_matrix_metadata_path(matrix_path), build_intensity_matrix_metadata_payload(spec))

    with pytest.raises(ValueError, match="Codebook gene-map contract error.*compiled gene map is empty before decoding FOV 7"):
        decoder.decode_fov(FOV_ID)


def test_decoder_rejects_empty_gene_list(tmp_path: Path) -> None:
    gene_list = tmp_path / "empty_genes.csv"
    gene_list.write_text("", encoding="utf-8")
    decoder = build_minimal_decoder(output_dir=tmp_path, gene_list=gene_list)

    with pytest.raises(ValueError, match="Codebook gene-map contract error.*gene list is empty"):
        decoder._compile_codebook()


def test_decoder_accepts_headered_gene_list(tmp_path: Path) -> None:
    gene_list = tmp_path / "headered_genes.csv"
    gene_list.write_text("gene,seq\nGeneA,AA\n", encoding="utf-8")
    decoder = build_minimal_decoder(output_dir=tmp_path, gene_list=gene_list)

    gene_map, compiled = decoder._compile_codebook()

    assert gene_map == {"0": "GeneA"}
    assert list(compiled["gene"]) == ["GeneA"]


def test_decoder_rejects_gene_list_that_compiles_to_zero_barcodes(tmp_path: Path) -> None:
    gene_list = tmp_path / "too_short_genes.csv"
    gene_list.write_text("GeneA,A\n", encoding="utf-8")
    decoder = build_minimal_decoder(output_dir=tmp_path, gene_list=gene_list)

    with pytest.raises(ValueError, match="Codebook gene-map contract error.*compiled to zero valid barcodes"):
        decoder._compile_codebook()


def test_decoder_rejects_empty_encoding_tables_before_generic_errors(tmp_path: Path) -> None:
    gene_list = tmp_path / "genes.csv"
    gene_list.write_text("GeneA,AA\n", encoding="utf-8")
    decoder = build_minimal_decoder(output_dir=tmp_path, gene_list=gene_list)
    decoder.cfg.codebook.encoding_tables = {}

    with pytest.raises(ValueError, match="Codebook gene-map contract error.*encoding_tables is empty"):
        decoder._compile_codebook()


def test_decoder_missing_gene_list_still_fails_loudly(tmp_path: Path) -> None:
    decoder = build_minimal_decoder(output_dir=tmp_path, gene_list=tmp_path / "missing.csv")

    with pytest.raises(FileNotFoundError, match="Gene list not found"):
        decoder._compile_codebook()


@pytest.mark.parametrize("n_spots", [0, 1, 2, 3, 4])
def test_signal_miner_qc_sampling_handles_small_spot_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    n_spots: int,
) -> None:
    miner = build_minimal_signal_miner(output_dir=tmp_path, qc_enabled=True)
    matrix = np.arange(n_spots * 2 * 2, dtype=np.float32).reshape(n_spots, 2, 2)
    spots_df = pd.DataFrame(
        {
            "z": np.arange(n_spots, dtype=np.float32),
            "y": np.arange(n_spots, dtype=np.float32),
            "x": np.arange(n_spots, dtype=np.float32),
            "intensity": np.arange(n_spots, dtype=np.float32),
        }
    )
    captured_indices: list[int] | None = None

    def fake_plot_spot_traces(_matrix, spot_indices, _rounds, _channels, output_path=None):
        nonlocal captured_indices
        _ = output_path
        captured_indices = [int(index) for index in np.asarray(spot_indices, dtype=np.int64).reshape(-1)]

    monkeypatch.setattr("pystar.visualization.plot_spot_traces", fake_plot_spot_traces)

    miner._generate_qc(matrix, spots_df, [1, 2], [0, 1], FOV_ID)

    debug_path = tmp_path / "Position7" / "output_pystar" / "extraction" / "debug_intensities_fov_7.csv"
    assert debug_path.exists()
    if n_spots == 0:
        assert captured_indices is None
    else:
        assert captured_indices is not None
        assert set(captured_indices).issubset(set(range(n_spots)))
        assert 1 <= len(captured_indices) <= n_spots


def test_signal_miner_qc_disabled_skips_sampling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    miner = build_minimal_signal_miner(output_dir=tmp_path, qc_enabled=False)
    called = False

    def fake_plot_spot_traces(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr("pystar.visualization.plot_spot_traces", fake_plot_spot_traces)
    miner._generate_qc(np.zeros((0, 1, 1), dtype=np.float32), empty_spot_table(), [1], [0], FOV_ID)

    assert called is False
    assert not (tmp_path / "Position7" / "output_pystar" / "extraction" / "debug_intensities_fov_7.csv").exists()
