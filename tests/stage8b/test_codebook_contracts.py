from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from pystar._artifact_schemas import (
    build_intensity_matrix_metadata_payload,
    build_intensity_matrix_spec,
    empty_spot_table,
    intensity_matrix_metadata_path,
)
from pystar._codebook_contracts import (
    CodebookContractError,
    build_reverse_lookups,
    compile_codebook_contract,
)
from pystar.decoding import Decoder
from pystar.infrastructure import ExperimentConfig
from pystar.io import get_fov_output_structure
from pystar.serialization import write_backend_metadata


FOV_ID = 7


def build_minimal_decoder_config(*, gene_list: Path, output_dir: Path, encoding_tables: dict[str, dict[str, int]] | None = None, topology: Any | None = None) -> ExperimentConfig:
    default_topology = SimpleNamespace(
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
                    topology=default_topology if topology is None else topology,
                    encoding_tables={"dinucleotide": {"AA": 1}} if encoding_tables is None else encoding_tables,
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


def build_minimal_decoder(*, output_dir: Path, gene_list: Path, encoding_tables: dict[str, dict[str, int]] | None = None, topology: Any | None = None) -> Decoder:
    decoder = Decoder.__new__(Decoder)
    decoder.cfg = build_minimal_decoder_config(
        gene_list=gene_list,
        output_dir=output_dir,
        encoding_tables=encoding_tables,
        topology=topology,
    )
    decoder.output_dir = output_dir
    return decoder


def test_compile_codebook_contract_accepts_headerless_gene_list_and_writes_debug_csv(tmp_path: Path) -> None:
    gene_list = tmp_path / "genes.csv"
    gene_list.write_text("GeneA,AA\nGeneB,AA\n", encoding="utf-8")
    cfg = build_minimal_decoder_config(gene_list=gene_list, output_dir=tmp_path)

    compiled = compile_codebook_contract(cfg.codebook, output_dir=tmp_path)

    assert list(compiled.dataframe["gene"]) == ["GeneA", "GeneB"]
    assert compiled.gene_map == {"0": "GeneB"}
    assert compiled.barcode_length == 1
    debug_path = tmp_path / "compiled_codebook_debug.csv"
    assert debug_path.exists()
    debug_df = pd.read_csv(debug_path)
    assert list(debug_df["gene"]) == ["GeneA", "GeneB"]
    assert list(debug_df["barcode"]) == [0, 0]


def test_compile_codebook_contract_accepts_headered_gene_list(tmp_path: Path) -> None:
    gene_list = tmp_path / "genes.csv"
    gene_list.write_text("gene,seq\nGeneA,AA\n", encoding="utf-8")
    cfg = build_minimal_decoder_config(gene_list=gene_list, output_dir=tmp_path)

    compiled = compile_codebook_contract(cfg.codebook)

    assert compiled.gene_map == {"0": "GeneA"}
    assert list(compiled.dataframe["gene"]) == ["GeneA"]
    assert "processed_seq" in compiled.dataframe.columns


def test_compile_codebook_contract_preserves_reverse_string_and_csv_slice_semantics(tmp_path: Path) -> None:
    gene_list = tmp_path / "genes.csv"
    gene_list.write_text("GeneA,AAGTCC\n", encoding="utf-8")
    topology = SimpleNamespace(
        func="reverse_string",
        structure=[
            SimpleNamespace(id="seg1", rounds=[1], csv_slice=[1, 2], anchor_base=None, encoding_table="pair"),
            SimpleNamespace(id="seg2", rounds=[2], csv_slice=[3, 4], anchor_base=None, encoding_table="pair"),
            SimpleNamespace(id="seg3", rounds=[3], csv_slice=[5, 6], anchor_base=None, encoding_table="pair"),
        ],
        physical_order=["seg1", "seg2", "seg3"],
    )
    cfg = build_minimal_decoder_config(
        gene_list=gene_list,
        output_dir=tmp_path,
        encoding_tables={"pair": {"CC": 1, "TG": 2, "AA": 3}},
        topology=topology,
    )

    compiled = compile_codebook_contract(cfg.codebook)

    assert list(compiled.dataframe["processed_seq"]) == ["CCTGAA"]
    assert list(compiled.dataframe["barcode"]) == ["012"]
    assert compiled.gene_map == {"012": "GeneA"}


def test_compile_codebook_contract_rejects_blank_gene_or_seq_fields(tmp_path: Path) -> None:
    gene_list = tmp_path / "genes.csv"
    gene_list.write_text("gene,seq\nGeneA,\n", encoding="utf-8")
    cfg = build_minimal_decoder_config(gene_list=gene_list, output_dir=tmp_path)

    with pytest.raises(CodebookContractError, match="Codebook gene-map contract error.*gene and seq values must be non-empty"):
        compile_codebook_contract(cfg.codebook)


def test_compile_codebook_contract_rejects_missing_gene_or_seq_header_fields(tmp_path: Path) -> None:
    gene_list = tmp_path / "genes.csv"
    gene_list.write_text("gene\nGeneA\n", encoding="utf-8")
    cfg = build_minimal_decoder_config(gene_list=gene_list, output_dir=tmp_path)

    with pytest.raises(CodebookContractError, match="Codebook gene-map contract error.*missing required columns.*seq"):
        compile_codebook_contract(cfg.codebook)


def test_compile_codebook_contract_rejects_empty_per_table_mapping(tmp_path: Path) -> None:
    gene_list = tmp_path / "genes.csv"
    gene_list.write_text("GeneA,AA\n", encoding="utf-8")
    cfg = build_minimal_decoder_config(
        gene_list=gene_list,
        output_dir=tmp_path,
        encoding_tables={"dinucleotide": {}},
    )

    with pytest.raises(CodebookContractError, match="Codebook gene-map contract error: encoding table 'dinucleotide' is empty or malformed"):
        compile_codebook_contract(cfg.codebook)


def test_compile_codebook_contract_rejects_missing_topology_segment_reference(tmp_path: Path) -> None:
    gene_list = tmp_path / "genes.csv"
    gene_list.write_text("GeneA,AA\n", encoding="utf-8")
    topology = SimpleNamespace(
        func="none",
        structure=[],
        physical_order=["missing_seq"],
    )
    cfg = build_minimal_decoder_config(gene_list=gene_list, output_dir=tmp_path, topology=topology)

    with pytest.raises(CodebookContractError, match="Topology physical_order references undefined segment ID: missing_seq"):
        compile_codebook_contract(cfg.codebook)


def test_compile_codebook_contract_rejects_missing_encoding_table_reference(tmp_path: Path) -> None:
    gene_list = tmp_path / "genes.csv"
    gene_list.write_text("GeneA,AA\n", encoding="utf-8")
    topology = SimpleNamespace(
        func="none",
        structure=[
            SimpleNamespace(
                id="seq",
                rounds=[1, 2],
                csv_slice=[1, 2],
                anchor_base=None,
                encoding_table="missing_table",
            )
        ],
        physical_order=["seq"],
    )
    cfg = build_minimal_decoder_config(gene_list=gene_list, output_dir=tmp_path, topology=topology)

    with pytest.raises(CodebookContractError, match="topology segment 'seq' references missing encoding table 'missing_table'"):
        compile_codebook_contract(cfg.codebook)


def test_compile_codebook_contract_rejects_zero_valid_barcodes(tmp_path: Path) -> None:
    gene_list = tmp_path / "genes.csv"
    gene_list.write_text("GeneA,A\n", encoding="utf-8")
    cfg = build_minimal_decoder_config(gene_list=gene_list, output_dir=tmp_path)

    with pytest.raises(CodebookContractError, match="compiled to zero valid barcodes"):
        compile_codebook_contract(cfg.codebook)


def test_decoder_init_consumes_compiled_codebook_and_preserves_compatibility_attributes(tmp_path: Path) -> None:
    gene_list = tmp_path / "genes.csv"
    gene_list.write_text("GeneA,AA\n", encoding="utf-8")
    decoder = Decoder(build_minimal_decoder_config(gene_list=gene_list, output_dir=tmp_path))

    assert decoder.compiled_codebook.gene_map == {"0": "GeneA"}
    assert decoder.gene_map == decoder.compiled_codebook.gene_map
    assert decoder.barcode_map.equals(decoder.compiled_codebook.dataframe)
    assert decoder.reverse_lookups == decoder.compiled_codebook.reverse_lookups
    assert (tmp_path / "compiled_codebook_debug.csv").exists()


def test_decoder_compile_codebook_wrapper_preserves_legacy_tuple_surface(tmp_path: Path) -> None:
    gene_list = tmp_path / "genes.csv"
    gene_list.write_text("GeneA,AA\n", encoding="utf-8")
    decoder = build_minimal_decoder(output_dir=tmp_path, gene_list=gene_list)

    gene_map, compiled_df = decoder._compile_codebook()

    assert gene_map == {"0": "GeneA"}
    assert list(compiled_df["gene"]) == ["GeneA"]
    assert (tmp_path / "compiled_codebook_debug.csv").exists()


def test_decode_fov_writes_canonical_debug_csv_while_preserving_legacy_root_copy(tmp_path: Path) -> None:
    gene_list = tmp_path / "genes.csv"
    gene_list.write_text("GeneA,AA\n", encoding="utf-8")
    decoder = Decoder(build_minimal_decoder_config(gene_list=gene_list, output_dir=tmp_path))
    legacy_debug_path = tmp_path / "compiled_codebook_debug.csv"
    assert legacy_debug_path.exists()

    paths = get_fov_output_structure(tmp_path, FOV_ID)
    spots_path = paths["spots"] / f"spots_fov_{FOV_ID}.csv"
    matrix_path = paths["extraction"] / f"intensity_matrix_fov_{FOV_ID}.npy"
    spec = build_intensity_matrix_spec(fov_id=FOV_ID, n_spots=0, rounds=[1, 2], channels=[0])

    empty_spot_table(extra_columns=("channel", "fov", "algo")).to_csv(spots_path, index=False)
    np.save(matrix_path, np.zeros(spec.expected_shape, dtype=np.float32))
    write_backend_metadata(
        intensity_matrix_metadata_path(matrix_path),
        build_intensity_matrix_metadata_payload(spec),
    )

    decoded = decoder.decode_fov(FOV_ID)

    canonical_debug_path = paths["root"] / "compiled_codebook_debug.csv"
    assert len(decoded) == 0
    assert canonical_debug_path.exists()
    pd.testing.assert_frame_equal(
        pd.read_csv(legacy_debug_path),
        pd.read_csv(canonical_debug_path),
    )


def test_duplicate_barcode_behavior_remains_last_write_wins(tmp_path: Path) -> None:
    gene_list = tmp_path / "genes.csv"
    gene_list.write_text("GeneA,AA\nGeneB,AA\n", encoding="utf-8")
    cfg = build_minimal_decoder_config(gene_list=gene_list, output_dir=tmp_path)

    compiled = compile_codebook_contract(cfg.codebook)

    assert list(compiled.dataframe["barcode"]) == ["0", "0"]
    assert compiled.gene_map == {"0": "GeneB"}


def test_reverse_lookup_builder_keeps_legacy_ambiguity_failure(tmp_path: Path) -> None:
    _ = tmp_path
    with pytest.raises(ValueError, match="Ambiguous encoding in table"):
        build_reverse_lookups({"ambiguous": {"AA": 1, "AG": 1}}, 1)
