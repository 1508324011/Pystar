"""Private codebook compilation contracts for decoder runtime.

This module owns the fail-loud gene-list, topology, and encoding-table
validation used to compile PyStar codebooks. It preserves existing userspace
behavior: the compiled debug CSV path remains unchanged, compiled gene maps
still use legacy ``dict(zip(...))`` last-write-wins semantics for duplicate
barcodes, and decoder runtime behavior is unchanged for valid inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, cast

import pandas as pd


GENE_HEADER_NAMES = {"gene", "gene_name", "genename", "name"}
SEQUENCE_HEADER_NAMES = {"seq", "sequence"}
ERROR_BARCODE = "ERROR_LEN"


class CodebookContractError(ValueError):
    """Fail-loud private codebook contract error."""


@dataclass(frozen=True)
class CompiledCodebook:
    """Validated compiled codebook consumed by :class:`pystar.decoding.Decoder`."""

    gene_list_path: Path
    dataframe: pd.DataFrame
    gene_map: dict[str, str]
    reverse_lookups: dict[str, dict[tuple[str, int], str]]
    barcode_length: int

    def write_debug_csv(self, output_dir: Path) -> Path:
        """Persist the legacy debug CSV payload without changing its path/name."""

        output_dir.mkdir(parents=True, exist_ok=True)
        debug_path = output_dir / "compiled_codebook_debug.csv"
        self.dataframe.to_csv(debug_path, index=False)
        print(f"   -> Compiled {len(self.dataframe)} barcodes. Debug info saved to {debug_path.name}")
        return debug_path


def _empty_gene_list_error(gene_list_path: Path) -> CodebookContractError:
    return CodebookContractError(
        f"Codebook gene-map contract error: gene list is empty at {gene_list_path}. "
        "Decoder requires at least one gene/sequence row that can compile to a barcode."
    )


def _missing_columns_error(gene_list_path: Path, missing_columns: list[str]) -> CodebookContractError:
    return CodebookContractError(
        f"Codebook gene-map contract error at {gene_list_path}: missing required columns "
        f"{missing_columns}. Expected columns ['gene', 'seq'] or a two-column gene list."
    )


def _non_empty_row_error(gene_list_path: Path) -> CodebookContractError:
    return CodebookContractError(
        f"Codebook gene-map contract error at {gene_list_path}: gene and seq values must be non-empty "
        "for every row before barcode compilation."
    )


def _header_alias_lookup(columns: list[Any], aliases: set[str]) -> Any | None:
    for column in columns:
        normalized = str(column).strip().lower()
        if normalized in aliases:
            return column
    return None


def _normalize_headered_gene_list(df: pd.DataFrame, gene_list_path: Path) -> pd.DataFrame | None:
    columns = list(df.columns)
    gene_column = _header_alias_lookup(columns, GENE_HEADER_NAMES)
    sequence_column = _header_alias_lookup(columns, SEQUENCE_HEADER_NAMES)

    if gene_column is None and sequence_column is None:
        return None

    missing_columns: list[str] = []
    if gene_column is None:
        missing_columns.append("gene")
    if sequence_column is None:
        missing_columns.append("seq")
    if missing_columns:
        raise _missing_columns_error(gene_list_path, missing_columns)

    normalized = cast(pd.DataFrame, df[[gene_column, sequence_column]].copy())
    normalized.columns = ["gene", "seq"]
    return normalized


def _drop_optional_header_row(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return df
    first_gene = str(df.iloc[0]["gene"]).strip().lower()
    first_seq = str(df.iloc[0]["seq"]).strip().lower()
    if first_gene in GENE_HEADER_NAMES and first_seq in SEQUENCE_HEADER_NAMES:
        return cast(pd.DataFrame, df.iloc[1:].reset_index(drop=True))
    return df


def _load_gene_list_dataframe(gene_list_path: Path) -> pd.DataFrame:
    if not gene_list_path.exists():
        raise FileNotFoundError(f"Gene list not found: {gene_list_path}")

    try:
        headered_candidate = pd.read_csv(gene_list_path)
    except pd.errors.EmptyDataError as exc:
        raise _empty_gene_list_error(gene_list_path) from exc
    except Exception:
        headered_candidate = None

    if isinstance(headered_candidate, pd.DataFrame):
        normalized_headered = _normalize_headered_gene_list(headered_candidate, gene_list_path)
        if normalized_headered is not None:
            df_genes = normalized_headered
        else:
            try:
                df_genes = cast(pd.DataFrame, pd.read_csv(gene_list_path, header=None, names=["gene", "seq"]))
            except pd.errors.EmptyDataError as exc:
                raise _empty_gene_list_error(gene_list_path) from exc
    else:
        try:
            df_genes = cast(pd.DataFrame, pd.read_csv(gene_list_path, header=None, names=["gene", "seq"]))
        except pd.errors.EmptyDataError as exc:
            raise _empty_gene_list_error(gene_list_path) from exc

    df_genes = _drop_optional_header_row(df_genes)
    if len(df_genes) == 0:
        raise _empty_gene_list_error(gene_list_path)

    missing_codebook_columns = [column for column in ("gene", "seq") if column not in df_genes.columns]
    if missing_codebook_columns:
        raise _missing_columns_error(gene_list_path, missing_codebook_columns)

    invalid_gene_mask = df_genes["gene"].isna() | (df_genes["gene"].astype(str).str.strip() == "")
    invalid_seq_mask = df_genes["seq"].isna() | (df_genes["seq"].astype(str).str.strip() == "")
    if bool(invalid_gene_mask.any()) or bool(invalid_seq_mask.any()):
        raise _non_empty_row_error(gene_list_path)

    normalized = df_genes.copy()
    normalized["gene"] = normalized["gene"].astype(str)
    normalized["seq"] = normalized["seq"].astype(str)
    return normalized


def validate_encoding_tables(
    encoding_tables: Any,
    *,
    gene_list_path: Path | None = None,
) -> dict[str, dict[str, int]]:
    """Validate configured encoding tables without changing runtime semantics."""

    if not isinstance(encoding_tables, dict) or not encoding_tables:
        raise CodebookContractError(
            "Codebook gene-map contract error: encoding_tables is empty. "
            "Decoder requires at least one non-empty encoding table before barcode compilation."
        )

    validated: dict[str, dict[str, int]] = {}
    for table_name, mapping in encoding_tables.items():
        if not isinstance(mapping, Mapping) or not mapping:
            raise CodebookContractError(
                f"Codebook gene-map contract error: encoding table {table_name!r} is empty or malformed. "
                "Decoder requires non-empty base-to-color mappings before barcode compilation."
            )
        normalized_mapping: dict[str, int] = {}
        for raw_key, raw_value in mapping.items():
            if isinstance(raw_value, bool) or not isinstance(raw_value, int):
                raise CodebookContractError(
                    f"Codebook gene-map contract error: encoding table {table_name!r} contains a non-integer "
                    f"color value for base pattern {raw_key!r}."
                )
            normalized_mapping[str(raw_key)] = int(raw_value)
        validated[str(table_name)] = normalized_mapping
    return validated


def create_encoder(mapping: Mapping[str, int], base_idx: int) -> Callable[[str], str]:
    """Build a legacy-compatible sequence-to-color encoder closure."""

    keys = list(mapping.keys())
    if not keys:
        raise CodebookContractError(
            "Codebook gene-map contract error: encoding table is empty or malformed. "
            "Decoder requires non-empty base-to-color mappings before barcode compilation."
        )
    window_size = len(keys[0])
    normalized_map = {k: str(v - base_idx) for k, v in mapping.items()}

    def encode(seq: str) -> str:
        if len(seq) == window_size:
            return normalized_map.get(seq, ".")
        if len(seq) < window_size:
            return "." * len(seq)
        return "".join(normalized_map.get(seq[i : i + window_size], ".") for i in range(len(seq) - window_size + 1))

    return encode


def build_single_reverse_lookup(encoding_table: Mapping[str, int], base_idx: int) -> dict[tuple[str, int], str]:
    """Build the legacy-compatible reverse lookup for one encoding table."""

    reverse: dict[tuple[str, int], str] = {}
    for base_pair, color in encoding_table.items():
        if len(base_pair) != 2:
            continue

        key = (base_pair[0], color - base_idx)
        next_base = base_pair[1]
        if key in reverse and reverse[key] != next_base:
            raise ValueError(f"Ambiguous encoding in table: {key} maps to both '{reverse[key]}' and '{next_base}'")
        reverse[key] = next_base
    return reverse


def build_reverse_lookups(
    encoding_tables: Any,
    base_idx: int,
    *,
    gene_list_path: Path | None = None,
) -> dict[str, dict[tuple[str, int], str]]:
    """Build reverse lookups from validated encoding tables."""

    validated_tables = validate_encoding_tables(encoding_tables, gene_list_path=gene_list_path)
    return {
        table_name: build_single_reverse_lookup(mapping, base_idx)
        for table_name, mapping in validated_tables.items()
    }


def _apply_topology_transform(df_genes: pd.DataFrame, topo: Any) -> pd.DataFrame:
    normalized = df_genes.copy()
    if getattr(topo, "func", "none") == "reverse_string":
        print(" [Decoder] Applying Topology: Reverse Sequence")
        normalized["processed_seq"] = normalized["seq"].apply(lambda s: s[::-1])
    else:
        normalized["processed_seq"] = normalized["seq"]
    return normalized


def _segment_definitions(topo: Any) -> dict[str, Any]:
    return {seg.id: seg for seg in topo.structure}


def _validate_topology_segments(
    topo: Any,
    *,
    encoders: Mapping[str, Callable[[str], str]],
    gene_list_path: Path,
) -> dict[str, Any]:
    segment_defs = _segment_definitions(topo)
    for seg_id in topo.physical_order:
        if seg_id not in segment_defs:
            raise CodebookContractError(
                f"Codebook gene-map contract error at {gene_list_path}: "
                f"Topology physical_order references undefined segment ID: {seg_id}"
            )
        encoding_table_name = segment_defs[seg_id].encoding_table
        if encoding_table_name not in encoders:
            raise CodebookContractError(
                f"Codebook gene-map contract error: topology segment {seg_id!r} references missing "
                f"encoding table {encoding_table_name!r}. Available encoding tables: {sorted(encoders)}"
            )
    return segment_defs


def _assemble_barcode(
    seq: str,
    *,
    topo: Any,
    segment_defs: Mapping[str, Any],
    encoders: Mapping[str, Callable[[str], str]],
) -> str:
    full_barcode = ""
    for seg_id in topo.physical_order:
        seg_def = segment_defs[seg_id]
        start_1b, end_1b = seg_def.csv_slice
        py_start = max(0, start_1b - 1)
        py_end = end_1b
        if py_end > len(seq):
            return ERROR_BARCODE

        sub_seq = seq[py_start:py_end]
        encoded_chunk = encoders[seg_def.encoding_table](sub_seq)
        _expected_rounds = len(seg_def.rounds)
        _ = _expected_rounds
        full_barcode += encoded_chunk
    return full_barcode


def compile_codebook_contract(codebook_cfg: Any, *, output_dir: Path | None = None) -> CompiledCodebook:
    """Compile a validated private codebook contract from config-like input."""

    gene_list_path = Path(codebook_cfg.gene_list)
    topo = codebook_cfg.topology
    df_genes = _load_gene_list_dataframe(gene_list_path)
    transformed = _apply_topology_transform(df_genes, topo)

    validated_tables = validate_encoding_tables(codebook_cfg.encoding_tables, gene_list_path=gene_list_path)
    encoders = {
        table_name: create_encoder(mapping, codebook_cfg.channel_base_index)
        for table_name, mapping in validated_tables.items()
    }
    segment_defs = _validate_topology_segments(topo, encoders=encoders, gene_list_path=gene_list_path)
    reverse_lookups = build_reverse_lookups(
        validated_tables,
        codebook_cfg.channel_base_index,
        gene_list_path=gene_list_path,
    )

    compiled_df = transformed.copy()
    compiled_df["barcode"] = compiled_df["processed_seq"].apply(
        lambda seq: _assemble_barcode(seq, topo=topo, segment_defs=segment_defs, encoders=encoders)
    )

    valid_df = cast(pd.DataFrame, compiled_df[compiled_df["barcode"] != ERROR_BARCODE].copy())
    if len(valid_df) < len(compiled_df):
        print(f" [Warning] {len(compiled_df) - len(valid_df)} genes failed barcode generation (Check sequence lengths).")

    if len(valid_df) == 0:
        raise CodebookContractError(
            f"Codebook gene-map contract error: gene list at {gene_list_path} compiled to zero valid barcodes. "
            "Check topology csv_slice/physical_order and gene sequence lengths before decoding."
        )

    gene_map = dict(zip(valid_df["barcode"], valid_df["gene"]))
    if not gene_map:
        raise CodebookContractError(
            f"Codebook gene-map contract error: compiled gene map is empty for {gene_list_path}. "
            "Decoder requires at least one barcode-to-gene mapping."
        )

    compiled = CompiledCodebook(
        gene_list_path=gene_list_path,
        dataframe=valid_df,
        gene_map=gene_map,
        reverse_lookups=reverse_lookups,
        barcode_length=len(next(iter(gene_map.keys()))),
    )
    if output_dir is not None:
        compiled.write_debug_csv(output_dir)
    return compiled
