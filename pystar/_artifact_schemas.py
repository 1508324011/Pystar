"""Private schema helpers for canonical PyStar runtime artifacts.

This module names the persisted spot/intensity/decoded contracts that already
exist on disk without changing userspace paths, filenames, or public imports.
The boundary is intentionally small: stage modules keep filesystem ownership,
while this module validates the dataframes/arrays that cross those boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd


INTENSITY_MATRIX_METADATA_SCHEMA_NAME = "pystar.private.intensity_matrix_metadata"
INTENSITY_MATRIX_METADATA_SCHEMA_VERSION = 1
SPOT_ROW_LINEAGE_SCHEMA_NAME = "pystar.private.spot_row_lineage"
SPOT_ROW_LINEAGE_SCHEMA_VERSION = 1
SPOT_ROW_LINEAGE_COLUMNS = ("z", "y", "x", "intensity", "channel", "fov", "algo")
SPOT_ROW_LINEAGE_NUMERIC_COLUMNS = ("z", "y", "x", "intensity", "channel", "fov")
SPOT_ROW_LINEAGE_OBJECT_COLUMNS = ("algo",)


def _path_text(path: Path | str | None) -> str:
    if path is None:
        return "<in-memory>"
    return str(path)


def _dedupe_columns(*column_groups: Sequence[str]) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for group in column_groups:
        for column in group:
            if column not in seen:
                ordered.append(column)
                seen.add(column)
    return tuple(ordered)


def _schema_error(
    artifact_name: str,
    *,
    fov_id: int,
    path: Path | str | None,
    context: str,
    detail: str,
    expected: str,
) -> ValueError:
    return ValueError(
        f"FOV {fov_id} {artifact_name} schema error during {context} at {_path_text(path)}: "
        f"{detail}. Expected schema: {expected}"
    )


def wrap_table_read_error(
    exc: Exception,
    artifact_name: str,
    *,
    fov_id: int,
    path: Path | str | None,
    context: str,
    expected: str,
) -> ValueError:
    """Normalize CSV/TSV parse failures into the Stage 7A fail-loud schema form."""

    return _schema_error(
        artifact_name,
        fov_id=fov_id,
        path=path,
        context=context,
        detail=f"unable to parse artifact table ({type(exc).__name__}: {exc})",
        expected=expected,
    )


def wrap_payload_read_error(
    exc: Exception,
    artifact_name: str,
    *,
    fov_id: int,
    path: Path | str | None,
    context: str,
    expected: str,
) -> ValueError:
    """Normalize JSON/text payload failures into the Stage 7 fail-loud form."""

    return _schema_error(
        artifact_name,
        fov_id=fov_id,
        path=path,
        context=context,
        detail=f"unable to load serialized payload ({type(exc).__name__}: {exc})",
        expected=expected,
    )


def wrap_array_read_error(
    exc: Exception,
    artifact_name: str,
    *,
    fov_id: int,
    path: Path | str | None,
    context: str,
    expected: str,
) -> ValueError:
    """Normalize NumPy array load failures into the Stage 7 fail-loud form."""

    return _schema_error(
        artifact_name,
        fov_id=fov_id,
        path=path,
        context=context,
        detail=f"unable to load array artifact ({type(exc).__name__}: {exc})",
        expected=expected,
    )


def _empty_dataframe(columns: Sequence[str], dtype_map: dict[str, Any]) -> pd.DataFrame:
    payload: dict[str, pd.Series] = {}
    for column in columns:
        payload[column] = pd.Series(dtype=dtype_map.get(column, object))
    return pd.DataFrame(payload)


def _validate_table_shape(
    df: Any,
    *,
    artifact_name: str,
    fov_id: int,
    path: Path | str | None,
    context: str,
) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise _schema_error(
            artifact_name,
            fov_id=fov_id,
            path=path,
            context=context,
            detail=f"artifact is not a pandas DataFrame (got {type(df)!r})",
            expected="a pandas DataFrame with explicit artifact columns",
        )
    return df.copy()


def _validate_required_columns(
    df: pd.DataFrame,
    *,
    required_columns: Sequence[str],
    artifact_name: str,
    fov_id: int,
    path: Path | str | None,
    context: str,
    expected: str,
) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise _schema_error(
            artifact_name,
            fov_id=fov_id,
            path=path,
            context=context,
            detail=f"missing required columns {missing}",
            expected=expected,
        )


def _coerce_numeric_columns(
    df: pd.DataFrame,
    *,
    numeric_columns: Sequence[str],
    artifact_name: str,
    fov_id: int,
    path: Path | str | None,
    context: str,
    expected: str,
    require_non_null: bool,
) -> pd.DataFrame:
    normalized = df.copy()
    for column in numeric_columns:
        try:
            normalized[column] = pd.to_numeric(normalized[column], errors="raise")
        except Exception as exc:  # pragma: no cover - pandas owns exact exception type
            raise _schema_error(
                artifact_name,
                fov_id=fov_id,
                path=path,
                context=context,
                detail=f"column {column!r} contains non-numeric values",
                expected=expected,
            ) from exc
        if require_non_null and bool(normalized[column].isna().any()):
            raise _schema_error(
                artifact_name,
                fov_id=fov_id,
                path=path,
                context=context,
                detail=f"column {column!r} contains missing values",
                expected=expected,
            )
    return normalized


def _validate_non_null_columns(
    df: pd.DataFrame,
    *,
    non_null_columns: Sequence[str],
    artifact_name: str,
    fov_id: int,
    path: Path | str | None,
    context: str,
    expected: str,
) -> None:
    for column in non_null_columns:
        if bool(df[column].isna().any()):
            raise _schema_error(
                artifact_name,
                fov_id=fov_id,
                path=path,
                context=context,
                detail=f"column {column!r} contains missing values",
                expected=expected,
            )


@dataclass(frozen=True)
class SpotTableSchema:
    """Contract for ``spots/spots_fov_{fov_id}.csv``."""

    required_columns: tuple[str, ...] = ("z", "y", "x", "intensity")
    compatibility_columns: tuple[str, ...] = ("channel", "fov", "algo")
    required_numeric_columns: tuple[str, ...] = ("z", "y", "x", "intensity")
    optional_numeric_columns: tuple[str, ...] = ("channel", "fov")

    def expected_description(self) -> str:
        return (
            f"required columns {list(self.required_columns)}; accepted compatibility columns "
            f"{list(self.compatibility_columns)}; numeric columns {list(self.required_numeric_columns)}"
        )


@dataclass(frozen=True)
class IntensityMatrixSpec:
    """Contract for ``extraction/intensity_matrix_fov_{fov_id}.npy``."""

    fov_id: int
    n_spots: int
    n_rounds: int
    n_channels: int
    rounds: tuple[int, ...]
    channels: tuple[int, ...]

    @property
    def expected_shape(self) -> tuple[int, int, int]:
        return (self.n_spots, self.n_rounds, self.n_channels)

    def expected_description(self) -> str:
        return (
            f"numeric rank-3 array with shape {self.expected_shape} == "
            f"(N_spots, N_rounds, N_seq_channels), rounds={list(self.rounds)}, channels={list(self.channels)}"
        )


@dataclass(frozen=True)
class SpotRowLineage:
    """Explicit row-identity contract for spot-table order across stages."""

    spot_count: int
    fingerprint: str
    columns: tuple[str, ...] = SPOT_ROW_LINEAGE_COLUMNS

    def expected_description(self) -> str:
        return spot_row_lineage_expected_description()


def build_spot_row_lineage(
    df: Any,
    *,
    fov_id: int,
    path: Path | str | None,
    context: str,
) -> SpotRowLineage:
    """Build a deterministic spot-row fingerprint from the validated spot table."""

    normalized = validate_spot_table(
        df,
        fov_id=fov_id,
        path=path,
        context=context,
    )
    row_count = int(len(normalized))
    digest = hashlib.sha256()
    digest.update(f"{SPOT_ROW_LINEAGE_SCHEMA_NAME}:{SPOT_ROW_LINEAGE_SCHEMA_VERSION}".encode("utf-8"))
    digest.update(np.asarray([row_count], dtype="<i8").tobytes())

    for column in SPOT_ROW_LINEAGE_NUMERIC_COLUMNS:
        digest.update(column.encode("utf-8"))
        digest.update(b"\0")
        if column in normalized.columns:
            numeric_values = np.asarray(pd.to_numeric(normalized[column], errors="raise"), dtype="<f8")
            mask = np.asarray(~np.isnan(numeric_values), dtype=np.uint8)
            values = np.nan_to_num(numeric_values, nan=0.0, copy=True)
        else:
            mask = np.zeros(row_count, dtype=np.uint8)
            values = np.zeros(row_count, dtype="<f8")
        digest.update(mask.tobytes())
        digest.update(values.tobytes())

    for column in SPOT_ROW_LINEAGE_OBJECT_COLUMNS:
        digest.update(column.encode("utf-8"))
        digest.update(b"\0")
        if column in normalized.columns:
            series = normalized[column]
        else:
            series = pd.Series([None] * row_count, dtype=object)
        mask = np.asarray(~series.isna(), dtype=np.uint8)
        digest.update(mask.tobytes())
        for raw_value in series:
            if pd.isna(raw_value):
                continue
            encoded = str(raw_value).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little", signed=False))
            digest.update(encoded)

    return SpotRowLineage(
        spot_count=row_count,
        fingerprint=f"sha256:{digest.hexdigest()}",
    )


def build_spot_row_lineage_payload(lineage: SpotRowLineage) -> dict[str, Any]:
    return {
        "schema_name": SPOT_ROW_LINEAGE_SCHEMA_NAME,
        "schema_version": SPOT_ROW_LINEAGE_SCHEMA_VERSION,
        "spot_count": int(lineage.spot_count),
        "columns": [str(column) for column in lineage.columns],
        "fingerprint": lineage.fingerprint,
    }


def validate_spot_row_lineage_payload(
    payload: Any,
    *,
    fov_id: int,
    path: Path | str | None,
    context: str,
) -> SpotRowLineage:
    artifact_name = "intensity matrix metadata sidecar"
    expected = spot_row_lineage_expected_description()
    if not isinstance(payload, dict):
        raise _schema_error(
            artifact_name,
            fov_id=fov_id,
            path=path,
            context=context,
            detail="field 'spot_row_lineage' must be a JSON object",
            expected=expected,
        )

    schema_name = payload.get("schema_name")
    if schema_name != SPOT_ROW_LINEAGE_SCHEMA_NAME:
        raise _schema_error(
            artifact_name,
            fov_id=fov_id,
            path=path,
            context=context,
            detail=(
                f"field 'spot_row_lineage.schema_name' is {schema_name!r}; "
                f"expected {SPOT_ROW_LINEAGE_SCHEMA_NAME!r}"
            ),
            expected=expected,
        )

    schema_version = _coerce_int_field(
        payload,
        "schema_version",
        artifact_name=artifact_name,
        fov_id=fov_id,
        path=path,
        context=context,
        expected=expected,
    )
    if schema_version != SPOT_ROW_LINEAGE_SCHEMA_VERSION:
        raise _schema_error(
            artifact_name,
            fov_id=fov_id,
            path=path,
            context=context,
            detail=(
                f"field 'spot_row_lineage.schema_version' is {schema_version}; "
                f"expected {SPOT_ROW_LINEAGE_SCHEMA_VERSION}"
            ),
            expected=expected,
        )

    spot_count = _coerce_int_field(
        payload,
        "spot_count",
        artifact_name=artifact_name,
        fov_id=fov_id,
        path=path,
        context=context,
        expected=expected,
    )
    if spot_count < 0:
        raise _schema_error(
            artifact_name,
            fov_id=fov_id,
            path=path,
            context=context,
            detail=f"field 'spot_row_lineage.spot_count' must be non-negative, got {spot_count}",
            expected=expected,
        )

    columns = _coerce_string_sequence(
        payload,
        "columns",
        artifact_name=artifact_name,
        fov_id=fov_id,
        path=path,
        context=context,
        expected=expected,
        expected_length=len(SPOT_ROW_LINEAGE_COLUMNS),
    )
    if columns != SPOT_ROW_LINEAGE_COLUMNS:
        raise _schema_error(
            artifact_name,
            fov_id=fov_id,
            path=path,
            context=context,
            detail=(
                f"field 'spot_row_lineage.columns' is {list(columns)}; "
                f"expected {list(SPOT_ROW_LINEAGE_COLUMNS)}"
            ),
            expected=expected,
        )

    fingerprint = payload.get("fingerprint")
    if not isinstance(fingerprint, str) or not _is_canonical_sha256(fingerprint):
        raise _schema_error(
            artifact_name,
            fov_id=fov_id,
            path=path,
            context=context,
            detail="field 'spot_row_lineage.fingerprint' must use canonical 'sha256:<hex>' form",
            expected=expected,
        )

    return SpotRowLineage(spot_count=int(spot_count), fingerprint=fingerprint, columns=columns)


def spot_row_lineage_from_intensity_metadata_payload(
    payload: dict[str, Any],
    *,
    fov_id: int,
    path: Path | str | None,
    context: str,
) -> SpotRowLineage | None:
    raw_payload = payload.get("spot_row_lineage")
    if raw_payload is None:
        return None
    return validate_spot_row_lineage_payload(
        raw_payload,
        fov_id=fov_id,
        path=path,
        context=context,
    )


def validate_spot_row_lineage_consumer_contract(
    persisted_lineage: SpotRowLineage,
    consumer_lineage: SpotRowLineage,
    *,
    fov_id: int,
    path: Path | str | None,
    context: str,
    spot_path: Path | str | None,
) -> None:
    artifact_name = "intensity matrix metadata sidecar"
    expected = consumer_lineage.expected_description()
    if persisted_lineage.spot_count != consumer_lineage.spot_count:
        raise _schema_error(
            artifact_name,
            fov_id=fov_id,
            path=path,
            context=context,
            detail=(
                f"spot_row_lineage.spot_count={persisted_lineage.spot_count} but loaded spot table at "
                f"{_path_text(spot_path)} has {consumer_lineage.spot_count} rows"
            ),
            expected=expected,
        )
    if persisted_lineage.columns != consumer_lineage.columns:
        raise _schema_error(
            artifact_name,
            fov_id=fov_id,
            path=path,
            context=context,
            detail=(
                f"spot_row_lineage.columns={list(persisted_lineage.columns)} but loaded spot table at "
                f"{_path_text(spot_path)} canonicalizes to {list(consumer_lineage.columns)}"
            ),
            expected=expected,
        )
    if persisted_lineage.fingerprint != consumer_lineage.fingerprint:
        raise _schema_error(
            artifact_name,
            fov_id=fov_id,
            path=path,
            context=context,
            detail=(
                f"spot_row_lineage fingerprint mismatch for loaded spot table at {_path_text(spot_path)}; "
                f"metadata fingerprint {persisted_lineage.fingerprint!r} does not match loaded spot fingerprint "
                f"{consumer_lineage.fingerprint!r}. Spot row order/content changed after mining"
            ),
            expected=expected,
        )


def build_intensity_matrix_spec(
    *,
    fov_id: int,
    n_spots: int,
    rounds: Sequence[int],
    channels: Sequence[int],
) -> IntensityMatrixSpec:
    """Construct an intensity-matrix spec from stage-owned ordering facts."""

    coerced_n_spots = int(n_spots)
    if coerced_n_spots < 0:
        raise _schema_error(
            "intensity matrix",
            fov_id=int(fov_id),
            path=None,
            context="spec construction",
            detail=f"field 'n_spots' must be non-negative, got {coerced_n_spots}",
            expected="numeric rank-3 array with non-negative shape (N_spots, N_rounds, N_seq_channels)",
        )

    return IntensityMatrixSpec(
        fov_id=int(fov_id),
        n_spots=coerced_n_spots,
        n_rounds=len(rounds),
        n_channels=len(channels),
        rounds=tuple(int(value) for value in rounds),
        channels=tuple(int(value) for value in channels),
    )


def intensity_matrix_metadata_path(matrix_path: Path | str) -> Path:
    """Return the private metadata sidecar path for an intensity matrix file."""

    raw_path = Path(matrix_path)
    return raw_path.with_name(f"{raw_path.stem}_metadata.json")


def intensity_matrix_metadata_expected_description() -> str:
    return (
        "JSON object with "
        f"schema_name={INTENSITY_MATRIX_METADATA_SCHEMA_NAME!r}, "
        f"schema_version={INTENSITY_MATRIX_METADATA_SCHEMA_VERSION}, "
        "integer fields 'fov_id'/'n_spots', integer lists 'round_order'/'channel_order', "
        "integer list 'matrix_shape' == [N_spots, N_rounds, N_seq_channels], "
        "and optional object field 'spot_row_lineage' carrying the explicit spot-row fingerprint contract"
    )


def _coerce_int_field(
    payload: dict[str, Any],
    field_name: str,
    *,
    artifact_name: str,
    fov_id: int,
    path: Path | str | None,
    context: str,
    expected: str,
) -> int:
    value = payload.get(field_name)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise _schema_error(
            artifact_name,
            fov_id=fov_id,
            path=path,
            context=context,
            detail=f"field {field_name!r} must be an integer",
            expected=expected,
        )
    return int(value)


def _coerce_int_sequence(
    payload: dict[str, Any],
    field_name: str,
    *,
    artifact_name: str,
    fov_id: int,
    path: Path | str | None,
    context: str,
    expected: str,
    expected_length: int | None = None,
) -> tuple[int, ...]:
    raw_values = payload.get(field_name)
    if not isinstance(raw_values, list):
        raise _schema_error(
            artifact_name,
            fov_id=fov_id,
            path=path,
            context=context,
            detail=f"field {field_name!r} must be a JSON list of integers",
            expected=expected,
        )
    coerced: list[int] = []
    for index, value in enumerate(raw_values):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise _schema_error(
                artifact_name,
                fov_id=fov_id,
                path=path,
                context=context,
                detail=f"field {field_name!r} contains a non-integer value at index {index}",
                expected=expected,
            )
        coerced.append(int(value))
    if expected_length is not None and len(coerced) != expected_length:
        raise _schema_error(
            artifact_name,
            fov_id=fov_id,
            path=path,
            context=context,
            detail=f"field {field_name!r} has length {len(coerced)} but expected {expected_length}",
            expected=expected,
        )
    return tuple(coerced)


def _coerce_string_sequence(
    payload: dict[str, Any],
    field_name: str,
    *,
    artifact_name: str,
    fov_id: int,
    path: Path | str | None,
    context: str,
    expected: str,
    expected_length: int | None = None,
) -> tuple[str, ...]:
    raw_values = payload.get(field_name)
    if not isinstance(raw_values, list):
        raise _schema_error(
            artifact_name,
            fov_id=fov_id,
            path=path,
            context=context,
            detail=f"field {field_name!r} must be a JSON list of strings",
            expected=expected,
        )
    coerced: list[str] = []
    for index, value in enumerate(raw_values):
        if not isinstance(value, str) or not value:
            raise _schema_error(
                artifact_name,
                fov_id=fov_id,
                path=path,
                context=context,
                detail=f"field {field_name!r} contains a non-string value at index {index}",
                expected=expected,
            )
        coerced.append(value)
    if expected_length is not None and len(coerced) != expected_length:
        raise _schema_error(
            artifact_name,
            fov_id=fov_id,
            path=path,
            context=context,
            detail=f"field {field_name!r} has length {len(coerced)} but expected {expected_length}",
            expected=expected,
        )
    return tuple(coerced)


def _is_canonical_sha256(value: str) -> bool:
    if not value.startswith("sha256:"):
        return False
    digest = value[7:]
    return len(digest) == 64 and all(char in "0123456789abcdef" for char in digest)


def spot_row_lineage_expected_description() -> str:
    return (
        "JSON object with "
        f"schema_name={SPOT_ROW_LINEAGE_SCHEMA_NAME!r}, "
        f"schema_version={SPOT_ROW_LINEAGE_SCHEMA_VERSION}, "
        "integer field 'spot_count', string list 'columns', and string field "
        "'fingerprint' in canonical 'sha256:<hex>' form over validated spot-row order "
        f"for columns {list(SPOT_ROW_LINEAGE_COLUMNS)}"
    )


def build_intensity_matrix_metadata_payload(
    spec: IntensityMatrixSpec,
    *,
    spot_row_lineage: SpotRowLineage | None = None,
) -> dict[str, Any]:
    """Serialize the minimal persisted ordering facts for an intensity matrix."""

    payload: dict[str, Any] = {
        "schema_name": INTENSITY_MATRIX_METADATA_SCHEMA_NAME,
        "schema_version": INTENSITY_MATRIX_METADATA_SCHEMA_VERSION,
        "fov_id": int(spec.fov_id),
        "n_spots": int(spec.n_spots),
        "round_order": [int(value) for value in spec.rounds],
        "channel_order": [int(value) for value in spec.channels],
        "matrix_shape": [int(value) for value in spec.expected_shape],
    }
    if spot_row_lineage is not None:
        payload["spot_row_lineage"] = build_spot_row_lineage_payload(spot_row_lineage)
    return payload


def validate_intensity_matrix_metadata_payload(
    payload: Any,
    *,
    fov_id: int,
    path: Path | str | None,
    context: str,
) -> IntensityMatrixSpec:
    """Validate a persisted intensity-matrix metadata payload and recover its spec."""

    artifact_name = "intensity matrix metadata sidecar"
    expected = intensity_matrix_metadata_expected_description()
    if not isinstance(payload, dict):
        raise _schema_error(
            artifact_name,
            fov_id=fov_id,
            path=path,
            context=context,
            detail=f"artifact payload is not a JSON object (got {type(payload)!r})",
            expected=expected,
        )

    schema_name = payload.get("schema_name")
    if schema_name != INTENSITY_MATRIX_METADATA_SCHEMA_NAME:
        raise _schema_error(
            artifact_name,
            fov_id=fov_id,
            path=path,
            context=context,
            detail=(
                f"field 'schema_name' is {schema_name!r}; "
                f"expected {INTENSITY_MATRIX_METADATA_SCHEMA_NAME!r}"
            ),
            expected=expected,
        )

    schema_version = _coerce_int_field(
        payload,
        "schema_version",
        artifact_name=artifact_name,
        fov_id=fov_id,
        path=path,
        context=context,
        expected=expected,
    )
    if schema_version != INTENSITY_MATRIX_METADATA_SCHEMA_VERSION:
        raise _schema_error(
            artifact_name,
            fov_id=fov_id,
            path=path,
            context=context,
            detail=(
                f"field 'schema_version' is {schema_version}; "
                f"expected {INTENSITY_MATRIX_METADATA_SCHEMA_VERSION}"
            ),
            expected=expected,
        )

    persisted_fov_id = _coerce_int_field(
        payload,
        "fov_id",
        artifact_name=artifact_name,
        fov_id=fov_id,
        path=path,
        context=context,
        expected=expected,
    )
    if persisted_fov_id != int(fov_id):
        raise _schema_error(
            artifact_name,
            fov_id=fov_id,
            path=path,
            context=context,
            detail=(
                f"field 'fov_id' is {persisted_fov_id}; "
                f"requested decoder/miner FOV is {int(fov_id)}"
            ),
            expected=expected,
        )

    n_spots = _coerce_int_field(
        payload,
        "n_spots",
        artifact_name=artifact_name,
        fov_id=fov_id,
        path=path,
        context=context,
        expected=expected,
    )
    if n_spots < 0:
        raise _schema_error(
            artifact_name,
            fov_id=fov_id,
            path=path,
            context=context,
            detail=f"field 'n_spots' must be non-negative, got {n_spots}",
            expected=expected,
        )
    round_order = _coerce_int_sequence(
        payload,
        "round_order",
        artifact_name=artifact_name,
        fov_id=fov_id,
        path=path,
        context=context,
        expected=expected,
    )
    channel_order = _coerce_int_sequence(
        payload,
        "channel_order",
        artifact_name=artifact_name,
        fov_id=fov_id,
        path=path,
        context=context,
        expected=expected,
    )
    matrix_shape = _coerce_int_sequence(
        payload,
        "matrix_shape",
        artifact_name=artifact_name,
        fov_id=fov_id,
        path=path,
        context=context,
        expected=expected,
        expected_length=3,
    )
    negative_dimensions = [value for value in matrix_shape if value < 0]
    if negative_dimensions:
        raise _schema_error(
            artifact_name,
            fov_id=fov_id,
            path=path,
            context=context,
            detail=f"field 'matrix_shape' contains negative dimensions {negative_dimensions}",
            expected=expected,
        )

    spec = build_intensity_matrix_spec(
        fov_id=persisted_fov_id,
        n_spots=n_spots,
        rounds=round_order,
        channels=channel_order,
    )
    if matrix_shape != spec.expected_shape:
        raise _schema_error(
            artifact_name,
            fov_id=fov_id,
            path=path,
            context=context,
            detail=(
                f"field 'matrix_shape' is {list(matrix_shape)} but derived shape from "
                f"n_spots/round_order/channel_order is {list(spec.expected_shape)}"
            ),
            expected=expected,
        )

    if "spot_row_lineage" in payload:
        _ = validate_spot_row_lineage_payload(
            payload["spot_row_lineage"],
            fov_id=fov_id,
            path=path,
            context=context,
        )

    return spec


def validate_intensity_matrix_consumer_contract(
    persisted_spec: IntensityMatrixSpec,
    consumer_spec: IntensityMatrixSpec,
    *,
    path: Path | str | None,
    context: str,
    matrix_path: Path | str | None,
) -> None:
    """Fail loudly when persisted ordering facts disagree with current consumer assumptions."""

    artifact_name = "intensity matrix metadata sidecar"
    expected = consumer_spec.expected_description()
    if persisted_spec.fov_id != consumer_spec.fov_id:
        raise _schema_error(
            artifact_name,
            fov_id=consumer_spec.fov_id,
            path=path,
            context=context,
            detail=(
                f"persisted fov_id {persisted_spec.fov_id} does not match consumer fov_id "
                f"{consumer_spec.fov_id} for matrix {_path_text(matrix_path)}"
            ),
            expected=expected,
        )
    if persisted_spec.n_spots != consumer_spec.n_spots:
        raise _schema_error(
            artifact_name,
            fov_id=consumer_spec.fov_id,
            path=path,
            context=context,
            detail=(
                f"sidecar declares n_spots={persisted_spec.n_spots} but consumer expects "
                f"n_spots={consumer_spec.n_spots}; sidecar shape is "
                f"{list(persisted_spec.expected_shape)} and consumer expected shape is "
                f"{list(consumer_spec.expected_shape)} for matrix {_path_text(matrix_path)}"
            ),
            expected=expected,
        )
    if persisted_spec.rounds != consumer_spec.rounds:
        raise _schema_error(
            artifact_name,
            fov_id=consumer_spec.fov_id,
            path=path,
            context=context,
            detail=(
                f"round_order mismatch: sidecar declares {list(persisted_spec.rounds)} with "
                f"shape {list(persisted_spec.expected_shape)} but consumer expects "
                f"{list(consumer_spec.rounds)} with shape {list(consumer_spec.expected_shape)} "
                f"for matrix {_path_text(matrix_path)}"
            ),
            expected=expected,
        )
    if persisted_spec.channels != consumer_spec.channels:
        raise _schema_error(
            artifact_name,
            fov_id=consumer_spec.fov_id,
            path=path,
            context=context,
            detail=(
                f"channel_order mismatch: sidecar declares {list(persisted_spec.channels)} with "
                f"shape {list(persisted_spec.expected_shape)} but consumer expects "
                f"{list(consumer_spec.channels)} with shape {list(consumer_spec.expected_shape)} "
                f"for matrix {_path_text(matrix_path)}"
            ),
            expected=expected,
        )


@dataclass(frozen=True)
class DecodedTableSchema:
    """Contract for the decoded CSV artifact family."""

    required_columns: tuple[str, ...] = ("z", "y", "x", "barcode", "quality", "intensity", "gene")
    compatibility_columns: tuple[str, ...] = (
        "channel",
        "fov",
        "algo",
        "pattern_valid",
        "in_codebook",
        "gating_mode",
    )
    required_numeric_columns: tuple[str, ...] = ("z", "y", "x", "quality", "intensity")
    required_non_null_columns: tuple[str, ...] = ("barcode", "gene")

    def expected_description(self) -> str:
        return (
            f"required columns {list(self.required_columns)}; accepted compatibility columns "
            f"{list(self.compatibility_columns)}; numeric columns {list(self.required_numeric_columns)}"
        )


def empty_spot_table(*, extra_columns: Sequence[str] = ()) -> pd.DataFrame:
    """Return an empty canonical spot table with stable columns/dtypes."""

    schema = SpotTableSchema()
    columns = _dedupe_columns(schema.required_columns, extra_columns)
    return _empty_dataframe(
        columns,
        {
            "z": np.float32,
            "y": np.float32,
            "x": np.float32,
            "intensity": np.float32,
            "channel": np.int64,
            "fov": np.int64,
            "algo": object,
        },
    )


def validate_spot_table(
    df: Any,
    *,
    fov_id: int,
    path: Path | str | None,
    context: str,
) -> pd.DataFrame:
    """Validate the canonical spot-table contract and return a normalized copy."""

    schema = SpotTableSchema()
    normalized = _validate_table_shape(
        df,
        artifact_name="spot table",
        fov_id=fov_id,
        path=path,
        context=context,
    )
    expected = schema.expected_description()
    _validate_required_columns(
        normalized,
        required_columns=schema.required_columns,
        artifact_name="spot table",
        fov_id=fov_id,
        path=path,
        context=context,
        expected=expected,
    )
    normalized = _coerce_numeric_columns(
        normalized,
        numeric_columns=schema.required_numeric_columns,
        artifact_name="spot table",
        fov_id=fov_id,
        path=path,
        context=context,
        expected=expected,
        require_non_null=True,
    )
    optional_numeric = [column for column in schema.optional_numeric_columns if column in normalized.columns]
    if optional_numeric:
        normalized = _coerce_numeric_columns(
            normalized,
            numeric_columns=optional_numeric,
            artifact_name="spot table",
            fov_id=fov_id,
            path=path,
            context=context,
            expected=expected,
            require_non_null=False,
        )
    if "fov" in normalized.columns and len(normalized) > 0:
        non_null_fov = normalized["fov"].dropna()
        if len(non_null_fov) > 0 and not bool((non_null_fov == int(fov_id)).all()):
            raise _schema_error(
                "spot table",
                fov_id=fov_id,
                path=path,
                context=context,
                detail=f"column 'fov' contains values that do not match requested fov_id={fov_id}",
                expected=expected,
            )
    return normalized


def validate_intensity_matrix(
    matrix: Any,
    spec: IntensityMatrixSpec,
    *,
    path: Path | str | None,
    context: str,
) -> npt.NDArray[np.generic]:
    """Validate the canonical intensity-matrix contract."""

    arr = np.asarray(matrix)
    expected = spec.expected_description()
    if arr.ndim != 3:
        raise _schema_error(
            "intensity matrix",
            fov_id=spec.fov_id,
            path=path,
            context=context,
            detail=f"rank mismatch: got ndim={arr.ndim}",
            expected=expected,
        )
    if not np.issubdtype(arr.dtype, np.number):
        raise _schema_error(
            "intensity matrix",
            fov_id=spec.fov_id,
            path=path,
            context=context,
            detail=f"dtype {arr.dtype!s} is not numeric",
            expected=expected,
        )
    actual_shape = tuple(int(value) for value in arr.shape)
    if actual_shape[0] != spec.n_spots:
        raise _schema_error(
            "intensity matrix",
            fov_id=spec.fov_id,
            path=path,
            context=context,
            detail=f"axis 0 mismatch: expected {spec.n_spots} spot rows, got {actual_shape[0]}",
            expected=expected,
        )
    if actual_shape[1] != spec.n_rounds:
        raise _schema_error(
            "intensity matrix",
            fov_id=spec.fov_id,
            path=path,
            context=context,
            detail=f"axis 1 mismatch: expected {spec.n_rounds} rounds, got {actual_shape[1]}",
            expected=expected,
        )
    if actual_shape[2] != spec.n_channels:
        raise _schema_error(
            "intensity matrix",
            fov_id=spec.fov_id,
            path=path,
            context=context,
            detail=f"axis 2 mismatch: expected {spec.n_channels} sequencing channels, got {actual_shape[2]}",
            expected=expected,
        )
    return arr


def empty_decoded_table(*, extra_columns: Sequence[str] = ()) -> pd.DataFrame:
    """Return an empty canonical decoded table with stable columns/dtypes."""

    schema = DecodedTableSchema()
    columns = _dedupe_columns(schema.required_columns, extra_columns)
    return _empty_dataframe(
        columns,
        {
            "z": np.float32,
            "y": np.float32,
            "x": np.float32,
            "barcode": object,
            "quality": np.float32,
            "intensity": np.float32,
            "gene": object,
            "channel": np.int64,
            "fov": np.int64,
            "algo": object,
            "pattern_valid": bool,
            "in_codebook": bool,
            "gating_mode": object,
        },
    )


def validate_decoded_table(
    df: Any,
    *,
    fov_id: int,
    path: Path | str | None,
    context: str,
) -> pd.DataFrame:
    """Validate the canonical decoded-table contract and return a normalized copy."""

    schema = DecodedTableSchema()
    normalized = _validate_table_shape(
        df,
        artifact_name="decoded table",
        fov_id=fov_id,
        path=path,
        context=context,
    )
    expected = schema.expected_description()
    _validate_required_columns(
        normalized,
        required_columns=schema.required_columns,
        artifact_name="decoded table",
        fov_id=fov_id,
        path=path,
        context=context,
        expected=expected,
    )
    normalized = _coerce_numeric_columns(
        normalized,
        numeric_columns=schema.required_numeric_columns,
        artifact_name="decoded table",
        fov_id=fov_id,
        path=path,
        context=context,
        expected=expected,
        require_non_null=True,
    )
    _validate_non_null_columns(
        normalized,
        non_null_columns=schema.required_non_null_columns,
        artifact_name="decoded table",
        fov_id=fov_id,
        path=path,
        context=context,
        expected=expected,
    )
    if "fov" in normalized.columns and len(normalized) > 0:
        non_null_fov = normalized["fov"].dropna()
        if len(non_null_fov) > 0 and not bool((non_null_fov == int(fov_id)).all()):
            raise _schema_error(
                "decoded table",
                fov_id=fov_id,
                path=path,
                context=context,
                detail=f"column 'fov' contains values that do not match requested fov_id={fov_id}",
                expected=expected,
            )
    return normalized
