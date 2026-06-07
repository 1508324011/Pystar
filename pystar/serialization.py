"""Small serialization helpers for backend metadata sidecars."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np


JSONScalar = str | int | float | bool | None
JSONValue = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


def json_safe(value: object) -> JSONValue:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        raw_dict = cast(dict[object, object], value)
        return {str(key): json_safe(item) for key, item in raw_dict.items()}
    if isinstance(value, (list, tuple)):
        raw_sequence = cast(list[object] | tuple[object, ...], value)
        return [json_safe(item) for item in raw_sequence]
    if isinstance(value, np.ndarray):
        return cast(JSONValue, value.tolist())
    if isinstance(value, np.integer):
        return value.item()
    if isinstance(value, np.floating):
        return value.item()
    return cast(JSONValue, value)


def write_backend_metadata(path: Path, payload: dict[str, object]) -> None:
    _ = path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
