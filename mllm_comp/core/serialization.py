"""Serialization helpers for manifests and traces."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any


def to_jsonable(value: Any) -> Any:
    """Convert dataclasses and common containers to JSON-compatible values."""

    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, Enum):
        return to_jsonable(value.value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(inner) for key, inner in value.items()}
    if isinstance(value, set):
        return [to_jsonable(inner) for inner in sorted(value, key=str)]
    if isinstance(value, (list, tuple)):
        return [to_jsonable(inner) for inner in value]
    return value


def atomic_write_json(
    path: str | os.PathLike[str],
    payload: Any,
    *,
    ensure_ascii: bool = True,
    sort_keys: bool = True,
) -> None:
    """Write JSON beside the target and atomically replace the old file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=target.parent,
        delete=False,
        suffix=".tmp",
    ) as handle:
        json.dump(
            to_jsonable(payload),
            handle,
            ensure_ascii=ensure_ascii,
            indent=2,
            sort_keys=sort_keys,
        )
        handle.write("\n")
        tmp_name = handle.name
    os.replace(tmp_name, target)


def atomic_write_text(path: str | os.PathLike[str], content: str) -> None:
    """Write UTF-8 text beside the target and atomically replace the old file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=target.parent,
        delete=False,
        suffix=".tmp",
    ) as handle:
        handle.write(content)
        if not content.endswith("\n"):
            handle.write("\n")
        tmp_name = handle.name
    os.replace(tmp_name, target)
