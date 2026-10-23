"""Stable semantic model-handle container."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StructuralHandles:
    """Maps semantic paths such as 'llm.layer[0].attn' to modules or metadata."""

    handles: dict[str, Any] = field(default_factory=dict)

    def get(self, path: str) -> Any:
        try:
            return self.handles[path]
        except KeyError as exc:
            raise KeyError(f"Unknown structural handle: {path}") from exc

    def paths(self) -> list[str]:
        return sorted(self.handles)

    def iter_indexed(self, prefix: str) -> list[tuple[int, Any]]:
        """Return indexed handles like 'llm.layer[0]' in ascending index order."""

        pattern = re.compile(rf"^{re.escape(prefix)}\[(\d+)\]$")
        indexed: list[tuple[int, Any]] = []
        for path, handle in self.handles.items():
            match = pattern.match(path)
            if match:
                indexed.append((int(match.group(1)), handle))
        return sorted(indexed, key=lambda item: item[0])
