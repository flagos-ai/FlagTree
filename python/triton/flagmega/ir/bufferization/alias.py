# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Typed provenance for MemSpan aliases."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class AliasKind(str, Enum):
    IDENTITY = "identity"
    INPLACE = "inplace"
    VIEW = "view"
    PARAMETER = "parameter"
    RESULT = "result"


@dataclass(frozen=True)
class AliasInfo:
    """Why a logical buffer view was produced.

    This is provenance only.  Alias truth is always computed from MemSpan.
    """

    source: str
    kind: AliasKind

    def to_data(self) -> dict[str, object]:
        return {"source": self.source, "kind": self.kind.value}

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> AliasInfo:
        return cls(str(data["source"]), AliasKind(str(data["kind"])))


__all__ = ["AliasInfo", "AliasKind"]
