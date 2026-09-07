# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Frozen compile-time constant recipes.

A recipe owns an ordinary, topologically ordered IR graph.  It is deliberately
not a nested function/region: high-level rewrites finish before recipes are
created, and TIR passes treat the recipe table as immutable artifact data.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Mapping

from triton.flagmega.errors import IRSchemaError

if TYPE_CHECKING:
    from triton.flagmega.ir.model import Node


class ConstantPhase(str, Enum):
    OPEN = "open"
    FROZEN = "frozen"


@dataclass(frozen=True)
class ConstantRecipe:
    """One maximal constant island and the values exported to the main graph."""

    id: str
    nodes: tuple[Node, ...]
    outputs: tuple[str, ...]
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.id:
            raise IRSchemaError("Constant recipe id must be non-empty.")
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(self, "outputs", tuple(str(value) for value in self.outputs))
        if not self.nodes or not self.outputs:
            raise IRSchemaError("A constant recipe requires nodes and at least one output.")
        object.__setattr__(self, "fingerprint", constant_recipe_fingerprint(self.nodes, self.outputs))

    @property
    def node_map(self) -> dict[str, Node]:
        return {node.id: node for node in self.nodes}

    def to_data(self) -> dict[str, object]:
        return {
            "id": self.id,
            "nodes": [node.to_data() for node in self.nodes],
            "outputs": list(self.outputs),
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> ConstantRecipe:
        from triton.flagmega.ir.model import Node

        # The fingerprint is derived instead of trusted.  An edited Python
        # checkpoint therefore acquires the right identity automatically.
        return cls(
            str(data["id"]),
            tuple(Node.from_data(value) for value in data.get("nodes", ())),
            tuple(str(value) for value in data.get("outputs", ())),
        )


def constant_recipe_fingerprint(nodes: tuple[Node, ...], outputs: tuple[str, ...]) -> str:
    """Hash recipe semantics independently of incidental node names."""

    positions = {node.id: index for index, node in enumerate(nodes)}
    payload = {
        "nodes": [
            {
                "op": node.op,
                "inputs": [positions[value] for value in node.inputs],
                "type": node.type.to_data(),
                "effect": node.effect.to_data(),
                "attrs": _plain(node.attrs),
            }
            for node in nodes
        ],
        "outputs": [positions[value] for value in outputs],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in sorted(value.items())}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_data"):
        return _plain(value.to_data())
    return value


__all__ = ["ConstantPhase", "ConstantRecipe", "constant_recipe_fingerprint"]
