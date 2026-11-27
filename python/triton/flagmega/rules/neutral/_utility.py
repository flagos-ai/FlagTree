# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Small construction helpers for handwritten neutral rules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from triton.flagmega.ir import Node, get_definition


def make_node(
    op: str,
    node_id: str,
    inputs: Sequence[Node],
    attrs: Mapping[str, object],
    metadata: Mapping[str, object],
) -> Node:
    prepared = get_definition(op).prepare(inputs, attrs)
    return Node(
        node_id,
        op,
        tuple(value.id for value in prepared.inputs),
        prepared.result_type,
        prepared.effect,
        prepared.attrs,
        metadata,
    )


def decomposition_metadata(source: Node, rule: str) -> dict[str, object]:
    return {
        **dict(source.metadata),
        "decomposed_from": source.id,
        "decomposition_rule": rule,
    }


__all__ = ["decomposition_metadata", "make_node"]
