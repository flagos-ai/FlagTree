# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Stable, readable identities for AutoDistribution layout relations."""

from __future__ import annotations

import re

from triton.flagmega.ir import (
    BlockCyclicSplit,
    ContiguousSplit,
    DistributedType,
    IRType,
    RefType,
    SBPBroadCast,
    SBPPartial,
    SBPSplit,
    TensorType,
    TupleType,
)


def distributed_candidate_id(
    node_id: str,
    family: str,
    return_type: IRType,
    input_types: tuple[IRType, ...],
) -> str:
    """Name a candidate by its type relation, never its enumeration ordinal.

    Agent-edited selection checkpoints survive insertion or reordering of
    unrelated legal candidates. The placement and complete staged SBP policy
    stay visible in the id, so replay on a different mesh fails explicitly.
    """

    family = _safe(family)
    placements = sorted({
        _placement_signature(placement)
        for value in (*input_types, return_type)
        for placement in _placements(value)
    })
    mesh = "" if not placements else ".mesh_" + "__".join(placements)
    inputs = "__".join(layout_signature(value) for value in input_types)
    output = layout_signature(return_type)
    return f"distribution.{node_id}.{family}{mesh}.in_{inputs}.out_{output}"


def layout_signature(value: IRType) -> str:
    if isinstance(value, DistributedType):
        policies = "_".join(_sbp_signature(policy) for policy in value.axis_policies)
        partial = (
            ""
            if value.partial is None
            else f"_partial_{_sbp_signature(value.partial)}"
        )
        return f"d_{policies}{partial}"
    if isinstance(value, TensorType):
        return "tensor"
    if isinstance(value, TupleType):
        return "tuple_" + "__".join(layout_signature(field) for field in value.fields)
    if isinstance(value, RefType):
        return f"ref_{_safe(value.name)}"
    return _safe(type(value).__name__)


def _sbp_signature(value) -> str:
    if isinstance(value, SBPBroadCast):
        return "b"
    if isinstance(value, SBPPartial):
        axes = "_".join(str(axis) for axis in value.axes)
        return f"p_{value.reduce_op.value}_h{axes}"
    if isinstance(value, SBPSplit):
        return "s_" + "_then_".join(
            _stage_signature(stage) for stage in value.stages
        )
    return _safe(type(value).__name__)


def _stage_signature(stage) -> str:
    axes = "_".join(str(axis) for axis in stage.hierarchy_axes)
    distribution = stage.distribution
    if isinstance(distribution, ContiguousSplit):
        granularity = (
            "auto"
            if distribution.granularity is None
            else _safe(str(distribution.granularity))
        )
        return f"c_h{axes}_g{granularity}"
    if isinstance(distribution, BlockCyclicSplit):
        return f"bc_h{axes}_b{distribution.block_size}"
    return f"{_safe(type(distribution).__name__)}_h{axes}"


def _placements(value: IRType):
    if isinstance(value, DistributedType):
        yield value.placement
    elif isinstance(value, TupleType):
        for field in value.fields:
            yield from _placements(field)


def _placement_signature(value) -> str:
    hierarchy = "x".join(str(extent) for extent in value.hierarchy)
    return f"{_safe(value.name)}_{hierarchy}_{_safe(value.hierarchy_levels)}"


def _safe(value: str) -> str:
    result = re.sub(r"[^0-9A-Za-z_]+", "_", str(value)).strip("_").lower()
    return result or "none"


__all__ = ["distributed_candidate_id", "layout_signature"]
