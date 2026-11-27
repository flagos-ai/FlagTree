# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Owner relationship of one selected TIR kernel invocation."""

from __future__ import annotations

from enum import Enum
from typing import Mapping


class KernelExecutionKind(str, Enum):
    """How a selected kernel relates placement owners."""

    LOCAL_SHARD = "local_shard"
    SYNCHRONIZED_LOCAL = "synchronized_local"
    COLLECTIVE = "collective"


def kernel_execution_kind(
    semantic_op: str,
    facts: Mapping[str, object],
) -> KernelExecutionKind:
    """Classify execution from selected implementation facts.

    This belongs to TIR rather than a renderer: memory-effect analysis,
    scheduling and every backend source emitter must agree on the same owner
    relationship.
    """

    collective = facts.get("collective_semantics")
    if isinstance(collective, str) and collective:
        return KernelExecutionKind.COLLECTIVE
    # Preserve safety for hand-edited pre-materialization checkpoints whose
    # explicit Boxing call has lost optional target facts.
    if semantic_op in {"distributed.boxing", "distributed.force_boxing"}:
        return KernelExecutionKind.COLLECTIVE
    raw_requires = facts.get("requires", ())
    requires = (
        (raw_requires,)
        if isinstance(raw_requires, str)
        else tuple(str(value) for value in raw_requires)
    )
    if "grid_sync" in requires or "cooperative_grid" in requires:
        return KernelExecutionKind.SYNCHRONIZED_LOCAL
    return KernelExecutionKind.LOCAL_SHARD


def kernel_execution_kind_for_call(module, node) -> KernelExecutionKind:
    """Resolve a graph kernel call through its materialized PrimFunction."""

    from triton.flagmega.ir.tir.kernel_dispatch import kernel_dispatch_for_call

    dispatch = kernel_dispatch_for_call(module, node)
    if dispatch is not None:
        return kernel_execution_kind(
            dispatch.semantic_op, dispatch.resolved_facts
        )
    if node.op == "tir.kernel":
        facts = node.attrs.get("facts", {})
        return kernel_execution_kind(
            str(node.attrs.get("semantic_op", "")),
            facts if isinstance(facts, Mapping) else {},
        )
    return KernelExecutionKind.LOCAL_SHARD


__all__ = [
    "KernelExecutionKind",
    "kernel_execution_kind",
    "kernel_execution_kind_for_call",
]
