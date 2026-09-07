# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Read-only adapter from first-class selected TIR to template descriptors."""

from __future__ import annotations

from dataclasses import replace

from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import IRModule, Node, kernel_dispatch_for_call


_ZERO_COPY_ADAPTERS = frozenset({
    "distributed.sharded_view",
    "tir.buffer_view",
})


def resolve_zero_copy_adapter(module: IRModule, value: str | Node) -> Node:
    """Resolve a logical view to the value that owns its physical storage.

    Selected and bufferized IR deliberately retain distribution views so a
    Python checkpoint still describes the logical coordinate map.  Template
    matchers that reason about the graph ABI may look through those views, but
    must not erase executable boxing operations.
    """

    node = module.node_map[value] if isinstance(value, str) else value
    visited: set[str] = set()
    while node.op in _ZERO_COPY_ADAPTERS:
        if node.id in visited or len(node.inputs) != 1:
            raise CodegenError(
                f"Malformed zero-copy adapter chain at {node.id!r}.",
                stage=module.stage,
                node_id=node.id,
            )
        visited.add(node.id)
        node = module.node_map[node.inputs[0]]
    return node


def selected_kernel_node(module: IRModule, value: str | Node) -> Node:
    """Return a compatibility view of one authoritative KernelDispatch.

    The returned node is never inserted into the module.  This narrow adapter
    lets existing template matchers consume ``attrs`` while selection data is
    owned solely by the called PrimFunction.
    """

    node = module.node_map[value] if isinstance(value, str) else value
    dispatch = kernel_dispatch_for_call(module, node)
    if dispatch is None:
        raise CodegenError(
            f"Node {node.id!r} is not a call to a single selected KernelDispatch PrimFunction.",
            stage=module.stage,
            node_id=node.id,
        )
    if dispatch.microkernel is None:
        raise CodegenError(
            f"Semantic TIR node {node.id!r} has no selected microkernel; resume at "
            "propose-microkernels/select-microkernels before code generation.",
            stage=module.stage,
            node_id=node.id,
        )
    selection = module.selection_map.get(f"tir.{node.id}")
    if (
        selection is not None
        and selection.candidate_id != dispatch.semantic_candidate
    ):
        raise CodegenError(
            f"TIR selection {selection.candidate_id!r} for node {node.id!r} does not "
            f"match materialized KernelDispatch semantic candidate "
            f"{dispatch.semantic_candidate!r}; edit or resume "
            "before lower-tir instead of changing only the final selection record.",
            stage=module.stage,
            node_id=node.id,
        )
    return replace(node, op="tir.kernel", attrs={
        "semantic_op": dispatch.semantic_op,
        "candidate": dispatch.microkernel.implementation,
        "parameters": dispatch.resolved_parameters,
        "facts": dispatch.resolved_facts,
        "semantic_attrs": dispatch.semantic_attrs,
    })


def selected_kernel_nodes(module: IRModule) -> tuple[Node, ...]:
    result = []
    for node in module.nodes:
        if kernel_dispatch_for_call(module, node) is not None:
            result.append(selected_kernel_node(module, node))
    return tuple(result)


__all__ = [
    "resolve_zero_copy_adapter",
    "selected_kernel_node",
    "selected_kernel_nodes",
]
