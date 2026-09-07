# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Remove graph functions and expressions unreachable from the module entry."""

from __future__ import annotations

from dataclasses import replace

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir import IRModule
from triton.flagmega.passes.functions.graph import function_nodes


def remove_unused_functions(module: IRModule) -> IRModule:
    """Keep exactly the graph program reachable from ``module.entry``.

    This is the flat-IR counterpart of nncase ``RemoveUnusedFunctions``.  A
    FlagMega module has one executable entry; graph functions not referenced
    by that entry and nodes not belonging to a retained function are dead
    program state, not independent compilation roots.

    Prim/execution functions are deliberately left alone.  This pass runs in
    high-level AutoPacking before those function kinds are materialized, and
    silently guessing their later reachability here would mix graph and TIR
    ownership rules.
    """

    function_map = module.function_map
    if module.entry not in function_map:
        raise IRVerificationError(
            f"Entry function @{module.entry} is not defined.", stage=module.stage
        )

    reachable_functions: set[str] = set()
    pending = [module.entry]
    while pending:
        function_name = pending.pop()
        if function_name in reachable_functions:
            continue
        function = function_map[function_name]
        reachable_functions.add(function_name)
        for node in function_nodes(module, function):
            if node.op not in {"builtin.call", "tir.call"}:
                continue
            callee = str(node.attrs.get("callee", ""))
            if callee in module.kernel_callable_map:
                continue
            if callee not in function_map:
                raise IRVerificationError(
                    f"Call {node.id!r} in @{function.name} references unknown "
                    f"graph function @{callee}.",
                    stage=module.stage,
                    node_id=node.id,
                )
            pending.append(callee)

    functions = tuple(
        function
        for function in module.functions
        if function.name in reachable_functions
    )
    live_nodes = {
        node.id
        for function in functions
        for node in function_nodes(module, function)
    }
    nodes = tuple(node for node in module.nodes if node.id in live_nodes)
    points = tuple(
        point
        for point in module.selection_points
        if point.owner is None or point.owner in live_nodes
    )
    point_ids = {point.id for point in points}
    selections = tuple(
        record
        for record in module.selections
        if record.point_id in point_ids
    )

    if (
        nodes == module.nodes
        and functions == module.functions
        and points == module.selection_points
        and selections == module.selections
    ):
        return module
    return replace(
        module,
        nodes=nodes,
        functions=functions,
        selection_points=points,
        selections=selections,
    )


__all__ = ["remove_unused_functions"]
