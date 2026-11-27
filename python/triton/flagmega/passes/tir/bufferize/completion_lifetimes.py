# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Propose allocation lifetimes that include unfinished peer accesses.

The reference epochs contain only semantic dependencies, so incidental reuse
does not become a mandatory completion boundary. The selected physical plan
still goes through the ordinary, unfiltered synchronization analysis.
"""

from .call_accesses import CallAccessResolver
from .graph import function_nodes
from .synchronization import _node_accesses, _plan_memory_synchronization, _requires_device
from triton.flagmega.ir.bufferization import MemorySharingScope


class CompletionCapacityUnavailable(Exception):
    """This optional proposal lacks a constructive capacity witness."""


def verify_completion_capacity(lifetimes_by_space, spaces):
    from .first_fit_allocator import first_fit_placement
    from .allocation import usable_capacity
    for name, lifetimes in lifetimes_by_space.items():
        space = spaces[name]
        _, peak = first_fit_placement(tuple(value for value in lifetimes if value.nbytes), space)
        if peak > usable_capacity(space):
            raise CompletionCapacityUnavailable(
                f"Completion-lifetime proposal has no first-fit capacity witness in {name!r}: {peak} bytes.")


def completion_lifetime_ends(module, plan, *, include_chip_local=False):
    leaf_functions = tuple(
        (function, function_nodes(module, function)) for function in module.functions
        if not plan.function_map[function.name].calls
    )
    resolver = CallAccessResolver(module, plan, _node_accesses)
    pending = []
    for function, nodes in leaf_functions:
        bindings = dict(plan.function_map[function.name].values)
        for index, node in enumerate(nodes):
            if node.op not in {"tir.call", "tir.kernel"}:
                continue
            for access in resolver.accesses(function.name, node, bindings):
                physical = plan.buffer_map[access.buffer].mem_span.buffer
                space = plan.memory_space_map[physical.memory_space]
                if (physical.function == function.name and physical.live_end is not None
                        and space.supports_lifetime_reuse
                        and space.sharing_scope is MemorySharingScope.CHIP
                        and (include_chip_local or _requires_device(access))):
                    pending.append((function.name, index, len(nodes), physical, _requires_device(access)))
    if not pending:
        return {}
    events = _plan_memory_synchronization(module, plan, semantic_dependencies_only=True).events
    indices = {function.name: {node.id: index for index, node in enumerate(nodes)}
               for function, nodes in leaf_functions}
    epochs = {}
    for event in events:
        if (event.function in indices and (event.scope == "device"
                or event.scope == "grid" and not event.axis_group_axes)):
            epochs.setdefault(event.function, []).append((indices[event.function][event.before], event.scope))
    result = {}
    for function, index, end, physical, needs_device in pending:
        completed = next((value for value, scope in epochs.get(function, ())
                          if value > index and (not needs_device or scope == "device")), end + 1)
        horizon = max(physical.live_end, completed - 1)
        if horizon > physical.live_end:
            result[physical.id] = max(result.get(physical.id, 0), horizon)
    return result


__all__ = ["completion_lifetime_ends"]
