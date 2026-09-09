# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Derive barrier-sensitive SAT tie-breaks from actual typed byte hazards.

This is not a latency cost model. A preference exists only where allocation
reuse is the sole reason a synchronization cut needs chip-wide coordination.
The final plans are compared again, because relocation can expose other
hazards; a change must dominate the original for every function and pool.
"""

from collections import defaultdict

from triton.flagmega.ir.bufferization import MemorySharingScope


def collect_reuse_preferences(module, plan):
    from .synchronization import _plan_memory_synchronization

    collected = defaultdict(set)
    _plan_memory_synchronization(module, plan, reuse_preferences=collected)
    return {key: tuple(sorted(pairs)) for key, pairs in sorted(collected.items())}


def record_reuse_preferences(plan, function, conflicts, collected):
    from .synchronization import _hazard_requirement

    reuse = []
    intrinsic = []
    for previous, current in conflicts:
        left = plan.buffer_map[previous.buffer].mem_span.buffer
        right = plan.buffer_map[current.buffer].mem_span.buffer
        space = plan.memory_space_map[left.memory_space]
        if (
            left.id != right.id
            and left.function == right.function == function
            and left.memory_space == right.memory_space
            and space.supports_lifetime_reuse
            and space.sharing_scope is MemorySharingScope.CHIP
        ):
            reuse.append((previous, current, left, right))
        else:
            intrinsic.append((previous, current))
    # Do not spend solver work avoiding reuse at an already necessary grid
    # cut. Semantic aliases, references and producer/consumer owner changes
    # remain governed by the original effects and synchronization analysis.
    if any(_hazard_requirement(previous, current)[0] == "grid" for previous, current in intrinsic):
        return
    for previous, current, left, right in reuse:
        if _hazard_requirement(previous, current)[0] == "grid":
            collected[(function, left.memory_space)].add(tuple(sorted((left.id, right.id))))


def dominates_memory_schedule(module, candidate, baseline):
    """Require no pool growth and no worse barrier counts in any function."""

    from .synchronization import plan_memory_synchronization

    for function in baseline.functions:
        for name, pool in function.memory_pool_map.items():
            if candidate.function_map[function.name].memory_pool_map[name].scope_bytes > pool.scope_bytes:
                return False

    def counts(plan):
        result = defaultdict(lambda: [0, 0, 0])
        for event in plan_memory_synchronization(module, plan).events:
            count = result[event.function]
            count[0] += event.scope == "grid"
            count[1] += event.scope == "grid" and not event.axis_group_axes
            count[2] += 1
        return result

    before, after = counts(baseline), counts(candidate)
    improved = False
    for name in before.keys() | after.keys():
        for old, new in zip(before[name], after[name]):
            if new > old:
                return False
            improved |= new < old
    return improved


__all__ = ["collect_reuse_preferences", "dominates_memory_schedule"]
