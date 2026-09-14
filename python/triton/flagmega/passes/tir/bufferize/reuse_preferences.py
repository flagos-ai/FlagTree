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

    for previous, current in conflicts:
        if _hazard_requirement(previous, current)[0] != "grid":
            continue
        left = plan.buffer_map[previous.buffer].mem_span.buffer
        right = plan.buffer_map[current.buffer].mem_span.buffer
        space = plan.memory_space_map[left.memory_space]
        if (
            left.id == right.id
            or left.function != right.function
            or left.memory_space != right.memory_space
            or not space.supports_lifetime_reuse
            or space.sharing_scope is not MemorySharingScope.CHIP
        ):
            # Semantic aliases, references and producer/consumer owner
            # changes remain governed by the original effects and
            # synchronization analysis; reallocation cannot remove them.
            continue
        collected[(left.function, left.memory_space)].add(tuple(sorted((left.id, right.id))))


def dominates_memory_schedule(module, candidate, baseline, *, bytes_budget: int = 0):
    """Require no unjustified pool growth and no worse barrier counts.

    A candidate may grow a function pool by at most ``bytes_budget`` bytes and
    only where the same function's barrier counts strictly improve; every
    growth byte must buy a synchronization reduction.
    """

    from .synchronization import plan_memory_synchronization

    growth_allowed = max(0, int(bytes_budget))
    for function in baseline.functions:
        for name, pool in function.memory_pool_map.items():
            grown = candidate.function_map[function.name].memory_pool_map[name].scope_bytes
            if grown > pool.scope_bytes + growth_allowed:
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
