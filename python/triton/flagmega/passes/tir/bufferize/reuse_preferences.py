# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Derive barrier-sensitive SAT tie-breaks from actual typed byte hazards.

This is not a latency cost model. A preference exists only where allocation
reuse is the sole reason a synchronization cut needs chip-wide coordination.
The final plans are compared again, because relocation can expose other
hazards; a change must dominate the original for the closed entry execution
while keeping every function pool within its explicit growth budget.
"""

from collections import defaultdict

from triton.flagmega.ir.bufferization import MemorySharingScope


def collect_reuse_preferences(module, plan):
    from .synchronization import _plan_memory_synchronization

    collected = defaultdict(set)
    _plan_memory_synchronization(module, plan, reuse_preferences=collected)
    return {key: tuple(sorted(pairs)) for key, pairs in sorted(collected.items())}


def record_reuse_preferences(plan, function, conflicts, collected):
    from .synchronization import _conflict_requirement

    candidates = []
    fixed = []
    for previous, current in conflicts:
        pair = _reallocatable_reuse_pair(plan, previous, current)
        if pair is None:
            fixed.append((previous, current))
        else:
            candidates.append(pair)
    # Relocation cannot remove or weaken a cut whose true dependencies still
    # require the same rendezvous. Pricing its incidental reuse pairs gives
    # SAT an objective unrelated to the synchronization we accept afterward.
    if fixed and _conflict_requirement(fixed) == _conflict_requirement(conflicts):
        return
    for owner_function, space, pair in candidates:
        collected[(owner_function, space)].add(pair)


def _reallocatable_reuse_pair(plan, previous, current):
    from .synchronization import _hazard_requirement

    if _hazard_requirement(previous, current)[0] not in {"grid", "device"}:
        return None
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
        # Semantic aliases, references and producer/consumer owner changes
        # remain governed by the original effects and synchronization analysis.
        return None
    return left.function, left.memory_space, tuple(sorted((left.id, right.id)))


def dominates_memory_schedule(module, candidate, baseline, *, bytes_budget: int = 0, pool_baseline=None):
    """Require no unjustified pool growth and no worse emitted rendezvous.

    A candidate may grow each function pool by at most ``bytes_budget`` bytes
    relative to ``pool_baseline`` (the original layout in a multi-round search),
    or to ``baseline`` when no separate pool baseline is supplied.
    Synchronization counts compose over the module's actual entry call graph,
    rather than requiring each private reusable body to improve separately.
    Static loops use their exact trip count; dynamic loops must prove no
    increase for any nonnegative trip count and a possible strict reduction.

    Count the executor's actual hierarchy, including the two chip fences
    surrounding a device rendezvous. A logical device event is not one
    primitive: treating it as one can reject fewer remote epochs merely
    because their replacement needs one additional CTA fence. Conversely,
    fewer chip events must not hide a new remote epoch.
    """

    from .synchronization import plan_memory_synchronization
    from .synchronization_cost import entry_synchronization_delta_bounds
    from triton.flagmega.ir.bufferization.synchronization_protocol import module_synchronization_protocol
    protocol = module_synchronization_protocol(module)

    growth_allowed = max(0, int(bytes_budget))
    for function in (pool_baseline if pool_baseline is not None else baseline).functions:
        for name, pool in function.memory_pool_map.items():
            grown = candidate.function_map[function.name].memory_pool_map[name].scope_bytes
            if grown > pool.scope_bytes + growth_allowed:
                return False

    def counts(plan):
        result = defaultdict(lambda: [0, 0, 0, 0])
        for event in plan_memory_synchronization(module, plan).events:
            count = result[event.function]
            # Mirrors entrypoints/_common.py.jinja::device_barrier: publish
            # all CTAs, rendezvous through the chip leader, then acquire on
            # all CTAs. These are operations, not fitted latency weights.
            device_grids = 2 if protocol is None else protocol.device_grid_barriers
            primitives = ((("grid", ()),) * device_grids + (("device", ()),)
                          if event.scope == "device" else ((event.scope, event.axis_group_axes),))
            for scope, axes in primitives:
                count[0] += scope == "device"
                count[1] += scope == "device" or scope == "grid" and not axes
                count[2] += scope in {"grid", "device"}
                # Grid rendezvous has target-declared local CTA phases. A
                # device leaf remains an independently counted operation;
                # its unknown internal work cannot increase when count[0]
                # does not increase. No measured latency weights are used.
                count[3] += protocol.grid_cta_barriers if protocol is not None and scope == "grid" else 1
        return result

    before, after = counts(baseline), counts(candidate)
    deltas = {name: tuple(new - old for old, new in zip(before[name], after[name]))
              for name in before.keys() | after.keys()}
    bounds = entry_synchronization_delta_bounds(module, deltas)
    return all(upper <= 0 for _, upper in bounds) and any(lower < 0 for lower, _ in bounds)


__all__ = ["collect_reuse_preferences", "dominates_memory_schedule"]
