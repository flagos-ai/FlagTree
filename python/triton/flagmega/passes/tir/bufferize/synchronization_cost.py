# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Compose emitted synchronization deltas over one closed entry execution."""

from math import inf

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir.ops.control.for_loop import static_trip_count
from triton.flagmega.passes.functions.graph import callee_first_functions, function_nodes


def entry_synchronization_delta_bounds(module, local_deltas):
    """Bound candidate-minus-baseline costs without expanding reusable bodies.

    Each vector component counts an emitted synchronization hierarchy. Calls
    compose once per call site; static For bodies compose exactly once per
    trip. An unknown trip count is any nonnegative integer, including zero.
    Bound each complete body before multiplying, so mandatory callee savings
    can pay for its local costs without assuming that a dynamic loop runs.

    Intervals deliberately forget correlations between dynamic loops. They
    prove nonincrease for every trip count, rather than pricing an assumed
    iteration count or expanding paths/polynomials in the call DAG.
    """
    if module.entry not in module.function_map:
        raise IRVerificationError(f"Entry function @{module.entry} is not defined.", stage=module.stage)
    for name in local_deltas:
        if name not in module.function_map:
            raise IRVerificationError(f"Synchronization cost references unknown function @{name}.", stage=module.stage)

    costs = {}
    for function in callee_first_functions(module):
        bounds = [(delta, delta) for delta in local_deltas.get(function.name, (0, 0, 0, 0))]
        for node in function_nodes(module, function):
            if node.op not in {"builtin.call", "tir.call", "control.for"}:
                continue
            callee = str(node.attrs["callee"])
            if callee in module.kernel_callable_map:
                continue
            trips = static_trip_count(node, module) if node.op == "control.for" else 1
            for index, (lower, upper) in enumerate(costs[callee]):
                if trips == 0:
                    lower = upper = 0
                elif trips is None:
                    lower = -inf if lower < 0 else 0
                    upper = inf if upper > 0 else 0
                else:
                    lower, upper = lower * trips, upper * trips
                old_lower, old_upper = bounds[index]
                bounds[index] = (old_lower + lower, old_upper + upper)
        costs[function.name] = tuple(bounds)
    return costs[module.entry]


__all__ = ["entry_synchronization_delta_bounds"]
