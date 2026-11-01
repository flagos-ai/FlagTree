# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Conservative SSA intervals for memory touched by concurrent programs."""

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir.program_order import program_ancestors


def program_use_ends(module, function, nodes):
    """Extend each call through every program not ordered after its completion.

    The supplied order is topological, but unrelated programs can execute in
    either order. If an allocation touched by P ends before a later allocation
    begins in Q, Q must follow P. Otherwise Q's last call contributes to P's
    interval and prevents those addresses from overlapping. Taking the maximum
    over *all* readers/writers gives ordinary inclusive interval allocation a
    sound interference proof without keeping every allocation to function end.

    Same-program calls retain their sequential order. This projection may be
    conservative within grouped programs; it never adds an execution edge.
    """
    programs = module.metadata.get("kernel_program_partition")
    ancestors = program_ancestors(programs)
    if function.name != module.entry or ancestors is None:
        return {}
    indexes = {node.id: index for index, node in enumerate(nodes)}
    by_call, ends, flat = {}, {}, []
    for program in programs:
        calls = program.get("calls")
        if not isinstance(calls, (tuple, list)) or not calls:
            raise IRVerificationError("Concurrent program lifetimes require non-empty call partitions.", stage=module.stage)
        for call in calls:
            if call not in indexes or call in by_call:
                raise IRVerificationError(
                    "Concurrent program lifetimes require unique actual entry call ids before allocation.", stage=module.stage)
            by_call[call] = program["symbol"]
            flat.append(indexes[call])
        ends[program["symbol"]] = max(indexes[call] for call in calls)
    if flat != sorted(flat) or any(node.op in {"tir.call", "builtin.call", "tir.kernel"}
                                  and node.id not in by_call for node in nodes):
        raise IRVerificationError(
            "Concurrent program lifetimes must cover all actual entry calls in original order before allocation.",
            stage=module.stage)
    unordered_ends = {
        symbol: max((end for other, end in ends.items()
                     if other != symbol and symbol not in ancestors[other]), default=-1)
        for symbol in ancestors
    }
    return {call: max(indexes[call], unordered_ends[symbol]) for call, symbol in by_call.items()}


__all__ = ["program_use_ends"]
