# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Prove explicit program dependencies from instantiated physical effects."""

from triton.flagmega.errors import CodegenError
from triton.flagmega.ir.program_order import program_ancestors
from triton.flagmega.passes.functions.graph import function_nodes
from triton.flagmega.passes.tir.bufferize.call_accesses import CallAccessResolver
from triton.flagmega.passes.tir.bufferize.synchronization import _node_accesses, _requires_device


def verify_program_dependencies(module, programs, runtime_binding, *, with_device_completion=False, with_accesses=False):
    ancestors = program_ancestors(programs)
    if ancestors is None:
        return None
    from triton.flagmega.ir import verify_buffer_plan

    plan = verify_buffer_plan(module)
    function = module.function_map[module.entry]
    nodes = {node.id: node for node in function_nodes(module, function)}
    bindings = dict(plan.function_map[function.name].values)
    resolver = CallAccessResolver(module, plan, _node_accesses)
    accesses = {}
    for program in programs:
        values = []
        for identity in program["calls"]:
            node = nodes.get(identity)
            if node is None or node.op not in {"tir.call", "tir.kernel"}:
                raise CodegenError("Concurrent kernel programs require complete straight-line call regions.")
            values.extend(resolver.accesses(function.name, node, bindings))
        accesses[program["symbol"]] = tuple(values)

    def interval(access):
        descriptor = plan.buffer_map[access.buffer]
        # Include all physical owner components, even when a local access
        # names only the executing CTA's component. This is conservative for
        # sparse owner subsets and handles strided compact-owner storage.
        owners = descriptor.mem_span.buffer.nbytes
        return access.offset, access.offset + max(access.nbytes, owners)

    for later_index, later in enumerate(programs):
        for earlier in programs[:later_index]:
            if earlier["symbol"] in ancestors[later["symbol"]]:
                continue
            if any(_requires_device(a) for a in accesses[earlier["symbol"]]) and any(
                _requires_device(a) for a in accesses[later["symbol"]]
            ):
                raise CodegenError(
                    "Concurrent peer-access programs require distinct distributed synchronization channels."
                )
            for left in accesses[earlier["symbol"]]:
                for right in accesses[later["symbol"]]:
                    if left.mode == right.mode == "read" or left.physical_id != right.physical_id:
                        continue
                    lo, hi = interval(left)
                    other_lo, other_hi = interval(right)
                    if lo < other_hi and other_lo < hi:
                        raise CodegenError(
                            "Concurrent kernel programs have an unordered physical memory hazard: "
                            f"{earlier['symbol']}:{left.buffer} and {later['symbol']}:{right.buffer}."
                        )
    externals = runtime_binding["arguments"]
    writes = {}
    for program in programs:
        written = {access.physical_id for access in accesses[program["symbol"]] if access.mode != "read"}
        writes[program["symbol"]] = tuple(
            index
            for index, binding in enumerate(externals)
            if str(plan.buffer_map[binding["buffer"]].physical_id) in written
        )
    # Kernel retirement orders local streams. It does not prove that another
    # chip has finished reading this chip's buffers. A program exposing peer
    # accesses needs an explicit terminal device rendezvous before successor
    # branches may reuse their storage.
    device_completion = {symbol: any(_requires_device(a) for a in values)
                         for symbol, values in accesses.items()}
    if with_accesses:
        return writes, device_completion, accesses
    return (writes, device_completion) if with_device_completion else writes


def elide_completed_device_barriers(events, symbol, programs, accesses, device_completion):
    """Remove only leading memory fences discharged by global predecessors.

    A peer program's terminal device rendezvous completes every ancestor on
    every chip. A newer chip-local producer is published to an ordered local
    consumer by stream completion. Prove those ranges independently when a
    single fence combines peer reuse and local RAW dependencies.
    """
    ancestors = program_ancestors(programs)
    earlier = []
    for program in programs:
        if program["symbol"] == symbol:
            break
        earlier.append(program["symbol"])
    completed = set().union(*(ancestors[g] | {g} for g in ancestors[symbol] if device_completion[g]))
    result, elided, leading = [], [], True
    for event in events:
        if event["kind"] != "barrier":
            leading = False
        ranges = event.get("ranges", ())
        if (leading and event.get("scope") == "device" and ranges
                and event.get("hazards") and event.get("before")):
            relevant, covered = set(), True
            for region in ranges:
                region_sources = set()
                matching = {}
                for previous in earlier:
                    values = tuple(a for a in accesses[previous]
                                   if a.physical_id == region["physical_id"]
                                   and a.offset < region["offset"] + region["nbytes"]
                                   and region["offset"] < a.offset + a.nbytes)
                    if values:
                        region_sources.add(previous)
                        matching[previous] = values
                unpublished = region_sources - completed
                local_readers = tuple(a for a in accesses[symbol]
                                      if a.physical_id == region["physical_id"]
                                      and a.offset < region["offset"] + region["nbytes"]
                                      and region["offset"] < a.offset + a.nbytes)
                local_completion = (
                    bool(local_readers)
                    and unpublished <= ancestors[symbol]
                    and not any(_requires_device(a) for a in local_readers)
                    and not any(_requires_device(a) for previous in unpublished for a in matching[previous])
                )
                covered &= bool(region_sources) and (not unpublished or local_completion)
                relevant.update(region_sources)
            if covered and relevant:
                elided.append(event)
                continue
        result.append(event)
    return result, elided
