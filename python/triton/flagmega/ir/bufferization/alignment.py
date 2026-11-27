# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Proven view alignment from physical roots and complete typed caller bindings.

A descriptor's historical ``alignment`` field is not an allocation guarantee.
This analysis starts at PhysicalBuffer contracts, meets *all* callers of a
callee root, and reduces through offsets and component/scope strides. Unknown
external roots, symbolic offsets and recursive edges cannot invent a proof.
The BufferPlan is unchanged; this is a cached, derived fact for codegen.
"""
from math import gcd


def proven_buffer_alignments(plan):
    incoming = {}
    roots = {}
    for function in plan.functions:
        for call in function.calls:
            incoming.setdefault(call.callee, []).append(call.call)
            for formal_id, actual_id in (*call.arguments, *call.results):
                formal = plan.buffer_map[formal_id]
                physical = formal.mem_span.buffer
                if physical.memory_space == 'external' and physical.function == call.callee:
                    roots.setdefault(physical.id, {}).setdefault(call.call, []).append((formal, actual_id))
    external_entries = {plan.buffer_map[buffer_id].physical_id
                        for _, ids in (*plan.entry_inputs, *plan.entry_outputs) for buffer_id in ids}
    root_cache, visiting = {}, set()

    def view_alignment(buffer):
        start = buffer.mem_span.start
        if not start.is_fixed:
            return 1
        alignment = gcd(root_alignment(buffer.mem_span.buffer), start.fixed_value)
        if buffer.component_stride_bytes:
            alignment = gcd(alignment, buffer.component_stride_bytes)
        space = plan.memory_space_map.get(buffer.mem_span.buffer.memory_space)
        if (space is not None and space.allocation_scope.value == 'function'
                and space.sharing_scope.value == 'block' and space.kind != 'shared'):
            alignment = gcd(alignment, plan.function_memory_space_bytes(buffer.function, space.name))
        return alignment

    def root_alignment(physical):
        if physical.id in root_cache:
            return root_cache[physical.id]
        declared = physical.alignment
        if physical.id in visiting or physical.id in external_entries:
            return declared
        calls = incoming.get(physical.function, ())
        bindings = roots.get(physical.id, {})
        if not calls or any(call not in bindings for call in calls):
            return declared
        visiting.add(physical.id)
        common = 0
        for call in calls:
            for formal, actual_id in bindings[call]:
                # Subtracting a formal view offset from its bound actual
                # address preserves only their gcd alignment.
                start = formal.mem_span.start
                actual = view_alignment(plan.buffer_map[actual_id])
                bound = gcd(actual, start.fixed_value) if start.is_fixed else 1
                common = gcd(common, bound)
        visiting.remove(physical.id)
        # The declared root contract remains a precondition of the function;
        # inferred alignment can only strengthen it when every caller proves it.
        root_cache[physical.id] = max(declared, common)
        return root_cache[physical.id]

    return {buffer.id: view_alignment(buffer) for buffer in plan.buffers}
