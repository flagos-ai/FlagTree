# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Instantiate reusable-function memory accesses in each caller's byte arena.

The function's parameter types alone cannot summarize its owner accesses:
a local shard parameter can be gathered by the body. Keep individual effects,
aliases and frame ranges instead of classifying the entire call as collective.
"""

from dataclasses import replace

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.passes.tir.bufferize.graph import function_nodes


def physical_identity(plan, function_name, descriptor):
    space = plan.memory_space_map[descriptor.mem_span.buffer.memory_space]
    if space.supports_lifetime_reuse and space.allocation_scope.value == "function" and space.kind != "shared":
        return f"{space.name}:@{function_name}"
    return str(descriptor.physical_id)


class CallAccessResolver:
    """Memoized, transitive summaries with concrete call-frame substitution.

Summaries conservatively retain all body accesses for the duration of a call.
Internal barriers do not discharge the caller's outstanding hazards. This is
sound for both RAW and frame/alias WAR/WAW, without adding a grid fence around
calls whose relevant operands are accessed only by their local owners.
"""

    def __init__(self, module, plan, leaf_accesses):
        self.module = module
        self.plan = plan
        self.leaf_accesses = leaf_accesses
        self.cache = {}
        self.active = set()

    def accesses(self, function_name, node, bindings):
        call = self.plan.call_map.get(node.id)
        if call is None:
            return self.leaf_accesses(self.module, self.plan, function_name, node, bindings)
        if call.caller != function_name:
            raise IRVerificationError("Memory access call binding belongs to a different caller.")
        formal_bindings = {}
        for formal_id, actual_id in (*call.arguments, *call.results):
            formal = self.plan.buffer_map[formal_id]
            actual = self.plan.buffer_map[actual_id]
            formal_bindings.setdefault(str(formal.physical_id), []).append((formal, actual))
        result = []
        for access in self._summary(call.callee):
            descriptor = self.plan.buffer_map[access.buffer]
            candidates = formal_bindings.get(str(descriptor.physical_id), ())
            if candidates:
                # Multiple ABI names can denote one alias group. Prefer the
                # exact buffer, then a containing span, rather than losing a
                # RefSlice's nonzero offset by mapping only its root name.
                pairs = sorted(candidates, key=lambda pair: pair[0].id != access.buffer)
                translated = None
                for formal, actual in pairs:
                    formal_span = formal.physical_access_span
                    start, end = formal_span.absolute_start.minimum, formal_span.absolute_end.maximum
                    if start is None or end is None:
                        raise IRVerificationError("Call memory effects require bounded formal spans.")
                    if start <= access.offset and access.offset + access.nbytes <= end:
                        actual_start = actual.physical_access_span.absolute_start.minimum
                        if actual_start is None:
                            raise IRVerificationError("Call memory effects require bounded actual spans.")
                        translated = replace(
                            access,
                            node=node.id,
                            buffer=actual.id,
                            storage=actual.storage,
                            physical_id=physical_identity(self.plan, function_name, actual),
                            offset=actual_start + access.offset - start,
                            distributed_storage_kind=actual.distributed_storage_kind,
                            sharing_scope=self.plan.memory_space_map[actual.mem_span.buffer.memory_space].sharing_scope,
                        )
                        break
                if translated is None:
                    raise IRVerificationError(f"Callee access {access.buffer!r} exceeds its bound formal span.")
                # A private callee selector is not a caller SSA value. Static
                # partitions stay exact; unresolved symbolic ones must not
                # establish disjointness across independent invocations.
                result.append(translated)
                continue
            space = descriptor.mem_span.buffer.memory_space
            pool = call.memory_pool_map.get(space)
            if pool is None:
                raise IRVerificationError(f"Call {node.id!r} has no binding for accessed pool {space!r}.")
            result.append(
                replace(access, node=node.id, storage=space, physical_id=f"{space}:@{function_name}",
                        offset=pool.offset + access.offset))
        return tuple(result)

    def _summary(self, function_name):
        if function_name in self.cache:
            return self.cache[function_name]
        if function_name in self.active:
            raise IRVerificationError("Memory effects cannot summarize recursive function calls.")
        self.active.add(function_name)
        try:
            bindings = dict(self.plan.function_map[function_name].values)
            accesses = tuple(access for node in function_nodes(self.module, self.module.function_map[function_name])
                             if node.op in {"tir.kernel", "tir.call"}
                             for access in self.accesses(function_name, node, bindings))
            self.cache[function_name] = accesses
            return accesses
        finally:
            self.active.remove(function_name)


__all__ = ["CallAccessResolver", "physical_identity"]
