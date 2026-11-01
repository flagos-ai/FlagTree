# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""NTT physical bufferization and memory-hazard realization."""

from __future__ import annotations

from dataclasses import replace

from triton.flagmega.ir import (
    IRModule,
    Node,
    kernel_dispatch_of,
    make_buffer_plan,
    verify_buffer_plan,
)
from triton.flagmega.passes.tir import (
    bind_prim_function_buffers,
    materialize_memory_synchronization,
    specialize_prim_functions_for_buffer_layouts,
)
from triton.flagmega.passes.tir.bufferize import plan_memory_synchronization
from triton.flagmega.passes.tir.bufferize.planner import BufferizationOptions
from triton.flagmega.passes.tir.bufferize.allocation_session import AllocationSession
from triton.flagmega.passes.tir.plan_function_memory import plan_function_memory


class NttBufferizationPolicy:
    def __init__(self, options: BufferizationOptions, *, synchronization_protocol_provider=None) -> None:
        self.options = options
        self.synchronization_protocol_provider = synchronization_protocol_provider

    def with_options(self, options):
        return type(self)(options, synchronization_protocol_provider=self.synchronization_protocol_provider)

    def with_optimization_level(self, level: str) -> NttBufferizationPolicy:
        return self.with_options(replace(self.options, optimization_level=level))

    def _with_protocol(self, module):
        from triton.flagmega.ir.bufferization.synchronization_protocol import PROTOCOL_METADATA
        if self.synchronization_protocol_provider is None:
            return module
        protocol = self.synchronization_protocol_provider(module)
        data = None if protocol is None else protocol.to_data()
        current = module.metadata.get(PROTOCOL_METADATA)
        if current == data:
            return module
        metadata = dict(module.metadata)
        if data is None:
            metadata.pop(PROTOCOL_METADATA, None)
        else:
            metadata[PROTOCOL_METADATA] = data
        return replace(module, metadata=metadata)

    def plan_function_memory(self, module: IRModule) -> IRModule:
        return plan_function_memory(self._with_protocol(module), self.options)

    def bufferize(self, module: IRModule) -> IRModule:
        module = self._with_protocol(module)
        session = AllocationSession(self.options)
        plan = make_buffer_plan(
            module,
            options=self.options,
            allocation_session=session,
        )
        specialized = specialize_prim_functions_for_buffer_layouts(module, plan)
        if specialized is not module:
            module = specialized
            # PrimFunction names own shared-workspace identities and kernel
            # call records, so a specialized call graph requires a fresh plan.
            plan = make_buffer_plan(module, options=self.options, allocation_session=session)
        module = bind_prim_function_buffers(module, plan=plan)
        nodes: list[Node] = []
        for node in module.nodes:
            if node.op == "builtin.weight":
                nodes.append(replace(
                    node,
                    op="tir.buffer",
                    attrs={**dict(node.attrs), "storage": "rdata", "alignment": plan.alignment},
                    metadata={**dict(node.metadata), "bufferized_from": "builtin.weight"},
                ))
            elif node.op == "builtin.const_asset":
                asset_name = self._constant_asset_name(module, node)
                nodes.append(replace(
                    node,
                    op="tir.buffer",
                    attrs={
                        "name": asset_name,
                        "source": "constant_recipe",
                        "key": node.id,
                        "storage": "rdata",
                        "alignment": plan.alignment,
                    },
                    metadata={
                        **dict(node.metadata),
                        "bufferized_from": "builtin.const_asset",
                        "constant_recipe": str(node.attrs["recipe"]),
                        "constant_output": str(node.attrs["output"]),
                    },
                ))
            elif node.op == "distributed.sharded_view":
                # Distribution materialization uses this node to carry the
                # logical coordinate map into buffer planning.  Once every
                # value has a concrete MemSpan, retain that information as a
                # TIR buffer view instead of leaking a distributed-dialect op
                # into bufferized TIR or silently erasing the alias.
                nodes.append(replace(
                    node,
                    op="tir.buffer_view",
                    attrs={"alias_kind": "sharded_view"},
                    metadata={
                        **dict(node.metadata),
                        "bufferized_from": "distributed.sharded_view",
                    },
                ))
            else:
                nodes.append(node)
        metadata = {
            **dict(module.metadata),
            "buffer_plan": plan.to_data(),
            "kernel_sequence": [
                node.id for node in nodes
                if node.op == "tir.kernel" or (
                    node.op == "tir.call"
                    and (
                        function := module.kernel_callable_map.get(str(node.attrs.get("callee", "")))
                    ) is not None
                    and kernel_dispatch_of(function) is not None
                )
            ],
        }
        return replace(module, nodes=tuple(nodes), metadata=metadata)

    def plan_memory_synchronization(self, module: IRModule) -> IRModule:
        plan = verify_buffer_plan(module)
        synchronization = plan_memory_synchronization(module, plan)
        materialized = materialize_memory_synchronization(
            module, synchronization
        )
        return replace(materialized, metadata={
            **dict(module.metadata),
            "memory_synchronization": synchronization.to_data(),
            "synchronization": [value.to_data() for value in synchronization.events],
        })

    @staticmethod
    def _constant_asset_name(module: IRModule, asset: Node) -> str:
        recipe_id = str(asset.attrs["recipe"])
        recipe = next(recipe for recipe in module.constant_recipes if recipe.id == recipe_id)
        reachable: set[str] = set()
        pending = [str(asset.attrs["output"])]
        while pending:
            node_id = pending.pop()
            if node_id in reachable:
                continue
            reachable.add(node_id)
            pending.extend(recipe.node_map[node_id].inputs)
        weights = [
            node for node in recipe.nodes
            if node.id in reachable and node.op == "builtin.weight"
        ]
        if len(weights) == 1:
            return str(weights[0].attrs["name"])
        return asset.id


__all__ = ["NttBufferizationPolicy"]
