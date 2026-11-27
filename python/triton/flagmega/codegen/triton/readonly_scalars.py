# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Prove uniform readonly FP32 values without materializing tensor payloads."""

from math import isfinite
import struct

from triton.flagmega.ir import DType, DistributedType, TensorType, logical_type


_VALUE_PRESERVING = frozenset({
    "distributed.sharded_view", "distributed.boxing", "tensors.reshape", "tensors.permute",
    "tensors.broadcast_to", "tensors.slice",
})


class ReadonlyScalarAnalysis:
    """Invocation-local proof, never an assumption about a weight or parameter.

    Only scalar-element FP32 splats are currently lowered as literals. Casts,
    packed padding, partial components and arbitrary constant arithmetic are
    deliberately outside this proof. Unproved inputs keep their normal loads.
    """

    def __init__(self, module, plan):
        self.nodes = module.node_map
        self.recipes = {recipe.id: recipe for recipe in module.constant_recipes}
        self.recipe_nodes = {}
        self.plan = plan
        self.values = {}

    def float32_splat(self, buffer_id):
        if buffer_id in self.values:
            return self.values[buffer_id]
        buffer = self.plan.buffer_map[buffer_id]
        value = None
        if (buffer.storage == "rdata" and buffer.dtype is DType.FLOAT32 and buffer.nbytes
                and (buffer.distributed_type is None or buffer.distributed_type.partial is None)):
            node = self.nodes.get(buffer.source_node)
            if node is not None:
                value = self._node_value(node, self.nodes)
        self.values[buffer_id] = value
        return value

    def _node_value(self, node, nodes):
        seen = set()
        while node is not None:
            key = (id(nodes), node.id)
            if key in seen:
                return None
            seen.add(key)
            tensor = logical_type(node.type)
            if (not isinstance(tensor, TensorType) or tensor.dtype is not DType.FLOAT32
                    or isinstance(node.type, DistributedType) and node.type.partial is not None):
                return None
            if node.op == "builtin.splat_const":
                try:
                    value = struct.unpack("f", struct.pack("f", node.attrs["value"]))[0]
                except (OverflowError, struct.error):
                    return None
                return value if isfinite(value) else None
            if node.op == "builtin.const_asset":
                recipe_id, output = str(node.attrs["recipe"]), str(node.attrs["output"])
            elif node.op == "tir.buffer" and node.metadata.get("bufferized_from") == "builtin.const_asset":
                recipe_id = str(node.metadata["constant_recipe"])
                output = str(node.metadata["constant_output"])
            elif node.op in _VALUE_PRESERVING and len(node.inputs) == 1:
                node = nodes.get(node.inputs[0])
                continue
            else:
                return None
            if recipe_id not in self.recipes:
                return None
            if recipe_id not in self.recipe_nodes:
                self.recipe_nodes[recipe_id] = self.recipes[recipe_id].node_map
            nodes = self.recipe_nodes[recipe_id]
            node = nodes.get(output)
        return None


def annotate_readonly_scalars(calls, module, plan):
    """Attach proved values to fresh call bindings, leaving the physical ABI intact."""
    analysis = ReadonlyScalarAnalysis(module, plan)
    for call in calls:
        for parameter in call["inputs"]:
            for binding in parameter["buffers"]:
                value = analysis.float32_splat(binding["actual"])
                if value is not None:
                    binding["float32_splat"] = value
    return calls


__all__ = ["ReadonlyScalarAnalysis"]
