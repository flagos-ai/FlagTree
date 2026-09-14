# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Expose owner sums, then fuse only private, owner-local expert stages."""

from collections import Counter
from dataclasses import replace

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir import DistributedType, VectorType, verify_module
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.op_fusion import has_ops
from triton.flagmega.ir.ops.nn.sparse_experts_combine import SparseExpertsWeightedSum
from triton.flagmega.ir.ops.ntt.sparse_experts import DispatchedExpertsGateUp, SparseExpertsDownCombine
from triton.flagmega.rules.neutral._utility import make_node


def lower_sparse_experts(module):
    nodes, changed = [], set()
    source = module.node_map
    for node in module.nodes:
        if node.op != "nn.sparse_experts_combine":
            nodes.append(node)
            continue
        local = make_node(SparseExpertsWeightedSum.op_name, node.id + ".weighted_sum",
                          tuple(source[key] for key in node.inputs), node.attrs, node.metadata)
        prefix = [local]
        if isinstance(local.type, DistributedType) and local.type.partial is not None:
            local = make_node("distributed.boxing", node.id + ".owner_sum", (local,),
                              {"new_type": replace(local.type, partial=None)}, node.metadata)
            prefix.append(local)
        if local.type != node.type:
            dtype = tensor_of(node.type).dtype
            if isinstance(dtype, VectorType):
                local = make_node("ntt.vectorized_cast", node.id, (local,),
                                  {"new_type": dtype, "vectorize_axes": (-1,)}, node.metadata)
            else:
                local = make_node("tensors.cast", node.id, (local,), {"dtype": dtype}, node.metadata)
            prefix.append(local)
        else:
            prefix[-1] = replace(local, id=node.id)
        assert prefix[-1].type == node.type
        nodes.extend(prefix)
        changed.add(node.id)
    module = _replace_nodes(module, nodes, changed)

    uses = Counter(key for node in module.nodes for key in node.inputs)
    uses.update(key for function in module.functions for key in (*function.parameters, *function.outputs))
    sources, removed, replaced = dict(module.node_map), set(), {}
    for node in module.nodes:
        if has_ops(node.attrs):
            continue
        inputs = tuple(sources[key] for key in node.inputs)
        if not inputs or uses[inputs[0].id] != 1 or has_ops(inputs[0].attrs):
            continue
        producer = inputs[0]
        if node.op == "nn.sparse_experts_gate_up" and producer.op == "nn.sparse_experts_dispatch":
            if producer.inputs[1] != node.inputs[1]:
                continue
            definition = DispatchedExpertsGateUp
            arguments = (sources[producer.inputs[0]], *inputs[1:])
            attrs = node.attrs
        elif node.op == "nn.sparse_experts_weighted_sum" and producer.op == "nn.sparse_experts_down":
            definition = SparseExpertsDownCombine
            arguments = (*(sources[key] for key in producer.inputs), inputs[1])
            attrs = {**producer.attrs, **node.attrs}
        elif node.op in {"tensors.cast", "ntt.vectorized_cast"} and producer.op == SparseExpertsDownCombine.op_name:
            definition = SparseExpertsDownCombine
            arguments = tuple(sources[key] for key in producer.inputs)
            attrs = {**producer.attrs, "output_dtype": tensor_of(node.type).dtype, "cast_output": True}
            # The coefficient rounding boundary is part of the fused contract.
            if producer.attrs["round_weighted_output"]:
                from triton.flagmega.ir.types import data_type
                if data_type(producer.attrs["output_dtype"]) != tensor_of(node.type).dtype:
                    continue
        else:
            continue
        try:
            fused = make_node(definition.op_name, node.id, arguments, attrs, node.metadata)
        except IRSchemaError:
            continue
        if fused.type != node.type:
            continue
        removed.add(producer.id)
        replaced[node.id] = sources[node.id] = fused
    return _replace_nodes(module, tuple(replaced.get(n.id, n) for n in module.nodes if n.id not in removed),
                          removed | replaced.keys())


def _replace_nodes(module, nodes, changed):
    if not changed:
        return module
    invalid = {point.id for point in module.selection_points if point.owner in changed}
    return verify_module(replace(module, nodes=tuple(nodes),
                                 selection_points=tuple(p for p in module.selection_points if p.id not in invalid),
                                 selections=tuple(s for s in module.selections if s.point_id not in invalid)))
