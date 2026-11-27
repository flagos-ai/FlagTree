# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Fuse private attention materialization with a typed sigmoid/mul epilogue."""

from collections import Counter
from dataclasses import replace

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir import verify_module, logical_type
from triton.flagmega.ir.op_fusion import has_ops
from triton.flagmega.ir.ops.ntt.paged_attention_gated_combine import PagedAttentionGatedCombine


def _sigmoid(node):
    return not has_ops(node.attrs) and (node.op == "math.sigmoid" or
                                      node.op == "math.vectorized_unary" and node.attrs["unary_op"] == "sigmoid")


def fuse_attention_gate(module):
    """Prove the output layout before lowering; retain all gate-side views."""
    verify_module(module)
    nodes = module.node_map
    uses = Counter(value for node in module.nodes for value in node.inputs)
    uses.update(value for function in module.functions for value in (*function.parameters, *function.outputs))
    removed, replacements = set(), {}
    for root in module.nodes:
        if not root.effect.is_pure or root.attrs.get("post_ops") or not (
            root.op == "math.mul" or root.op == "math.vectorized_binary" and root.attrs["binary_op"] == "mul"
        ):
            continue
        for gate_index in (0, 1):
            gate_name = ("lhs", "rhs")[gate_index]
            pre = root.attrs.get("pre_ops", {})
            gate = nodes[root.inputs[gate_index]]
            private = []
            if pre:
                body = pre.get(gate_name)
                if set(pre) != {gate_name} or body is None or len(body.nodes) != 2:
                    continue
                if not _sigmoid(body.nodes[1]) or body.nodes[1].inputs != (body.parameter.id,) or body.output != body.nodes[1].id:
                    continue
            elif _sigmoid(gate) and uses[gate.id] == 1:
                private.append(gate.id)
                gate = nodes[gate.inputs[0]]
            else:
                continue
            value = nodes[root.inputs[1 - gate_index]]
            while value.op == "distributed.sharded_view" and uses[value.id] == 1 and not has_ops(value.attrs):
                private.append(value.id)
                value = nodes[value.inputs[0]]
            if (value.op != "ntt.paged_attention_combine" or uses[value.id] != 1 or has_ops(value.attrs)
                    or logical_type(value.type) != logical_type(root.type)):
                continue
            arguments = (*(nodes[key] for key in value.inputs), gate)
            try:
                prepared = PagedAttentionGatedCombine.prepare(arguments, {**value.attrs, "output_type": root.type})
            except IRSchemaError:
                continue
            if prepared.result_type != root.type:
                continue
            removed.update((*private, value.id))
            replacements[root.id] = replace(
                root, op=PagedAttentionGatedCombine.op_name, inputs=tuple(arg.id for arg in arguments),
                attrs=prepared.attrs, effect=prepared.effect,
                metadata={"introduced_by": "FuseAttentionGate"})
            break
    if not replacements:
        return module
    invalid_points = {point.id for point in module.selection_points if point.owner in removed or point.owner in replacements}
    return verify_module(replace(
        module, nodes=tuple(replacements.get(node.id, node) for node in module.nodes if node.id not in removed),
        selection_points=tuple(point for point in module.selection_points if point.id not in invalid_points),
        selections=tuple(selection for selection in module.selections if selection.point_id not in invalid_points)))
