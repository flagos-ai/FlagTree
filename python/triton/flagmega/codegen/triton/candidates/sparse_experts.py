# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Explicit local expert stages with preserved per-route numerical boundaries."""

from triton.flagmega.errors import CodegenError
from itertools import product

from triton.flagmega.ir import DistributedType, VectorType, local_tensor_type, local_shard_descriptor, logical_type
from triton.flagmega.ir.ops.core import get_definition

from .core import TritonCandidateProposal


class SparseExpertsCandidateProvider:
    op_names = frozenset({"nn.sparse_experts_gate_up", "nn.sparse_experts_down",
                          "ntt.dispatched_experts_gate_up", "ntt.sparse_experts_down_combine",
                          "nn.sparse_experts_dispatch", "nn.sparse_experts_weighted_sum"})

    def propose(self, node, context):
        family = {"ntt.dispatched_experts_gate_up": "sparse_experts_gate_up",
                  "ntt.sparse_experts_down_combine": "sparse_experts_down"}.get(node.op, node.op.removeprefix("nn."))
        candidates = tuple(
            context.configure_implementation(implementation)
            for implementation in context.implementations(family, indexing="local", rounding="explicit")
            if _supports_local_n(implementation, node.type)
            and _supports_weight_tiles(implementation, node, context))
        if not candidates:
            return None
        return TritonCandidateProposal(
            candidates, context.choose_default(family, candidates, portable_fallback=f"tir.{family}.simt"))


def _supports_local_n(implementation, result_type):
    minimum = implementation.contract.get("min_local_n", 0)
    if type(minimum) is not int or minimum < 0:
        raise CodegenError("Expert min_local_n must be a non-negative integer.")
    if minimum == 0:
        return True
    tensor = local_tensor_type(result_type) if isinstance(result_type, DistributedType) else result_type
    extent = tensor.shape[-1].maximum
    lanes = tensor.dtype.lane_count if isinstance(tensor.dtype, VectorType) else 1
    return extent is not None and extent * lanes >= minimum


def _supports_weight_tiles(implementation, node, context):
    if not implementation.contract.get("requires_affine_weight_tiles", False):
        return True
    alignment = implementation.contract["weight_alignment_bytes"]
    definition = get_definition(node.op)
    for index, parameter in enumerate(definition.input_parameters):
        if parameter.name not in {"gate_weight", "up_weight", "down_weight"}:
            continue
        value = context.module.node_map[node.inputs[index]].type
        weight = logical_type(value)
        if (weight.dtype.value != implementation.contract["required_weight_dtype"]
                or any(not extent.is_fixed for extent in weight.shape)
                or weight.layout.strides or weight.layout.order):
            return False
        shape = tuple(extent.fixed_value for extent in weight.shape)
        if (max(shape) > implementation.contract["max_weight_axis_extent"]
                or shape[-1] * weight.dtype.itemsize % alignment):
            return False
        if isinstance(value, DistributedType):
            for owner in product(*(range(extent) for extent in value.placement.hierarchy)):
                descriptor = local_shard_descriptor(value, owner)
                if any(axis.affine_stride is None for axis in descriptor.axes) or descriptor.axes[-1].affine_stride != 1:
                    return False
                # Expert and row strides are already aligned by the dense K extent.
                if descriptor.axes[-1].map_local_to_global(0).fixed_value * weight.dtype.itemsize % alignment:
                    return False
    return True
