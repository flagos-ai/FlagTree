# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Rank-2 scalar matrix multiplication."""

from __future__ import annotations

from typing import Mapping, Sequence

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import all_broadcast, placement_of, split_policy, tensor_of
from triton.flagmega.ir.distributed_type import SBPSplit
from triton.flagmega.ir.model import DistributedType, IRType, Node, SBP, SBPPartial, TensorType, tensor_type
from triton.flagmega.ir.ops.core import OpCost, OpDefinition, attribute_parameter, input_parameter, op_definition, tensor_nbytes
from triton.flagmega.ir.type_pattern import is_tensor
from triton.flagmega.ir.types import VectorType


@op_definition("math.matmul", namespace="math", functional_name="matmul", display_name="Math.MatMul")
class MatMul(OpDefinition):
    const_evaluable = True
    lhs = input_parameter(is_tensor())
    rhs = input_parameter(is_tensor())
    transpose_a = attribute_parameter(default=False)
    transpose_b = attribute_parameter(default=False)

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        if any(not isinstance(attrs[name], bool) for name in ("transpose_a", "transpose_b")):
            raise IRSchemaError("F.math.matmul transpose flags must be bool.")
        return attrs

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        lhs_type = cls.lhs.type_of(inputs)
        rhs_type = cls.rhs.type_of(inputs)
        lhs = tensor_of(lhs_type)
        rhs = tensor_of(rhs_type)
        if lhs.rank != 2 or rhs.rank != 2 or lhs.dtype != rhs.dtype or isinstance(lhs.dtype, VectorType):
            raise IRSchemaError("F.math.matmul requires rank-2 scalar tensors with matching dtype.")
        lhs_m_axis, lhs_k_axis = (1, 0) if attrs["transpose_a"] else (0, 1)
        rhs_k_axis, rhs_n_axis = (1, 0) if attrs["transpose_b"] else (0, 1)
        lm, lk = lhs.shape[lhs_m_axis], lhs.shape[lhs_k_axis]
        rk, rn = rhs.shape[rhs_k_axis], rhs.shape[rhs_n_axis]
        if lk != rk:
            raise IRSchemaError(f"F.math.matmul reduction dimensions differ: {lk} versus {rk}.")
        logical_output = tensor_type(lhs.dtype, (lm, rn), layout=lhs.layout)
        placement = placement_of(lhs_type, rhs_type)
        if placement is None:
            return logical_output
        if not isinstance(lhs_type, DistributedType) or not isinstance(rhs_type, DistributedType):
            raise IRSchemaError("Distributed MatMul requires both operands to name a placement.")

        lhs_m_split = split_policy(lhs_type, lhs_m_axis)
        rhs_n_split = split_policy(rhs_type, rhs_n_axis)
        lhs_k_split = split_policy(lhs_type, lhs_k_axis)
        rhs_k_split = split_policy(rhs_type, rhs_k_axis)
        if lhs_m_split is not None and all_broadcast(rhs_type):
            return DistributedType(logical_output, (lhs_m_split, SBP.broadcast()), placement)
        if rhs_n_split is not None and all_broadcast(lhs_type):
            return DistributedType(logical_output, (SBP.broadcast(), rhs_n_split), placement)
        if lhs_k_split is not None and lhs_k_split == rhs_k_split:
            other_lhs = lhs_type.axis_policies[lhs_m_axis]
            other_rhs = rhs_type.axis_policies[rhs_n_axis]
            output_policies = (other_lhs, other_rhs)
            reduction_axes = tuple(sorted(set(lhs_k_split.hierarchy_axes)))
            output_axes = {
                hierarchy_axis
                for policy in output_policies
                if isinstance(policy, SBPSplit)
                for hierarchy_axis in policy.hierarchy_axes
            }
            if (
                lhs_type.partial is None
                and rhs_type.partial is None
                and all(
                    policy == SBP.broadcast() or isinstance(policy, SBPSplit)
                    for policy in output_policies
                )
                and not output_axes.intersection(reduction_axes)
            ):
                return DistributedType(
                    logical_output,
                    output_policies,
                    placement,
                    partial=SBPPartial(reduction_axes),
                )
        if all_broadcast(lhs_type) and all_broadcast(rhs_type):
            return DistributedType(
                logical_output, (SBP.broadcast(), SBP.broadcast()), placement)
        raise IRSchemaError(
            "Distributed MatMul operand policies do not form a legal M, N, K, "
            "or disjoint output/reduction partition."
        )

    @classmethod
    def evaluate(cls, node, arguments, context):
        lhs = cls.lhs.read(arguments)
        rhs = cls.rhs.read(arguments)
        if node.attrs["transpose_a"]:
            lhs = lhs.transpose(-2, -1)
        if node.attrs["transpose_b"]:
            rhs = rhs.transpose(-2, -1)
        return lhs @ rhs

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        assert isinstance(node.type, TensorType)
        size = tensor_nbytes(node.type)
        return OpCost(flops=None, bytes_read=None, bytes_written=size, notes=("matmul",))


__all__ = ["MatMul"]
