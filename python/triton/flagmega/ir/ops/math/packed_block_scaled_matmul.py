# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""N-major/K-vector packed block-scaled FP8 matmul."""

from __future__ import annotations

from typing import Mapping, Sequence

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.dim_expr import equivalent_dim
from triton.flagmega.ir.distributed_inference import all_broadcast, placement_of, split_policy, tensor_of
from triton.flagmega.ir.distributed_type import SBP, SBPPartial, scale_split_units
from triton.flagmega.ir.model import DistributedType, IRType, Node, TensorType, tensor_type
from triton.flagmega.ir.ops.core import (
    OpCost,
    OpDefinition,
    attribute_parameter,
    input_parameter,
    op_definition,
    tensor_nbytes,
)
from triton.flagmega.ir.ops.math._block_scaled import block_scaled_linear
from triton.flagmega.ir.type_pattern import has_rank, is_tensor
from triton.flagmega.ir.types import VectorType


@op_definition(
    "math.packed_block_scaled_matmul",
    namespace="math",
    functional_name="packed_block_scaled_matmul",
    display_name="Math.PackedBlockScaledMatMul",
)
class PackedBlockScaledMatMul(OpDefinition):
    const_evaluable = True
    value = input_parameter(is_tensor() & has_rank(2))
    weight = input_parameter(is_tensor() & has_rank(2))
    weight_scale = input_parameter(is_tensor() & has_rank(2))
    weight_block_n = attribute_parameter()
    weight_block_k = attribute_parameter()
    k_pack = attribute_parameter()
    k_vector = attribute_parameter()
    packed_layout = attribute_parameter(default="n_major_k_packed")

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        result = {
            "weight_block_n": int(attrs["weight_block_n"]),
            "weight_block_k": int(attrs["weight_block_k"]),
            "k_pack": int(attrs["k_pack"]),
            "k_vector": int(attrs["k_vector"]),
            "packed_layout": str(attrs["packed_layout"]),
        }
        if any(result[name] <= 0 for name in ("weight_block_n", "weight_block_k", "k_pack", "k_vector")):
            raise IRSchemaError("PackedBlockScaledMatMul block/vector sizes must be positive.")
        if result["packed_layout"] != "n_major_k_packed":
            raise IRSchemaError("PackedBlockScaledMatMul requires n_major_k_packed layout.")
        return result

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        value_type = cls.value.type_of(inputs)
        weight_type = cls.weight.type_of(inputs)
        value = tensor_of(value_type)
        weight = tensor_of(weight_type)
        if not isinstance(weight.dtype, VectorType):
            raise IRSchemaError("PackedBlockScaledMatMul weight must use VectorType.")
        expected_lanes = (int(attrs["k_pack"]), int(attrs["k_vector"]))
        if weight.dtype.lanes != expected_lanes:
            raise IRSchemaError(f"Packed weight lanes must be {expected_lanes}, got {weight.dtype.lanes}.")
        logical_k = weight.shape[1] * weight.dtype.lane_count
        if not equivalent_dim(value.shape[1], logical_k):
            raise IRSchemaError("PackedBlockScaledMatMul value/weight logical K dimensions must match.")
        output = tensor_type(value.dtype, (value.shape[0], weight.shape[0]))
        scale_type = cls.weight_scale.type_of(inputs)
        placement = placement_of(value_type, weight_type, scale_type)
        if placement is None:
            return output
        distributed_inputs = (value_type, weight_type, scale_type)
        if not all(isinstance(item, DistributedType) for item in distributed_inputs):
            raise IRSchemaError(
                "Distributed PackedBlockScaledMatMul requires every tensor operand to name a placement.")
        if all(all_broadcast(item) for item in distributed_inputs):
            return DistributedType(output, (SBP.broadcast(), SBP.broadcast()), placement)
        output_split = split_policy(weight_type, 0)
        reduction_split = split_policy(weight_type, 1)
        value_reduction_split = split_policy(value_type, 1)
        logical_reduction_split = (
            None
            if reduction_split is None
            else scale_split_units(reduction_split, weight.dtype.lane_count, 1)
        )
        if output_split is not None and all_broadcast(value_type) and all_broadcast(scale_type):
            return DistributedType(output, (SBP.broadcast(), output_split), placement)
        if (
            reduction_split is not None
            and logical_reduction_split == value_reduction_split
            and all_broadcast(scale_type)
        ):
            return DistributedType(
                output,
                (SBP.broadcast(), SBP.broadcast()),
                placement,
                partial=SBPPartial(reduction_split.hierarchy_axes),
            )
        raise IRSchemaError(
            "PackedBlockScaledMatMul distributed inputs do not describe replicated, output-, or K-sharding.")

    @classmethod
    def evaluate(cls, node, arguments, context):
        weight = cls.weight.read(arguments)
        logical_weight = weight.reshape(weight.shape[0], -1)
        return block_scaled_linear(
            cls.value.read(arguments),
            logical_weight,
            cls.weight_scale.read(arguments),
            block_n=int(cls.weight_block_n.read(arguments, node.attrs)),
            block_k=int(cls.weight_block_k.read(arguments, node.attrs)),
            output_dtype=context.torch_dtype(tensor_of(node.type).dtype),
        )

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        size = tensor_nbytes(node.type) if isinstance(node.type, TensorType) else None
        return OpCost(flops=None, bytes_read=None, bytes_written=size, notes=("packed-block-fp8-matmul",))


__all__ = ["PackedBlockScaledMatMul"]
