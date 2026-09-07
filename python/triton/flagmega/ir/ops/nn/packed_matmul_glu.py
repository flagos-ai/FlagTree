# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Packed fused block-scaled MatMul/GLU operation."""

from __future__ import annotations

from typing import Mapping, Sequence

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.dim_expr import equivalent_dim
from triton.flagmega.ir.distributed_inference import all_broadcast, placement_of, split_policy, tensor_of
from triton.flagmega.ir.distributed_type import SBP
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
    "nn.packed_matmul_glu",
    namespace="nn",
    functional_name="packed_matmul_glu",
    display_name="NN.PackedMatMulGlu",
)
class PackedMatMulGlu(OpDefinition):
    value = input_parameter(is_tensor() & has_rank(2))
    gate_weight = input_parameter(is_tensor() & has_rank(2))
    up_weight = input_parameter(is_tensor() & has_rank(2))
    gate_scale = input_parameter(is_tensor() & has_rank(2))
    up_scale = input_parameter(is_tensor() & has_rank(2))
    activation = attribute_parameter()
    weight_block_n = attribute_parameter()
    weight_block_k = attribute_parameter()
    k_pack = attribute_parameter()
    k_vector = attribute_parameter()
    packed_layout = attribute_parameter(default="n_major_k_packed")

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        result = {
            "activation": str(attrs["activation"]),
            "weight_block_n": int(attrs["weight_block_n"]),
            "weight_block_k": int(attrs["weight_block_k"]),
            "k_pack": int(attrs["k_pack"]),
            "k_vector": int(attrs["k_vector"]),
            "packed_layout": str(attrs["packed_layout"]),
        }
        if result["activation"] != "silu":
            raise IRSchemaError("PackedMatMulGlu currently supports only SiLU.")
        if any(result[name] <= 0 for name in ("weight_block_n", "weight_block_k", "k_pack", "k_vector")):
            raise IRSchemaError("PackedMatMulGlu block/vector sizes must be positive.")
        if result["packed_layout"] != "n_major_k_packed":
            raise IRSchemaError("PackedMatMulGlu requires n_major_k_packed layout.")
        return result

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        value_type = cls.value.type_of(inputs)
        gate_type = cls.gate_weight.type_of(inputs)
        up_type = cls.up_weight.type_of(inputs)
        value = tensor_of(value_type)
        gate = tensor_of(gate_type)
        up = tensor_of(up_type)
        if gate != up or not isinstance(gate.dtype, VectorType):
            raise IRSchemaError("PackedMatMulGlu gate/up weights must have identical VectorType tensor types.")
        expected_lanes = (int(attrs["k_pack"]), int(attrs["k_vector"]))
        if gate.dtype.lanes != expected_lanes:
            raise IRSchemaError(f"Packed weight lanes must be {expected_lanes}, got {gate.dtype.lanes}.")
        logical_k = gate.shape[1] * gate.dtype.lane_count
        if not equivalent_dim(value.shape[1], logical_k):
            raise IRSchemaError("PackedMatMulGlu value/weight logical K dimensions must match.")
        output = tensor_type(value.dtype, (value.shape[0], gate.shape[0]))
        gate_scale_type = cls.gate_scale.type_of(inputs)
        up_scale_type = cls.up_scale.type_of(inputs)
        distributed_inputs = (
            value_type, gate_type, up_type, gate_scale_type, up_scale_type)
        placement = placement_of(*distributed_inputs)
        if placement is None:
            return output
        if not all(isinstance(item, DistributedType) for item in distributed_inputs):
            raise IRSchemaError(
                "Distributed PackedMatMulGlu requires every tensor operand to name a placement.")
        if all(all_broadcast(item) for item in distributed_inputs):
            return DistributedType(output, (SBP.broadcast(), SBP.broadcast()), placement)
        gate_split = split_policy(gate_type, 0)
        if (
            gate_split is None
            or gate_split != split_policy(up_type, 0)
            or not all_broadcast(value_type)
            or not all_broadcast(gate_scale_type)
            or not all_broadcast(up_scale_type)
        ):
            raise IRSchemaError("PackedMatMulGlu gate/up weights must use the same output split.")
        return DistributedType(output, (SBP.broadcast(), gate_split), placement)

    @classmethod
    def evaluate(cls, node, arguments, context):
        gate_weight = cls.gate_weight.read(arguments)
        up_weight = cls.up_weight.read(arguments)
        common = {
            "block_n": int(cls.weight_block_n.read(arguments, node.attrs)),
            "block_k": int(cls.weight_block_k.read(arguments, node.attrs)),
            "output_dtype": context.torch_dtype(tensor_of(node.type).dtype),
        }
        gate = block_scaled_linear(
            cls.value.read(arguments), gate_weight.reshape(gate_weight.shape[0], -1),
            cls.gate_scale.read(arguments), **common)
        up = block_scaled_linear(
            cls.value.read(arguments), up_weight.reshape(up_weight.shape[0], -1),
            cls.up_scale.read(arguments), **common)
        return context.torch.nn.functional.silu(gate) * up

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        size = tensor_nbytes(node.type) if isinstance(node.type, TensorType) else None
        return OpCost(flops=None, bytes_read=None, bytes_written=size, notes=("packed-two-matmuls+glu",))


__all__ = ["PackedMatMulGlu"]
