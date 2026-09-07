# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Fused block-scaled MatMulGlu definition and its local behaviors."""

from typing import Mapping, Sequence

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import all_broadcast, placement_of, split_policy, tensor_of
from triton.flagmega.ir.model import DistributedType, IRType, Node, SBP, tensor_type
from triton.flagmega.ir.type_pattern import has_rank, is_tensor
from triton.flagmega.ir.ops.math._block_scaled import block_scaled_linear
from triton.flagmega.ir.ops.core import (
    OpCost,
    OpDefinition,
    attribute_parameter,
    input_parameter,
    op_definition,
    tensor_nbytes,
)


@op_definition(
    "nn.matmul_glu",
    namespace="nn",
    functional_name="matmul_glu",
    display_name="NN.MatMulGlu",
)
class MatMulGlu(OpDefinition):
    value = input_parameter(is_tensor() & has_rank(2))
    gate_weight = input_parameter(is_tensor() & has_rank(2))
    up_weight = input_parameter(is_tensor() & has_rank(2))
    gate_scale = input_parameter(is_tensor() & has_rank(2))
    up_scale = input_parameter(is_tensor() & has_rank(2))
    activation = attribute_parameter()
    weight_block_n = attribute_parameter()
    weight_block_k = attribute_parameter()

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        required = {"activation", "weight_block_n", "weight_block_k"}
        if set(attributes) != required:
            raise IRSchemaError(f"MatMulGlu requires attributes {sorted(required)}.")
        if str(attributes["activation"]) != "silu":
            raise IRSchemaError("P0 MatMulGlu only supports SiLU activation.")
        block_n = int(attributes["weight_block_n"])
        block_k = int(attributes["weight_block_k"])
        if block_n <= 0 or block_k <= 0:
            raise IRSchemaError("MatMulGlu block sizes must be positive.")
        return {"activation": "silu", "weight_block_n": block_n, "weight_block_k": block_k}

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        input_types = tuple(parameter.type_of(inputs) for parameter in cls.input_parameters)
        value_type, gate_type, up_type, gate_scale_type, up_scale_type = input_types
        value = tensor_of(value_type)
        gate = tensor_of(gate_type)
        up = tensor_of(up_type)
        if up != gate:
            raise IRSchemaError("MatMulGlu gate/up weight types must match.")
        if value.shape[1] != gate.shape[1]:
            raise IRSchemaError("MatMulGlu value/weight K dimensions must match.")
        output = tensor_type(value.dtype, (value.shape[0], gate.shape[0]))
        placement = placement_of(*input_types)
        if placement is None:
            return output
        if not all(isinstance(item, DistributedType) for item in input_types):
            raise IRSchemaError("Distributed MatMulGlu requires every tensor operand to name a placement.")
        if all(all_broadcast(item) for item in input_types):
            return DistributedType(output, (SBP.broadcast(), SBP.broadcast()), placement)
        gate_split = split_policy(gate_type, 0)
        if (
            gate_split is not None
            and gate_split == split_policy(up_type, 0)
            and all_broadcast(value_type)
            and all_broadcast(gate_scale_type)
            and all_broadcast(up_scale_type)
        ):
            return DistributedType(output, (SBP.broadcast(), gate_split), placement)
        raise IRSchemaError("MatMulGlu distributed inputs do not describe replicated or output sharding.")

    @classmethod
    def evaluate(cls, node, arguments, context):
        gate = block_scaled_linear(
            cls.value.read(arguments), cls.gate_weight.read(arguments), cls.gate_scale.read(arguments),
            block_n=int(cls.weight_block_n.read(arguments, node.attrs)),
            block_k=int(cls.weight_block_k.read(arguments, node.attrs)),
            output_dtype=context.torch_dtype(node.type.dtype),
        )
        up = block_scaled_linear(
            cls.value.read(arguments), cls.up_weight.read(arguments), cls.up_scale.read(arguments),
            block_n=int(cls.weight_block_n.read(arguments, node.attrs)),
            block_k=int(cls.weight_block_k.read(arguments, node.attrs)),
            output_dtype=context.torch_dtype(node.type.dtype),
        )
        return context.torch.nn.functional.silu(gate) * up

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        size = tensor_nbytes(node.type) if hasattr(node.type, "shape") else None
        return OpCost(flops=None, bytes_read=None, bytes_written=size, notes=("two-block-fp8-matmuls+glu",))
