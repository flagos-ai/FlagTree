# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Block-scaled FP8 matmul definition and its local behaviors."""

from typing import Mapping, Sequence

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import all_broadcast, placement_of, split_policy, tensor_of
from triton.flagmega.ir.model import DistributedType, IRType, Node, SBP, SBPPartial, tensor_type
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
    "math.block_scaled_matmul",
    namespace="math",
    functional_name="block_scaled_matmul",
    display_name="Math.BlockScaledMatMul",
)
class BlockScaledMatMul(OpDefinition):
    const_evaluable = True
    value = input_parameter(is_tensor() & has_rank(2))
    weight = input_parameter(is_tensor() & has_rank(2))
    weight_scale = input_parameter(is_tensor() & has_rank(2))
    weight_block_n = attribute_parameter()
    weight_block_k = attribute_parameter()

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        if set(attributes) != {"weight_block_n", "weight_block_k"}:
            raise IRSchemaError("BlockScaledMatMul requires weight_block_n and weight_block_k.")
        block_n = int(attributes["weight_block_n"])
        block_k = int(attributes["weight_block_k"])
        if block_n <= 0 or block_k <= 0:
            raise IRSchemaError("BlockScaledMatMul block sizes must be positive.")
        return {"weight_block_n": block_n, "weight_block_k": block_k}

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        value_type = cls.value.type_of(inputs)
        weight_type = cls.weight.type_of(inputs)
        scale_type = cls.weight_scale.type_of(inputs)
        value = tensor_of(value_type)
        weight = tensor_of(weight_type)
        if value.shape[1] != weight.shape[1]:
            raise IRSchemaError("BlockScaledMatMul value/weight K dimensions must match.")
        output = tensor_type(value.dtype, (value.shape[0], weight.shape[0]))
        placement = placement_of(value_type, weight_type, scale_type)
        if placement is None:
            return output
        distributed_inputs = (value_type, weight_type, scale_type)
        if not all(isinstance(item, DistributedType) for item in distributed_inputs):
            raise IRSchemaError("Distributed BlockScaledMatMul requires every tensor operand to name a placement.")
        if all(all_broadcast(item) for item in distributed_inputs):
            return DistributedType(output, (SBP.broadcast(), SBP.broadcast()), placement)
        output_split = split_policy(weight_type, 0)
        if output_split is not None and all_broadcast(value_type) and all_broadcast(scale_type):
            return DistributedType(output, (SBP.broadcast(), output_split), placement)
        reduction_split = split_policy(weight_type, 1)
        if (
            reduction_split is not None
            and reduction_split == split_policy(value_type, 1)
            and all_broadcast(scale_type)
        ):
            return DistributedType(
                output,
                (SBP.broadcast(), SBP.broadcast()),
                placement,
                partial=SBPPartial(reduction_split.hierarchy_axes),
            )
        raise IRSchemaError("BlockScaledMatMul distributed inputs do not describe replicated, output-, or K-sharding.")

    @classmethod
    def evaluate(cls, node, arguments, context):
        return block_scaled_linear(
            cls.value.read(arguments),
            cls.weight.read(arguments),
            cls.weight_scale.read(arguments),
            block_n=int(cls.weight_block_n.read(arguments, node.attrs)),
            block_k=int(cls.weight_block_k.read(arguments, node.attrs)),
            output_dtype=context.torch_dtype(node.type.dtype),
        )

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        output_size = tensor_nbytes(node.type) if hasattr(node.type, "shape") else None
        return OpCost(flops=None, bytes_read=None, bytes_written=output_size, notes=("block-fp8-matmul",))
