# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Block-scaled FP8 matmul definition and its local behaviors."""

from math import prod
from typing import Mapping, Sequence

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.dim_expr import try_div_exactly
from triton.flagmega.ir.distributed_inference import all_broadcast, placement_of, split_policy, tensor_of
from triton.flagmega.ir.distributed_type import scale_split_units
from triton.flagmega.ir.model import SBP, DistributedType, IRType, Node, SBPPartial, tensor_type
from triton.flagmega.ir.ops.core import (
    OpCost,
    OpDefinition,
    attribute_parameter,
    input_parameter,
    op_definition,
    tensor_nbytes,
)
from triton.flagmega.ir.ops.math._block_scaled import (
    block_scaled_linear,
    normalize_block_scaled_precision,
    normalize_output_lanes,
    verify_block_scaled_precision_inputs,
)
from triton.flagmega.ir.ops.math._quantized import matrix_width
from triton.flagmega.ir.type_pattern import has_rank, is_tensor
from triton.flagmega.ir.types import DType, VectorType


@op_definition(
    "math.block_scaled_matmul",
    namespace="math",
    functional_name="block_scaled_matmul",
    display_name="Math.BlockScaledMatMul",
)
class BlockScaledMatMul(OpDefinition):
    """Project from FP8 storage with explicit operand quantization/rounding.

    ``activation_quantization='none'`` preserves the original activation.
    ``weight_dequant_dtype`` optionally rounds FP32(code * block_scale) to
    F16/BF16 before multiplication, independently of the stored FP8 dtype.
    Defaults retain dynamic activation quantization and unrounded weights.
    """

    const_evaluable = True
    value = input_parameter(is_tensor() & has_rank(2))
    weight = input_parameter(is_tensor() & has_rank(2))
    weight_scale = input_parameter(is_tensor() & has_rank(2))
    weight_block_n = attribute_parameter()
    weight_block_k = attribute_parameter()
    weight_encoding = attribute_parameter(default=None)
    output_data_type = attribute_parameter(default=None)
    output_lanes = attribute_parameter(default=())
    activation_quantization = attribute_parameter(default="dynamic_fp8")
    weight_dequant_dtype = attribute_parameter(default=None)

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        block_n = int(attrs["weight_block_n"])
        block_k = int(attrs["weight_block_k"])
        if block_n <= 0 or block_k <= 0:
            raise IRSchemaError("BlockScaledMatMul block sizes must be positive.")
        result = {"weight_block_n": block_n, "weight_block_k": block_k,
                  "output_lanes": normalize_output_lanes(attrs["output_lanes"], "BlockScaledMatMul")}
        encoding = attrs.get("weight_encoding")
        if encoding is not None:
            if not isinstance(encoding, str):
                raise IRSchemaError("BlockScaledMatMul weight_encoding must be a string or None.")
            result["weight_encoding"] = encoding
        if attrs.get("output_data_type") is not None:
            result["output_data_type"] = DType(attrs["output_data_type"]).value
        result.update(normalize_block_scaled_precision(attrs, "BlockScaledMatMul"))
        return result

    @classmethod
    def ir_attrs(cls, attrs: Mapping[str, object]) -> Mapping[str, object]:
        return {key: value for key, value in attrs.items() if key != "output_lanes" or value}

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        value_type = cls.value.type_of(inputs)
        weight_type = cls.weight.type_of(inputs)
        scale_type = cls.weight_scale.type_of(inputs)
        value = tensor_of(value_type)
        weight = tensor_of(weight_type)
        if isinstance(value.dtype, VectorType) or isinstance(weight.dtype, VectorType):
            raise IRSchemaError("BlockScaledMatMul value/weight must use scalar element types.")
        verify_block_scaled_precision_inputs(value, weight, tensor_of(scale_type), attrs, "BlockScaledMatMul")
        if value.shape[1] != matrix_width(weight, attrs.get("weight_encoding")):
            raise IRSchemaError("BlockScaledMatMul value/weight K dimensions must match.")
        lanes = tuple(attrs.get("output_lanes", ()))
        output_n = try_div_exactly(weight.shape[0], prod(lanes))
        if output_n is None:
            raise IRSchemaError("BlockScaledMatMul N must be divisible by output_lanes.")
        scalar_dtype = DType(attrs["output_data_type"]) if attrs.get("output_data_type") else value.dtype
        output = tensor_type(VectorType(scalar_dtype, lanes) if lanes else scalar_dtype, (value.shape[0], output_n))
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
            packed_split = scale_split_units(output_split, 1, prod(lanes))
            if packed_split is None:
                raise IRSchemaError("BlockScaledMatMul N split must align with output_lanes.")
            return DistributedType(output, (SBP.broadcast(), packed_split), placement)
        reduction_split = split_policy(weight_type, 1)
        if (attrs.get("weight_encoding") is None and reduction_split is not None
                and reduction_split == split_policy(value_type, 1) and all_broadcast(scale_type)):
            return DistributedType(
                output,
                (SBP.broadcast(), SBP.broadcast()),
                placement,
                partial=SBPPartial(reduction_split.hierarchy_axes),
            )
        raise IRSchemaError("BlockScaledMatMul distributed inputs do not describe replicated, output-, or K-sharding.")

    @classmethod
    def evaluate(cls, node, arguments, context):
        dtype = tensor_of(node.type).dtype
        output = block_scaled_linear(
            cls.value.read(arguments),
            cls.weight.read(arguments),
            cls.weight_scale.read(arguments),
            block_n=int(cls.weight_block_n.read(arguments, node.attrs)),
            block_k=int(cls.weight_block_k.read(arguments, node.attrs)),
            output_dtype=context.torch_dtype(dtype.elem_type if isinstance(dtype, VectorType) else dtype),
            activation_quantization=node.attrs.get("activation_quantization", "dynamic_fp8"),
            weight_dequant_dtype=node.attrs.get("weight_dequant_dtype"),
        )
        lanes = tuple(node.attrs.get("output_lanes", ()))
        return output.reshape(output.shape[0], output.shape[1] // prod(lanes), *lanes)

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        output_size = tensor_nbytes(node.type) if hasattr(node.type, "shape") else None
        return OpCost(flops=None, bytes_read=None, bytes_written=output_size, notes=("block-fp8-matmul",))
