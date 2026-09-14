# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Materialize split attention and apply an explicit sigmoid gate."""

from dataclasses import replace

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.memory_effect import MemoryEffect
from triton.flagmega.ir.ops.core import OpDefinition, attribute_parameter, input_parameter, op_definition, tensor_nbytes
from triton.flagmega.ir.ops.ntt.paged_attention_combine import PagedAttentionCombine
from triton.flagmega.ir.type_pattern import is_tensor
from triton.flagmega.ir.types import DType, VectorType


@op_definition("ntt.paged_attention_gated_combine", namespace="ntt",
               functional_name="paged_attention_gated_combine", display_name="NTT.PagedAttentionGatedCombine")
class PagedAttentionGatedCombine(OpDefinition):
    max_state = input_parameter(is_tensor(), memory_effect=MemoryEffect.READ.across_partial_owners())
    sum_state = input_parameter(is_tensor(), memory_effect=MemoryEffect.READ.across_partial_owners())
    acc_state = input_parameter(is_tensor(), memory_effect=MemoryEffect.READ.across_partial_owners())
    gate = input_parameter(is_tensor())
    layout = attribute_parameter()
    hidden_size = attribute_parameter()
    output_data_type = attribute_parameter()
    output_type = attribute_parameter()
    split_hierarchy_axis = attribute_parameter()
    split_count = attribute_parameter()

    @classmethod
    def normalize_attrs(cls, attrs):
        return PagedAttentionCombine.normalize_attrs(attrs)

    @classmethod
    def ir_attrs(cls, attrs):
        return PagedAttentionCombine.ir_attrs(attrs)

    @classmethod
    def infer_type(cls, inputs, attrs):
        output = PagedAttentionCombine.infer_type(inputs[:3], attrs)
        dtype = tensor_of(output).dtype
        scalar = dtype.elem_type if isinstance(dtype, VectorType) else dtype
        if scalar not in {DType.BFLOAT16, DType.FLOAT16, DType.FLOAT32}:
            raise IRSchemaError("Gated attention combine requires a floating-point result")
        if cls.gate.type_of(inputs) != output:
            raise IRSchemaError("Gated attention combine requires gate and output to have identical types and owners")
        return output

    @classmethod
    def evaluate(cls, node, arguments, context):
        # Both producer results round before multiplication, as in unfused IR.
        value = PagedAttentionCombine.evaluate(node, arguments[:3], context)
        gate = cls.gate.read(arguments)
        return (value * gate.float().sigmoid().to(gate.dtype)).to(value.dtype)

    @classmethod
    def cost(cls, node):
        base = PagedAttentionCombine.cost(node)
        return replace(base, model="flagmega.paged-attention-gated-combine/v1",
                       notes=(*base.notes, "sigmoid-gate-epilogue", "no-attention-intermediate"),
                       bytes_written=tensor_nbytes(tensor_of(node.type)))
