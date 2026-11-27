# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Explicit FP32 split projection and BF16-rounded complete GLU boundary."""
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir import DistributedType, SBP, tensor_type
from triton.flagmega.ir.distributed_inference import placement_of, tensor_of
from triton.flagmega.ir.ops.core import OpCost, OpDefinition, input_parameter, op_definition, tensor_nbytes
from triton.flagmega.ir.type_pattern import has_rank, is_tensor
from triton.flagmega.ir.types import DType


def _shape(value, dtype, shape, label):
    tensor = tensor_of(value)
    if tensor.dtype != dtype or tuple(d.fixed_value if d.is_fixed else None for d in tensor.shape) != shape:
        raise IRSchemaError(f'{label} requires {dtype} {shape}, got {tensor}.')
    if isinstance(value, DistributedType) and value.partial is not None:
        raise IRSchemaError(f'{label} cannot consume an unresolved partial value.')


@op_definition('ntt.split_k_dense_glu_projection', namespace='ntt', functional_name='split_k_dense_glu_projection',
               display_name='NTT.SplitKDenseGluProjection')
class SplitKDenseGluProjection(OpDefinition):
    supports_broadcast_lifting = False
    value = input_parameter(is_tensor() & has_rank(2))
    weight = input_parameter(is_tensor() & has_rank(2))

    @classmethod
    def infer_type(cls, inputs, attrs):
        a, w = (v.type for v in inputs)
        _shape(a, DType.BFLOAT16, (1, 2048), 'Split GU input')
        _shape(w, DType.BFLOAT16, (2048, 12288), 'Split GU paired weight')
        output = tensor_type(DType.FLOAT32, (2, 12288))
        placement = placement_of(a, w)
        if placement is None:
            return output
        if placement.hierarchy != (48, 2) or placement.hierarchy_levels != 'bb':
            raise IRSchemaError('Split GU projection requires its explicit 48-by-2 physical domain.')
        if not all(
                isinstance(v, DistributedType) and all(p == SBP.broadcast() for p in v.axis_policies) for v in (a, w)):
            raise IRSchemaError('Split GU projection requires canonical full global input and paired weight.')
        return DistributedType(output, (SBP.split_contiguous((1, )), SBP.split_contiguous((0, ))), placement)

    @classmethod
    def evaluate(cls, node, arguments, context):
        a, w = cls.value.read(arguments), cls.weight.read(arguments)
        return context.torch.stack(
            tuple(a[:, s * 1024:(s + 1) * 1024].float().matmul(w[s * 1024:(s + 1) * 1024].float()).reshape(12288)
                  for s in range(2)))

    @classmethod
    def cost(cls, node):
        return OpCost(flops=50331648, bytes_written=tensor_nbytes(node.type),
                      notes=('two-ordered-1024-FP32-split-projections', ))


@op_definition('ntt.split_k_dense_glu_epilogue', namespace='ntt', functional_name='split_k_dense_glu_epilogue',
               display_name='NTT.SplitKDenseGluEpilogue')
class SplitKDenseGluEpilogue(OpDefinition):
    supports_broadcast_lifting = False
    partial = input_parameter(is_tensor() & has_rank(2))

    @classmethod
    def infer_type(cls, inputs, attrs):
        p = inputs[0].type
        _shape(p, DType.FLOAT32, (2, 12288), 'Split GU epilogue partial')
        output = tensor_type(DType.BFLOAT16, (1, 6144))
        placement = placement_of(p)
        if placement is None:
            return output
        if (placement.hierarchy != (24, ) or placement.hierarchy_levels != 'b'
                or any(policy != SBP.broadcast() for policy in p.axis_policies)):
            raise IRSchemaError('Split GU epilogue requires canonical complete partials in its 24-owner domain.')
        return DistributedType(output, (SBP.broadcast(), SBP.split_contiguous((0, ))), placement)

    @classmethod
    def evaluate(cls, node, arguments, context):
        p = cls.partial.read(arguments)
        complete = p[0].float() + p[1].float()
        gate = complete[:6144].to(context.torch.bfloat16).float()
        up = complete[6144:].to(context.torch.bfloat16).float()
        return (context.torch.nn.functional.silu(gate) * up).to(context.torch.bfloat16).reshape(1, 6144)

    @classmethod
    def cost(cls, node):
        return OpCost(bytes_read=98304, bytes_written=tensor_nbytes(node.type),
                      notes=('sum-two-FP32-rows-round-G-and-U-BF16-FP32-SiLU-final-BF16', ))


__all__ = ['SplitKDenseGluProjection', 'SplitKDenseGluEpilogue']
