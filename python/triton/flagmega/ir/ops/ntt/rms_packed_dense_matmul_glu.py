# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Complete-row FP32 RMS, a BF16 materialization, and packed dense GLU."""
from math import isfinite, prod

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import all_broadcast, tensor_of
from triton.flagmega.ir.model import DistributedType, Node
from triton.flagmega.ir.memory_effect import MemoryEffect
from triton.flagmega.ir.ops.core import (
    OpCost, OpCostFactors, OpDefinition, attribute_parameter, input_parameter,
    op_definition, tensor_nbytes,
)
from triton.flagmega.ir.ops.nn.norm_apply import NormApply
from triton.flagmega.ir.ops.nn.norm_stats import NormStats
from triton.flagmega.ir.ops.nn.packed_dense_matmul_glu import PackedDenseMatMulGlu
from triton.flagmega.ir.type_pattern import has_rank, is_tensor
from triton.flagmega.ir.types import DType


@op_definition('ntt.rms_packed_dense_matmul_glu', namespace='ntt',
               functional_name='rms_packed_dense_matmul_glu',
               display_name='NTT.RmsPackedDenseMatMulGlu')
class RmsPackedDenseMatMulGlu(OpDefinition):
    """The raw operand is never the BF16 input of the inner GLU.

    ``norm_tree_group_size`` and ``norm_partial_group_size`` describe two
    contiguous butterflies; a third reduces their remaining partials.
    Every butterfly uses descending XOR partners. Scale/bias are explicit values,
    including when a selected implementation proves that bias is zero.
    The evaluator uses separate FP32 operations; fast native lowering may
    contract multiply/add and use approximate reciprocal square root.
    """
    raw = input_parameter(is_tensor() & has_rank(2), memory_effect=MemoryEffect.CHIP_READ)
    scale = input_parameter(is_tensor() & has_rank(1), memory_effect=MemoryEffect.CHIP_READ)
    bias = input_parameter(is_tensor() & has_rank(1), memory_effect=MemoryEffect.CHIP_READ)
    gate_weight = input_parameter(is_tensor() & has_rank(4))
    up_weight = input_parameter(is_tensor() & has_rank(4))
    epsilon = attribute_parameter()
    norm_tree_group_size = attribute_parameter()
    norm_partial_group_size = attribute_parameter()
    norm_math_mode = attribute_parameter(default='fast')
    round_before_scale = attribute_parameter(default=False)
    activation = attribute_parameter(default='silu')
    round_activation = attribute_parameter(default=True)
    gate_clamp_max = attribute_parameter(default=None)
    up_clamp = attribute_parameter(default=None)
    packed_layout = attribute_parameter(default='k_major_n8_k16')
    reduction_order = attribute_parameter(default='relaxed')
    supports_broadcast_lifting = False

    @classmethod
    def normalize_attrs(cls, attributes):
        attrs = super().normalize_attrs(attributes)
        for name in ('norm_tree_group_size', 'norm_partial_group_size'):
            group = attrs[name]
            if type(group) is not int or group <= 0 or group & (group - 1):
                raise IRSchemaError(f'RmsPackedDenseMatMulGlu {name} must be a positive power of two.')
        epsilon = float(attrs['epsilon'])
        if not isfinite(epsilon) or epsilon <= 0:
            raise IRSchemaError('RmsPackedDenseMatMulGlu epsilon must be finite and positive.')
        norm = NormApply.normalize_attrs(norm_apply_attrs(attrs))
        glu = PackedDenseMatMulGlu.normalize_attrs(glu_attrs(attrs))
        return {**glu, 'epsilon': epsilon,
                'norm_tree_group_size': attrs['norm_tree_group_size'],
                'norm_partial_group_size': attrs['norm_partial_group_size'],
                'norm_math_mode': norm.get('math_mode', 'fast'),
                'round_before_scale': norm['round_before_scale']}

    @classmethod
    def infer_type(cls, inputs, attrs):
        raw = tensor_of(cls.raw.type_of(inputs))
        if raw.dtype is not DType.FLOAT32 or raw.rank != 2:
            raise IRSchemaError('RmsPackedDenseMatMulGlu raw must be scalar FP32 [M,K]; use a proven byte-preserving view for vectors.')
        k = raw.shape[1].fixed_value
        group = attrs['norm_tree_group_size'] * attrs['norm_partial_group_size']
        if k is None or k <= 0 or k & (k - 1) or k % group:
            raise IRSchemaError('RmsPackedDenseMatMulGlu fixed K must be a power of two divisible by both tree levels.')
        for name, operand in (('scale', cls.scale.read(inputs)), ('bias', cls.bias.read(inputs))):
            value = tensor_of(operand.type)
            if value.dtype is not DType.BFLOAT16 or value.rank != 1 or value.shape != raw.shape[1:]:
                raise IRSchemaError(f'RmsPackedDenseMatMulGlu {name} must be BF16 [K].')
        for operand in inputs[:3]:
            if isinstance(operand.type, DistributedType) and (
                    not all_broadcast(operand.type) or operand.type.partial is not None):
                raise IRSchemaError('RmsPackedDenseMatMulGlu raw/affine inputs must be materialized all-broadcast values.')
        _, _, glu = composition_nodes(inputs, attrs)
        return glu.type

    @classmethod
    def evaluate(cls, node, arguments, context):
        inputs = tuple(Node(key, 'builtin.var', (), context.types[key], attrs={'name': key}) for key in node.inputs)
        stats, normalized, glu = composition_nodes(inputs, node.attrs)
        values = tuple(arguments)
        with context.type_scope((*inputs, stats, normalized, glu)):
            square_sum = hierarchical_square_sum(values[0], node.attrs, context.torch)
            norm_value = NormApply.evaluate(normalized, (values[0], square_sum, values[1], values[2]), context)
            return PackedDenseMatMulGlu.evaluate(glu, (norm_value, values[3], values[4]), context)

    @classmethod
    def cost(cls, node):
        return OpCost(flops=None, bytes_read=None, bytes_written=tensor_nbytes(tensor_of(node.type)),
                      notes=('complete-row-rms+bf16-materialization+packed-glu', 'normalization-replicated-per-output-owner'))

    @classmethod
    def cost_factors(cls, inputs, attrs, return_type):
        from triton.flagmega.ir.distributed_type import local_tensor_type
        local = lambda t: local_tensor_type(t) if isinstance(t, DistributedType) else tensor_of(t)
        tensors = tuple(local(value.type) for value in inputs)
        output = local(return_type)
        if any(not d.is_fixed for t in (*tensors, output) for d in t.shape):
            return None
        m, k = (d.fixed_value for d in tensors[0].shape)
        n = output.shape[1].fixed_value
        return OpCostFactors(
            simt_fma_operations=2 * m * n * k,
            elementwise_operations=7 * m * k + 5 * m * n,
            block_local_memory_load_bytes=sum(prod(d.fixed_value for d in t.shape) * t.dtype.itemsize for t in tensors),
            block_local_memory_store_bytes=m * n * output.dtype.itemsize,
        )


def norm_stats_attrs(attrs):
    # Shape inference only: the new operation owns the explicit three-level
    # numerical contract; existing NormStats has no such multi-level attr.
    return {'axis': -1, 'use_mean': False}


def hierarchical_square_sum(raw, attrs, torch):
    """Separate-RN hierarchy reference, not a bit oracle for fast native FMA."""
    first, second = attrs['norm_tree_group_size'], attrs['norm_partial_group_size']
    square = torch.zeros_like(raw, dtype=torch.float32) + raw.float().square()
    outer = tuple(square.shape[:-1])
    for group in (first, second, square.shape[-1] // (first * second)):
        square = square.reshape(*outer, -1, group)
        lanes = torch.arange(group, device=raw.device)
        mask = group // 2
        while mask:
            square = square + square.index_select(-1, lanes ^ mask)
            mask //= 2
        square = square[..., 0]
    # The ordinary collective adds the complete sum to its positive-zero seed.
    return (square + 0.0).reshape(1, *outer, 1)


def norm_apply_attrs(attrs):
    return {'axis': -1, 'epsilon': attrs['epsilon'], 'use_mean': False,
            'round_before_scale': attrs.get('round_before_scale', False),
            'input_dtype': 'float32', 'output_dtype': 'bfloat16',
            'math_mode': attrs.get('norm_math_mode', 'fast')}


def glu_attrs(attrs):
    return {key: attrs[key] for key in (
        'activation', 'round_activation', 'gate_clamp_max', 'up_clamp', 'packed_layout', 'reduction_order'
    ) if key in attrs}


def composition_nodes(inputs, attrs):
    """Typed reference composition, also reused by selection's shape proof."""
    raw, scale, bias, gate, up = inputs
    stats_attrs = NormStats.normalize_attrs(norm_stats_attrs(attrs))
    stats = Node('<rms_glu_stats>', 'builtin.var', (),
                 NormStats.infer_type((raw,), stats_attrs), attrs={'name': '<rms_glu_stats>'})
    apply_attrs = NormApply.normalize_attrs(norm_apply_attrs(attrs))
    normalized = Node('<rms_glu_normalized>', NormApply.op_name, (raw.id, stats.id, scale.id, bias.id),
                      NormApply.infer_type((raw, stats, scale, bias), apply_attrs), attrs=apply_attrs)
    packed_attrs = PackedDenseMatMulGlu.normalize_attrs(glu_attrs(attrs))
    glu = Node('<rms_glu_result>', PackedDenseMatMulGlu.op_name, (normalized.id, gate.id, up.id),
               PackedDenseMatMulGlu.infer_type((normalized, gate, up), packed_attrs), attrs=packed_attrs)
    return stats, normalized, glu


__all__ = ['RmsPackedDenseMatMulGlu', 'composition_nodes', 'glu_attrs', 'norm_apply_attrs', 'norm_stats_attrs',
           'hierarchical_square_sum']
