# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Sum-partial RMS statistics, complete-row BF16 normalization, and packed GLU."""
from dataclasses import replace
from math import prod

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import all_broadcast, tensor_of
from triton.flagmega.ir.distributed_type import ReduceOp, chip_data_type, local_tensor_type
from triton.flagmega.ir.model import DistributedType, Node
from triton.flagmega.ir.memory_effect import MemoryEffect
from triton.flagmega.ir.ops.core import OpCost, OpCostFactors, OpDefinition, attribute_parameter, input_parameter, op_definition, tensor_nbytes
from triton.flagmega.ir.ops.nn.norm_apply import NormApply
from triton.flagmega.ir.ops.nn.packed_dense_matmul_glu import PackedDenseMatMulGlu
from triton.flagmega.ir.ops.ntt.rms_packed_dense_matmul_glu import RmsPackedDenseMatMulGlu, composition_nodes
from triton.flagmega.ir.type_pattern import has_rank, is_tensor
from triton.flagmega.ir.types import DType


def _compatible_statistics_domain(partial, statistics):
    # Removing Partial here compares coordinate domains only. The real input
    # retains its original owners, additive meaning and across-owner read effect.
    domain = replace(partial, partial=None)
    if domain == statistics:
        return True
    # New cross-placement support is confined to block meshes on one chip.
    # Existing same-placement device hierarchies keep their original contract.
    if (not isinstance(statistics, DistributedType)
            or set(domain.placement.hierarchy_levels) != {'b'}
            or set(statistics.placement.hierarchy_levels) != {'b'}
            or domain.tensor != statistics.tensor):
        return False
    chip_domain = chip_data_type(domain)
    return chip_domain is not None and chip_domain == chip_data_type(statistics)


@op_definition('ntt.partial_rms_packed_dense_matmul_glu', namespace='ntt',
               functional_name='partial_rms_packed_dense_matmul_glu',
               display_name='NTT.PartialRmsPackedDenseMatMulGlu')
class PartialRmsPackedDenseMatMulGlu(OpDefinition):
    """Consume actual Sum-partial owners; the sixth input is not broadcast data.

    Each partial is the retained local ``norm_tree_group_size`` statistic.
    The consumer reduces contiguous groups of ``norm_partial_group_size``
    with descending-XOR butterflies, then their remaining groups identically.
    Raw/affine normalization and its BF16 materialization precede ordinary GLU.
    Statistics may retain another block mesh on the same chip; this is a read
    of their original owners, not a redistribution or broadcast of partials.
    Fast native lowering retains its FMA/approximate-rsqrt contract; reference
    evaluation is mathematical and does not promise native bitwise rounding.
    """
    raw = input_parameter(is_tensor() & has_rank(2), memory_effect=MemoryEffect.CHIP_READ)
    scale = input_parameter(is_tensor() & has_rank(1), memory_effect=MemoryEffect.CHIP_READ)
    bias = input_parameter(is_tensor() & has_rank(1), memory_effect=MemoryEffect.CHIP_READ)
    gate_weight = input_parameter(is_tensor() & has_rank(4))
    up_weight = input_parameter(is_tensor() & has_rank(4))
    partial_stats = input_parameter(is_tensor() & has_rank(3), memory_effect=MemoryEffect.READ.across_partial_owners())
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
        return RmsPackedDenseMatMulGlu.normalize_attrs(attributes)

    @classmethod
    def infer_type(cls, inputs, attrs):
        output = RmsPackedDenseMatMulGlu.infer_type(inputs[:5], attrs)
        partial = cls.partial_stats.type_of(inputs)
        raw = cls.raw.type_of(inputs)
        stats, _, _ = composition_nodes(inputs[:5], attrs)
        if (not isinstance(raw, DistributedType) or not isinstance(partial, DistributedType)
                or partial.tensor.dtype is not DType.FLOAT32
                or partial.partial is None or partial.partial.reduce_op is not ReduceOp.SUM
                or not partial.partial.axes or partial.exclusive is not None
                or not all_broadcast(replace(partial, partial=None))
                or not _compatible_statistics_domain(partial, stats.type)):
            raise IRSchemaError('PartialRmsPackedDenseMatMulGlu requires scalar FP32 Sum-partial broadcast statistics with a compatible chip domain and RMS shape.')
        owners = prod(partial.placement.hierarchy[axis] for axis in partial.partial.axes)
        k = tensor_of(raw).shape[-1].fixed_value
        if owners * attrs['norm_tree_group_size'] != k:
            raise IRSchemaError('PartialRmsPackedDenseMatMulGlu owner count must cover exactly one local statistic per raw tree group.')
        return output

    @classmethod
    def evaluate(cls, node, arguments, context):
        inputs = tuple(Node(key, 'builtin.var', (), context.types[key], attrs={'name': key}) for key in node.inputs)
        stats, normalized, glu = composition_nodes(inputs[:5], node.attrs)
        # Distributed reference arguments denote their full additive logical
        # value, exactly as GatherReduceNormApply; physical peer loads belong
        # to the execution backend, not an invented broadcast [owners] type.
        with context.type_scope((*inputs, stats, normalized, glu)):
            norm = NormApply.evaluate(normalized, (arguments[0], arguments[5], arguments[1], arguments[2]), context)
            return PackedDenseMatMulGlu.evaluate(glu, (norm, arguments[3], arguments[4]), context)

    @classmethod
    def cost(cls, node):
        return OpCost(flops=None, bytes_read=None, bytes_written=tensor_nbytes(tensor_of(node.type)),
                      notes=('sum-partial-rms+bf16-materialization+packed-glu', 'normalization-replicated-per-output-owner'))

    @classmethod
    def cost_factors(cls, inputs, attrs, return_type):
        local = lambda t: local_tensor_type(t) if isinstance(t, DistributedType) else tensor_of(t)
        tensors = tuple(local(value.type) for value in inputs)
        output = local(return_type)
        if any(not d.is_fixed for t in (*tensors, output) for d in t.shape):
            return None
        m, k = (d.fixed_value for d in tensors[0].shape)
        n = output.shape[1].fixed_value
        owners = prod(inputs[5].type.placement.hierarchy[axis] for axis in inputs[5].type.partial.axes)
        return OpCostFactors(simt_fma_operations=2 * m * n * k,
            elementwise_operations=5 * m * k + m * (owners - 1) + 5 * m * n,
            block_local_memory_load_bytes=sum(prod(d.fixed_value for d in t.shape) * t.dtype.itemsize for t in tensors[:5]) + m * owners * 4,
            block_local_memory_store_bytes=m * n * output.dtype.itemsize)


__all__ = ['PartialRmsPackedDenseMatMulGlu']
