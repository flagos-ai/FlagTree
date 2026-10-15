# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Publish head-major [K | V] chunks without materializing either field."""

from math import prod

from triton.flagmega.errors import EvaluationError, IRSchemaError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.distributed_type import SBPSplit, local_shape
from triton.flagmega.ir.memory_effect import MemoryEffect
from triton.flagmega.ir.model import DistributedType, DType, effect, tensor_type
from triton.flagmega.ir.ops.core import (
    OpCost,
    OpCostFactors,
    OpDefinition,
    ParameterKind,
    attribute_parameter,
    input_parameter,
    op_definition,
    tensor_nbytes,
)
from triton.flagmega.ir.ops.nn._attention_layout import normalize_attention_layout, to_seq_head_dim
from triton.flagmega.ir.ops.nn._paged_attention_state import (
    PagedAttentionState,
    paged_attention_state_config_from_type,
)
from triton.flagmega.ir.ops.nn.qwen3_paged_attention import _scalar_bool, _scalar_int
from triton.flagmega.ir.ops.tensors.unpack import unpack_physical
from triton.flagmega.ir.type_pattern import has_dtype, has_rank, is_ref, is_tensor
from triton.flagmega.ir.types import VectorType


@op_definition(
    "nn.update_paged_attention_kv_cache_pair", namespace="nn",
    functional_name="update_paged_attention_kv_cache_pair",
    display_name="NN.UpdatePagedAttentionKVCachePair",
)
class UpdatePagedAttentionKVCachePair(OpDefinition):
    slots = input_parameter(is_tensor() & has_rank(3))
    state = input_parameter(
        is_ref(), parameter_kind=ParameterKind.ATTRIBUTE,
        memory_effect=MemoryEffect.CHIP_READ_WRITE.partitioned_by_argument(2),
    )
    layer_id = input_parameter(is_tensor() & has_rank(0) & has_dtype(DType.INT32),
                               parameter_kind=ParameterKind.ATTRIBUTE)
    advance_sequence = input_parameter(is_tensor() & has_rank(0) & has_dtype(DType.BOOL),
                                       parameter_kind=ParameterKind.ATTRIBUTE)
    layout = attribute_parameter()

    @classmethod
    def normalize_attrs(cls, attributes):
        attrs = super().normalize_attrs(attributes)
        return {"layout": normalize_attention_layout(attrs["layout"])}

    @classmethod
    def infer_type(cls, inputs, attrs):
        slots_type = cls.slots.type_of(inputs)
        slots = tensor_of(slots_type)
        config = paged_attention_state_config_from_type(cls.state.type_of(inputs))
        scalar = slots.dtype.elem_type if isinstance(slots.dtype, VectorType) else slots.dtype
        lanes = prod(slots.dtype.lanes) if isinstance(slots.dtype, VectorType) else 1
        layout = tuple(attrs["layout"])
        if scalar != DType.BFLOAT16:
            raise IRSchemaError("UpdatePagedAttentionKVCachePair requires bfloat16 slots.")
        if any(not dim.is_fixed for dim in slots.shape):
            raise IRSchemaError("UpdatePagedAttentionKVCachePair requires static chunk geometry.")
        if (slots.shape[layout.index("seq")].fixed_value <= 0
                or slots.shape[layout.index("head")].fixed_value != config.num_kv_heads
                or slots.shape[layout.index("dim")].fixed_value * lanes != 2 * config.head_dim):
            raise IRSchemaError("UpdatePagedAttentionKVCachePair slots must have scalar shape [T, H, 2D] matching the cache.")
        if isinstance(slots_type, DistributedType):
            if slots_type.partial is not None:
                raise IRSchemaError("UpdatePagedAttentionKVCachePair requires materialized slots.")
            # Every CHIP-scoped cache allocation must receive the whole chunk.
            # Block owners may partition any input axis; a chip partition
            # requires publication/Boxing before this stateful consumer.
            split_axes = {axis for policy in slots_type.axis_policies if isinstance(policy, SBPSplit)
                          for axis in policy.hierarchy_axes}
            exclusive_axes = () if slots_type.exclusive is None else slots_type.exclusive.axes
            if any(slots_type.placement.hierarchy_levels[axis] == "c"
                   for axis in (*split_axes, *exclusive_axes)):
                raise IRSchemaError("UpdatePagedAttentionKVCachePair requires complete slots on every chip.")
        return cls.state.type_of(inputs)

    @classmethod
    def infer_effect(cls, inputs, attrs):
        return effect("read_write", "paged_attention_kv_cache")

    @classmethod
    def evaluate(cls, node, arguments, context):
        slots = cls.slots.read(arguments)
        slots_type = tensor_of(context.types[cls.slots.read(node.inputs)])
        layout = tuple(node.attrs["layout"])
        if isinstance(slots_type.dtype, VectorType):
            slots = unpack_physical(slots, slots_type.rank, slots_type.dtype.lanes,
                                    (layout.index("dim"),) * len(slots_type.dtype.lanes))
        slots = to_seq_head_dim(slots, layout)
        state = cls.state.read(arguments)
        if not isinstance(state, PagedAttentionState):
            raise EvaluationError("UpdatePagedAttentionKVCachePair state must evaluate to PagedAttentionState.")
        dim = state.config.head_dim
        state.append(slots[..., :dim], slots[..., dim:],
                     layer_id=_scalar_int(cls.layer_id.read(arguments)),
                     advance_sequence=_scalar_bool(cls.advance_sequence.read(arguments)))
        return state

    @classmethod
    def cost(cls, node):
        return OpCost(notes=("paged-cache-pair-direct-write",))

    @classmethod
    def cost_factors(cls, inputs, attrs, return_type):
        value = cls.slots.type_of(inputs)
        tensor = tensor_of(value)
        shape = local_shape(value) if isinstance(value, DistributedType) else tensor.shape
        size = tensor_nbytes(tensor_type(tensor.dtype, shape))
        if size is None:
            return None
        return OpCostFactors(block_local_memory_load_bytes=size,
                             block_local_memory_store_bytes=size, grid_synchronizations=1)


__all__ = ["UpdatePagedAttentionKVCachePair"]
