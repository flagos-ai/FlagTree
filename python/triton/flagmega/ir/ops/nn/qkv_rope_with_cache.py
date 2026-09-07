# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Fused Q/K normalization, RoPE, layout conversion, and KV-cache update."""

from __future__ import annotations

from typing import Mapping, Sequence

from triton.flagmega.errors import EvaluationError, IRSchemaError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.memory_effect import MemoryEffect
from triton.flagmega.ir.model import (
    DistributedType,
    DType,
    Effect,
    IRType,
    Node,
    TensorType,
    TupleType,
    effect,
)
from triton.flagmega.ir.ops.core import (
    OpCost,
    OpDefinition,
    ParameterKind,
    attribute_parameter,
    input_parameter,
    op_definition,
    tensor_elements,
    tensor_nbytes,
)
from triton.flagmega.ir.ops.nn._attention_layout import (
    normalize_attention_layout,
    require_decode_token,
    to_seq_head_dim,
)
from triton.flagmega.ir.ops.nn._norm import (
    norm_apply_value,
    norm_stats_value,
    normalize_axis,
)
from triton.flagmega.ir.ops.nn._paged_attention_state import PagedAttentionState
from triton.flagmega.ir.ops.nn._paged_attention_state import (
    paged_attention_state_config_from_type,
)
from triton.flagmega.ir.ops.nn.norm_apply import NormApply
from triton.flagmega.ir.ops.nn.norm_stats import NormStats
from triton.flagmega.ir.ops.nn.qwen3_paged_attention import _scalar_bool, _scalar_int
from triton.flagmega.ir.ops.nn.rope import RoPE
from triton.flagmega.ir.ops.nn.update_paged_attention_kv_cache import (
    UpdatePagedAttentionKVCache,
)
from triton.flagmega.ir.ops.tensors.pack import Pack, pack_physical
from triton.flagmega.ir.ops.tensors.unpack import Unpack, unpack_physical
from triton.flagmega.ir.type_pattern import (
    has_dtype,
    has_rank,
    is_ref,
    is_tensor,
    is_tuple,
)
from triton.flagmega.ir.types import VectorType


@op_definition(
    "nn.qkv_rope_with_cache",
    namespace="nn",
    functional_name="qkv_rope_with_cache",
    display_name="NN.QKVRoPEWithCache",
)
class QKVRoPEWithCache(OpDefinition):
    """Semantic fusion formed before target vectorization and distribution."""

    qkv = input_parameter(
        is_tuple(), memory_effect=MemoryEffect.READ.across_partial_owners()
    )
    q_scale = input_parameter(is_tensor())
    k_scale = input_parameter(is_tensor())
    q_bias = input_parameter(is_tensor())
    k_bias = input_parameter(is_tensor())
    cos = input_parameter(is_tensor())
    sin = input_parameter(is_tensor())
    state = input_parameter(
        is_ref(), parameter_kind=ParameterKind.ATTRIBUTE,
        memory_effect=MemoryEffect.CHIP_READ_WRITE.partitioned_by_argument(8),
    )
    layer_id = input_parameter(
        is_tensor() & has_rank(0) & has_dtype(DType.INT32),
        parameter_kind=ParameterKind.ATTRIBUTE,
    )
    advance_sequence = input_parameter(
        is_tensor() & has_rank(0) & has_dtype(DType.BOOL),
        parameter_kind=ParameterKind.ATTRIBUTE,
    )
    q_axis = attribute_parameter()
    q_epsilon = attribute_parameter()
    q_use_mean = attribute_parameter()
    q_round_before_scale = attribute_parameter(default=False)
    k_axis = attribute_parameter()
    k_epsilon = attribute_parameter()
    k_use_mean = attribute_parameter()
    k_round_before_scale = attribute_parameter(default=False)
    qkv_layout = attribute_parameter()
    attention_layout = attribute_parameter()

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        q_axis = attrs["q_axis"]
        k_axis = attrs["k_axis"]
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (q_axis, k_axis)
        ):
            raise IRSchemaError("QKVRoPEWithCache normalization axes must be integers.")
        q_epsilon = float(attrs["q_epsilon"])
        k_epsilon = float(attrs["k_epsilon"])
        if q_epsilon <= 0 or k_epsilon <= 0:
            raise IRSchemaError("QKVRoPEWithCache epsilon values must be positive.")
        if any(not isinstance(attrs[f"{prefix}_round_before_scale"], bool) for prefix in ("q", "k")):
            raise IRSchemaError("QKVRoPEWithCache rounding policies must be boolean.")
        return {
            "q_axis": q_axis,
            "q_epsilon": q_epsilon,
            "q_use_mean": bool(attrs["q_use_mean"]),
            "q_round_before_scale": attrs["q_round_before_scale"],
            "k_axis": k_axis,
            "k_epsilon": k_epsilon,
            "k_use_mean": bool(attrs["k_use_mean"]),
            "k_round_before_scale": attrs["k_round_before_scale"],
            "qkv_layout": normalize_attention_layout(attrs["qkv_layout"]),
            "attention_layout": normalize_attention_layout(
                attrs["attention_layout"]
            ),
        }

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        qkv_type = cls.qkv.type_of(inputs)
        if len(qkv_type.fields) != 3 or not all(
            isinstance(field, (TensorType, DistributedType))
            for field in qkv_type.fields
        ):
            raise IRSchemaError(
                "QKVRoPEWithCache qkv must be a tuple of three tensor values."
            )
        cache_config = paged_attention_state_config_from_type(
            cls.state.type_of(inputs)
        )
        q, k, v = (
            Node(f"__qkv_field_{index}", "builtin.var", (), field)
            for index, field in enumerate(qkv_type.fields)
        )
        qkv_layout = tuple(attrs["qkv_layout"])
        attention_layout = tuple(attrs["attention_layout"])
        q = _logical_attention_node(q, cache_config.lanes, qkv_layout, "q")
        k = _logical_attention_node(k, cache_config.lanes, qkv_layout, "k")
        v = _logical_attention_node(v, cache_config.lanes, qkv_layout, "v")
        q_scale = _logical_norm_parameter_node(
            cls.q_scale.read(inputs), q, int(attrs["q_axis"]),
            cache_config.lanes, qkv_layout, "q_scale",
        )
        k_scale = _logical_norm_parameter_node(
            cls.k_scale.read(inputs), k, int(attrs["k_axis"]),
            cache_config.lanes, qkv_layout, "k_scale",
        )
        q_bias = _logical_norm_parameter_node(
            cls.q_bias.read(inputs), q, int(attrs["q_axis"]),
            cache_config.lanes, qkv_layout, "q_bias",
        )
        k_bias = _logical_norm_parameter_node(
            cls.k_bias.read(inputs), k, int(attrs["k_axis"]),
            cache_config.lanes, qkv_layout, "k_bias",
        )
        cos = _logical_rope_parameter_node(
            cls.cos.read(inputs), q, cache_config.lanes, qkv_layout, "cos"
        )
        sin = _logical_rope_parameter_node(
            cls.sin.read(inputs), q, cache_config.lanes, qkv_layout, "sin"
        )
        state = cls.state.read(inputs)
        layer_id = cls.layer_id.read(inputs)
        advance = cls.advance_sequence.read(inputs)

        q_norm = _infer_norm(
            q,
            q_scale,
            q_bias,
            axis=int(attrs["q_axis"]),
            epsilon=float(attrs["q_epsilon"]),
            use_mean=bool(attrs["q_use_mean"]),
            prefix="q",
        )
        k_norm = _infer_norm(
            k,
            k_scale,
            k_bias,
            axis=int(attrs["k_axis"]),
            epsilon=float(attrs["k_epsilon"]),
            use_mean=bool(attrs["k_use_mean"]),
            prefix="k",
        )
        q_rope = _infer_rope(q_norm, cos, sin, "q")
        k_rope = _infer_rope(k_norm, cos, sin, "k")
        q_output = _transform_attention_layout_type(
            q_rope.type,
            qkv_layout,
            attention_layout,
            cache_config.lanes,
        )
        k_slots_type = _transform_attention_layout_type(
            k_rope.type,
            qkv_layout,
            attention_layout,
            cache_config.lanes,
        )
        v_slots_type = _transform_attention_layout_type(
            v.type,
            qkv_layout,
            attention_layout,
            cache_config.lanes,
        )
        false_advance = Node(
            "__qkv_key_advance",
            "builtin.scalar_const",
            (),
            advance.type,
            attrs={"value": False},
        )
        key_slots = Node("__qkv_key_slots", "builtin.var", (), k_slots_type)
        key_state_type = UpdatePagedAttentionKVCache.infer_type(
            (key_slots, state, layer_id, false_advance),
            {
                "cache_kind": "key",
                "layout": attrs["attention_layout"],
            },
        )
        key_state = Node("__qkv_key_state", "builtin.var", (), key_state_type)
        value_slots = Node("__qkv_value_slots", "builtin.var", (), v_slots_type)
        value_state_type = UpdatePagedAttentionKVCache.infer_type(
            (value_slots, key_state, layer_id, advance),
            {
                "cache_kind": "value",
                "layout": attrs["attention_layout"],
            },
        )
        return TupleType((q_output, value_state_type))

    @classmethod
    def infer_effect(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> Effect:
        return effect("read_write", "paged_attention_kv_cache")

    @classmethod
    def evaluate(cls, node, arguments, context):
        qkv = cls.qkv.read(arguments)
        if not isinstance(qkv, (tuple, list)) or len(qkv) != 3:
            raise EvaluationError("QKVRoPEWithCache expects three Q/K/V values.")
        qkv_type = context.types[cls.qkv.read(node.inputs)]
        if not isinstance(qkv_type, TupleType):
            raise EvaluationError("QKVRoPEWithCache qkv input has no tuple type.")
        cache_config = paged_attention_state_config_from_type(
            context.types[cls.state.read(node.inputs)]
        )
        qkv_layout = tuple(node.attrs["qkv_layout"])
        q_type, k_type, v_type = qkv_type.fields
        q_value = _logical_attention_value(
            qkv[0], q_type, cache_config.lanes, qkv_layout, "q"
        )
        k_value = _logical_attention_value(
            qkv[1], k_type, cache_config.lanes, qkv_layout, "k"
        )
        v_value = _logical_attention_value(
            qkv[2], v_type, cache_config.lanes, qkv_layout, "v"
        )
        q_scale = _logical_norm_parameter_value(
            cls.q_scale.read(arguments),
            context.types[cls.q_scale.read(node.inputs)],
            tensor_of(q_type).rank,
            int(node.attrs["q_axis"]),
            cache_config.lanes,
            qkv_layout,
            "q_scale",
        )
        q_bias = _logical_norm_parameter_value(
            cls.q_bias.read(arguments),
            context.types[cls.q_bias.read(node.inputs)],
            tensor_of(q_type).rank,
            int(node.attrs["q_axis"]),
            cache_config.lanes,
            qkv_layout,
            "q_bias",
        )
        k_scale = _logical_norm_parameter_value(
            cls.k_scale.read(arguments),
            context.types[cls.k_scale.read(node.inputs)],
            tensor_of(k_type).rank,
            int(node.attrs["k_axis"]),
            cache_config.lanes,
            qkv_layout,
            "k_scale",
        )
        k_bias = _logical_norm_parameter_value(
            cls.k_bias.read(arguments),
            context.types[cls.k_bias.read(node.inputs)],
            tensor_of(k_type).rank,
            int(node.attrs["k_axis"]),
            cache_config.lanes,
            qkv_layout,
            "k_bias",
        )
        cos = _logical_rope_parameter_value(
            cls.cos.read(arguments),
            context.types[cls.cos.read(node.inputs)],
            tensor_of(q_type).rank,
            cache_config.lanes,
            qkv_layout,
            "cos",
        )
        sin = _logical_rope_parameter_value(
            cls.sin.read(arguments),
            context.types[cls.sin.read(node.inputs)],
            tensor_of(q_type).rank,
            cache_config.lanes,
            qkv_layout,
            "sin",
        )
        q = _normalize_and_rope_value(
            q_value,
            q_scale,
            q_bias,
            cos,
            sin,
            axis=int(node.attrs["q_axis"]),
            epsilon=float(node.attrs["q_epsilon"]),
            use_mean=bool(node.attrs["q_use_mean"]),
            round_before_scale=bool(node.attrs.get("q_round_before_scale", False)),
            torch=context.torch,
        )
        k = _normalize_and_rope_value(
            k_value,
            k_scale,
            k_bias,
            cos,
            sin,
            axis=int(node.attrs["k_axis"]),
            epsilon=float(node.attrs["k_epsilon"]),
            use_mean=bool(node.attrs["k_use_mean"]),
            round_before_scale=bool(node.attrs.get("k_round_before_scale", False)),
            torch=context.torch,
        )
        q = _physical_attention_value(
            q, qkv_layout, tuple(node.attrs["attention_layout"]), cache_config.lanes
        )
        k = _permute_attention_value(
            k, qkv_layout, tuple(node.attrs["attention_layout"])
        )
        v = _permute_attention_value(
            v_value, qkv_layout, tuple(node.attrs["attention_layout"])
        )
        state = cls.state.read(arguments)
        if not isinstance(state, PagedAttentionState):
            raise EvaluationError(
                "QKVRoPEWithCache state must evaluate to PagedAttentionState."
            )
        state.update(
            require_decode_token(
                to_seq_head_dim(k, tuple(node.attrs["attention_layout"])),
                operation="QKVRoPEWithCache key",
            ),
            cache_kind="key",
            layer_id=_scalar_int(cls.layer_id.read(arguments)),
            advance_sequence=False,
        )
        state.update(
            require_decode_token(
                to_seq_head_dim(v, tuple(node.attrs["attention_layout"])),
                operation="QKVRoPEWithCache value",
            ),
            cache_kind="value",
            layer_id=_scalar_int(cls.layer_id.read(arguments)),
            advance_sequence=_scalar_bool(cls.advance_sequence.read(arguments)),
        )
        return q, state

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        output = node.type.fields[0] if isinstance(node.type, TupleType) else None
        elements = tensor_elements(output) if output is not None else None
        size = tensor_nbytes(output) if output is not None else None
        return OpCost(
            flops=None if elements is None else elements * 9,
            bytes_read=None if size is None else size * 4,
            bytes_written=size,
            notes=("fused-qk-normalization-rope-cache-update",),
        )


def _infer_norm(
    value: Node,
    scale: Node,
    bias: Node,
    *,
    axis: int,
    epsilon: float,
    use_mean: bool,
    prefix: str,
) -> Node:
    stats_attrs = {"axis": axis, "use_mean": use_mean}
    stats_type = NormStats.infer_type((value,), stats_attrs)
    stats = Node(f"__qkv_{prefix}_stats", "nn.norm_stats", (value.id,), stats_type, attrs=stats_attrs)
    norm_attrs = {"axis": axis, "epsilon": epsilon, "use_mean": use_mean}
    norm_type = NormApply.infer_type((value, stats, scale, bias), norm_attrs)
    return Node(
        f"__qkv_{prefix}_norm",
        "nn.norm_apply",
        (value.id, stats.id, scale.id, bias.id),
        norm_type,
        attrs=norm_attrs,
    )


def _infer_rope(value: Node, cos: Node, sin: Node, prefix: str) -> Node:
    result_type = RoPE.infer_type((value, cos, sin), {})
    return Node(
        f"__qkv_{prefix}_rope",
        "nn.rope",
        (value.id, cos.id, sin.id),
        result_type,
    )


def _logical_attention_node(
    value: Node,
    lane: int,
    layout: tuple[str, str, str],
    label: str,
) -> Node:
    return _logical_vector_node(
        value,
        expected_lanes=(lane,),
        axes=(layout.index("dim"),),
        label=label,
    )


def _logical_norm_parameter_node(
    parameter: Node,
    logical_input: Node,
    axis: int,
    lane: int,
    layout: tuple[str, str, str],
    label: str,
) -> Node:
    input_tensor = tensor_of(logical_input.type)
    normalized_axis = normalize_axis(axis, input_tensor.rank)
    parameter_axis = layout.index("dim") - normalized_axis
    if parameter_axis < 0:
        raise IRSchemaError(
            f"QKVRoPEWithCache {label} vector axis is outside the normalized suffix."
        )
    return _logical_vector_node(
        parameter,
        expected_lanes=(lane,),
        axes=(parameter_axis,),
        label=label,
    )


def _logical_rope_parameter_node(
    parameter: Node,
    logical_input: Node,
    lane: int,
    layout: tuple[str, str, str],
    label: str,
) -> Node:
    input_rank = tensor_of(logical_input.type).rank
    parameter_rank = tensor_of(parameter.type).rank
    parameter_axis = layout.index("dim") - (input_rank - parameter_rank)
    if parameter_axis < 0 or parameter_axis >= parameter_rank:
        raise IRSchemaError(
            f"QKVRoPEWithCache {label} cannot represent the rotary dimension."
        )
    return _logical_vector_node(
        parameter,
        expected_lanes=(2, lane),
        axes=(parameter_axis, parameter_axis),
        label=label,
    )


def _logical_vector_node(
    value: Node,
    *,
    expected_lanes: tuple[int, ...],
    axes: tuple[int, ...],
    label: str,
) -> Node:
    value_type = tensor_of(value.type)
    if not isinstance(value_type.dtype, VectorType):
        return value
    if value_type.dtype.lanes != expected_lanes:
        raise IRSchemaError(
            f"QKVRoPEWithCache {label} vector lanes {value_type.dtype.lanes} "
            f"do not match cache lanes {expected_lanes}."
        )
    logical_type = Unpack.infer_type((value,), {"axes": axes})
    return Node(
        f"__qkv_logical_{label}",
        "tensors.unpack",
        (value.id,),
        logical_type,
        attrs={"axes": axes},
    )


def _logical_attention_value(
    value,
    value_type: IRType,
    lane: int,
    layout: tuple[str, str, str],
    label: str,
):
    return _logical_vector_value(
        value,
        value_type,
        expected_lanes=(lane,),
        axes=(layout.index("dim"),),
        label=label,
    )


def _logical_norm_parameter_value(
    value,
    value_type: IRType,
    input_rank: int,
    axis: int,
    lane: int,
    layout: tuple[str, str, str],
    label: str,
):
    parameter_axis = layout.index("dim") - normalize_axis(axis, input_rank)
    return _logical_vector_value(
        value,
        value_type,
        expected_lanes=(lane,),
        axes=(parameter_axis,),
        label=label,
    )


def _logical_rope_parameter_value(
    value,
    value_type: IRType,
    input_rank: int,
    lane: int,
    layout: tuple[str, str, str],
    label: str,
):
    parameter_rank = tensor_of(value_type).rank
    parameter_axis = layout.index("dim") - (input_rank - parameter_rank)
    return _logical_vector_value(
        value,
        value_type,
        expected_lanes=(2, lane),
        axes=(parameter_axis, parameter_axis),
        label=label,
    )


def _logical_vector_value(
    value,
    value_type: IRType,
    *,
    expected_lanes: tuple[int, ...],
    axes: tuple[int, ...],
    label: str,
):
    tensor = tensor_of(value_type)
    if not isinstance(tensor.dtype, VectorType):
        return value
    if tensor.dtype.lanes != expected_lanes:
        raise EvaluationError(
            f"QKVRoPEWithCache {label} vector lanes {tensor.dtype.lanes} "
            f"do not match cache lanes {expected_lanes}."
        )
    return unpack_physical(value, tensor.rank, tensor.dtype.lanes, axes)


def _physical_attention_value(
    value,
    input_layout: tuple[str, str, str],
    output_layout: tuple[str, str, str],
    lane: int,
):
    transposed = _permute_attention_value(value, input_layout, output_layout)
    return pack_physical(
        transposed,
        len(output_layout),
        (lane,),
        (output_layout.index("dim"),),
    )


def _transform_attention_layout_type(
    value_type: IRType,
    input_layout: tuple[str, str, str],
    output_layout: tuple[str, str, str],
    lane: int,
) -> IRType:
    permutation = tuple(input_layout.index(axis) for axis in output_layout)
    tensor = tensor_of(value_type)
    transformed = TensorType(
        tensor.dtype,
        tuple(tensor.shape[axis] for axis in permutation),
        tensor.layout,
    )
    transposed: IRType
    if not isinstance(value_type, DistributedType):
        transposed = transformed
    else:
        transposed = DistributedType(
            transformed,
            tuple(value_type.axis_policies[axis] for axis in permutation),
            value_type.placement,
            value_type.partial,
        )
    return Pack.infer_type(
        (Node("__qkv_layout", "builtin.var", (), transposed),),
        {"lanes": (lane,), "axes": (output_layout.index("dim"),)},
    )


def _normalize_and_rope_value(
    value,
    scale,
    bias,
    cos,
    sin,
    *,
    axis: int,
    epsilon: float,
    use_mean: bool,
    torch,
    round_before_scale: bool = False,
):
    stats = norm_stats_value(value, axis=axis, use_mean=use_mean)
    normalized = norm_apply_value(
        value,
        stats,
        scale,
        bias,
        axis=axis,
        epsilon=epsilon,
        use_mean=use_mean,
        round_before_scale=round_before_scale,
    )
    half = normalized.shape[-1] // 2
    rotated = torch.cat((-normalized[..., half:], normalized[..., :half]), dim=-1)
    result = normalized * cos.to(dtype=normalized.dtype) + rotated * sin.to(
        dtype=normalized.dtype
    )
    return result


def _permute_attention_value(
    value,
    input_layout: tuple[str, str, str],
    output_layout: tuple[str, str, str],
):
    permutation = tuple(input_layout.index(axis) for axis in output_layout)
    if permutation == (0, 1, 2):
        return value
    lane_axes = tuple(range(3, value.ndim))
    return value.permute(*permutation, *lane_axes).contiguous()


__all__ = ["QKVRoPEWithCache"]
