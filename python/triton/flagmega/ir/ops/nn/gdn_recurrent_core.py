# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""State update and gated-value stage of decomposed Gated DeltaNet."""

import math
from typing import Mapping, Sequence

from triton.flagmega.errors import EvaluationError, IRSchemaError
from triton.flagmega.ir.distributed_inference import all_broadcast, placement_of, tensor_of
from triton.flagmega.ir.distributed_type import SBPBroadCast, SBPSplit
from triton.flagmega.ir.model import DistributedType, Effect, IRType, Node, SBP, TupleType, effect, tensor_type
from triton.flagmega.ir.ops.core import (
    OpCost,
    OpDefinition,
    attribute_parameter,
    input_parameter,
    op_definition,
)
from triton.flagmega.ir.ops.nn._gdn_state import GatedDeltaNetState
from triton.flagmega.ir.type_pattern import has_rank, is_ref, is_tensor


@op_definition(
    "nn.gdn_recurrent_core",
    namespace="nn",
    functional_name="gated_delta_net_recurrent_core",
    display_name="NN.GatedDeltaNetRecurrentCore",
)
class GatedDeltaNetRecurrentCore(OpDefinition):
    """Update recurrent state and produce the local gated value activation."""

    state = input_parameter(is_ref(), memory_effect="read_write")
    qkv = input_parameter(is_tensor() & has_rank(2))
    z = input_parameter(is_tensor() & has_rank(2))
    projection_input = input_parameter(is_tensor() & has_rank(2))
    b_weight = input_parameter(is_tensor() & has_rank(2))
    a_weight = input_parameter(is_tensor() & has_rank(2))
    a_log = input_parameter(is_tensor())
    dt_bias = input_parameter(is_tensor())
    norm_weight = input_parameter(is_tensor())
    num_key_heads = attribute_parameter()
    num_value_heads = attribute_parameter()
    key_head_dim = attribute_parameter()
    value_head_dim = attribute_parameter()
    epsilon = attribute_parameter()

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        values = {
            "num_key_heads": int(attrs["num_key_heads"]),
            "num_value_heads": int(attrs["num_value_heads"]),
            "key_head_dim": int(attrs["key_head_dim"]),
            "value_head_dim": int(attrs["value_head_dim"]),
            "epsilon": float(attrs["epsilon"]),
        }
        if any(values[name] <= 0 for name in (
            "num_key_heads", "num_value_heads", "key_head_dim", "value_head_dim"
        )):
            raise IRSchemaError("GatedDeltaNetRecurrentCore dimensions must be positive.")
        if values["num_value_heads"] % values["num_key_heads"]:
            raise IRSchemaError("GatedDeltaNetRecurrentCore value heads must divide by key heads.")
        if values["epsilon"] <= 0:
            raise IRSchemaError("GatedDeltaNetRecurrentCore epsilon must be positive.")
        return values

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        qkv_type = cls.qkv.type_of(inputs)
        z_type = cls.z.type_of(inputs)
        state_type = cls.state.type_of(inputs)
        z_tensor = tensor_of(z_type)
        output = tensor_type(
            z_tensor.dtype,
            (z_tensor.shape[0], int(attrs["num_value_heads"]) * int(attrs["value_head_dim"])),
        )
        placement = placement_of(*(parameter.type_of(inputs) for parameter in cls.input_parameters))
        if placement is None:
            return TupleType((output, state_type))
        tensor_inputs = tuple(
            parameter.type_of(inputs)
            for parameter in cls.input_parameters
            if parameter is not cls.state
        )
        if not all(isinstance(value, DistributedType) for value in tensor_inputs):
            raise IRSchemaError(
                "Distributed GatedDeltaNetRecurrentCore requires every tensor input to name its placement."
            )
        assert isinstance(qkv_type, DistributedType) and isinstance(z_type, DistributedType)
        if not all_broadcast(qkv_type):
            raise IRSchemaError("GatedDeltaNetRecurrentCore QKV must be replicated.")
        for parameter in (
            cls.projection_input, cls.b_weight, cls.a_weight,
            cls.a_log, cls.dt_bias, cls.norm_weight,
        ):
            if not all_broadcast(parameter.type_of(inputs)):
                raise IRSchemaError(
                    f"GatedDeltaNetRecurrentCore {parameter.name} must be replicated."
                )
        head_axes = _materialized_or_partial_axes(z_type, 1)
        expected_z = _split_on(tensor_of(z_type), 1, placement, head_axes)
        if z_type.partial is None and z_type != expected_z:
            raise IRSchemaError("GatedDeltaNetRecurrentCore Z must be value-head split.")
        return TupleType((_split_on(output, 1, placement, head_axes), state_type))

    @classmethod
    def infer_effect(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> Effect:
        del inputs, attrs
        return effect("read_write", "gated_delta_net_state")

    @classmethod
    def evaluate(cls, node, arguments, context):
        return gated_delta_net_recurrent_core(
            state=cls.state.read(arguments),
            qkv=cls.qkv.read(arguments),
            z=cls.z.read(arguments),
            projection_input=cls.projection_input.read(arguments),
            b_weight=cls.b_weight.read(arguments),
            a_weight=cls.a_weight.read(arguments),
            a_log=cls.a_log.read(arguments),
            dt_bias=cls.dt_bias.read(arguments),
            norm_weight=cls.norm_weight.read(arguments),
            attrs=node.attrs,
            torch=context.torch,
        )

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        del node
        return OpCost(
            flops=None,
            bytes_read=None,
            bytes_written=None,
            notes=("stateful-rank1-update",),
        )


def gated_delta_net_recurrent_core(
    *,
    state,
    qkv,
    z,
    projection_input,
    b_weight,
    a_weight,
    a_log,
    dt_bias,
    norm_weight,
    attrs,
    torch=None,
):
    torch = torch or _torch()
    if not isinstance(state, GatedDeltaNetState):
        raise EvaluationError(
            f"GatedDeltaNet state must be GatedDeltaNetState, got {type(state).__name__}."
        )
    num_key_heads = int(attrs["num_key_heads"])
    num_value_heads = int(attrs["num_value_heads"])
    key_head_dim = int(attrs["key_head_dim"])
    value_head_dim = int(attrs["value_head_dim"])
    epsilon = float(attrs["epsilon"])
    key_dim = num_key_heads * key_head_dim
    value_dim = num_value_heads * value_head_dim
    conv_dim = (2 * key_dim) + value_dim
    repeats = num_value_heads // num_key_heads
    if int(qkv.shape[-1]) != conv_dim or int(z.shape[-1]) != value_dim:
        raise EvaluationError("GatedDeltaNet recurrent QKV/Z dimensions do not match attributes.")
    recurrent_state = state.recurrent_layer(0).float()
    b_projection = torch.matmul(
        projection_input,
        b_weight.to(dtype=projection_input.dtype).transpose(0, 1),
    )
    a_projection = torch.matmul(
        projection_input,
        a_weight.to(dtype=projection_input.dtype).transpose(0, 1),
    )
    outputs = []
    for token in range(qkv.shape[0]):
        current = qkv[token]
        query = current[:key_dim].reshape(num_key_heads, key_head_dim)
        key = current[key_dim:2 * key_dim].reshape(num_key_heads, key_head_dim)
        value = current[2 * key_dim:].reshape(num_value_heads, value_head_dim)
        query = query.repeat_interleave(repeats, dim=0).float()
        key = key.repeat_interleave(repeats, dim=0).float()
        query = query / torch.clamp(torch.linalg.vector_norm(query, dim=-1, keepdim=True), min=1e-12)
        key = key / torch.clamp(torch.linalg.vector_norm(key, dim=-1, keepdim=True), min=1e-12)
        beta = torch.sigmoid(b_projection[token].float())
        decay_log = -torch.exp(a_log.float()) * torch.nn.functional.softplus(
            a_projection[token].float() + dt_bias.float()
        )
        decayed_state = recurrent_state * torch.exp(decay_log).reshape(num_value_heads, 1, 1)
        recalled = (decayed_state * key.unsqueeze(-1)).sum(dim=1)
        delta = (value.float() - recalled) * beta.unsqueeze(-1)
        recurrent_state = decayed_state + key.unsqueeze(-1) * delta.unsqueeze(1)
        scaled_query = query * (1.0 / math.sqrt(key_head_dim))
        core = (recurrent_state * scaled_query.unsqueeze(-1)).sum(dim=1)
        inverse_rms = torch.rsqrt(core.pow(2).mean(dim=-1, keepdim=True) + epsilon)
        normalized = core * inverse_rms * norm_weight.float()
        gate = torch.nn.functional.silu(z[token].float().reshape(num_value_heads, value_head_dim))
        outputs.append((normalized * gate).reshape(1, value_dim).to(dtype=z.dtype))
    state.update_recurrent_layer(recurrent_state, 0)
    return torch.cat(outputs, dim=0), state


def _torch():
    try:
        import torch
    except ImportError as error:
        raise EvaluationError("Gated DeltaNet evaluation requires PyTorch.") from error
    return torch


def _materialized_or_partial_axes(value: DistributedType, tensor_axis: int) -> tuple[int, ...]:
    placement_axes = set(range(value.placement.rank))
    split = value.axis_policies[tensor_axis]
    split_axes = set(split.hierarchy_axes) if isinstance(split, SBPSplit) else set()
    if any(
        not isinstance(policy, SBPBroadCast)
        for index, policy in enumerate(value.axis_policies)
        if index != tensor_axis
    ):
        raise IRSchemaError("GatedDeltaNetRecurrentCore supports only value distribution.")
    if value.partial is None:
        if not isinstance(split, SBPSplit) or split_axes != placement_axes:
            raise IRSchemaError(
                "GatedDeltaNetRecurrentCore materialized Z values must cover the placement."
            )
        return tuple(sorted(split_axes))
    partial_axes = set(value.partial.axes)
    if split_axes & partial_axes or split_axes | partial_axes != placement_axes:
        raise IRSchemaError(
            "GatedDeltaNetRecurrentCore partial Z must partition all placement axes."
        )
    return tuple(range(value.placement.rank))


def _split_on(tensor, tensor_axis: int, placement, axes: tuple[int, ...]) -> DistributedType:
    policies = [SBP.broadcast() for _ in tensor.shape]
    policies[tensor_axis] = SBP.split_contiguous(axes)
    return DistributedType(tensor, tuple(policies), placement)


__all__ = ["GatedDeltaNetRecurrentCore", "gated_delta_net_recurrent_core"]
