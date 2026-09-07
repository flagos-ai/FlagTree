# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Stateful convolution stage of decomposed Gated DeltaNet."""

from typing import Mapping, Sequence

from triton.flagmega.errors import EvaluationError, IRSchemaError
from triton.flagmega.ir.distributed_inference import placement_of, tensor_of
from triton.flagmega.ir.distributed_type import SBPBroadCast, SBPSplit
from triton.flagmega.ir.model import DistributedType, Effect, IRType, Node, SBP, TupleType, effect
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
    "nn.gdn_convolution",
    namespace="nn",
    functional_name="gated_delta_net_convolution",
    display_name="NN.GatedDeltaNetConvolution",
)
class GatedDeltaNetConvolution(OpDefinition):
    """Apply depthwise state convolution to an already projected QKV."""

    qkv = input_parameter(is_tensor() & has_rank(2))
    state = input_parameter(is_ref(), memory_effect="read_write")
    # Checkpoints store depthwise Conv1D weights as [channels, 1, kernel].
    # Keep the operand tensor-generic here because packing may move channel or
    # kernel factors into VectorType lanes without changing the operation.
    conv_weight = input_parameter(is_tensor())
    conv_kernel_size = attribute_parameter()

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        kernel = int(attrs["conv_kernel_size"])
        if kernel < 2:
            raise IRSchemaError("GatedDeltaNetConvolution conv_kernel_size must be at least two.")
        return {"conv_kernel_size": kernel}

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        qkv_type = cls.qkv.type_of(inputs)
        state_type = cls.state.type_of(inputs)
        conv_weight_type = cls.conv_weight.type_of(inputs)
        qkv = tensor_of(qkv_type)
        conv_weight = tensor_of(conv_weight_type)
        kernel = int(attrs["conv_kernel_size"])
        if conv_weight.rank not in (2, 3):
            raise IRSchemaError("GatedDeltaNetConvolution weight must be rank 2 or checkpoint rank 3.")
        if qkv.shape[1] != conv_weight.shape[0] or conv_weight.shape[-1].fixed_value != kernel:
            raise IRSchemaError(
                "GatedDeltaNetConvolution QKV channels and convolution weight shape do not match."
            )
        placement = placement_of(qkv_type, conv_weight_type)
        if placement is None:
            return TupleType((qkv_type, state_type))
        if not isinstance(qkv_type, DistributedType) or not isinstance(conv_weight_type, DistributedType):
            raise IRSchemaError(
                "Distributed GatedDeltaNetConvolution requires QKV and convolution weight "
                "to name the same placement."
            )
        channel_axes = _materialized_or_partial_axes(qkv_type, 1)
        expected_qkv = _split_on(qkv, 1, placement, channel_axes)
        if qkv_type.partial is None and qkv_type != expected_qkv:
            raise IRSchemaError(
                "GatedDeltaNetConvolution QKV must be broadcast on tokens and channel-split."
            )
        expected_weight = _split_on(conv_weight, 0, placement, channel_axes)
        if conv_weight_type != expected_weight:
            raise IRSchemaError(
                "GatedDeltaNetConvolution weight channel ownership must match QKV channels."
            )
        return TupleType((expected_qkv, state_type))

    @classmethod
    def infer_effect(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> Effect:
        del inputs, attrs
        return effect("read_write", "gated_delta_net_state")

    @classmethod
    def evaluate(cls, node, arguments, context):
        return gated_delta_net_convolution(
            qkv=cls.qkv.read(arguments),
            state=cls.state.read(arguments),
            conv_weight=cls.conv_weight.read(arguments),
            conv_kernel_size=int(node.attrs["conv_kernel_size"]),
            torch=context.torch,
        )

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        del node
        return OpCost(
            flops=None,
            bytes_read=None,
            bytes_written=None,
            notes=("stateful-depthwise-convolution",),
        )


def gated_delta_net_convolution(
    *,
    qkv,
    state,
    conv_weight,
    conv_kernel_size: int,
    torch=None,
):
    torch = torch or _torch()
    if not isinstance(state, GatedDeltaNetState):
        raise EvaluationError(
            f"GatedDeltaNet state must be GatedDeltaNetState, got {type(state).__name__}."
        )
    conv_dim = int(qkv.shape[-1])
    if state.config.conv_dim != conv_dim or state.config.conv_kernel_size != conv_kernel_size:
        raise EvaluationError(
            "Packed GatedDeltaNet convolution state configuration does not match the operation."
        )
    conv_state = state.convolution_layer(0)
    weight = conv_weight.reshape(conv_dim, conv_kernel_size).to(dtype=qkv.dtype)
    outputs = []
    for token in range(qkv.shape[0]):
        current = qkv[token].reshape(conv_dim, 1)
        history = torch.cat((conv_state, current), dim=1)
        conv_state = history[:, 1:]
        value = (history * weight).sum(dim=1)
        outputs.append(torch.nn.functional.silu(value).to(dtype=qkv.dtype))
    state.update_convolution_layer(conv_state, 0)
    return torch.stack(outputs), state


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
        raise IRSchemaError("GatedDeltaNetConvolution supports only channel distribution.")
    if value.partial is None:
        if not isinstance(split, SBPSplit) or split_axes != placement_axes:
            raise IRSchemaError(
                "GatedDeltaNetConvolution materialized QKV channels must cover the placement."
            )
        return tuple(sorted(split_axes))
    partial_axes = set(value.partial.axes)
    if split_axes & partial_axes or split_axes | partial_axes != placement_axes:
        raise IRSchemaError(
            "GatedDeltaNetConvolution partial QKV must partition all placement axes."
        )
    return tuple(range(value.placement.rank))


def _split_on(tensor, tensor_axis: int, placement, axes: tuple[int, ...]) -> DistributedType:
    policies = [SBP.broadcast() for _ in tensor.shape]
    policies[tensor_axis] = SBP.split_contiguous(axes)
    return DistributedType(tensor, tuple(policies), placement)


__all__ = ["GatedDeltaNetConvolution", "gated_delta_net_convolution"]
