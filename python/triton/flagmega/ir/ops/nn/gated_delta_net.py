# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Fused GatedDeltaNet definition and its local behaviors."""

from typing import Mapping, Sequence

from triton.flagmega.ir.model import Effect, IRType, Node, TupleType, effect
from triton.flagmega.ir.type_pattern import has_rank, is_ref, is_tensor
from triton.flagmega.ir.ops.core import (
    OpCost,
    OpDefinition,
    attribute_parameter,
    input_parameter,
    op_definition,
)
from triton.flagmega.ir.ops.nn._gdn_attrs import GDN_ATTRIBUTE_NAMES, normalize_gdn_attrs
from triton.flagmega.ir.ops.nn.gdn_convolution import gated_delta_net_convolution
from triton.flagmega.ir.ops.nn.gdn_recurrent_core import gated_delta_net_recurrent_core
from triton.flagmega.ir.ops.math._block_scaled import block_scaled_linear


@op_definition(
    "nn.gated_delta_net",
    namespace="nn",
    functional_name="gated_delta_net",
    display_name="NN.GatedDeltaNet",
)
class GatedDeltaNet(OpDefinition):
    value = input_parameter(is_tensor() & has_rank(2))
    state = input_parameter(is_ref(), memory_effect="read_write")
    qkv_weight = input_parameter(is_tensor())
    qkv_scale = input_parameter(is_tensor())
    z_weight = input_parameter(is_tensor())
    z_scale = input_parameter(is_tensor())
    b_weight = input_parameter(is_tensor())
    a_weight = input_parameter(is_tensor())
    conv_weight = input_parameter(is_tensor())
    a_log = input_parameter(is_tensor())
    dt_bias = input_parameter(is_tensor())
    norm_weight = input_parameter(is_tensor())
    output_weight = input_parameter(is_tensor())
    output_scale = input_parameter(is_tensor())
    num_key_heads = attribute_parameter()
    num_value_heads = attribute_parameter()
    key_head_dim = attribute_parameter()
    value_head_dim = attribute_parameter()
    conv_kernel_size = attribute_parameter()
    epsilon = attribute_parameter()
    weight_block_n = attribute_parameter()
    weight_block_k = attribute_parameter()

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        assert tuple(parameter.name for parameter in cls.attribute_parameters) == GDN_ATTRIBUTE_NAMES
        return normalize_gdn_attrs(attributes)

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        value = cls.value.type_of(inputs)
        state = cls.state.type_of(inputs)
        return TupleType((value, state))

    @classmethod
    def infer_effect(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> Effect:
        return effect("read_write", "gated_delta_net_state")

    @classmethod
    def evaluate(cls, node, arguments, context):
        return gated_delta_net(
            hidden=cls.value.read(arguments),
            state=cls.state.read(arguments),
            qkv_weight=cls.qkv_weight.read(arguments),
            qkv_scale=cls.qkv_scale.read(arguments),
            z_weight=cls.z_weight.read(arguments),
            z_scale=cls.z_scale.read(arguments),
            b_weight=cls.b_weight.read(arguments),
            a_weight=cls.a_weight.read(arguments),
            conv_weight=cls.conv_weight.read(arguments),
            a_log=cls.a_log.read(arguments),
            dt_bias=cls.dt_bias.read(arguments),
            norm_weight=cls.norm_weight.read(arguments),
            output_weight=cls.output_weight.read(arguments),
            output_scale=cls.output_scale.read(arguments),
            attrs=node.attrs,
            torch=context.torch,
        )

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        return OpCost(flops=None, bytes_read=None, bytes_written=None, notes=("stateful-gdn",))


def gated_delta_net(
    *,
    hidden,
    state,
    qkv_weight,
    qkv_scale,
    z_weight,
    z_scale,
    b_weight,
    a_weight,
    conv_weight,
    a_log,
    dt_bias,
    norm_weight,
    output_weight,
    output_scale,
    attrs,
    torch=None,
):
    block_n = int(attrs["weight_block_n"])
    block_k = int(attrs["weight_block_k"])
    qkv = block_scaled_linear(
        hidden,
        qkv_weight,
        qkv_scale,
        block_n=block_n,
        block_k=block_k,
        output_dtype=hidden.dtype,
    )
    z = block_scaled_linear(
        hidden,
        z_weight,
        z_scale,
        block_n=block_n,
        block_k=block_k,
        output_dtype=hidden.dtype,
    )
    convolved, state = gated_delta_net_convolution(
        qkv=qkv,
        state=state,
        conv_weight=conv_weight,
        conv_kernel_size=int(attrs["conv_kernel_size"]),
        torch=torch,
    )
    gated, state = gated_delta_net_recurrent_core(
        state=state,
        qkv=convolved,
        z=z,
        projection_input=hidden,
        b_weight=b_weight,
        a_weight=a_weight,
        a_log=a_log,
        dt_bias=dt_bias,
        norm_weight=norm_weight,
        attrs=attrs,
        torch=torch,
    )
    output = block_scaled_linear(
        gated,
        output_weight,
        output_scale,
        block_n=block_n,
        block_k=block_k,
        output_dtype=hidden.dtype,
    )
    return output, state


__all__ = ["GatedDeltaNet", "gated_delta_net"]
