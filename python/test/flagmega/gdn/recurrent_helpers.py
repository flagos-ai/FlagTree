# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Model-independent inputs and an explicit-step recurrent numerical oracle."""

import torch

from triton.flagmega import ir as fm
from triton.flagmega.ir.ops.nn._gdn_state import GatedDeltaNetStateConfig, create_gdn_state


def recurrent_case(*, tokens=3, dimension=8, value_dimension=None, **attributes):
    value_dimension = dimension if value_dimension is None else value_dimension
    config = GatedDeltaNetStateConfig(1, 2, 4, dimension, value_dimension, 4, 32)
    attrs = {
        "num_key_heads": 2, "num_value_heads": 4, "key_head_dim": dimension, "value_head_dim": value_dimension,
        "epsilon": 1e-6, **attributes
    }
    generator = torch.Generator().manual_seed(1409)
    qkv = torch.randn((tokens, config.conv_dim), generator=generator).bfloat16()
    # Exercise epsilon below the norm, not only unit-scale vectors for which
    # adding epsilon can disappear at float32 precision.
    qkv[:, :4 * dimension] *= .0001
    source = torch.zeros((tokens, 32), dtype=torch.bfloat16)
    source[:, 0] = 1
    b_weight = torch.zeros((4, 32), dtype=torch.bfloat16)
    b_weight[:, 0] = torch.tensor([.1, .3, .7, 1.1]).bfloat16()
    a_weight = torch.zeros_like(b_weight)
    a_weight[:, 0] = torch.tensor([-.7, .3, -.1, .9]).bfloat16()
    state = create_gdn_state(config)
    initial = torch.randn((4, dimension, value_dimension), generator=generator) / 8
    state.update_recurrent_layer(initial)
    state.convolution.fill_(17)
    values = {
        "state": state, "qkv": qkv, "z": torch.randn(
            (tokens, 4 * value_dimension), generator=generator).bfloat16(), "projection_input": source, "b_weight":
        b_weight, "a_weight": a_weight, "a_log": torch.tensor([-.2, .5, -.1, .7]), "dt_bias":
        torch.tensor([.1, -.5, .3, -.2]).bfloat16(), "norm_weight": torch.randn(value_dimension,
                                                                                generator=generator).bfloat16()
    }
    types = {
        name:
        config.ref_type if name == "state" else fm.tensor_type(str(value.dtype).removeprefix("torch."), value.shape)
        for name, value in values.items()
    }
    return config, attrs, types, values


def recurrent_module(types, attrs):

    class Graph(fm.Module):

        def forward(self):
            inputs = {name: self.input(name, value_type, id=name) for name, value_type in types.items()}
            result = fm.F.nn.gated_delta_net_recurrent_core(**inputs, **attrs, name="recurrent")
            self.function("main", tuple(inputs.values()), (result, ))

    return Graph(dialect="high_level", stage="imported", entry="main").build()


def recurrent_reference(values, attrs):
    state = values["state"].recurrent_layer().clone()
    heads, value_heads = attrs["num_key_heads"], attrs["num_value_heads"]
    key_dim, value_dim = attrs["key_head_dim"], attrs["value_head_dim"]
    key_total = heads * key_dim
    a = values["projection_input"] @ values["a_weight"].T
    b = values["projection_input"] @ values["b_weight"].T
    outputs, states = [], []
    for index, packed in enumerate(values["qkv"]):
        q, k, v = torch.split(packed, (key_total, key_total, value_heads * value_dim))
        normalized = []
        for source in (q, k):
            source = source.reshape(heads, key_dim).repeat_interleave(value_heads // heads, dim=0).float()
            epsilon = attrs.get("qk_norm_epsilon", 1e-12)
            squared = source.square().sum(-1, keepdim=True)
            denominator = (squared + epsilon).sqrt() if attrs.get(
                "qk_norm_mode", "clamp") == "add" else squared.sqrt().clamp_min(epsilon)
            value = source / denominator
            if attrs.get("round_normalized_qk", False):
                value = value.bfloat16().float()
            normalized.append(value)
        q, k = normalized
        beta = torch.sigmoid(b[index].float())
        if attrs.get("round_beta", True):
            beta = beta.bfloat16().float()
        decay = torch.exp(-torch.exp(values["a_log"]) *
                          torch.nn.functional.softplus(a[index].float() + values["dt_bias"].float()))
        state = state * decay[:, None, None]
        recalled = torch.einsum("hkv,hk->hv", state, k)
        update = (v.reshape(value_heads, value_dim).float() - recalled) * beta[:, None]
        state = state + k[:, :, None] * update[:, None, :]
        core = torch.einsum("hkv,hk->hv", state, q * key_dim**-.5)
        if attrs.get("round_core", False):
            core = core.bfloat16().float()
        rms = torch.rsqrt(core.square().mean(-1, keepdim=True) + attrs["epsilon"])
        gate = torch.nn.functional.silu(values["z"][index].float().reshape(value_heads, value_dim))
        outputs.append((core * rms * values["norm_weight"].float() * gate).flatten().bfloat16())
        states.append(state.clone())
    return torch.stack(outputs), states
