# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Convolution product and pre-activation rounding are separate IR contracts."""

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega import pattern_match as pm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.ir.ops.nn._gdn_state import GatedDeltaNetStateConfig, create_gdn_state
from triton.flagmega.ir.ops.nn.gdn_convolution import GatedDeltaNetConvolution
from triton.flagmega.ir.ops.nn.gdn_convolution import gated_delta_net_convolution
from python.test.flagmega.ir.ops.primitive_helpers import primitive_module


def _module(kernel=4, **attrs):
    config = GatedDeltaNetStateConfig(1, 2, 4, 8, 8, kernel, 32)
    types = (fm.tensor_type("bfloat16", (3, config.conv_dim)), config.ref_type,
             fm.tensor_type("bfloat16", (config.conv_dim, 1, kernel)))
    return primitive_module(GatedDeltaNetConvolution, types, conv_kernel_size=kernel, **attrs), config


@pytest.mark.parametrize("kernel", [2, 4, 6])
@pytest.mark.parametrize("round_products", [False, True])
@pytest.mark.parametrize("round_before_activation", [False, True])
@pytest.mark.parametrize("order", ["current_first", "chronological"])
def test_independent_rounding_boundaries_with_nonzero_history(kernel, round_products, round_before_activation, order):
    module, config = _module(kernel, round_products=round_products, round_before_activation=round_before_activation,
                             accumulation_order=order)
    generator = torch.Generator().manual_seed(214)
    qkv = torch.randn((3, config.conv_dim), generator=generator).bfloat16()
    weight = torch.randn((config.conv_dim, 1, kernel), generator=generator).bfloat16()
    history = torch.randn((config.conv_dim, kernel - 1), generator=generator).bfloat16()
    state = create_gdn_state(config)
    state.update_convolution_layer(history)
    state.recurrent.fill_(17)
    expected = []
    for current in qkv:
        window = torch.cat((history, current[:, None]), dim=1)
        products = window.float() * weight[:, 0].float()
        if round_products:
            products = products.bfloat16().float()
        # Products of two BF16 values are exactly representable in FP32.
        # Accumulate each explicitly ordered step with one FP32 rounding.
        value = torch.zeros(config.conv_dim)
        indices = (kernel - 1, *range(kernel - 1)) if order == "current_first" else range(kernel)
        for index in indices:
            value = (value.double() + products[:, index].double()).float()
        if round_before_activation:
            value = value.bfloat16().float()
        expected.append(torch.nn.functional.silu(value).bfloat16())
        history = window[:, 1:]
    (actual,
     returned_state), = TorchEvaluator(DictWeightResolver({})).run(module,
                                                                   {"qkv": qkv, "state": state, "conv_weight": weight})
    torch.testing.assert_close(actual, torch.stack(expected), rtol=0, atol=0)
    torch.testing.assert_close(state.convolution_layer(), history, rtol=0, atol=0)
    assert returned_state is state
    assert torch.all(state.recurrent == 17)


@pytest.mark.parametrize("name", ["round_products", "round_before_activation"])
@pytest.mark.parametrize("value", [1, 0, "true", None])
def test_rounding_flags_reject_non_boolean_values(name, value):
    with pytest.raises(IRSchemaError, match=name):
        _module(**{name: value})


def test_old_constructor_retains_nncase_rounding_and_python_resume():
    module, _ = _module()
    node = module.node_map["output"]
    assert node.attrs["round_products"] is True
    assert node.attrs["round_before_activation"] is True
    assert node.attrs["accumulation_order"] == "current_first"
    namespace = {}
    exec(fm.module_source(module), namespace)
    assert namespace["MODULE"].semantic_hash == module.semantic_hash


def test_explicit_rounding_roundtrips_and_matches_named_pattern():
    module, _ = _module(round_products=False, round_before_activation=False, accumulation_order="chronological")
    namespace = {}
    exec(fm.module_source(module), namespace)
    resumed = namespace["MODULE"]
    assert resumed.semantic_hash == module.semantic_hash
    node = resumed.node_map["output"]
    matching = pm.F.nn.is_gated_delta_net_convolution(round_products=False, round_before_activation=False,
                                                      accumulation_order="chronological")
    wrong = pm.F.nn.is_gated_delta_net_convolution(round_products=True)
    assert pm.try_match_root(node, matching, resumed) is not None
    assert pm.try_match_root(node, wrong, resumed) is None
    wrong_order = pm.F.nn.is_gated_delta_net_convolution(accumulation_order="current_first")
    assert pm.try_match_root(node, wrong_order, resumed) is None


@pytest.mark.parametrize("order", ["reverse", "", 0, None])
def test_invalid_accumulation_order_is_rejected(order):
    with pytest.raises(IRSchemaError, match="accumulation_order"):
        _module(accumulation_order=order)


def test_default_evaluator_matches_current_first_device_accumulation():
    # The established device kernel visits the current token before history.
    # A vectorized Torch sum instead cancels the large history values first.
    _, config = _module()
    state = create_gdn_state(config)
    history = torch.tensor([1.e20, -1.e20, 1.], dtype=torch.bfloat16).repeat(config.conv_dim, 1)
    state.update_convolution_layer(history)
    qkv = torch.ones((1, config.conv_dim), dtype=torch.bfloat16)
    weight = torch.ones((config.conv_dim, 4), dtype=torch.bfloat16)
    actual, _ = gated_delta_net_convolution(qkv=qkv, state=state, conv_weight=weight, conv_kernel_size=4)
    expected = torch.nn.functional.silu(qkv.float()).bfloat16()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
