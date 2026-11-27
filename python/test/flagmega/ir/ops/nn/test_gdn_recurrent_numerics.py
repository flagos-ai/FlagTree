# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Recurrent storage boundaries must agree between evaluation and kernels."""

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega import pattern_match as pm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator

from triton.flagmega.ir.ops.nn._gdn_state import GatedDeltaNetStateConfig, create_gdn_state
from triton.flagmega.ir.ops.nn.gdn_recurrent_core import gated_delta_net_recurrent_core
from python.test.flagmega.gdn.recurrent_helpers import recurrent_case, recurrent_module, recurrent_reference


def test_default_evaluator_rounds_beta_like_the_device_kernel():
    config = GatedDeltaNetStateConfig(1, 2, 4, 8, 8, 4, 32)
    state = create_gdn_state(config)
    generator = torch.Generator().manual_seed(1409)
    qkv = torch.randn((1, config.conv_dim), generator=generator).bfloat16()
    z = torch.ones((1, 32), dtype=torch.bfloat16)
    source = torch.zeros((1, 32), dtype=torch.bfloat16)
    source[:, 0] = 1
    b_weight = torch.zeros((4, 32), dtype=torch.bfloat16)
    b_weight[:, 0] = torch.tensor([.1, .3, .7, 1.1]).bfloat16()
    gated_delta_net_recurrent_core(
        state=state, qkv=qkv, z=z, projection_input=source, b_weight=b_weight, a_weight=torch.zeros_like(b_weight),
        a_log=torch.zeros(4), dt_bias=torch.zeros(4, dtype=torch.bfloat16), norm_weight=torch.ones(8),
        attrs={"num_key_heads": 2, "num_value_heads": 4, "key_head_dim": 8, "value_head_dim": 8, "epsilon": 1e-6})
    key = qkv[0, 16:32].reshape(2, 8).repeat_interleave(2, dim=0).float()
    key = key / torch.linalg.vector_norm(key, dim=-1, keepdim=True).clamp_min(1e-12)
    value = qkv[0, 32:].reshape(4, 8).float()
    beta = torch.sigmoid((source @ b_weight.T).float()).bfloat16().float()[0]
    expected = key[:, :, None] * (value * beta[:, None])[:, None, :]
    torch.testing.assert_close(state.recurrent_layer(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["clamp", "add"])
@pytest.mark.parametrize("round_qk", [False, True])
@pytest.mark.parametrize("round_beta", [False, True])
@pytest.mark.parametrize("round_core", [False, True])
def test_independent_rounding_boundaries_and_multistep_state(mode, round_qk, round_beta, round_core):
    _, attrs, types, values = recurrent_case(qk_norm_mode=mode, qk_norm_epsilon=1e-6, round_normalized_qk=round_qk,
                                             round_beta=round_beta, round_core=round_core)
    module = recurrent_module(types, attrs)
    expected, states = recurrent_reference(values, attrs)
    (actual, state), = TorchEvaluator(DictWeightResolver({})).run(module, values)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(state.recurrent_layer(), states[-1], rtol=2e-6, atol=1e-7)
    assert torch.all(state.convolution == 17)
    assert state is values["state"]


@pytest.mark.parametrize("name,value", [("qk_norm_mode", "rms"), ("qk_norm_epsilon", 0), ("qk_norm_epsilon", -1),
                                        ("qk_norm_epsilon", float("nan")), ("qk_norm_epsilon", float("inf")),
                                        ("round_beta", 1), ("round_core", "true"), ("round_normalized_qk", None)])
def test_invalid_recurrent_numerical_attributes(name, value):
    _, attrs, types, _ = recurrent_case(**{name: value})
    with pytest.raises(IRSchemaError, match=name):
        recurrent_module(types, attrs)


def test_recurrent_numerical_attrs_survive_python_and_named_patterns():
    _, attrs, types, _ = recurrent_case(qk_norm_mode="add", qk_norm_epsilon=1e-6, round_normalized_qk=True,
                                        round_beta=False, round_core=True)
    module = recurrent_module(types, attrs)
    namespace = {}
    exec(fm.module_source(module), namespace)
    resumed = namespace["MODULE"]
    assert resumed.semantic_hash == module.semantic_hash
    pattern = pm.F.nn.is_gated_delta_net_recurrent_core(**attrs)
    assert pm.try_match_root(resumed.node_map["recurrent"], pattern, resumed) is not None
    wrong = pm.F.nn.is_gated_delta_net_recurrent_core(round_beta=True)
    assert pm.try_match_root(resumed.node_map["recurrent"], wrong, resumed) is None


def test_legacy_constructor_keeps_existing_device_defaults():
    _, attrs, types, _ = recurrent_case()
    module = recurrent_module(types, attrs)
    actual = module.node_map["recurrent"].attrs
    assert actual["qk_norm_mode"] == "clamp"
    assert actual["qk_norm_epsilon"] == 1e-12
    assert actual["round_beta"] is True
    assert actual["round_normalized_qk"] is actual["round_core"] is False
