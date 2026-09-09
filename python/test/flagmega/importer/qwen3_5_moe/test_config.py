# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega.errors import ImporterError
from triton.flagmega.importer.qwen3_5_moe import Qwen35MoeConfig
from python.test.flagmega.importer.qwen3_5_moe.helpers import configuration


@pytest.mark.parametrize("key,value", [
    ("num_experts_per_tok", 5),
    ("num_experts_per_tok", True),
    ("num_hidden_layers", 0),
    ("layer_types", ["linear_attention"]),
    ("layer_types", ["unknown"] * 4),
    ("hidden_act", "gelu"),
    ("attention_bias", True),
    ("dtype", "float32"),
    ("mamba_ssm_dtype", "bfloat16"),
    ("num_key_value_heads", 3),
    ("linear_num_key_heads", 3),
    ("attn_output_gate", False),
    ("rms_norm_eps", float("nan")),
    ("mlp_only_layers", [1]),
    ("rope_parameters", {"partial_rotary_factor": .3}),
    ("rope_parameters", {"rope_type": "dynamic"}),
])
def test_config_rejects_unsupported_or_inconsistent_semantics(key, value):
    source = configuration()
    source["text_config"][key] = value
    with pytest.raises(ImporterError):
        Qwen35MoeConfig.parse(source)


def test_config_can_derive_layer_kinds_from_interval():
    source = configuration()
    del source["text_config"]["layer_types"]
    source["text_config"]["full_attention_interval"] = 2
    config = Qwen35MoeConfig.parse(source)
    assert config.layer_types == ("linear_attention", "full_attention") * 2
    assert config.rotary_dim == 4
