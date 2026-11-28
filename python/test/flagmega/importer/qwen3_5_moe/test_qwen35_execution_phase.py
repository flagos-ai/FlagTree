# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Explicit token extents and phase-specific reusable import functions."""

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import ImporterError
from triton.flagmega.importer import Qwen35MoeImporter, import_model
from python.test.flagmega.importer.qwen3_5_moe.helpers import checkpoint


@pytest.mark.parametrize("tokens", (1, 3, 65))
def test_prefill_phase_is_explicit_and_keeps_reusable_complete_decoder(tokens):
    module = import_model(checkpoint(), mode="prefill", num_tokens=tokens)
    assert module.metadata["execution_phase"] == "prefill"
    assert module.metadata["tokens_per_call"] == tokens
    assert module.metadata["num_hidden_layers"] == 4
    assert set(module.function_map) == {"main", "prefill_linear", "prefill_attention"}
    assert module.node_map["input_ids"].type == fm.tensor_type("int32", (tokens, ))
    for name in ("prefill_linear", "prefill_attention"):
        assert module.node_map[name + "_hidden"].type.shape[0].fixed_value == tokens
        assert module.function_map[name].attrs["reusable"]
    # LM head/sampling only consume the last prompt row, not every token.
    assert module.node_map["logits"].type == fm.tensor_type("float32", (1, 32))
    assert module.node_map["prefill_last_hidden"].attrs["starts"] == (tokens - 1, )
    namespace = {}
    exec(fm.module_source(module), namespace)
    assert namespace["MODULE"].semantic_hash == module.semantic_hash


def test_decode_and_single_token_prefill_remain_distinct_contracts():
    decode = import_model(checkpoint())
    prefill = import_model(checkpoint(), mode="prefill", num_tokens=1)
    assert decode.metadata["execution_phase"] == "decode"
    assert decode.semantic_hash != prefill.semantic_hash
    assert decode.node_map["input_ids"].type == prefill.node_map["input_ids"].type


@pytest.mark.parametrize(
    "options", ({"execution_phase": "auto"}, {"execution_phase": "decode", "num_tokens": 2},
                {"execution_phase": "prefill", "num_tokens": 0}, {"execution_phase": "prefill", "num_tokens": True}))
def test_invalid_phase_or_token_extent_is_rejected(options):
    with pytest.raises(ImporterError):
        Qwen35MoeImporter(checkpoint(), **options)
