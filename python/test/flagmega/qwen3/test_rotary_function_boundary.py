# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Position embeddings are a token-step input, not a per-layer computation."""

import pytest
import torch

from triton.flagmega.evaluator import TorchEvaluator, CheckpointWeightResolver, create_paged_attention_state
from triton.flagmega.importer import Qwen3ModelImporter
from triton.flagmega.ir.ops.nn.rotary_embedding import RotaryEmbedding
from triton.flagmega.passes.functions.graph import function_nodes
from .helpers import full_checkpoint


@pytest.mark.parametrize("layers", [1, 3])
def test_import_computes_position_embeddings_in_main_and_passes_to_reusable_decoder(layers):
    module = Qwen3ModelImporter(full_checkpoint(num_hidden_layers=layers)).import_module()
    main = function_nodes(module, module.function_map["main"])
    decoder = function_nodes(module, module.function_map["decode_layer"])
    assert sum(node.op == "nn.rotary_embedding" for node in main) == 1
    assert not any(node.op == "nn.rotary_embedding" for node in decoder)
    calls = [node for node in main if node.op == "builtin.call"]
    assert len(calls) == layers
    assert len({node.inputs[-2:] for node in calls}) == 1
    assert module.node_map["decode_layer_query_rope"].inputs[-2:] == (
        "decode_layer_rotary_cos", "decode_layer_rotary_sin",
    )


def test_rotary_is_evaluated_once_per_token_step_and_before_sequence_advance(monkeypatch):
    checkpoint = full_checkpoint(num_hidden_layers=3)
    importer = Qwen3ModelImporter(checkpoint, block_size=4, num_blocks=2)
    module = importer.import_module()
    state = create_paged_attention_state(importer.state_config)
    positions = []
    original = RotaryEmbedding.evaluate

    def evaluate(cls, node, arguments, context):
        positions.append(cls.state.read(arguments).sequence_length)
        return original(node, arguments, context)

    monkeypatch.setattr(RotaryEmbedding, "evaluate", classmethod(evaluate))
    evaluator = TorchEvaluator(CheckpointWeightResolver(checkpoint))
    for token in (3, 4):
        evaluator.run(module, {"input_ids": torch.tensor([token], dtype=torch.int32),
                               "paged_attention_kv_cache": state})
    assert positions == [0, 1]
    assert state.sequence_length == 2
