# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import (
    DictWeightResolver,
    PagedAttentionStateConfig,
    TorchEvaluator,
    create_paged_attention_state,
)
from triton.flagmega import pattern_match as pm


class _RotaryEmbeddingGraph(fm.Module):
    def __init__(self):
        super().__init__(dialect="high_level", stage="imported", entry="main")

    def forward(self):
        reference = self.input("reference", fm.tensor_type("bfloat16", (1, 16)))
        state = self.input(
            "state", PagedAttentionStateConfig(1, 2, 8, block_size=4, num_blocks=2).ref_type)
        pair = fm.F.nn.rotary_embedding(
            reference,
            state,
            head_dim=8,
            theta=10000.0,
            name="rotary",
        )
        cos = fm.F.tensors.get_item(pair, 0, name="cos")
        sin = fm.F.tensors.get_item(pair, 1, name="sin")
        self.function("main", (reference, state), (cos, sin))


def test_rotary_embedding_uses_cache_position_and_has_named_parameters():
    module = _RotaryEmbeddingGraph().build()
    state = create_paged_attention_state(
        PagedAttentionStateConfig(1, 2, 8, block_size=4, num_blocks=2))
    state.seq_lens[0] = 3
    reference = torch.zeros((1, 16), dtype=torch.bfloat16)

    cos, sin = TorchEvaluator(DictWeightResolver({})).run(
        module, {"reference": reference, "state": state})

    indices = torch.arange(0, 8, 2, dtype=torch.float32)
    angle = 3 * (10000.0 ** (-indices / 8))
    frequency = torch.cat((angle, angle)).reshape(1, 1, 8)
    torch.testing.assert_close(cos, frequency.cos())
    torch.testing.assert_close(sin, frequency.sin())
    definition = fm.get_definition("nn.rotary_embedding")
    assert [value.name for value in definition.input_parameters] == ["reference", "state"]
    assert pm.try_match_root(
        module.node_map["rotary"],
        pm.F.nn.is_rotary_embedding(head_dim=8),
        module,
    ) is not None
