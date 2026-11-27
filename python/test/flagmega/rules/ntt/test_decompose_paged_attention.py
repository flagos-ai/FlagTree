# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import inspect

import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import (
    DictWeightResolver,
    PagedAttentionStateConfig,
    TorchEvaluator,
    create_paged_attention_state,
)
from triton.flagmega.rules import DataflowRewriter
from triton.flagmega.rules.ntt.decompose_paged_attention import (
    decompose_paged_attention_rule,
)


class _Graph(fm.Module):
    def __init__(self, *, sequence=1):
        super().__init__(dialect="high_level", stage="packed", entry="main")
        self.sequence = sequence

    def forward(self):
        query = self.input(
            "query", fm.tensor_type("bfloat16", (self.sequence, 4, 8))
        )
        state = self.input(
            "state",
            PagedAttentionStateConfig(
                1, 2, 8, block_size=4, num_blocks=2
            ).ref_type,
        )
        layer = self.input("layer", fm.tensor_type("int32", ()))
        attention = fm.F.nn.paged_attention(
            query,
            state,
            layer,
            scale=8**-0.5,
            layout=("seq", "head", "dim"),
            hidden_size=32,
            name="attention",
            metadata={"source": "unit"},
        )
        self.function("main", (query, state, layer), (attention,))


def _rewrite(module: fm.IRModule) -> fm.IRModule:
    return DataflowRewriter((decompose_paged_attention_rule(0, 8),)).rewrite(module)


def test_rule_builds_explicit_partial_state_tuple_and_combine_with_stable_root():
    rewritten = _rewrite(_Graph().build())

    partial = next(node for node in rewritten.nodes if node.op == "ntt.paged_attention_partial")
    output = rewritten.node_map["attention"]
    projections = tuple(
        rewritten.node_map[input_id] for input_id in output.inputs
    )
    assert output.op == "ntt.paged_attention_combine"
    assert output.type == fm.tensor_type("bfloat16", (1, 4, 8))
    assert tuple(node.op for node in projections) == ("builtin.get_item",) * 3
    assert tuple(node.attrs["index"] for node in projections) == (0, 1, 2)
    assert all(node.inputs == (partial.id,) for node in projections)
    assert not partial.effect.is_pure
    assert output.effect.is_pure
    assert output.metadata["source"] == "unit"
    assert output.metadata["decomposition_rule"] == "DecomposePagedAttention"


def test_rule_does_not_form_an_empty_query_reduction():
    original = _Graph(sequence=0).build()

    assert _rewrite(original) == original


def test_decomposed_reference_evaluator_matches_original_attention():
    original = _Graph().build()
    rewritten = _rewrite(original)
    state = create_paged_attention_state(
        PagedAttentionStateConfig(1, 2, 8, block_size=4, num_blocks=2)
    )
    key = torch.arange(16, dtype=torch.float32).reshape(2, 8).to(torch.bfloat16)
    value = (key.float() + 10).to(torch.bfloat16)
    state.append(key, value, layer_id=0, advance_sequence=True)
    feeds = {
        "query": torch.randn(
            (1, 4, 8), generator=torch.Generator().manual_seed(13)
        ).to(torch.bfloat16),
        "state": state,
        "layer": torch.tensor(0, dtype=torch.int32),
    }
    evaluator = TorchEvaluator(DictWeightResolver({}))

    expected = evaluator.run(original, feeds)[0]
    actual = evaluator.run(rewritten, feeds)[0]

    torch.testing.assert_close(actual, expected)


def test_rule_source_is_target_neutral():
    source = inspect.getsource(decompose_paged_attention_rule).lower()
    for spelling in ("nvidia", "sm90", "cuda", "mma", "tma", "warp"):
        assert spelling not in source
