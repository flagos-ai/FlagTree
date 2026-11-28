# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The explicit partial/combine contract applies to every causal query row."""

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.evaluator import PagedAttentionStateConfig, create_paged_attention_state
from triton.flagmega.rules import DataflowRewriter
from triton.flagmega.rules.ntt.decompose_paged_attention import decompose_paged_attention_rule
from python.test.flagmega.rules.ntt.test_decompose_paged_attention import _Graph


@pytest.mark.parametrize("tokens", (2, 3, 5))
def test_chunk_rule_preserves_rows_effects_and_numerical_values(tmp_path, tokens):
    original = _Graph(sequence=tokens).build()
    rewritten = DataflowRewriter((decompose_paged_attention_rule(0, 8), )).rewrite(original)
    output = rewritten.node_map["attention"]
    assert output.op == "ntt.paged_attention_combine"
    assert output.type == original.node_map["attention"].type
    partial = next(node for node in rewritten.nodes if node.op == "ntt.paged_attention_partial")
    assert partial.inputs == original.node_map["attention"].inputs
    assert not partial.effect.is_pure and output.effect.is_pure
    fm.emit_module(rewritten, tmp_path / "rewritten.py")
    rewritten = fm.load_module(tmp_path / "rewritten.py")
    state = create_paged_attention_state(PagedAttentionStateConfig(1, 2, 8, block_size=4, num_blocks=2))
    generator = torch.Generator().manual_seed(52)
    state.append(
        torch.randn((tokens, 2, 8), generator=generator).bfloat16(),
        torch.randn((tokens, 2, 8), generator=generator).bfloat16(), layer_id=0)
    feeds = {
        "query": torch.randn((tokens, 4, 8), generator=generator).bfloat16(), "state": state, "layer":
        torch.tensor(0, dtype=torch.int32)
    }
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(evaluator.run(rewritten, feeds)[0], evaluator.run(original, feeds)[0], rtol=0, atol=0)
