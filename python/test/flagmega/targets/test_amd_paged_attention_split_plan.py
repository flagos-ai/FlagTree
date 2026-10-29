# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.importer import import_qwen3_model
from triton.flagmega.options import CompileOptions
from triton.flagmega.rules.ntt.decompose_paged_attention import (
    head_divisible_paged_attention_split_plan,
)

pytest.importorskip("torch")

from python.test.flagmega.qwen3.helpers import full_checkpoint_wmma  # noqa: E402

MESH = (fm.Placement((8, 16), "yx", "bb"),)


@pytest.mark.parametrize("heads,expected", [
    ((24,), (1, 16)),  # 24 % 16 != 0 with the default y split; 24 % 8 == 0 with x
    ((16,), None),  # the default y split already shards 16 heads over x
    ((32,), None),
    ((12,), None),  # neither axis divides: keep the default
    ((24, 16), (1, 16)),
])
def test_head_divisible_split_plan(heads, expected):
    assert head_divisible_paged_attention_split_plan(MESH, heads) == expected


def _partial_split(target, num_attention_heads):
    checkpoint = full_checkpoint_wmma(
        num_hidden_layers=1, hidden=num_attention_heads * 8, intermediate=128,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=4, head_dim=8, vocab=64,
    )
    module = import_qwen3_model(checkpoint)
    module = Compiler(CompileOptions(target=target)).compile(module, stop_after="decompose-paged-attention").module
    partials = [node for node in module.nodes if node.op == "ntt.paged_attention_partial"]
    assert partials
    return {(node.attrs["split_hierarchy_axis"], node.attrs["split_count"]) for node in partials}


def test_rdna_decode_splits_context_on_the_axis_that_keeps_heads_sharded():
    assert _partial_split("amd-gfx1201", 24) == {(1, 16)}
    assert _partial_split("amd-gfx1100", 24) == {(1, 16)}


@pytest.mark.parametrize("target,heads", [
    ("amd-gfx1201", 16),  # already shardable: unchanged
    ("nvidia-sm90", 24),  # other machines keep the nncase default
])
def test_default_split_is_kept_elsewhere(target, heads):
    assert _partial_split(target, heads) == {(0, 8)}
