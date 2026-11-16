# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The paged-attention context split must come from the attention work.

The regression locked here is a distribution-ownership defect, not a tile
defect.  ``paged_attention_split_plan`` used to return the *smaller mesh
extent* as the context-split count, so on a ``(8, 16)`` mesh the remaining
extent that carries the query-head axis was 16 while the model has only 8 KV
heads: two participants owned, fetched and issued the same KV stream, and the
context critical path was cut 8 ways instead of 16.  The split count must
follow the GQA group and the cache geometry instead, which leaves exactly one
KV stream per participant and one whole GQA group local to each.

Everything here is CPU-only and runs the real ``decompose-paged-attention``
stage; it makes no statement about measured time.
"""

import math

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import PagedAttentionStateConfig
from triton.flagmega.stages import get_stage

# Qwen3-1.7B decode on one H800: 16 query heads over 8 KV heads (GQA group 2),
# 28 layers, 32 pages of 256 tokens, 128-element heads.  These are the b0018
# baseline cache dimensions (``bf16<8>[32,28,2,256,8,16]``).
QWEN3_CACHE = PagedAttentionStateConfig(28, 8, 128, block_size=256, num_blocks=32)
MHA_CACHE = PagedAttentionStateConfig(28, 16, 128, block_size=256, num_blocks=32)
QWEN3_MESH = (fm.Placement((8, 16), "yx", "bb"),)
ATTENTION_TILE = 64


class _AttentionModule(fm.Module):
    def __init__(self, *, query_heads=16, cache=QWEN3_CACHE):
        super().__init__(
            dialect="high_level", stage="stats_threaded", entry="main"
        )
        self.query_heads = query_heads
        self.cache = cache

    def forward(self):
        head_dim = self.cache.head_dim
        query = self.input(
            "query", fm.tensor_type("bfloat16", (1, self.query_heads, head_dim))
        )
        state = self.input("state", self.cache.ref_type)
        layer = self.input("layer", fm.tensor_type("int32", ()))
        attention = fm.F.nn.paged_attention(
            query,
            state,
            layer,
            scale=head_dim**-0.5,
            layout=("seq", "head", "dim"),
            hidden_size=self.query_heads * head_dim,
            name="attention",
            metadata={"source": "unit"},
        )
        self.function("main", (query, state, layer), (attention,))


class _MeshTarget:
    """The only target services the decomposition stage consults."""

    def __init__(self, placements):
        self._placements = tuple(placements)

    def distributed_placements(self, module):
        return self._placements

    def verify(self, module):
        return None


def _split_plan_of(module, placements=QWEN3_MESH):
    """Run the real stage and read back the plan it recorded on the partial."""

    decomposed = get_stage("decompose-paged-attention").run(
        module, _MeshTarget(placements)
    )
    partial = next(
        node for node in decomposed.nodes
        if node.op == "ntt.paged_attention_partial"
    )
    return (
        int(partial.attrs["split_hierarchy_axis"]),
        int(partial.attrs["split_count"]),
    )


def test_decomposed_partial_splits_context_sixteen_ways():
    assert _split_plan_of(_AttentionModule().build()) == (1, 16)


def test_each_kv_stream_has_exactly_one_owner():
    placement = QWEN3_MESH[0]
    split_axis, split_count = _split_plan_of(_AttentionModule().build())

    head_owners = placement.size // split_count

    # One owner per KV stream: no participant re-reads another participant's
    # KV bytes, and no participant has to stream more than one KV head.
    assert head_owners == QWEN3_CACHE.num_kv_heads
    # The owner key (context_shard, kv_head) stays a bijection onto the mesh,
    # so every partial-state span keeps exactly one physical writer.
    assert split_count * head_owners == placement.size
    # Each owner holds a whole GQA group locally.
    assert 16 // head_owners == 16 // QWEN3_CACHE.num_kv_heads == 2
    assert placement.hierarchy[split_axis] == split_count


def test_multi_head_attention_keeps_the_single_stream_axis():
    # Without a GQA group the 16-way axis would hand one participant two KV
    # streams, so the work-derived plan keeps the 8-way split for this mesh.
    module = _AttentionModule(query_heads=16, cache=MHA_CACHE).build()

    assert _split_plan_of(module) == (0, 8)


@pytest.mark.parametrize("context_length", (128, 1024, 4096))
def test_plan_is_independent_of_the_runtime_context_length(context_length):
    # The decode context length is a runtime value (``slot_mapping``), so one
    # compiled artifact serves every scenario on this cache and the plan is the
    # same for all of them.  That is what makes ctx128 the measurement that
    # isolates the participant-count cost: the tile plan there is identical for
    # 8 and 16 shards, so only the rendezvous group size differs.
    assert _split_plan_of(_AttentionModule().build()) == (1, 16)

    tiles = math.ceil((context_length + 1) / ATTENTION_TILE)
    eight = math.ceil(tiles / 8)
    sixteen = math.ceil(tiles / 16)
    if context_length == 128:
        assert (tiles, eight, sixteen) == (3, 1, 1)
    else:
        assert sixteen < eight


def test_split_plan_refuses_to_plan_without_the_attention_work():
    from triton.flagmega.targets.pyntt import paged_attention_split_plan

    with pytest.raises(TypeError, match="PagedAttentionWork"):
        paged_attention_split_plan(QWEN3_MESH, None)


def test_module_work_carries_the_gqa_group_and_cache_geometry():
    from triton.flagmega.targets.pyntt import (
        PagedAttentionWork,
        paged_attention_work,
    )

    work = paged_attention_work(_AttentionModule().build())

    assert work == PagedAttentionWork(
        num_query_heads=16,
        num_kv_heads=8,
        max_context_length=8192,
        cache_block_size=256,
    )
    assert work.query_heads_per_kv_head == 2
    assert work.context_pages == 32


def test_split_plan_still_rejects_an_unusable_mesh():
    from triton.flagmega.targets.pyntt import (
        paged_attention_split_plan,
        paged_attention_work,
    )

    work = paged_attention_work(_AttentionModule().build())
    for placements in (
        (),
        (fm.Placement((1, 1), "yx", "bb"),),
        (fm.Placement((8,), "x", "b"), fm.Placement((8, 16), "yx", "bb")),
    ):
        with pytest.raises(ValueError, match="paged-attention decomposition"):
            paged_attention_split_plan(placements, work)


class _NoAttentionModule(fm.Module):
    def __init__(self):
        super().__init__(
            dialect="high_level", stage="stats_threaded", entry="main"
        )

    def forward(self):
        value = self.input("value", fm.tensor_type("float32", (4, 8)))
        result = fm.F.math.silu(value, name="result")
        self.function("main", (value,), (result,))


def test_stage_is_a_no_op_without_any_paged_attention():
    # A module with no attention has no attention work; the stage must not
    # demand one.  Reading the work used to raise here, which broke every
    # attention-free module that passes through this stage.
    module = _NoAttentionModule().build()

    decomposed = get_stage("decompose-paged-attention").run(
        module, _MeshTarget(QWEN3_MESH)
    )

    assert not any(
        node.op.startswith("ntt.paged_attention") for node in decomposed.nodes
    )
