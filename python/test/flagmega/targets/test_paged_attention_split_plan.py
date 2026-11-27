# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.targets.pyntt import (
    PagedAttentionWork,
    paged_attention_split_plan,
)

# 16 query heads over 8 KV heads (GQA group 2), 32 pages of 256 tokens.
GQA_WORK = PagedAttentionWork(16, 8, 8192, 256)
# 16 query heads over 16 KV heads (no group).
MHA_WORK = PagedAttentionWork(16, 16, 8192, 256)


def test_split_plan_leaves_one_kv_stream_per_owner():
    placements = (
        fm.Placement((8, 16), "yx", "bb"),
        fm.Placement((8, 16), "yx", "bb"),
    )

    # Splitting the 8-wide axis would leave 16 participants for 8 KV streams,
    # so the work-derived plan takes the 16-wide axis instead.
    assert paged_attention_split_plan(placements, GQA_WORK) == (1, 16)


def test_split_plan_keeps_the_stream_matched_axis_without_a_query_group():
    placements = (
        fm.Placement((8, 16), "yx", "bb"),
        fm.Placement((8, 16), "yx", "bb"),
    )

    # 16 KV streams and 16 remaining participants: one stream each, and the
    # alternative would hand every participant two streams.
    assert paged_attention_split_plan(placements, MHA_WORK) == (0, 8)


def test_split_plan_ignores_non_block_and_inconsistent_extent_axes():
    placements = (
        fm.Placement((8, 16, 4), "yxc", "cbb"),
        fm.Placement((8, 16, 8), "yxc", "cbb"),
    )

    assert paged_attention_split_plan(placements, GQA_WORK) == (1, 16)


def test_split_plan_requires_the_attention_workload():
    placements = (fm.Placement((8, 16), "yx", "bb"), )

    with pytest.raises(TypeError, match="PagedAttentionWork"):
        paged_attention_split_plan(placements, None)


@pytest.mark.parametrize(
    "placements",
    (
        (),
        (fm.Placement((1, 1), "yx", "bb"), ),
        (
            fm.Placement((8, ), "x", "b"),
            fm.Placement((8, 16), "yx", "bb"),
        ),
    ),
)
def test_split_plan_rejects_missing_compatible_physical_block_axis(placements):
    with pytest.raises(ValueError, match="paged-attention decomposition"):
        paged_attention_split_plan(placements, GQA_WORK)


@pytest.mark.parametrize(
    "work",
    (
        (16, 0, 8192, 256),
        (16, 5, 8192, 256),
        (16, 8, 8192, 0),
        (16, 8, 100, 256),
    ),
)
def test_work_rejects_inconsistent_head_and_cache_geometry(work):
    with pytest.raises(ValueError, match="PagedAttentionWork"):
        PagedAttentionWork(*work)


def test_split_plan_maximizes_context_shards_when_one_kv_stream_is_shared():
    # A latent/MLA cache has a single KV stream, so every participant reads it
    # whatever the plan does.  The work then says: cut the context as finely as
    # the mesh allows, because that is the only term left to reduce.  This is a
    # deliberate change from the old smallest-extent rule and is pinned here.
    placements = (fm.Placement((8, 16), "yx", "bb"), )
    latent = PagedAttentionWork(128, 1, 8192, 256)

    assert paged_attention_split_plan(placements, latent) == (1, 16)


def test_split_plan_prefers_the_axis_that_matches_the_kv_stream_count():
    # Three block axes: only leaving 8 participants beside the split matches
    # the 8 KV streams, and that is the 16-wide axis (4 * 8 * 16 = 512).
    placements = (fm.Placement((4, 8, 16), "zyx", "bbb"), )
    work = PagedAttentionWork(16, 8, 8192, 256)

    assert paged_attention_split_plan(placements, work) == (2, 16)


def test_kv_stream_ownership_outranks_the_cache_page_bound():
    # The cache-page term orders candidates of equal KV-stream ownership; it
    # never overrides ownership.  A two-page cache would leave 14 of 16 context
    # shards idle, and the 16-way split is still right, because duplicating a
    # KV stream costs the *active* participants on every tile they read.
    placements = (fm.Placement((8, 16), "yx", "bb"), )
    two_page_cache = PagedAttentionWork(16, 8, 512, 256)

    assert paged_attention_split_plan(placements, two_page_cache) == (1, 16)


def test_equal_extent_axes_resolve_to_the_lowest_axis():
    # A symmetric mesh gives both axes the same ownership quality and the same
    # usable shard count, so the plan stays deterministic on the lowest axis.
    placements = (fm.Placement((4, 4), "yx", "bb"), )
    work = PagedAttentionWork(16, 4, 8192, 256)

    assert paged_attention_split_plan(placements, work) == (0, 4)
