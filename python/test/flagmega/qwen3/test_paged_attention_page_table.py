# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A scheduler's logical pages need not be consecutive physical pages."""

import pytest

from triton.flagmega.errors import EvaluationError
from triton.flagmega.evaluator import PagedAttentionStateConfig, create_paged_attention_state

torch = pytest.importorskip("torch")


def make_state():
    return create_paged_attention_state(
        PagedAttentionStateConfig(1, 1, 8, block_size=2, num_blocks=3))


def test_permuted_pages_append_and_gather_across_boundary():
    state = make_state()
    state.block_table.copy_(torch.tensor([[2, 0, 1]], dtype=torch.int32))
    keys = [torch.full((1, 8), i, dtype=torch.bfloat16) for i in (1, 2, 3)]
    for key in keys:
        state.append(key, key + 4, layer_id=0)
    key, value = state.gather(layer_id=0)
    torch.testing.assert_close(key, torch.stack(keys), rtol=0, atol=0)
    torch.testing.assert_close(value, torch.stack(keys) + 4, rtol=0, atol=0)
    torch.testing.assert_close(state.kv_caches[2, 0, 0, 0].flatten(), keys[0].flatten())
    torch.testing.assert_close(state.kv_caches[0, 0, 0, 0].flatten(), keys[2].flatten())


@pytest.mark.parametrize("table", [[-1, 0, 1], [3, 0, 1]])
def test_out_of_bounds_page_is_rejected(table):
    state = make_state()
    state.block_table.copy_(torch.tensor([table], dtype=torch.int32))
    with pytest.raises(EvaluationError, match="physical page"):
        state.validate()


def test_inactive_slots_may_repeat_but_active_pages_must_not_alias():
    state = make_state()
    state.block_table.copy_(torch.tensor([[2, 0, 0]], dtype=torch.int32))
    state.validate()
    state.seq_lens.fill_(3)
    state.validate()
    state.seq_lens.fill_(5)
    with pytest.raises(EvaluationError, match="alias"):
        state.validate()
