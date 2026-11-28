# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Single-request chunks retain the packed cache ABI and advance once."""

import pytest
import torch

from triton.flagmega.errors import EvaluationError
from triton.flagmega.evaluator import PagedAttentionStateConfig, create_paged_attention_state


def make_state():
    state = create_paged_attention_state(PagedAttentionStateConfig(3, 2, 8, block_size=4, num_blocks=4))
    state.kv_caches.fill_(-9)
    state.block_table.copy_(torch.tensor([[2, 0, 3, 1]], dtype=torch.int32))
    state.seq_lens.fill_(3)
    return state


@pytest.mark.parametrize("tokens", (1, 3, 6))
@pytest.mark.parametrize("advance", (False, True))
@pytest.mark.parametrize("kind,index", (("key", 0), ("value", 1)))
def test_update_chunk_crosses_pages_without_touching_other_layers(tokens, advance, kind, index):
    state = make_state()
    slots = torch.arange(tokens * 16).reshape(tokens, 2, 8).bfloat16()
    expected = state.kv_caches.clone()
    for row in range(tokens):
        block, offset = divmod(3 + row, 4)
        expected[state.block_table[0, block], 1, index, offset].copy_(slots[row].reshape(2, 1, 8))
    assert state.update(slots, cache_kind=kind, layer_id=1, advance_sequence=advance) == 3
    torch.testing.assert_close(state.kv_caches, expected, rtol=0, atol=0)
    assert state.slot_mapping.tolist() == [3]
    assert state.query_start_loc.tolist() == [0, tokens]
    assert state.sequence_length == 3 + (tokens if advance else 0)


def test_append_chunk_updates_both_fields_and_advances_once_per_all_layers():
    state = make_state()
    for tokens in (3, 5):
        start = state.sequence_length
        key = torch.full((tokens, 2, 8), float(start), dtype=torch.bfloat16)
        value = key + 1
        for layer in range(3):
            assert state.append(key, value, layer_id=layer, advance_sequence=layer == 2) == start
            keys, values = state.gather(layer_id=layer, length=start + tokens)
            torch.testing.assert_close(keys[start:], key, rtol=0, atol=0)
            torch.testing.assert_close(values[start:], value, rtol=0, atol=0)
        assert state.sequence_length == start + tokens
        assert state.slot_mapping.item() == start


@pytest.mark.parametrize("failure", ("capacity", "alias", "empty", "shape"))
def test_invalid_chunk_does_not_partially_modify_state(failure):
    state = make_state()
    slots = torch.ones((3, 2, 8), dtype=torch.bfloat16)
    if failure == "capacity":
        state.seq_lens.fill_(14)
    elif failure == "alias":
        state.block_table[0, 1] = state.block_table[0, 0]
    elif failure == "empty":
        slots = slots[:0]
    elif failure == "shape":
        slots = slots[:, :1]
    before = state.clone()
    with pytest.raises(EvaluationError):
        state.update(slots, cache_kind="value", layer_id=1, advance_sequence=True)
    for field in ("kv_caches", "query_start_loc", "seq_lens", "slot_mapping", "block_table"):
        torch.testing.assert_close(getattr(state, field), getattr(before, field), rtol=0, atol=0)


def test_append_validates_both_shapes_before_writing_key():
    state = make_state()
    before = state.clone()
    with pytest.raises(EvaluationError):
        state.append(torch.ones((3, 2, 8)), torch.ones((2, 2, 8)), layer_id=1)
    torch.testing.assert_close(state.kv_caches, before.kv_caches, rtol=0, atol=0)
