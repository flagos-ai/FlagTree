# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega.errors import EvaluationError
from triton.flagmega.evaluator import PagedAttentionStateConfig, create_paged_attention_state
from triton.flagmega.ir import DType, VectorType


torch = pytest.importorskip("torch")


def test_state_uses_nncase_axis_order_and_explicit_vector_lane():
    config = PagedAttentionStateConfig(2, 1, 8, block_size=2, num_blocks=2, lanes=8)

    assert config.cache_type.dtype == VectorType(DType.BFLOAT16, (8,))
    assert tuple(dimension.fixed_value for dimension in config.cache_type.shape) == (2, 2, 2, 2, 1, 1)
    assert config.storage_shape == (2, 2, 2, 2, 1, 1, 8)
    assert tuple(name for name, _ in config.ref_type.fields) == (
        "kv_caches", "query_start_loc", "seq_lens", "slot_mapping", "block_table")


def test_append_and_gather_cross_a_physical_block_boundary():
    config = PagedAttentionStateConfig(1, 1, 8, block_size=2, num_blocks=2)
    state = create_paged_attention_state(config)
    keys = [torch.full((1, 8), value, dtype=torch.bfloat16) for value in (1.0, 2.0, 3.0)]
    values = [torch.full((1, 8), value, dtype=torch.bfloat16) for value in (4.0, 5.0, 6.0)]

    for key, value in zip(keys, values):
        state.append(key, value, layer_id=0)

    gathered_key, gathered_value = state.gather(layer_id=0)
    torch.testing.assert_close(gathered_key, torch.stack(keys), rtol=0, atol=0)
    torch.testing.assert_close(gathered_value, torch.stack(values), rtol=0, atol=0)
    assert state.sequence_length == 3
    assert state.slot_mapping.item() == 2
    assert torch.equal(state.kv_caches[1, 0, 0, 0].reshape(1, 8), keys[2])


def test_state_rejects_append_past_capacity():
    state = create_paged_attention_state(PagedAttentionStateConfig(1, 1, 8, block_size=1, num_blocks=1))
    slot = torch.zeros((1, 8), dtype=torch.bfloat16)
    state.append(slot, slot, layer_id=0)

    with pytest.raises(EvaluationError, match="capacity"):
        state.append(slot, slot, layer_id=0)


def test_static_state_validation_is_capture_safe_without_host_value_reads():
    state = create_paged_attention_state(PagedAttentionStateConfig(1, 1, 8, block_size=2, num_blocks=2))
    state.seq_lens.fill_(-1)
    state.block_table.copy_(torch.tensor([[1, 0]], dtype=torch.int32))

    state.validate(dynamic_values=False)
    with pytest.raises(EvaluationError, match="sequence length"):
        state.validate()
