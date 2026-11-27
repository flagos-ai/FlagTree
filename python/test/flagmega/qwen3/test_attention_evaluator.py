# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega.evaluator import PagedAttentionStateConfig, create_paged_attention_state, qwen3_paged_attention


torch = pytest.importorskip("torch")


def test_two_decode_steps_read_history_and_update_the_same_cache():
    config = PagedAttentionStateConfig(1, 1, 8, block_size=2, num_blocks=2)
    state = create_paged_attention_state(config)
    q_weight = torch.zeros((16, 16), dtype=torch.bfloat16)
    k_weight = torch.zeros((8, 16), dtype=torch.bfloat16)
    v_weight = torch.zeros((8, 16), dtype=torch.bfloat16)
    v_weight[:, :8] = torch.eye(8, dtype=torch.bfloat16)
    norm_weight = torch.ones((8,), dtype=torch.bfloat16)
    output_weight = torch.eye(16, dtype=torch.bfloat16)
    first = torch.arange(16, dtype=torch.bfloat16).reshape(1, 16)
    second = (torch.arange(16, dtype=torch.bfloat16) + 16).reshape(1, 16)

    first_output, returned_state = qwen3_paged_attention(
        first, state, q_weight, k_weight, v_weight, norm_weight, norm_weight, output_weight,
        layer_id=0, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        epsilon=1e-6, rope_theta=10000.0, torch=torch)
    second_output, returned_state_2 = qwen3_paged_attention(
        second, state, q_weight, k_weight, v_weight, norm_weight, norm_weight, output_weight,
        layer_id=0, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        epsilon=1e-6, rope_theta=10000.0, torch=torch)

    first_value = first[:, :8]
    second_mean = (first[:, :8].float() + second[:, :8].float()) / 2
    torch.testing.assert_close(first_output, first_value.repeat(1, 2), rtol=0, atol=0)
    torch.testing.assert_close(second_output.float(), second_mean.repeat(1, 2), rtol=0, atol=0)
    assert returned_state is state and returned_state_2 is state
    assert state.sequence_length == 2


def test_qk_norm_and_rope_change_second_step_attention_weights():
    generator = torch.Generator().manual_seed(9)
    state = create_paged_attention_state(PagedAttentionStateConfig(1, 1, 8, block_size=4, num_blocks=1))
    weights = [torch.randn(shape, generator=generator).to(torch.bfloat16) * 0.1 for shape in (
        (16, 16), (8, 16), (8, 16), (16, 16))]
    norm = torch.ones((8,), dtype=torch.bfloat16)
    first_hidden = torch.randn((1, 16), generator=generator).to(torch.bfloat16)
    second_hidden = torch.randn((1, 16), generator=generator).to(torch.bfloat16)

    qwen3_paged_attention(
        first_hidden, state, weights[0], weights[1], weights[2], norm, norm, weights[3],
        layer_id=0, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        epsilon=1e-6, rope_theta=10000.0, torch=torch)
    with_history, _ = qwen3_paged_attention(
        second_hidden, state, weights[0], weights[1], weights[2], norm, norm, weights[3],
        layer_id=0, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        epsilon=1e-6, rope_theta=10000.0, torch=torch)

    fresh = create_paged_attention_state(state.config)
    no_history, _ = qwen3_paged_attention(
        second_hidden, fresh, weights[0], weights[1], weights[2], norm, norm, weights[3],
        layer_id=0, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        epsilon=1e-6, rope_theta=10000.0, torch=torch)
    assert not torch.equal(with_history, no_history)
