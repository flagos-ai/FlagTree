# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Unused future-token decay ratios must never enter a block's arithmetic."""

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.ir.ops.nn.delta_rule_block_update import delta_rule_block_update
from python.test.flagmega.codegen.triton.kernels.delta_rule_block_update.helpers import execute_block_update


def strong_decay(torch):
    tokens, dimension = 32, 16
    query = torch.ones((tokens, 1, dimension), dtype=torch.bfloat16) / 16
    key = torch.zeros_like(query)
    value = torch.ones_like(query)
    coefficient = (torch.eye(64) / 2).bfloat16().reshape(1, 1, 64, 64)
    # Step over the FP32 subnormal range so the CPU oracle and the native
    # device's flush-to-zero arithmetic agree exactly on the carried values.
    prefix = -40 * torch.arange(1, 65).float().reshape(1, 1, 64)
    initial = torch.ones((1, dimension, dimension))
    expected = torch.exp2(prefix[0, 0, :tokens])[:, None, None].expand_as(value).bfloat16()
    return (query, key, value, coefficient, prefix), initial, expected


def test_eager_strong_decay_keeps_zero_key_update_finite():
    torch = pytest.importorskip("torch")
    values, initial, expected = strong_decay(torch)
    output, state = delta_rule_block_update(*values, initial, scale=1., torch=torch)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert bool(torch.isfinite(state).all())
    assert torch.count_nonzero(state) == 0


def test_device_strong_decay_keeps_zero_key_update_finite(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")
    values, initial, expected = strong_decay(torch)
    state_type = fm.RefType("state", (("untouched", fm.tensor_type("float32", (1,))),
                                     ("matrix", fm.tensor_type("float32", initial.shape))))
    output, state, resources = execute_block_update(
        tmp_path, torch, values, state_type, {"untouched": torch.tensor([23.]), "matrix": initial},
        attrs={"scale": 1.})
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert bool(torch.isfinite(state["matrix"]).all())
    assert torch.count_nonzero(state["matrix"]) == 0
    assert state["untouched"].item() == 23.
    assert resources["spill_bytes"] == 0


def test_two_full_blocks_in_reusable_function_keep_strict_resources(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")
    query = torch.full((128, 4, 128), 1 / 128, dtype=torch.bfloat16)
    key = torch.zeros_like(query)
    value = torch.ones((128, 8, 128), dtype=torch.bfloat16)
    beta = torch.full((128, 8), 0.5)
    alpha = torch.ones_like(beta)
    initial = torch.ones((8, 128, 128))
    state_type = fm.RefType("state", (("untouched", fm.tensor_type("float32", (1,))),
                                     ("matrix", fm.tensor_type("float32", initial.shape))))
    output, state, resources = execute_block_update(
        tmp_path, torch, (query, key, value, beta, alpha), state_type,
        {"untouched": torch.tensor([23.]), "matrix": initial}, head_axes=(0, 1),
        attrs={"scale": 1.}, compose=True, reusable=True)
    torch.testing.assert_close(output, value, rtol=0, atol=0)
    torch.testing.assert_close(state["matrix"], initial, rtol=0, atol=0)
    assert state["untouched"].item() == 23.
    assert resources["spill_bytes"] == 0


def test_broadcast_packed_state_in_reusable_function_keeps_strict_resources(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")
    query = torch.full((128, 16, 128), 1 / 128, dtype=torch.bfloat16)
    key = torch.zeros_like(query)
    value = torch.ones((128, 32, 128), dtype=torch.bfloat16)
    beta = torch.full((128, 32), 0.5)
    alpha = torch.ones_like(beta)
    initial = torch.ones((1, 32, 128, 32, 4))
    field = fm.tensor_type(fm.VectorType(fm.DType.FLOAT32, (4,)), initial.shape[:-1])
    state_type = fm.RefType("state", (("untouched", fm.tensor_type("float32", (1,))), ("matrix", field)))
    output, state, resources = execute_block_update(
        tmp_path, torch, (query, key, value, beta, alpha), state_type,
        {"untouched": torch.tensor([23.]), "matrix": initial}, placement_shape=(8, 16),
        attrs={"scale": 1., "state_layout": ("layer", "head", "value", "key"), "state_vector_axes": ("key",)},
        compose=True, reusable=True)
    torch.testing.assert_close(output, value, rtol=0, atol=0)
    torch.testing.assert_close(state["matrix"], initial, rtol=0, atol=0)
    assert state["untouched"].item() == 23.
    assert resources["spill_bytes"] == 0
