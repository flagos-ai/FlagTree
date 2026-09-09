# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega.ir.ops.nn.delta_rule_log_prefix import DeltaRuleLogPrefix
from python.test.flagmega.ir.ops.primitive_helpers import evaluate


@pytest.mark.parametrize("block,group", [(8, 1), (8, 4), (32, 16), (64, 32), (128, 32)])
@pytest.mark.parametrize("tokens", [0, 1, 31, 65, 129])
def test_prefix_resets_blocks_and_repeats_last_prefix_in_padding(block, group, tokens):
    torch = pytest.importorskip("torch")
    alpha = torch.tensor([0.5, 1., 2.]).repeat(tokens, 1)
    before = alpha.clone()
    actual = evaluate(DeltaRuleLogPrefix, (alpha, ), block_size=block, scan_group_size=group)
    blocks = (tokens + block - 1) // block
    expected = torch.empty((blocks, 3, block))
    for index in range(blocks):
        steps = torch.arange(1, block + 1).clamp(max=min(block, tokens - index * block))
        expected[index] = torch.tensor([-1., 0., 1.])[:, None] * steps[None, :]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(alpha, before)


def test_epsilon_applies_only_to_real_tokens():
    torch = pytest.importorskip("torch")
    actual = evaluate(DeltaRuleLogPrefix, (torch.zeros((1, 2)), ), block_size=8, scan_group_size=4, epsilon=2.)
    torch.testing.assert_close(actual, torch.ones((1, 2, 8)), rtol=0, atol=0)


def test_zero_epsilon_log_domain_propagates_instead_of_silent_clamping():
    torch = pytest.importorskip("torch")
    actual = evaluate(DeltaRuleLogPrefix, (torch.tensor([[0., -1.]]), ), epsilon=0.)
    assert torch.isneginf(actual[0, 0]).all()
    assert torch.isnan(actual[0, 1]).all()
