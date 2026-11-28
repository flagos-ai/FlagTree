# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch

from triton.flagmega.ir.ops.nn.delta_rule_coefficients import DeltaRuleCoefficients
from python.test.flagmega.ir.ops.primitive_helpers import evaluate


@pytest.mark.parametrize("block", [8, 16, 32, 64])
def test_two_token_lower_triangle_preserves_beta_and_fp16_inverse(block):
    dtype = torch.bfloat16
    key = torch.tensor([[[.5]], [[.75]]], dtype=dtype)
    beta = torch.tensor([[.3, .5], [.7, .9]])
    original_key, original_beta = key.clone(), beta.clone()
    result = evaluate(DeltaRuleCoefficients, (key, beta), block_size=block)
    expected = torch.zeros((1, 2, block, block), dtype=dtype)
    expected[0, :, 0, 0] = beta[0].to(dtype)
    expected[0, :, 1, 1] = beta[1].to(dtype)
    expected[0, :, 1, 0] = (-((.5 * .75 * beta[1]).half().float()) * beta[0]).to(dtype)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert torch.equal(key, original_key) and torch.equal(beta, original_beta)


@pytest.mark.parametrize("block", [8, 16, 32, 64])
def test_chunk_boundaries_grouped_heads_and_nontrivial_inverse(block):
    dtype = torch.bfloat16
    generator = torch.Generator().manual_seed(209)
    tokens, key_heads, heads, dim = 2 * block + 3, 2, 4, 16
    key = (.03 * torch.randn((tokens, key_heads, dim), generator=generator)).to(dtype)
    beta = torch.rand((tokens, heads), generator=generator)
    actual = evaluate(DeltaRuleCoefficients, (key, beta), block_size=block)
    repeated = key.repeat_interleave(2, dim=1).double()
    for chunk, begin in enumerate(range(0, tokens, block)):
        valid = min(block, tokens - begin)
        assert not torch.any(actual[chunk, :, valid:])
        assert not torch.any(actual[chunk, :, :, valid:])
        k = repeated[begin:begin + valid].transpose(0, 1)
        b = beta[begin:begin + valid].T.double()
        lower = torch.tril((k @ k.transpose(-1, -2)) * b[..., None], diagonal=-1)
        # Independent high-precision algebra is only an accuracy bound; exact
        # intermediate rounding is checked separately, not by this tolerance.
        expected = torch.linalg.solve_triangular(lower + torch.eye(valid), torch.diag_embed(b), upper=False)
        error = actual[chunk, :, :valid, :valid].double() - expected
        assert float(error.abs().max()) < 0.002
        assert not torch.any(torch.triu(actual[chunk], diagonal=1))


def test_empty_token_stream_and_zero_beta():
    empty = evaluate(DeltaRuleCoefficients, (torch.empty((0, 2, 8), dtype=torch.bfloat16), torch.empty((0, 4))))
    assert empty.shape == (0, 4, 64, 64)
    zeros = evaluate(DeltaRuleCoefficients, (torch.ones((67, 2, 8), dtype=torch.bfloat16), torch.zeros((67, 4))))
    assert not torch.any(zeros)
