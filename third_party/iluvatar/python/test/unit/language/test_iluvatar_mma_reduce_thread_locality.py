"""Regression tests for preserving the initial value in MMA sum reductions."""

import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_corex

pytestmark = pytest.mark.skipif(not is_corex(), reason="Iluvatar backend only")


@triton.jit
def _rescaled_sum(Q, K, Alpha, Out, steps, INITIAL: tl.constexpr, SCALED: tl.constexpr):
    rows = tl.arange(0, 16)
    dims = tl.arange(0, 128)
    cols = tl.arange(0, 32)
    q = tl.load(Q + rows[:, None] * 128 + dims[None, :])
    acc = tl.full((16, ), INITIAL, tl.float32)
    for step in range(steps):
        k = tl.load(K + step * 128 * 32 + dims[:, None] * 32 + cols[None, :])
        scores = tl.dot(q, k)
        if SCALED:
            alpha = tl.load(Alpha + step * 16 + rows)
            acc = acc * alpha + tl.sum(scores, 1)
        else:
            acc = acc + tl.sum(scores, 1)
    tl.store(Out + rows, acc)


@pytest.mark.parametrize("initial", [0.0, 1.0, -2.0])
@pytest.mark.parametrize("scaled", [False, True])
@pytest.mark.parametrize("steps", [0, 1, 3])
@pytest.mark.parametrize("first_alpha", [0.0, 0.5, 1.0])
def test_mma_reduce_initial_value(initial, scaled, steps, first_alpha, device):
    q = torch.full((16, 128), 0.125, device=device, dtype=torch.bfloat16)
    k = torch.ones((3, 128, 32), device=device, dtype=torch.bfloat16)
    alpha = torch.tensor([first_alpha, 0.5, 0.25], dtype=torch.float32)[:, None].expand(3, 16).contiguous()
    out = torch.empty((16, ), device=device, dtype=torch.float32)

    _rescaled_sum[(1, )](q, k, alpha.to(device), out, steps, initial, scaled)

    # Every row of Q @ K sums to 128 * 0.125 * 32 = 512, exactly.
    # In the failing rewrite, the initial value is added after the loop
    # without the product of the per-iteration scale factors.
    expected = torch.full((16, ), initial, dtype=torch.float64)
    for step in range(steps):
        expected = expected * (alpha[step].double() if scaled else 1.0) + 512.0
    torch.testing.assert_close(out.cpu().double(), expected, rtol=0, atol=0)


@triton.jit
def _causal_attention(Q, K, V, Out, n, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    dims = tl.arange(0, D)
    q = tl.load(Q + rows[:, None] * D + dims[None, :], rows[:, None] < n, 0)
    maximum = tl.full((BM, ), float("-inf"), tl.float32)
    denominator = tl.full((BM, ), 1.0, tl.float32)
    numerator = tl.zeros((BM, D), tl.float32)
    for block in range(tl.cdiv(n, BN)):
        cols = block * BN + tl.arange(0, BN)
        k = tl.load(K + cols[None, :] * D + dims[:, None], cols[None, :] < n, 0)
        v = tl.load(V + cols[:, None] * D + dims[None, :], cols[:, None] < n, 0)
        scores = tl.dot(q, k) * (D**-0.5)
        scores = tl.where((cols[None, :] < n) & (cols[None, :] <= rows[:, None]), scores, float("-inf"))
        next_maximum = tl.maximum(maximum, tl.max(scores, 1))
        next_maximum = tl.where(next_maximum > float("-inf"), next_maximum, 0.0)
        probabilities = tl.exp(scores - next_maximum[:, None])
        alpha = tl.exp(maximum - next_maximum)
        denominator = denominator * alpha + tl.sum(probabilities, 1)
        numerator = numerator * alpha[:, None] + tl.dot(probabilities.to(v.dtype), v)
        maximum = next_maximum
    tl.store(Out + rows[:, None] * D + dims[None, :], numerator / denominator[:, None], rows[:, None] < n)


@pytest.mark.parametrize("length", [1, 5, 33])
def test_causal_attention_nonzero_initial_denominator(length, device):
    q = torch.zeros((length, 128), device=device, dtype=torch.bfloat16)
    k = torch.zeros_like(q)
    v = torch.ones_like(q)
    out = torch.empty_like(q)

    _causal_attention[(triton.cdiv(length, 16), )](q, k, v, out, length, 128, 16, 32)

    # Q=K=0 and V=1 imply an output of 1 for every valid query. The first
    # alpha is zero, which must cancel the denominator's initial value.
    # The failing compiler instead produces [1/2, 2/3, 3/4, ...].
    torch.testing.assert_close(out, torch.ones_like(out), rtol=0, atol=0)
