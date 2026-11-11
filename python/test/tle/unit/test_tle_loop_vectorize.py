"""Compare vectorized and scalar loops for supported patterns and regressions."""

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

pytestmark = pytest.mark.skip_if_tle_unsupported("gpu.range")


def test_loop_vectorize_store(device):

    @triton.jit(do_not_specialize=["N"])
    def kernel(X, Y, N, VF: tl.constexpr):
        for i in tle.range(N, loop_vectorize_factor=VF):
            tl.store(Y + i, tl.load(X + i))

    n = 9  # Two vector groups and one scalar tail iteration for VF=4.
    x = torch.arange(1, n + 1, dtype=torch.float32, device=device)
    y_scalar = torch.zeros(n, device=device)
    y_vector = torch.zeros_like(y_scalar)

    kernel[(1, )](x, y_scalar, n, VF=None, num_warps=4)
    kernel[(1, )](x, y_vector, n, VF=4, num_warps=4)
    torch.testing.assert_close(y_vector, y_scalar, rtol=0, atol=0)


def test_loop_vectorize_nested_loop_fallback(device):

    @triton.jit(do_not_specialize=["N"])
    def kernel(X, Y, N, VF: tl.constexpr):
        for i in tle.range(N, loop_vectorize_factor=VF):
            for j in tl.range(2):
                tl.store(Y + i, tl.load(X + i) + j)

    n = 9
    x = torch.arange(1, n + 1, dtype=torch.float32, device=device)
    y_scalar = torch.zeros(n, device=device)
    y_vector = torch.zeros_like(y_scalar)

    kernel[(1, )](x, y_scalar, n, VF=None, num_warps=4)
    kernel[(1, )](x, y_vector, n, VF=4, num_warps=4)
    torch.testing.assert_close(y_vector, y_scalar, rtol=0, atol=0)


@pytest.mark.parametrize("mode", [-1, 1])
def test_loop_vectorize_invariant_if(mode, device):

    @triton.jit(do_not_specialize=["N", "MODE"])
    def kernel(X, Y, N, MODE, VF: tl.constexpr):
        for i in tle.range(N, loop_vectorize_factor=VF):
            if MODE > 0:
                tl.store(Y + i, tl.load(X + i))

    n = 9
    x = torch.arange(1, n + 1, dtype=torch.float32, device=device)
    y_scalar = torch.zeros(n, device=device)
    y_vector = torch.zeros_like(y_scalar)

    kernel[(1, )](x, y_scalar, n, mode, VF=None, num_warps=4)
    kernel[(1, )](x, y_vector, n, mode, VF=4, num_warps=4)
    torch.testing.assert_close(y_vector, y_scalar, rtol=0, atol=0)


def test_loop_vectorize_varying_if_fallback(device):

    @triton.jit(do_not_specialize=["N"])
    def kernel(X, Y, N, VF: tl.constexpr):
        for i in tle.range(N, loop_vectorize_factor=VF):
            if i % 2 == 0:
                tl.store(Y + i, tl.load(X + i))

    n = 9
    x = torch.arange(1, n + 1, dtype=torch.float32, device=device)
    y_scalar = torch.zeros(n, device=device)
    y_vector = torch.zeros_like(y_scalar)

    kernel[(1, )](x, y_scalar, n, VF=None, num_warps=4)
    kernel[(1, )](x, y_vector, n, VF=4, num_warps=4)
    torch.testing.assert_close(y_vector, y_scalar, rtol=0, atol=0)


def test_loop_vectorize_last_value(device):

    @triton.jit(do_not_specialize=["N"])
    def kernel(X, Y, N, VF: tl.constexpr):
        value = -7.
        for i in tle.range(N, loop_vectorize_factor=VF):
            value = tl.load(X + i)
        tl.store(Y, value)

    n = 9
    x = torch.arange(1, n + 1, dtype=torch.float32, device=device)
    y_scalar = torch.zeros(1, device=device)
    y_vector = torch.zeros_like(y_scalar)

    kernel[(1, )](x, y_scalar, n, VF=None, num_warps=4)
    kernel[(1, )](x, y_vector, n, VF=4, num_warps=4)
    torch.testing.assert_close(y_vector, y_scalar, rtol=0, atol=0)


def test_loop_vectorize_read_before_overwrite_fallback(device):

    @triton.jit(do_not_specialize=["N"])
    def kernel(X, Y, N, VF: tl.constexpr):
        value = -7.
        for i in tle.range(N, loop_vectorize_factor=VF):
            tl.store(Y + i, value)
            value = tl.load(X + i)

    n = 9
    x = torch.arange(1, n + 1, dtype=torch.float32, device=device)
    y_scalar = torch.zeros(n, device=device)
    y_vector = torch.zeros_like(y_scalar)

    kernel[(1, )](x, y_scalar, n, VF=None, num_warps=4)
    kernel[(1, )](x, y_vector, n, VF=4, num_warps=4)
    torch.testing.assert_close(y_vector, y_scalar, rtol=0, atol=0)


def test_loop_vectorize_sum(device):

    @triton.jit(do_not_specialize=["N"])
    def kernel(X, Y, N, VF: tl.constexpr):
        total = 5.5
        for i in tle.range(N, loop_vectorize_factor=VF):
            total += tl.load(X + i)
        tl.store(Y, total)

    n = 9
    x = torch.arange(1, n + 1, dtype=torch.float32, device=device) * 0.1
    y_scalar = torch.zeros(1, device=device)
    y_vector = torch.zeros_like(y_scalar)

    kernel[(1, )](x, y_scalar, n, VF=None, num_warps=4)
    kernel[(1, )](x, y_vector, n, VF=4, num_warps=4)
    # Vectorization changes the order of FP32 additions.
    torch.testing.assert_close(y_vector, y_scalar, rtol=1e-5, atol=1e-5)


def test_loop_vectorize_intermediate_sum_fallback(device):

    @triton.jit(do_not_specialize=["N"])
    def kernel(X, Y, N, VF: tl.constexpr):
        total = 5.5
        for i in tle.range(N, loop_vectorize_factor=VF):
            total += tl.load(X + i)
            tl.store(Y + i, total)

    n = 9
    x = torch.arange(1, n + 1, dtype=torch.float32, device=device)
    y_scalar = torch.zeros(n, device=device)
    y_vector = torch.zeros_like(y_scalar)

    kernel[(1, )](x, y_scalar, n, VF=None, num_warps=4)
    kernel[(1, )](x, y_vector, n, VF=4, num_warps=4)
    torch.testing.assert_close(y_vector, y_scalar, rtol=0, atol=0)


@pytest.mark.parametrize("reduction", ["sum", "max"])
def test_loop_vectorize_flat_reduction(reduction, device):
    # Regression: axis=None inserts a reorderable reshape that must not mix rows.
    @triton.jit(do_not_specialize=["N"])
    def kernel(X, Y, N, REDUCTION: tl.constexpr, VF: tl.constexpr):
        offsets = tl.arange(0, 32)
        for i in tle.range(N, loop_vectorize_factor=VF):
            x = tl.load(X + i * 32 + offsets)
            if REDUCTION == "sum":
                value = tl.sum(x)
            else:
                value = tl.max(x)
            tl.store(Y + i, value)

    n = 9
    x = torch.arange(n * 32, dtype=torch.float32, device=device)
    y_scalar = torch.zeros(n, device=device)
    y_vector = torch.zeros_like(y_scalar)

    kernel[(1, )](x, y_scalar, n, REDUCTION=reduction, VF=None, num_warps=4)
    kernel[(1, )](x, y_vector, n, REDUCTION=reduction, VF=4, num_warps=4)
    torch.testing.assert_close(y_vector, y_scalar, rtol=0, atol=0)


@pytest.mark.parametrize("reduction", ["sum", "max"])
def test_loop_vectorize_reshaped_reduction_fallback(reduction, device):
    # Regression: flattening [8,4] before reduction must retain scalar execution.
    @triton.jit(do_not_specialize=["N"])
    def kernel(X, Y, N, REDUCTION: tl.constexpr, VF: tl.constexpr):
        offsets = tl.arange(0, 8)[:, None] * 4 + tl.arange(0, 4)[None, :]
        for i in tle.range(N, loop_vectorize_factor=VF):
            x = tl.load(X + i * 32 + offsets)
            if REDUCTION == "sum":
                value = tl.sum(x)
            else:
                value = tl.max(x)
            tl.store(Y + i, value)

    n = 9
    x = torch.arange(n * 32, dtype=torch.float32, device=device)
    y_scalar = torch.zeros(n, device=device)
    y_vector = torch.zeros_like(y_scalar)

    kernel[(1, )](x, y_scalar, n, REDUCTION=reduction, VF=None, num_warps=4)
    kernel[(1, )](x, y_vector, n, REDUCTION=reduction, VF=4, num_warps=4)
    torch.testing.assert_close(y_vector, y_scalar, rtol=0, atol=0)


@pytest.mark.parametrize("mode", [-1, 1])
def test_loop_vectorize_scalar_select(mode, device):
    # Regression: a runtime scalar condition must broadcast to the block values.
    @triton.jit(do_not_specialize=["N", "MODE"])
    def kernel(X, Y, N, MODE, VF: tl.constexpr):
        offsets = tl.arange(0, 32)
        for i in tle.range(N, loop_vectorize_factor=VF):
            x = tl.load(X + i * 32 + offsets)
            a = x * 2
            b = x + 1
            if MODE > 0:
                value = a
            else:
                value = b
            tl.store(Y + i * 32 + offsets, value)

    n = 9
    x = torch.arange(n * 32, dtype=torch.float32, device=device)
    y_scalar = torch.zeros_like(x)
    y_vector = torch.zeros_like(x)

    kernel[(1, )](x, y_scalar, n, mode, VF=None, num_warps=4)
    kernel[(1, )](x, y_vector, n, mode, VF=4, num_warps=4)
    torch.testing.assert_close(y_vector, y_scalar, rtol=0, atol=0)
