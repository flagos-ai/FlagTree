# flagtree tle
"""Generic tle.cumsum (tle-lite) integration tests on tsingmicro (txda).

tle.cumsum is a generic tle-lite primitive; on tsingmicro it lowers through
the shared tle dialect to the hardware scan. Current constraints:
float input only (integer scan unsupported by hardware), forward scan only
(reverse=True unsupported), rank-1 tensors only (shared op constraint).
"""

import pytest
import torch

import triton
import triton.experimental.tle.language as tle
import triton.language as tl


def _is_txda():
    try:
        import torch_txda  # noqa: F401
    except ImportError:
        return False
    target = triton.runtime.driver.active.get_current_target()
    return getattr(target, "backend", None) == "txda"


pytestmark = pytest.mark.skipif(not _is_txda(), reason="requires TsingMicro (txda) backend")


@triton.jit
def _cumsum_1d(x_ptr, exclusive_ptr, total_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    exclusive, total = tle.cumsum(x, axis=0)
    tl.store(exclusive_ptr + offs, exclusive, mask=mask)
    tl.store(total_ptr, total)


@triton.jit
def _cumsum_1d_reverse(x_ptr, exclusive_ptr, total_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    exclusive, total = tle.cumsum(x, axis=0, reverse=True)
    tl.store(exclusive_ptr + offs, exclusive, mask=offs < n)
    tl.store(total_ptr, total)


@triton.jit
def _cumsum_2d(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr):
    m = tl.arange(0, M)
    n = tl.arange(0, N)
    x = tl.load(x_ptr + m[:, None] * N + n[None, :])
    exclusive, total = tle.cumsum(x, axis=1)
    tl.store(out_ptr + m[:, None] * N + n[None, :], exclusive)


def _exclusive_expected(x, dtype):
    cs = torch.cumsum(x, dim=0, dtype=dtype)
    zero = torch.zeros(1, device=x.device, dtype=dtype)
    return torch.cat([zero, cs[:-1]])


def test_cumsum_1d_masked():
    torch.manual_seed(42)
    n, block = 100, 128
    x = torch.randn(n, device="txda", dtype=torch.float32)

    exclusive = torch.zeros(block, device="txda", dtype=torch.float32)
    total = torch.zeros(1, device="txda", dtype=torch.float32)
    _cumsum_1d[(1, )](x, exclusive, total, n, BLOCK=block)

    expected = _exclusive_expected(x, torch.float32)
    torch.testing.assert_close(exclusive[:n], expected)
    torch.testing.assert_close(total[0], x.sum(dim=0, dtype=torch.float32))


def test_cumsum_1d_full_block():
    torch.manual_seed(43)
    n = 64
    x = torch.randn(n, device="txda", dtype=torch.float32)

    exclusive = torch.zeros(n, device="txda", dtype=torch.float32)
    total = torch.zeros(1, device="txda", dtype=torch.float32)
    _cumsum_1d[(1, )](x, exclusive, total, n, BLOCK=n)

    expected = _exclusive_expected(x, torch.float32)
    torch.testing.assert_close(exclusive, expected, atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(total[0], x.sum(dim=0), atol=2e-6, rtol=1e-5)


def test_cumsum_2d_unsupported():
    """The shared tle.exclusive_cumsum op only accepts rank-1 tensors."""
    torch.manual_seed(46)
    m, n = 8, 32
    x = torch.randn(m, n, device="txda", dtype=torch.float32)
    out = torch.zeros(m, n, device="txda", dtype=torch.float32)
    with pytest.raises(Exception):
        _cumsum_2d[(1, )](x, out, M=m, N=n)


def test_cumsum_reverse_unsupported():
    torch.manual_seed(45)
    x = torch.randn(64, device="txda", dtype=torch.float32)
    exclusive = torch.zeros(64, device="txda", dtype=torch.float32)
    total = torch.zeros(1, device="txda", dtype=torch.float32)
    with pytest.raises(Exception):
        _cumsum_1d_reverse[(1, )](x, exclusive, total, 64, BLOCK=64)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
