"""An outstanding WGMMA result can cross a loop edge without an ordinary read."""

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton._internal_testing import is_hopper

pytestmark = pytest.mark.skipif(not is_hopper(), reason="requires NVIDIA Hopper WGMMA")


@triton.jit
def _pending_product_loop(lhs, rhs, result):
    row = tl.arange(0, 64)[:, None]
    col = tl.arange(0, 16)[:, None]
    kk = tl.arange(0, 256)[None, :]
    a = tle.gpu.alloc([2, 64, 256], tl.bfloat16)
    b = tle.gpu.alloc([16, 256], tl.bfloat16)
    for slot in tl.static_range(0, 2):
        tle.gpu.copy(lhs + slot * 64 * 256 + row * 256 + kk, a.slot(slot), [64, 256], is_async=True)
    tle.gpu.copy(rhs + col * 256 + kk, b, [16, 256], is_async=True)
    tle.gpu.async_commit_group()
    tle.gpu.async_wait_group(0)
    tl.debug_barrier()
    previous = tle.gpu.wgmma(a.slot(0), b, trans_b=True)
    total = tl.zeros((64, 16), tl.float32)
    for step in tl.range(1, 33):
        pending = tle.gpu.wgmma(a.slot(step % 2), b, trans_b=True)
        ready = tle.gpu.wgmma_wait(1, previous)
        total += ready
        previous = pending
    total += tle.gpu.wgmma_wait(0, previous)
    column = tl.arange(0, 16)[None, :]
    tl.store(result + row * 16 + column, total)


@pytest.mark.parametrize("warps", [4, 8])
def test_pending_product_loop_preserves_async_register_values(warps):
    generator = torch.Generator().manual_seed(1540)
    lhs = torch.randint(-2, 3, (2, 64, 256), generator=generator).float() / 2
    rhs = torch.randint(-2, 3, (16, 256), generator=generator).float() / 2
    expected = 17 * (lhs[0] @ rhs.T) + 16 * (lhs[1] @ rhs.T)
    lhs, rhs = lhs.cuda().to(torch.bfloat16), rhs.cuda().to(torch.bfloat16)
    result = torch.empty((64, 16), dtype=torch.float32, device="cuda")
    for _ in range(32):
        compiled = _pending_product_loop[(1, )](lhs, rhs, result, num_warps=warps)
        torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
    assert "wgmma.wait_group.sync.aligned 1" in compiled.asm["ptx"]
