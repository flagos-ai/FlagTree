"""Indexed local_ptr accesses must match PH1 bulk shared-memory addressing."""

import pytest
import torch
import torch_musa  # noqa: F401
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

from test_tle_utils import require_mthreads_libtriton

require_mthreads_libtriton()


@triton.jit
def _work(x, out, COLS: tl.constexpr, WRITE: tl.constexpr):
    layout: tl.constexpr = tle.gpu.swizzled_shared_layout(16, 1, 8, [1, 0], [1, 1], [1, 1], [1, 1], [1, 0])
    shared = tle.gpu.alloc((64, COLS), tl.bfloat16, layout=layout)
    sync = tle.gpu.alloc_barrier(arrive_count=8)
    rows = tl.broadcast_to(tl.arange(0, 64)[:, None], (64, COLS))
    cols = tl.broadcast_to(tl.arange(0, COLS)[None, :], (64, COLS))
    value = tl.load(x + rows * COLS + cols)
    if WRITE:
        # A bijection prevents write races while forcing indexed addressing.
        tl.store(tle.gpu.local_ptr(shared, (rows, cols ^ 16)), value)
    else:
        tl.store(tle.gpu.local_ptr(shared), value)
    tle.gpu.barrier_arrive(sync, phaseIdx=0)
    tle.gpu.barrier_wait(sync, phaseIdx=0)
    if WRITE:
        result = tl.load(tle.gpu.local_ptr(shared))
    else:
        # Non-bijective indices must not be rewritten to a full local_load.
        result = tl.load(tle.gpu.local_ptr(shared, (rows, cols // 2)))
    tl.store(out + rows * COLS + cols, result)


@pytest.mark.skipif(not torch.musa.is_available(), reason='MUSA device required')
@pytest.mark.parametrize('cols', [128, 256])
@pytest.mark.parametrize('write', [False, True])
def test_local_ptr_ph1_layout_runtime(cols, write):
    torch.manual_seed(731)
    x_cpu = torch.randn(64, cols).to(torch.bfloat16)
    x = x_cpu.to('musa')
    out = torch.empty_like(x)
    index = torch.arange(cols) ^ 16 if write else torch.arange(cols) // 2
    expected = x_cpu[:, index]
    for _ in range(3):
        out.fill_(float('nan'))
        _work[(1,)](x, out, cols, write, num_warps=8)
        torch.musa.synchronize()
        torch.testing.assert_close(out.cpu(), expected, atol=0, rtol=0)

