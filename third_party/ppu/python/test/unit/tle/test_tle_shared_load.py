"""PPU runtime tests of shared-memory reads into a layout that every lane of a warp shares: the addresses are
warp-uniform, and the loaded values stay in scalar registers."""

import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_ppu

tle = pytest.importorskip("triton.experimental.tle.language", reason="tle language unavailable")

pytestmark = pytest.mark.skipif(not is_ppu(), reason="requires the PPU backend")


@triton.jit
def _row_gather_kernel(x, idx, out, R: tl.constexpr, C: tl.constexpr):
    """out[r] = x[idx[r]]: the row indices are staged in shared memory and read in the layout of the gather rows, which
    every lane of a warp shares."""
    rows = tl.arange(0, R)
    cols = tl.arange(0, C)
    buf = tle.gpu.alloc([R], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    tl.store(tle.gpu.local_ptr(buf, (rows, )), tl.load(idx + rows))
    ids = tl.load(tle.gpu.local_ptr(buf, (rows, )))
    tl.store(out + rows[:, None] * C + cols[None, :], tl.load(x + ids[:, None] * C + cols[None, :]))


def test_lane_broadcast_shared_load_is_warp_uniform():
    rows, cols, n = 8, 256, 1000
    x = torch.randn(n, cols, device="cuda")
    idx = torch.randint(0, n, (rows, ), device="cuda", dtype=torch.int32)
    out = torch.empty(rows, cols, device="cuda")
    ck = _row_gather_kernel[(1, )](x, idx, out, R=rows, C=cols, num_warps=8)
    torch.testing.assert_close(out, x[idx.long()], atol=0, rtol=0)
    # the index read is the only local load: its addresses come from a warp ID marked warp-uniform
    assert ck.asm["llir"].count("ppu.to.uniform.b32") == 1


@triton.jit
def _overwritten_buffer_kernel(x, idx, idx2, out, R: tl.constexpr, C: tl.constexpr):
    """The row indices are read from shared memory, the buffer is then overwritten, and the first indices are used
    after that: the read must not be re-issued after the write."""
    rows = tl.arange(0, R)
    cols = tl.arange(0, C)
    buf = tle.gpu.alloc([R], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    tl.store(tle.gpu.local_ptr(buf, (rows, )), tl.load(idx + rows))
    ids = tl.load(tle.gpu.local_ptr(buf, (rows, )))
    tl.store(tle.gpu.local_ptr(buf, (rows, )), tl.load(idx2 + rows))
    ids2 = tl.load(tle.gpu.local_ptr(buf, (rows, )))
    gathered = tl.load(x + ids[:, None] * C + cols[None, :]) + tl.load(x + ids2[:, None] * C + cols[None, :]) * 0.0
    tl.store(out + rows[:, None] * C + cols[None, :], gathered)


def test_shared_load_is_not_moved_past_a_write():
    rows, cols, n = 8, 256, 1000
    x = torch.randn(n, cols, device="cuda")
    idx = torch.randint(0, n, (rows, ), device="cuda", dtype=torch.int32)
    idx2 = (idx + 1) % n
    out = torch.zeros(rows, cols, device="cuda")
    _overwritten_buffer_kernel[(1, )](x, idx, idx2, out, R=rows, C=cols, num_warps=4)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, x[idx.long()], rtol=0, atol=0)
