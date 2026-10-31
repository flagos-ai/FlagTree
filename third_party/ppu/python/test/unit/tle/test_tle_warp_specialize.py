"""PPU runtime tests of TLE warp specialization.

* ``tle.gpu.warp_specialize`` with named-barrier handover (``alloc_barriers`` +
  ``barrier_wait`` / ``barrier_arrive`` without a phase): lowered to the PPU hardware
  named barriers, one per barrier slot, with the partitions running concurrently.
"""

import re

import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_ppu

tle = pytest.importorskip("triton.experimental.tle.language", reason="tle language unavailable")

pytestmark = pytest.mark.skipif(not is_ppu(), reason="requires the PPU backend")


@triton.jit
def _wait2(bars, i):
    if i == 0:
        tle.gpu.barrier_wait(bars[0])
    else:
        tle.gpu.barrier_wait(bars[1])


@triton.jit
def _arrive2(bars, i):
    if i == 0:
        tle.gpu.barrier_arrive(bars[0])
    else:
        tle.gpu.barrier_arrive(bars[1])


@triton.jit
def _ring_consumer(ring, full, empty, y, n_tiles, BLOCK: tl.constexpr):
    for t in range(n_tiles):
        s = t % 2
        _wait2(full, s)
        v = tl.load(tle.gpu.local_ptr(ring.slot(s)))
        _arrive2(empty, s)
        tl.store(y + t * BLOCK + tl.arange(0, BLOCK), v * 3.0)


@triton.jit
def _ring_producer(ring, full, empty, x, n_tiles, BLOCK: tl.constexpr):
    for t in range(n_tiles):
        s = t % 2
        if t >= 2:
            _wait2(empty, s)
        v = tl.load(x + t * BLOCK + tl.arange(0, BLOCK))
        tl.store(tle.gpu.local_ptr(ring.slot(s)), v)
        _arrive2(full, s)


@triton.jit
def _ring_kernel(x, y, n_tiles, BLOCK: tl.constexpr, THREADS: tl.constexpr):
    ring = tle.gpu.alloc([2, BLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.smem)
    full = tle.gpu.alloc_barriers(2, arrive_count=THREADS)
    empty = tle.gpu.alloc_barriers(2, arrive_count=THREADS)
    tle.gpu.warp_specialize(
        [(_ring_consumer, (ring, full, empty, y, n_tiles, BLOCK)),
         (_ring_producer, (ring, full, empty, x, n_tiles, BLOCK))],
        [4],
        [128],
    )


@pytest.mark.parametrize("n_tiles", [1, 2, 3, 64])
def test_warp_specialize_named_barrier_ring(n_tiles):
    block = 128
    x = torch.randn(n_tiles * block, device="cuda")
    y = torch.zeros_like(x)
    ck = _ring_kernel[(4, )](x, y, n_tiles, BLOCK=block, THREADS=256, num_warps=4)
    torch.cuda.synchronize()
    torch.testing.assert_close(y, 3 * x)
    assert ck.metadata.num_warps == 8
    assert "ttg.warp_specialize" in ck.asm["ttgir"]
    # one hardware named barrier per barrier slot, waited for and arrived at by
    # the THREADS of both groups, with the shared-memory fence
    llir = ck.asm["llir"]
    assert re.search(r"__ppukernel_barrier_sync_cnt_void\(i32 [^,]+, i32 256, i32 1\)", llir)
    assert re.search(r"__ppukernel_barrier_arrive_void\(i32 [^,]+, i32 256, i32 1\)", llir)


# ---------------------------------------------------------------- lowering paths of warp_specialize


@triton.jit
def _sum_rows_partition(x, y, rows, BLOCK: tl.constexpr):
    # A reduction over several warps converts layouts through shared memory,
    # and debug_barrier is a CTA barrier: both become barriers of this warp
    # group, with its thread count.
    for r in range(rows):
        v = tl.load(x + r * BLOCK + tl.arange(0, BLOCK))
        tl.store(y + r, tl.sum(v))
        tl.debug_barrier()


@triton.jit
def _sum_rows_kernel(x, y, z, rows, BLOCK: tl.constexpr, WORKER_WARPS: tl.constexpr):
    tle.gpu.warp_specialize([(_sum_rows_partition, (x, y, rows, BLOCK)), (_sum_rows_partition, (x, z, rows, BLOCK))],
                            [WORKER_WARPS], [128])


@pytest.mark.parametrize("worker_warps", [1, 4])
def test_reductions_in_default_and_worker_partitions(worker_warps):
    """The default warps reduce on barrier 0 with their own thread count while
    the worker runs; a single-warp worker reduces on a 32-thread barrier."""
    rows, block = 64, 256
    x = torch.randn(rows, block, device="cuda")
    y = torch.zeros(rows, device="cuda")
    z = torch.zeros(rows, device="cuda")
    ck = _sum_rows_kernel[(1, )](x, y, z, rows, BLOCK=block, WORKER_WARPS=worker_warps, num_warps=4)
    torch.cuda.synchronize()
    torch.testing.assert_close(y, x.sum(1))
    torch.testing.assert_close(z, x.sum(1))
    llir = ck.asm["llir"]
    assert re.search(r"__ppukernel_barrier_sync_cnt_void\(i32 0, i32 128, i32 31\)", llir)
    assert re.search(rf"__ppukernel_barrier_sync_cnt_void\(i32 2, i32 {32 * worker_warps}, i32 31\)", llir)


@triton.jit(noinline=True)
def _sum_row_helper(x, y, r, BLOCK: tl.constexpr):
    v = tl.load(x + r * BLOCK + tl.arange(0, BLOCK))
    tl.store(y + r, tl.sum(v))


@triton.jit
def _helper_partition(x, y, rows, BLOCK: tl.constexpr):
    for r in range(rows):
        _sum_row_helper(x, y, r, BLOCK)


@triton.jit
def _helper_kernel(x, y, z, rows, BLOCK: tl.constexpr):
    tle.gpu.warp_specialize([(_helper_partition, (x, y, rows, BLOCK)), (_helper_partition, (x, z, rows, BLOCK))], [4],
                            [128])


def test_reduction_in_noinline_helper():
    """A helper that is not inlined gets the barrier of its caller's warp
    group as an argument."""
    rows, block = 64, 256
    x = torch.randn(rows, block, device="cuda")
    y = torch.zeros(rows, device="cuda")
    z = torch.zeros(rows, device="cuda")
    _helper_kernel[(1, )](x, y, z, rows, BLOCK=block, num_warps=4)
    torch.cuda.synchronize()
    torch.testing.assert_close(y, x.sum(1))
    torch.testing.assert_close(z, x.sum(1))


@triton.jit
def _capture_partition(v, y, BLOCK: tl.constexpr):
    tl.store(y + tl.arange(0, BLOCK), tl.zeros([BLOCK], dtype=tl.float32) + v)


@triton.jit
def _capture_kernel(x, y, BLOCK: tl.constexpr):
    v = tl.load(x) + tl.load(x + 1)
    tle.gpu.warp_specialize([(_idle, ()), (_capture_partition, (v, y, BLOCK))], [4], [128])


def test_capture_passed_through_shared_memory():
    """A value loaded before the warp_specialize reaches the worker through
    shared memory (a special register or constant would be recomputed)."""
    block = 256
    x = torch.randn(2, device="cuda")
    y = torch.zeros(block, device="cuda")
    ck = _capture_kernel[(1, )](x, y, BLOCK=block, num_warps=4)
    torch.cuda.synchronize()
    torch.testing.assert_close(y, torch.full((block, ), (x[0] + x[1]).item(), device="cuda"))
    assert "partition0(%" in ck.asm["ttgir"]


@triton.jit
def _looped_kernel(x, y, z, rounds, BLOCK: tl.constexpr):
    for i in range(rounds):
        tle.gpu.warp_specialize([(_sum_rows_partition, (x + i * BLOCK, y + i, 1, BLOCK)),
                                 (_sum_rows_partition, (x + i * BLOCK, z + i, 1, BLOCK))], [4], [128])


def test_warp_specialize_in_a_loop():
    """Each execution publishes a new partition state to the workers and
    collects them again; without a pipe, nothing carries over."""
    rounds, block = 16, 256
    x = torch.randn(rounds, block, device="cuda")
    y = torch.zeros(rounds, device="cuda")
    z = torch.zeros(rounds, device="cuda")
    _looped_kernel[(1, )](x, y, z, rounds, BLOCK=block, num_warps=4)
    torch.cuda.synchronize()
    torch.testing.assert_close(y, x.sum(1))
    torch.testing.assert_close(z, x.sum(1))


@triton.jit
def _idle():
    pass


@triton.jit
def _async_load_partition(x, y, n_tiles, BLOCK: tl.constexpr):
    for t in range(n_tiles):
        src = tl.make_block_ptr(x, (n_tiles * 64, BLOCK), (BLOCK, 1), (t * 64, 0), (64, BLOCK), (1, 0))
        dst = tl.make_block_ptr(y, (n_tiles * 64, BLOCK), (BLOCK, 1), (t * 64, 0), (64, BLOCK), (1, 0))
        tile = tle.load(src, boundary_check=(0, 1), is_async=True)
        tl.store(dst, (tile * 2).to(y.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def _async_load_kernel(x, y, n_tiles, BLOCK: tl.constexpr):
    tle.gpu.warp_specialize([(_idle, ()), (_async_load_partition, (x, y, n_tiles, BLOCK))], [4], [128])


def test_async_load_in_worker_partition():
    """The async copy addresses shared memory by the warp's index inside its warp group."""
    n_tiles, block = 4, 64
    x = torch.randn(n_tiles * 64, block, device="cuda", dtype=torch.bfloat16)
    y = torch.zeros_like(x)
    _async_load_kernel[(1, )](x, y, n_tiles, BLOCK=block, num_warps=4)
    torch.cuda.synchronize()
    torch.testing.assert_close(y, 2 * x)
