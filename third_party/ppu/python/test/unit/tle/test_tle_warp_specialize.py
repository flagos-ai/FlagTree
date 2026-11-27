"""PPU runtime tests of TLE warp specialization and shared-memory views.

* ``tle.gpu.warp_specialize`` with named-barrier handover (``alloc_barriers`` +
  ``barrier_wait`` / ``barrier_arrive`` without a phase): lowered to the PPU hardware
  named barriers, one per barrier slot, with the partitions running concurrently.
* Lowering paths of the partitions: reductions and ``debug_barrier`` in the default region and
  in a worker, a ``noinline`` helper, captures, a ``warp_specialize`` in a loop, async and AIU
  loads in a partition.
* ``buffered_tensor.subslice`` as a dot operand of a swizzled ring slot, and the shared-memory
  read order around a refilled ring slot (``FLAGTREE_KEEP_SMEM_READ_ORDER``).
* A sparse-attention block (gathered rows, ``q k^T``, online softmax, ``p v``) split over two
  and over three warp groups, with ``tle.pipe`` rings between them.
"""

import os
import re
import subprocess
import sys
import textwrap

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


@triton.jit
def _dot_partition(a, b, o, B: tl.constexpr):
    r = tl.arange(0, B)
    x = tl.load(a + r[:, None] * B + r[None, :])
    y = tl.load(b + r[:, None] * B + r[None, :])
    tl.store(o + r[:, None] * B + r[None, :], tl.dot(x, y))


@triton.jit
def _dot_partitions_kernel(a, b, o, B: tl.constexpr, WORKER_WARPS: tl.constexpr):
    tle.gpu.warp_specialize([(_dot_partition, (a, b, o, B)), (_dot_partition, (a, b, o + B * B, B))], [WORKER_WARPS],
                            [128])


@pytest.mark.parametrize("worker_warps", [1, 2, 4, 8])
@pytest.mark.parametrize("default_warps", [4, 8])
def test_dot_in_partition_of_another_warp_count(default_warps, worker_warps):
    """A tl.dot in the default region and in a worker partition takes the result layout of its own warp group, so
    a worker may run fewer or more warps than the default region (a producer need not match its consumer)."""
    block = 64
    a = torch.randn(block, block, device="cuda", dtype=torch.bfloat16)
    b = torch.randn_like(a)
    o = torch.zeros(2 * block * block, device="cuda")
    _dot_partitions_kernel[(1, )](a, b, o, block, worker_warps, num_warps=default_warps)
    torch.cuda.synchronize()
    ref = a.float() @ b.float()
    torch.testing.assert_close(o[:block * block].view(block, block), ref, rtol=0, atol=0.5)
    torch.testing.assert_close(o[block * block:].view(block, block), ref, rtol=0, atol=0.5)


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
    group as an argument. The frontend specializes it per warp group (name
    suffix _TLEWSNW<n>, ttg.num-warps = n), so its barriers count that group's
    threads; the lowering rejects a helper called from a group of another size
    (third_party/ppu/test/TritonGPU/warp-specialize-errors.mlir)."""
    rows, block = 64, 256
    x = torch.randn(rows, block, device="cuda")
    y = torch.zeros(rows, device="cuda")
    z = torch.zeros(rows, device="cuda")
    _helper_kernel[(1, )](x, y, z, rows, BLOCK=block, num_warps=4)
    torch.cuda.synchronize()
    torch.testing.assert_close(y, x.sum(1))
    torch.testing.assert_close(z, x.sum(1))


@triton.jit
def _idle():
    pass


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


@triton.jit
def _aiu_tile_partition(x, y, rows, cols, row_stride, tile, BT: tl.constexpr, BV: tl.constexpr):
    src = tl.make_block_ptr(x, (rows, cols), (row_stride, 1), (tile * BT, 0), (BT, BV), (1, 0))
    t = tle.load(src, boundary_check=(0, 1), is_async=True)
    offs_t = tile * BT + tl.arange(0, BT)
    tl.store(y + offs_t[:, None] * BV + tl.arange(0, BV)[None, :], t, mask=offs_t[:, None] < rows)


@triton.jit
def _aiu_even_tiles(x, y, rows, cols, row_stride, BT: tl.constexpr, BV: tl.constexpr):
    _aiu_tile_partition(x, y, rows, cols, row_stride, 2 * tl.program_id(0), BT, BV)


@triton.jit
def _aiu_odd_tiles(x, y, rows, cols, row_stride, BT: tl.constexpr, BV: tl.constexpr):
    _aiu_tile_partition(x, y, rows, cols, row_stride, 2 * tl.program_id(0) + 1, BT, BV)


@triton.jit
def _aiu_partitions_kernel(x, y, rows, cols, row_stride, BT: tl.constexpr, BV: tl.constexpr,
                           WORKER_WARPS: tl.constexpr):
    tle.gpu.warp_specialize([(_aiu_even_tiles, (x, y, rows, cols, row_stride, BT, BV)),
                             (_aiu_odd_tiles, (x, y, rows, cols, row_stride, BT, BV))], [WORKER_WARPS], [128])


@pytest.mark.parametrize("worker_warps", [4, 8])
@pytest.mark.parametrize("rows,cols,row_stride", [
    (256, 128, 128),
    (256, 128, 256),
    # a column check with a stride that is not a multiple of the row length is
    # copied by the threads of the warp group, not by the AIU
    (200, 100, 128),
])
def test_aiu_load_in_default_and_worker_partitions(rows, cols, row_stride, worker_warps):
    """The default warps and a worker group each promote their own block-pointer
    loads to AIU copies, including a row stride the descriptor needs a form for
    (and the thread copy of one it cannot express), and each fills its tile
    through the warp's index inside its own warp group."""
    bt, bv = 64, 128
    x = torch.randn(rows, row_stride, device="cuda", dtype=torch.bfloat16)
    y = torch.full((rows, bv), float("nan"), device="cuda", dtype=torch.bfloat16)
    tiles = triton.cdiv(rows, bt)
    ck = _aiu_partitions_kernel[(triton.cdiv(tiles, 2), )](x, y, rows, cols, row_stride, bt, bv, worker_warps,
                                                           num_warps=4, num_stages=1)
    torch.cuda.synchronize()
    expected = torch.zeros(rows, bv, device="cuda", dtype=torch.bfloat16)
    expected[:, :cols] = x[:, :cols]
    torch.testing.assert_close(y, expected, rtol=0, atol=0)
    assert "aiu_load" in ck.asm["ttir"]
    assert "ppu.cp.async.aiu" in ck.asm["llir"]


@triton.constexpr_function
def _ring_layout(vec, max_phase):
    return tle.gpu.swizzled_shared_layout(vec, 1, max_phase, [2, 1, 0], [1] * 3, [1] * 3, [1] * 3, [2, 1, 0])


@triton.jit
def _subslice_ring_kernel(a, b, out, n_blk, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """out = a @ sum_k b[k]: the tiles of b go through a three-slot swizzled ring; the dot reads the two column halves
    of a slot as subslices."""
    offs_m = tl.arange(0, BM)
    offs_k = tl.arange(0, BK)
    offs_n = tl.arange(0, BN)
    ring = tle.gpu.alloc([3, BK, BN], dtype=tl.bfloat16, layout=_ring_layout(8, 8), scope=tle.gpu.smem)
    a_t = tl.load(a + offs_m[:, None] * BK + offs_k[None, :])
    b_ptr = b + offs_k[:, None] * BN + offs_n[None, :]
    acc0 = tl.zeros([BM, BN // 2], dtype=tl.float32)
    acc1 = tl.zeros([BM, BN // 2], dtype=tl.float32)
    for g in range(n_blk):
        slot = ring.slot(g % 3)
        tle.gpu.copy(b_ptr + g * BK * BN, slot, [BK, BN])
        acc0 = tl.dot(a_t, tl.load(tle.gpu.local_ptr(slot.subslice(0, BN // 2, 1))), acc0)
        acc1 = tl.dot(a_t, tl.load(tle.gpu.local_ptr(slot.subslice(BN // 2, BN // 2, 1))), acc1)
    o_ptr = out + offs_m[:, None] * BN + tl.arange(0, BN // 2)[None, :]
    tl.store(o_ptr, acc0)
    tl.store(o_ptr + BN // 2, acc1)


@pytest.mark.parametrize("n_blk", [1, 2, 7])
def test_subslice_dot_operands_of_ring_slot(n_blk):
    bm, bk, bn = 64, 64, 128
    a = torch.randn(bm, bk, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n_blk, bk, bn, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(bm, bn, device="cuda", dtype=torch.float32)
    _subslice_ring_kernel[(1, )](a, b, out, n_blk, BM=bm, BN=bn, BK=bk, num_warps=4, num_stages=1)
    torch.cuda.synchronize()
    ref = sum(a.float() @ b[k].float() for k in range(n_blk))
    torch.testing.assert_close(out, ref, atol=1e-2 * n_blk, rtol=1e-2)


@triton.jit
def _staged_operand_kernel(a, b, out, n_blk, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """out = a @ sum_k b[k]: a is staged through ring slot 2 and read before the loop, which then refills that slot
    with tiles of b. The loop-invariant read of a must not be moved into the loop past the refills."""
    offs_m = tl.arange(0, BM)
    offs_k = tl.arange(0, BK)
    offs_n = tl.arange(0, BN)
    ring = tle.gpu.alloc([3, BK, BN], dtype=tl.bfloat16, layout=_ring_layout(8, 8), scope=tle.gpu.smem)
    a_slot = ring.slot(2).subslice(0, BK, 1)
    tle.gpu.copy(a + offs_m[:, None] * BK + offs_k[None, :], a_slot, [BM, BK])
    a_t = tl.load(tle.gpu.local_ptr(a_slot))
    b_ptr = b + offs_k[:, None] * BN + offs_n[None, :]
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for g in range(n_blk):
        tle.gpu.copy(b_ptr + g * BK * BN, ring.slot(g % 3), [BK, BN])
        acc = tl.dot(a_t, tl.load(tle.gpu.local_ptr(ring.slot(g % 3))), acc)
    tl.store(out + offs_m[:, None] * BN + offs_n[None, :], acc)


@pytest.mark.parametrize("n_blk", [3, 7])
def test_loop_invariant_operand_of_refilled_slot(n_blk):
    bm, bk, bn = 64, 64, 128
    a = torch.randn(bm, bk, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n_blk, bk, bn, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(bm, bn, device="cuda", dtype=torch.float32)
    _staged_operand_kernel[(1, )](a, b, out, n_blk, BM=bm, BN=bn, BK=bk, num_warps=4, num_stages=1)
    torch.cuda.synchronize()
    ref = sum(a.float() @ b[k].float() for k in range(n_blk))
    torch.testing.assert_close(out, ref, atol=1e-2 * n_blk, rtol=1e-2)


_READ_ORDER_SCRIPT = textwrap.dedent("""
    import importlib.util
    import sys
    import torch

    spec = importlib.util.spec_from_file_location("ws", sys.argv[1])
    ws = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ws)
    bm, bk, bn, n_blk = 64, 64, 128, 3
    a = torch.randn(bm, bk, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n_blk, bk, bn, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(bm, bn, device="cuda", dtype=torch.float32)
    ck = ws._staged_operand_kernel[(1, )](a, b, out, n_blk, BM=bm, BN=bn, BK=bk, num_warps=4, num_stages=1)
    ttgir = ck.asm["ttgir"]
    print("loads before the loop:", ttgir[:ttgir.index("scf.for")].count("ttg.local_load"))
""")


@pytest.mark.parametrize("keep", ["1", "0"])
def test_keep_smem_read_order_knob(tmp_path, keep):
    """FLAGTREE_KEEP_SMEM_READ_ORDER reaches the passes: off, the read of a is re-issued inside the loop, after the
    refills, as upstream does. A subprocess, since the in-process kernel cache does not key on the knob."""
    script = tmp_path / "read_order.py"
    script.write_text(_READ_ORDER_SCRIPT)
    env = dict(os.environ, FLAGTREE_KEEP_SMEM_READ_ORDER=keep, TRITON_CACHE_DIR=str(tmp_path / "cache"))
    result = subprocess.run([sys.executable, str(script), __file__], capture_output=True, text=True, timeout=300,
                            env=env)
    assert result.returncode == 0, result.stderr
    assert f"loads before the loop: {keep}" in result.stdout


# ---------------------------------------------------------------- a group that moves data, groups that compute
#
# A sparse-attention block with K == V, as in MLA: for each row block, gather BK rows of a table by index, then
# s = q k^T, an online softmax and acc += p k. The pipes carry the gathered rows and, in the three-group form, the
# scores. Two shapes: the default warps compute while a worker gathers, and the compute itself is split between a
# QK group and a softmax + PV group.

ATTN_BK = 64
ATTN_BM = 64
ATTN_TABLE_ROWS = 2048
ATTN_SM_SCALE = 1.4426950408889634  # log2(e): the softmax below uses exp2


@triton.jit
def _attn_gather_rows(kv, idx, pid, t, K, D: tl.constexpr, BK: tl.constexpr):
    ids = tl.load(idx + pid * K + t * BK + tl.arange(0, BK))
    return kv + ids.to(tl.int64)[:, None] * D + tl.arange(0, D)[None, :]


@triton.jit
def _attn_softmax_update(s, m_i, l_i, acc):
    m_new = tl.maximum(m_i, tl.max(s, 1))
    alpha = tl.math.exp2(m_i - m_new)
    p = tl.math.exp2(s - m_new[:, None])
    return p, m_new, l_i * alpha + tl.sum(p, 1), acc * alpha[:, None]


@triton.jit
def _attn_gather(writer, kv, idx, K, D: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    for t in range(K // BK):
        slot = writer.acquire(t)
        tle.gpu.copy(_attn_gather_rows(kv, idx, pid, t, K, D, BK), slot.k, [BK, D])
        writer.commit(t)


@triton.jit
def _attn_compute(kv_reader, q, out, K, sm, D: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    q_t = tl.load(q + offs_m[:, None] * D + offs_d[None, :])
    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    for t in range(K // BK):
        kb = tl.load(tle.gpu.local_ptr(kv_reader.wait(t).slot.k))
        s = tl.dot(q_t, tl.trans(kb)) * sm
        p, m_i, l_i, acc = _attn_softmax_update(s, m_i, l_i, acc)
        acc = tl.dot(p.to(kb.dtype), kb, acc)
        kv_reader.release(t)
    tl.store(out + offs_m[:, None] * D + offs_d[None, :], (acc / l_i[:, None]).to(out.dtype.element_ty))


@triton.jit
def _attn_two_groups_kernel(q, kv, idx, out, K, sm, D: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr,
                            CAP: tl.constexpr):
    ring = tle.gpu.alloc([CAP, BK, D], dtype=tl.bfloat16, layout=_ring_layout(8, 8), scope=tle.gpu.smem)
    pipe = tle.pipe(capacity=CAP, k=ring)
    tle.gpu.warp_specialize([(_attn_compute, (pipe.reader(), q, out, K, sm, D, BM, BK)),
                             (_attn_gather, (pipe.writer(), kv, idx, K, D, BK))], [4], [128])


@triton.jit
def _attn_softmax_pv(kv_reader, s_reader, out, K, D: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    for t in range(K // BK):
        s = tl.load(tle.gpu.local_ptr(s_reader.wait(t).slot.s))
        s_reader.release(t)
        p, m_i, l_i, acc = _attn_softmax_update(s, m_i, l_i, acc)
        kb = tl.load(tle.gpu.local_ptr(kv_reader.wait(t).slot.k))
        acc = tl.dot(p.to(kb.dtype), kb, acc)
        kv_reader.release(t)
    tl.store(out + offs_m[:, None] * D + offs_d[None, :], (acc / l_i[:, None]).to(out.dtype.element_ty))


@triton.jit
def _attn_qk(q, kv_reader, s_writer, K, sm, D: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BM + tl.arange(0, BM)
    q_t = tl.load(q + offs_m[:, None] * D + tl.arange(0, D)[None, :])
    for t in range(K // BK):
        kb = tl.load(tle.gpu.local_ptr(kv_reader.wait(t).slot.k))
        s = tl.dot(q_t, tl.trans(kb)) * sm
        kv_reader.release(t)
        tl.store(tle.gpu.local_ptr(s_writer.acquire(t).s), s)
        s_writer.commit(t)


@triton.jit
def _attn_three_groups_kernel(q, kv, idx, out, K, sm, D: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr,
                              CAP: tl.constexpr):
    ring = tle.gpu.alloc([CAP, BK, D], dtype=tl.bfloat16, layout=_ring_layout(8, 8), scope=tle.gpu.smem)
    s_ring = tle.gpu.alloc([1, BM, BK], dtype=tl.float32, layout=_ring_layout(4, 8), scope=tle.gpu.smem)
    kv_pipe = tle.pipe(capacity=CAP, readers=("qk", "pv"), k=ring)
    s_pipe = tle.pipe(capacity=1, s=s_ring)
    tle.gpu.warp_specialize([(_attn_softmax_pv, (kv_pipe.reader("pv"), s_pipe.reader(), out, K, D, BM, BK)),
                             (_attn_qk, (q, kv_pipe.reader("qk"), s_pipe.writer(), K, sm, D, BM, BK)),
                             (_attn_gather, (kv_pipe.writer(), kv, idx, K, D, BK))], [4, 4], [128, 128])


def _attn_inputs(d, n_tiles, n_blocks=4):
    q = torch.randn(n_blocks * ATTN_BM, d, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(ATTN_TABLE_ROWS, d, device="cuda", dtype=torch.bfloat16)
    idx = torch.randint(0, ATTN_TABLE_ROWS, (n_blocks, n_tiles * ATTN_BK), device="cuda", dtype=torch.int32)
    # exp2 with log2(e) folded into the scale is the natural-base softmax of q k^T / sqrt(d)
    ref = torch.empty(n_blocks * ATTN_BM, d, device="cuda", dtype=torch.float32)
    for b in range(n_blocks):
        keys = kv[idx[b].long()].float()
        scores = q[b * ATTN_BM:(b + 1) * ATTN_BM].float() @ keys.T / d**0.5
        ref[b * ATTN_BM:(b + 1) * ATTN_BM] = torch.softmax(scores, dim=-1) @ keys
    return q, kv, idx, ref


def _check_attn(out, ref):
    # p is rounded to bf16 before the second dot, so compare against the largest output value
    torch.testing.assert_close(out.float(), ref, rtol=0, atol=2e-2 * ref.abs().max().item())


@pytest.mark.parametrize("n_tiles", [1, 5])
@pytest.mark.parametrize("capacity", [2, 3])
@pytest.mark.parametrize("d", [128, 256])
def test_gather_group_feeds_compute_group(d, capacity, n_tiles):
    """The default warps run both dots and the online softmax on rows that a worker gathers into a pipe ring: the
    row-index loads, the gather copies and the dots sit in different warp groups."""
    q, kv, idx, ref = _attn_inputs(d, n_tiles)
    out = torch.empty_like(q)
    ck = _attn_two_groups_kernel[(idx.shape[0], )](q, kv, idx, out, idx.shape[1], ATTN_SM_SCALE / d**0.5, d, ATTN_BM,
                                                   ATTN_BK, capacity, num_warps=4, num_stages=1)
    torch.cuda.synchronize()
    _check_attn(out, ref)
    assert ck.metadata.num_warps == 8
    assert "ttng.wait_barrier_named" in ck.asm["ttgir"]


@pytest.mark.parametrize("n_tiles", [1, 5])
@pytest.mark.parametrize("capacity", [2, 3])
@pytest.mark.parametrize("d", [128, 256])
def test_qk_group_and_softmax_pv_group_share_gathered_rows(d, capacity, n_tiles):
    """Three groups: one gathers, one computes the scores, one runs the softmax and P V. The gathered rows have two
    readers (the QK group and the PV group), and the scores pass from the QK group to the PV group through a second
    pipe."""
    q, kv, idx, ref = _attn_inputs(d, n_tiles)
    out = torch.empty_like(q)
    ck = _attn_three_groups_kernel[(idx.shape[0], )](q, kv, idx, out, idx.shape[1], ATTN_SM_SCALE / d**0.5, d, ATTN_BM,
                                                     ATTN_BK, capacity, num_warps=4, num_stages=1)
    torch.cuda.synchronize()
    _check_attn(out, ref)
    assert ck.metadata.num_warps == 12
    assert "ttng.wait_barrier_named" in ck.asm["ttgir"]
