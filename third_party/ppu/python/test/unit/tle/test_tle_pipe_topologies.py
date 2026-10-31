"""PPU runtime tests of TLE pipe topologies with three warp groups.

The groups are the default warps (D) and two worker partitions (P0, P1); the
pipes between them cover what FlagGems' warp-specialized attention kernels use:

* a one_shot pipe written by D and read by D and P0;
* an SPMC pipe written by P1 and read by D and P0 (register-store payload);
* two cp.async pipes written by P1 for D and for P0, one of them SPMC;
* an exchange of partial results between D and P0, one pipe per direction;
* a capacity-2 ring shared by two pipes, each direction using its own stage.

With 16 pipe barriers needed and 12 hardware barriers left, the later pipes
use the software barriers in shared memory.
"""

import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_ppu

tle = pytest.importorskip("triton.experimental.tle.language", reason="tle language unavailable")

pytestmark = pytest.mark.skipif(not is_ppu(), reason="requires the PPU backend")


@triton.jit
def _producer(k0_w, k1_w, valid_w, x, n, BLOCK: tl.constexpr):
    for t in range(n):
        base = x + t * 2 * BLOCK + tl.arange(0, BLOCK)
        s0 = k0_w.acquire(t)
        tle.gpu.copy(base, s0.k, [BLOCK])
        k0_w.commit(t)
        s1 = k1_w.acquire(t)
        tle.gpu.copy(base + BLOCK, s1.k, [BLOCK])
        k1_w.commit(t)
        sv = valid_w.acquire(t)
        tl.store(tle.gpu.local_ptr(sv.v), tl.full([BLOCK], 1.0, tl.float32) * (t + 1))
        valid_w.commit(t)


@triton.jit
def _consumer0(q_w, q_r, k0_r, k1_r, valid_r, m0_w, m1_r, l0_w, l1_r, x, y, n, BLOCK: tl.constexpr):
    # Q: written once by this group, read by this group and by consumer 1.
    sq = q_w.acquire(0)
    tle.gpu.copy(x + tl.arange(0, BLOCK), sq.q, [BLOCK])
    q_w.commit(0)
    q = tl.load(tle.gpu.local_ptr(q_r.wait(0).slot.q))
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for t in range(n):
        k0 = tl.load(tle.gpu.local_ptr(k0_r.wait(t).slot.k))
        k0_r.release(t)
        k1 = tl.load(tle.gpu.local_ptr(k1_r.wait(t).slot.k))
        k1_r.release(t)
        v = tl.load(tle.gpu.local_ptr(valid_r.wait(t).slot.v))
        valid_r.release(t)
        part = q * k0 + k1 * v
        # Exchange the partial with consumer 1: m0 goes there, m1 comes back.
        tl.store(tle.gpu.local_ptr(m0_w.acquire(t).m), part)
        m0_w.commit(t)
        other = tl.load(tle.gpu.local_ptr(m1_r.wait(t).slot.m))
        m1_r.release(t)
        acc += part + other
    # Final exchange through the two-stage ring: stage 0 here, stage 1 there.
    tl.store(tle.gpu.local_ptr(l0_w.acquire(0).l), acc)
    l0_w.commit(0)
    peer = tl.load(tle.gpu.local_ptr(l1_r.wait(1).slot.l))
    l1_r.release(1)
    tl.store(y + tl.arange(0, BLOCK), acc + peer)


@triton.jit
def _consumer1(q_r, k1_r, valid_r, m1_w, m0_r, l1_w, l0_r, y, n, BLOCK: tl.constexpr):
    q = tl.load(tle.gpu.local_ptr(q_r.wait(0).slot.q))
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for t in range(n):
        k1 = tl.load(tle.gpu.local_ptr(k1_r.wait(t).slot.k))
        k1_r.release(t)
        v = tl.load(tle.gpu.local_ptr(valid_r.wait(t).slot.v))
        valid_r.release(t)
        part = q * k1 * v
        tl.store(tle.gpu.local_ptr(m1_w.acquire(t).m), part)
        m1_w.commit(t)
        other = tl.load(tle.gpu.local_ptr(m0_r.wait(t).slot.m))
        m0_r.release(t)
        acc += part - other
    tl.store(tle.gpu.local_ptr(l1_w.acquire(1).l), acc)
    l1_w.commit(1)
    peer = tl.load(tle.gpu.local_ptr(l0_r.wait(0).slot.l))
    l0_r.release(0)
    tl.store(y + BLOCK + tl.arange(0, BLOCK), acc - peer)


@triton.jit
def _topology_kernel(x, y, n, BLOCK: tl.constexpr):
    sq = tle.gpu.alloc([1, BLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.smem)
    sk0 = tle.gpu.alloc([1, BLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.smem)
    sk1 = tle.gpu.alloc([1, BLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.smem)
    sv = tle.gpu.alloc([1, BLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.smem)
    sm0 = tle.gpu.alloc([1, BLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.smem)
    sm1 = tle.gpu.alloc([1, BLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.smem)
    sl = tle.gpu.alloc([2, BLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.smem)
    q_pipe = tle.pipe(capacity=1, name="q", readers=("wg0", "wg1"), one_shot=True, q=sq)
    k0_pipe = tle.pipe(capacity=1, name="k0", k=sk0)
    k1_pipe = tle.pipe(capacity=1, name="k1", readers=("qk", "remote"), k=sk1)
    valid_pipe = tle.pipe(capacity=1, name="valid", readers=("wg0", "wg1"), v=sv)
    m0_pipe = tle.pipe(capacity=1, name="m0", m=sm0)
    m1_pipe = tle.pipe(capacity=1, name="m1", m=sm1)
    l0_pipe = tle.pipe(capacity=2, name="l0", l=sl)
    l1_pipe = tle.pipe(capacity=2, name="l1", l=sl)
    tle.gpu.warp_specialize(
        [
            (_consumer0, (q_pipe.writer(), q_pipe.reader("wg0"), k0_pipe.reader(), k1_pipe.reader("remote"),
                          valid_pipe.reader("wg0"), m0_pipe.writer(), m1_pipe.reader(), l0_pipe.writer(),
                          l1_pipe.reader(), x, y, n, BLOCK)),
            (_consumer1, (q_pipe.reader("wg1"), k1_pipe.reader("qk"), valid_pipe.reader("wg1"), m1_pipe.writer(),
                          m0_pipe.reader(), l1_pipe.writer(), l0_pipe.reader(), y, n, BLOCK)),
            (_producer, (k0_pipe.writer(), k1_pipe.writer(), valid_pipe.writer(), x, n, BLOCK)),
        ],
        [4, 4],
        [128, 128],
    )


@pytest.mark.parametrize("n", [1, 2, 5, 40])
def test_pipe_three_group_topology(n):
    block = 128
    x = torch.randn(max(n, 1) * 2 * block, device="cuda")
    y = torch.zeros(2 * block, device="cuda")
    ck = _topology_kernel[(1, )](x, y, n, BLOCK=block, num_warps=4)
    torch.cuda.synchronize()
    q = x[:block]
    tiles = x[:n * 2 * block].view(n, 2, block)
    k0, k1 = tiles[:, 0], tiles[:, 1]
    v = torch.arange(1, n + 1, device="cuda", dtype=torch.float32)[:, None]
    part0 = q * k0 + k1 * v
    part1 = q * k1 * v
    acc0 = (part0 + part1).sum(0)
    acc1 = (part1 - part0).sum(0)
    expected = torch.cat([acc0 + acc1, acc1 - acc0])
    torch.testing.assert_close(y, expected, rtol=1e-4, atol=1e-3)
    ttgir = ck.asm["ttgir"]
    assert "tle.pipe" not in ttgir
    assert "ttng.wait_barrier_named" in ttgir and "scf.while" in ttgir
    assert ck.metadata.num_warps == 12
