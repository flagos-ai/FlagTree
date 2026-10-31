"""PPU runtime tests of TLE pipes.

``tle.pipe`` lowers to PPU hardware named barriers: one full and one empty
barrier per stage, the readers arriving on the empty barriers once at start. A
stage filled by ``tle.gpu.copy`` is a cp.async group that the writer drains
before it publishes the stage. Covered here:

* warp-specialized producer / consumer rings with cp.async and register-store
  payloads, capacities 1, 2 and 4, tile counts around the capacity;
* ``writer.close`` and ``is_closed``;
* one reader per warp group (SPMC) and multi-field pipes;
* ``one_shot`` pipes;
* a pipe whose writer and readers are the same warps (software pipelining
  without warp specialization);
* more pipe barriers than the hardware has: the later pipes fall back to
  software barriers in shared memory.
"""

import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_ppu

tle = pytest.importorskip("triton.experimental.tle.language", reason="tle language unavailable")

pytestmark = pytest.mark.skipif(not is_ppu(), reason="requires the PPU backend")


def _check_lowered(ck, *, copies):
    ttgir = ck.asm["ttgir"]
    assert "tle.pipe" not in ttgir
    assert "ttng.wait_barrier_named" in ttgir and "ttng.arrive_barrier_named" in ttgir
    if copies:
        assert "ttg.async_copy_global_to_local" in ttgir and "ttg.async_wait" in ttgir


# ---------------------------------------------------------------- ring: copy payload


@triton.jit
def _copy_producer(writer, x, n_tiles, BLOCK: tl.constexpr):
    for t in range(n_tiles):
        slot = writer.acquire(t)
        tle.gpu.copy(x + t * BLOCK + tl.arange(0, BLOCK), slot.data, [BLOCK])
        writer.commit(t)


@triton.jit
def _ring_consumer(reader, y, n_tiles, BLOCK: tl.constexpr):
    for t in range(n_tiles):
        wait = reader.wait(t)
        v = tl.load(tle.gpu.local_ptr(wait.slot.data))
        reader.release(t)
        tl.store(y + t * BLOCK + tl.arange(0, BLOCK), v * 3.0)


@triton.jit
def _copy_ring_kernel(x, y, n_tiles, CAPACITY: tl.constexpr, BLOCK: tl.constexpr):
    ring = tle.gpu.alloc([CAPACITY, BLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.smem)
    pipe = tle.pipe(capacity=CAPACITY, data=ring)
    tle.gpu.warp_specialize(
        [(_ring_consumer, (pipe.reader(), y, n_tiles, BLOCK)), (_copy_producer, (pipe.writer(), x, n_tiles, BLOCK))],
        [4],
        [128],
    )


@pytest.mark.parametrize("capacity", [1, 2, 4])
@pytest.mark.parametrize("n_tiles", [1, 2, 3, 4, 7, 64])
def test_pipe_ring_copy(capacity, n_tiles):
    block = 128
    x = torch.randn(n_tiles * block, device="cuda")
    y = torch.zeros_like(x)
    ck = _copy_ring_kernel[(4, )](x, y, n_tiles, CAPACITY=capacity, BLOCK=block, num_warps=4)
    torch.cuda.synchronize()
    torch.testing.assert_close(y, 3 * x)
    _check_lowered(ck, copies=True)


# A producer that also consumes the pipe can finish one slot while younger
# copies stay in flight. The remote reader still needs a completed-copy handoff.
@triton.jit
def _mixed_copy(writer, reader, ack, x, local_out, n, DEPTH: tl.constexpr, BLOCK: tl.constexpr, MODE: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    for t in range(tl.minimum(n, DEPTH)):
        initial = writer.acquire(t)
        tle.gpu.copy(x + t * BLOCK + tl.reshape(offsets, [2, BLOCK // 2]), initial.data, [2, BLOCK // 2])
        writer.commit(t)
    if MODE == 1:
        tle.gpu.barrier_wait(ack[0])
    for t in range(n):
        if MODE != 2 or t + 1 < n:
            current = reader.wait(t).slot
            value = tl.reshape(tl.load(tle.gpu.local_ptr(current.data)), [BLOCK])
            reader.release(t)
            tl.store(local_out + t * BLOCK + offsets, value * 2)
        if t + DEPTH < n:
            following = writer.acquire(t + DEPTH)
            tle.gpu.copy(x + (t + DEPTH) * BLOCK + tl.reshape(offsets, [2, BLOCK // 2]), following.data,
                         [2, BLOCK // 2])
            writer.commit(t + DEPTH)


@triton.jit
def _mixed_remote(reader, ack, y, n, BLOCK: tl.constexpr, MODE: tl.constexpr):
    for t in range(n):
        wait = reader.wait(t)
        value = tl.reshape(tl.load(tle.gpu.local_ptr(wait.slot.data)), [BLOCK])
        reader.release(t)
        tl.store(y + t * BLOCK + tl.arange(0, BLOCK), value * 3)
        if MODE == 1 and t == 0:
            tle.gpu.barrier_arrive(ack[0])


@triton.jit
def _mixed_copy_kernel(x, local_out, remote_out, n, CAPACITY: tl.constexpr, DEPTH: tl.constexpr, BLOCK: tl.constexpr,
                       MODE: tl.constexpr = 0):
    layout: tl.constexpr = tle.gpu.swizzled_shared_layout(4, 1, 1, [2, 1, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1],
                                                          [2, 1, 0])
    ring = tle.gpu.alloc([CAPACITY, 2, BLOCK // 2], dtype=tl.float32, layout=layout, scope=tle.gpu.smem)
    pipe = tle.pipe(capacity=CAPACITY, readers=("local", "remote"), data=ring)
    ack = tle.gpu.alloc_barriers(1, arrive_count=256)
    tle.gpu.warp_specialize([(_mixed_copy,
                              (pipe.writer(), pipe.reader("local"), ack, x, local_out, n, DEPTH, BLOCK, MODE)),
                             (_mixed_remote, (pipe.reader("remote"), ack, remote_out, n, BLOCK, MODE))], [4], [128])


@pytest.mark.parametrize("capacity,depth", [(2, 2), (3, 2), (4, 3)])
@pytest.mark.parametrize("n_tiles", [0, 1, 2, 3, 4, 7, 33])
def test_pipe_mixed_readers_keep_copies_in_flight(capacity, depth, n_tiles):
    block = 512
    x = torch.randn(max(n_tiles, 1) * block, device="cuda")
    local_out, remote_out = torch.zeros_like(x), torch.zeros_like(x)
    ck = _mixed_copy_kernel[(4, )](x, local_out, remote_out, n_tiles, capacity, depth, block, num_warps=4)
    torch.cuda.synchronize()
    if n_tiles:
        torch.testing.assert_close(local_out, 2 * x)
        torch.testing.assert_close(remote_out, 3 * x)
    else:
        assert torch.count_nonzero(local_out).item() == 0
        assert torch.count_nonzero(remote_out).item() == 0
    _check_lowered(ck, copies=True)
    if n_tiles >= 2:
        assert "num = 1 : i32, ppugpu.pipe_wait" in ck.asm["ttgir"]


@pytest.mark.parametrize("mode", [1, 2])
def test_pipe_mixed_readers_drain_before_peer_wait_and_exit(mode):
    block, n_tiles = 512, 7
    x = torch.randn(n_tiles * block, device="cuda")
    local_out, remote_out = torch.zeros_like(x), torch.zeros_like(x)
    _mixed_copy_kernel[(4, )](x, local_out, remote_out, n_tiles, 3, 2, block, MODE=mode, num_warps=4)
    torch.cuda.synchronize()
    torch.testing.assert_close(remote_out, 3 * x)
    local_tiles = n_tiles if mode == 1 else n_tiles - 1
    torch.testing.assert_close(local_out[:local_tiles * block], 2 * x[:local_tiles * block])
    if mode == 2:
        assert torch.count_nonzero(local_out[local_tiles * block:]).item() == 0


@pytest.mark.parametrize("capacity,depth,block,specialized", [(2, 2, 512, True), (3, 2, 512, True), (4, 3, 256, False),
                                                              (4, 3, 1024, True)])
def test_pipe_fifo_specialization_preserves_overflow_fallback(capacity, depth, block, specialized):
    n_tiles = 7
    x = torch.randn(n_tiles * block, device="cuda")
    local_out, remote_out = torch.zeros_like(x), torch.zeros_like(x)
    # Compile the boundary without executing its billions of iterations. Both
    # bounds are odd, so the ordinary run uses the same specialization key.
    ck = _mixed_copy_kernel.warmup(x, local_out, remote_out, 2**31 - 1, capacity, depth, block, grid=(1, ), num_warps=4)
    bound = f"arith.constant {2**31 - 1 - depth} : i32"
    assert (bound in ck.asm["ttgir"]) == specialized
    if specialized:
        assert "arith.cmpi sle" in ck.asm["ttgir"]
        assert "iter_args" in ck.asm["ttgir"]  # The general FIFO remains available.
    executed = _mixed_copy_kernel[(1, )](x, local_out, remote_out, n_tiles, capacity, depth, block, num_warps=4)
    assert executed.hash == ck.hash
    torch.cuda.synchronize()
    torch.testing.assert_close(local_out, 2 * x)
    torch.testing.assert_close(remote_out, 3 * x)


# ---------------------------------------------------------------- ring: register-store payload


@triton.jit
def _store_producer(writer, x, n_tiles, BLOCK: tl.constexpr):
    for t in range(n_tiles):
        slot = writer.acquire(t)
        v = tl.load(x + t * BLOCK + tl.arange(0, BLOCK))
        tl.store(tle.gpu.local_ptr(slot.data), v + 1.0)
        writer.commit(t)


@triton.jit
def _store_ring_kernel(x, y, n_tiles, CAPACITY: tl.constexpr, BLOCK: tl.constexpr):
    ring = tle.gpu.alloc([CAPACITY, BLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.smem)
    pipe = tle.pipe(capacity=CAPACITY, data=ring)
    tle.gpu.warp_specialize(
        [(_ring_consumer, (pipe.reader(), y, n_tiles, BLOCK)), (_store_producer, (pipe.writer(), x, n_tiles, BLOCK))],
        [4],
        [128],
    )


@pytest.mark.parametrize("capacity", [1, 2])
@pytest.mark.parametrize("n_tiles", [1, 3, 32])
def test_pipe_ring_store(capacity, n_tiles):
    block = 128
    x = torch.randn(n_tiles * block, device="cuda")
    y = torch.zeros_like(x)
    ck = _store_ring_kernel[(2, )](x, y, n_tiles, CAPACITY=capacity, BLOCK=block, num_warps=4)
    torch.cuda.synchronize()
    torch.testing.assert_close(y, 3 * (x + 1))
    _check_lowered(ck, copies=False)
    assert "ttg.async_copy_global_to_local" not in ck.asm["ttgir"]


# ---------------------------------------------------------------- close / is_closed


@triton.jit
def _closing_producer(writer, x, n_tiles, BLOCK: tl.constexpr):
    for t in range(n_tiles):
        slot = writer.acquire(t)
        tle.gpu.copy(x + t * BLOCK + tl.arange(0, BLOCK), slot.data, [BLOCK])
        writer.commit(t)
    writer.close(n_tiles)


@triton.jit
def _closing_consumer(reader, y, closed, n_tiles, BLOCK: tl.constexpr):
    # One wait more than the producer commits: the last one observes the close.
    for t in range(n_tiles + 1):
        wait = reader.wait(t)
        if wait.is_closed:
            tl.store(closed + t, 1)
        else:
            v = tl.load(tle.gpu.local_ptr(wait.slot.data))
            tl.store(y + t * BLOCK + tl.arange(0, BLOCK), v)
        reader.release(t)


@triton.jit
def _closing_kernel(x, y, closed, n_tiles, CAPACITY: tl.constexpr, BLOCK: tl.constexpr):
    ring = tle.gpu.alloc([CAPACITY, BLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.smem)
    pipe = tle.pipe(capacity=CAPACITY, data=ring)
    tle.gpu.warp_specialize(
        [(_closing_consumer, (pipe.reader(), y, closed, n_tiles, BLOCK)),
         (_closing_producer, (pipe.writer(), x, n_tiles, BLOCK))],
        [4],
        [128],
    )


@pytest.mark.parametrize("capacity", [1, 2, 4])
@pytest.mark.parametrize("n_tiles", [0, 1, 5])
def test_pipe_close(capacity, n_tiles):
    block = 64
    x = torch.randn(max(n_tiles, 1) * block, device="cuda")
    y = torch.zeros_like(x)
    closed = torch.zeros(n_tiles + 1, dtype=torch.int32, device="cuda")
    ck = _closing_kernel[(1, )](x, y, closed, n_tiles, CAPACITY=capacity, BLOCK=block, num_warps=4)
    torch.cuda.synchronize()
    expected_closed = torch.zeros(n_tiles + 1, dtype=torch.int32, device="cuda")
    expected_closed[n_tiles] = 1
    torch.testing.assert_close(closed, expected_closed)
    if n_tiles:
        torch.testing.assert_close(y, x)
    _check_lowered(ck, copies=True)


# ---------------------------------------------------------------- SPMC + two fields


@triton.jit
def _two_field_producer(writer, x, n_tiles, BLOCK: tl.constexpr):
    for t in range(n_tiles):
        slot = writer.acquire(t)
        tle.gpu.copy(x + t * BLOCK + tl.arange(0, BLOCK), slot.a, [BLOCK])
        v = tl.load(x + t * BLOCK + tl.arange(0, BLOCK))
        tl.store(tle.gpu.local_ptr(slot.b), v * 2.0)
        writer.commit(t)


@triton.jit
def _two_field_consumer(reader, y, n_tiles, BLOCK: tl.constexpr, SCALE: tl.constexpr):
    for t in range(n_tiles):
        wait = reader.wait(t)
        a = tl.load(tle.gpu.local_ptr(wait.slot.a))
        b = tl.load(tle.gpu.local_ptr(wait.slot.b))
        reader.release(t)
        tl.store(y + t * BLOCK + tl.arange(0, BLOCK), (a + b) * SCALE)


@triton.jit
def _spmc_kernel(x, y0, y1, n_tiles, CAPACITY: tl.constexpr, BLOCK: tl.constexpr):
    ring_a = tle.gpu.alloc([CAPACITY, BLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.smem)
    ring_b = tle.gpu.alloc([CAPACITY, BLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.smem)
    pipe = tle.pipe(capacity=CAPACITY, readers=("left", "right"), a=ring_a, b=ring_b)
    tle.gpu.warp_specialize(
        [(_two_field_producer, (pipe.writer(), x, n_tiles, BLOCK)),
         (_two_field_consumer, (pipe.reader("left"), y0, n_tiles, BLOCK, 1.0)),
         (_two_field_consumer, (pipe.reader("right"), y1, n_tiles, BLOCK, 10.0))],
        [4, 4],
        [128, 128],
    )


@pytest.mark.parametrize("capacity", [1, 2])
@pytest.mark.parametrize("n_tiles", [1, 2, 9])
def test_pipe_spmc_two_fields(capacity, n_tiles):
    block = 64
    x = torch.randn(n_tiles * block, device="cuda")
    y0 = torch.zeros_like(x)
    y1 = torch.zeros_like(x)
    ck = _spmc_kernel[(1, )](x, y0, y1, n_tiles, CAPACITY=capacity, BLOCK=block, num_warps=4)
    torch.cuda.synchronize()
    torch.testing.assert_close(y0, 3 * x)
    torch.testing.assert_close(y1, 30 * x)
    _check_lowered(ck, copies=True)
    assert ck.metadata.num_warps == 12


# ---------------------------------------------------------------- one_shot


@triton.jit
def _one_shot_producer(writer, x, BLOCK: tl.constexpr):
    v = tl.load(x + tl.arange(0, BLOCK))
    tl.store(tle.gpu.local_ptr(writer.acquire(0).data), v)
    writer.commit(0)


@triton.jit
def _one_shot_consumer(reader, y, BLOCK: tl.constexpr):
    wait = reader.wait(0)
    v = tl.load(tle.gpu.local_ptr(wait.slot.data))
    tl.store(y + tl.arange(0, BLOCK), v + 1.0)


@triton.jit
def _one_shot_kernel(x, y, BLOCK: tl.constexpr):
    buf = tle.gpu.alloc([1, BLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.smem)
    pipe = tle.pipe(capacity=1, one_shot=True, data=buf)
    tle.gpu.warp_specialize(
        [(_one_shot_consumer, (pipe.reader(), y, BLOCK)), (_one_shot_producer, (pipe.writer(), x, BLOCK))],
        [4],
        [128],
    )


def test_pipe_one_shot():
    block = 256
    x = torch.randn(block, device="cuda")
    y = torch.zeros_like(x)
    ck = _one_shot_kernel[(1, )](x, y, BLOCK=block, num_warps=4)
    torch.cuda.synchronize()
    torch.testing.assert_close(y, x + 1)
    ttgir = ck.asm["ttgir"]
    assert "tle.pipe" not in ttgir
    # A one_shot pipe has no empty barriers: one full barrier, arrived once and waited once.
    assert ttgir.count("ttng.arrive_barrier_named") == 1 and ttgir.count("ttng.wait_barrier_named") == 1


# ---------------------------------------------------------------- same warps: software pipelining


@triton.jit
def _same_warps_kernel(x, y, n_tiles, BLOCK: tl.constexpr):
    ring = tle.gpu.alloc([2, BLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.smem)
    pipe = tle.pipe(capacity=2, data=ring)
    writer = pipe.writer()
    reader = pipe.reader()
    slot = writer.acquire(0)
    tle.gpu.copy(x + tl.arange(0, BLOCK), slot.data, [BLOCK])
    writer.commit(0)
    for t in range(n_tiles):
        if t + 1 < n_tiles:
            nxt = writer.acquire(t + 1)
            tle.gpu.copy(x + (t + 1) * BLOCK + tl.arange(0, BLOCK), nxt.data, [BLOCK])
            writer.commit(t + 1)
        wait = reader.wait(t)
        v = tl.load(tle.gpu.local_ptr(wait.slot.data))
        reader.release(t)
        tl.store(y + t * BLOCK + tl.arange(0, BLOCK), v * 3.0)


@pytest.mark.parametrize("n_tiles", [1, 2, 5, 32])
def test_pipe_same_warps(n_tiles):
    block = 128
    x = torch.randn(n_tiles * block, device="cuda")
    y = torch.zeros_like(x)
    ck = _same_warps_kernel[(2, )](x, y, n_tiles, BLOCK=block, num_warps=4)
    torch.cuda.synchronize()
    torch.testing.assert_close(y, 3 * x)
    ttgir = ck.asm["ttgir"]
    assert "tle.pipe" not in ttgir
    # Same warps on both ends: ordering barriers only, no hardware hand-over.
    assert "ttng.wait_barrier_named" not in ttgir and "ttg.local_barrier" in ttgir
    assert "ttg.async_wait" in ttgir


# ---------------------------------------------------------------- software-barrier fallback


@triton.jit
def _quad_producer(w0, w1, w2, w3, x, n_tiles, BLOCK: tl.constexpr):
    for t in range(n_tiles):
        base = x + t * 4 * BLOCK + tl.arange(0, BLOCK)
        s0 = w0.acquire(t)
        tle.gpu.copy(base, s0.data, [BLOCK])
        w0.commit(t)
        s1 = w1.acquire(t)
        tle.gpu.copy(base + BLOCK, s1.data, [BLOCK])
        w1.commit(t)
        s2 = w2.acquire(t)
        tle.gpu.copy(base + 2 * BLOCK, s2.data, [BLOCK])
        w2.commit(t)
        s3 = w3.acquire(t)
        tle.gpu.copy(base + 3 * BLOCK, s3.data, [BLOCK])
        w3.commit(t)


@triton.jit
def _quad_consumer(r0, r1, r2, r3, y, n_tiles, BLOCK: tl.constexpr):
    for t in range(n_tiles):
        base = y + t * 4 * BLOCK + tl.arange(0, BLOCK)
        v0 = tl.load(tle.gpu.local_ptr(r0.wait(t).slot.data))
        r0.release(t)
        v1 = tl.load(tle.gpu.local_ptr(r1.wait(t).slot.data))
        r1.release(t)
        v2 = tl.load(tle.gpu.local_ptr(r2.wait(t).slot.data))
        r2.release(t)
        v3 = tl.load(tle.gpu.local_ptr(r3.wait(t).slot.data))
        r3.release(t)
        tl.store(base, v0 * 1.0)
        tl.store(base + BLOCK, v1 * 2.0)
        tl.store(base + 2 * BLOCK, v2 * 3.0)
        tl.store(base + 3 * BLOCK, v3 * 4.0)


@triton.jit
def _quad_ring_kernel(x, y, n_tiles, CAPACITY: tl.constexpr, BLOCK: tl.constexpr):
    # 4 pipes x 2 x CAPACITY barriers exceed the 13 hardware barriers left to TLE.
    ring0 = tle.gpu.alloc([CAPACITY, BLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.smem)
    ring1 = tle.gpu.alloc([CAPACITY, BLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.smem)
    ring2 = tle.gpu.alloc([CAPACITY, BLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.smem)
    ring3 = tle.gpu.alloc([CAPACITY, BLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.smem)
    p0 = tle.pipe(capacity=CAPACITY, name="p0", data=ring0)
    p1 = tle.pipe(capacity=CAPACITY, name="p1", data=ring1)
    p2 = tle.pipe(capacity=CAPACITY, name="p2", data=ring2)
    p3 = tle.pipe(capacity=CAPACITY, name="p3", data=ring3)
    tle.gpu.warp_specialize(
        [(_quad_consumer, (p0.reader(), p1.reader(), p2.reader(), p3.reader(), y, n_tiles, BLOCK)),
         (_quad_producer, (p0.writer(), p1.writer(), p2.writer(), p3.writer(), x, n_tiles, BLOCK))],
        [4],
        [128],
    )


@pytest.mark.parametrize("n_tiles", [1, 3, 8, 33])
def test_pipe_software_barrier_fallback(n_tiles):
    block = 64
    x = torch.randn(n_tiles * 4 * block, device="cuda")
    y = torch.zeros_like(x)
    ck = _quad_ring_kernel[(4, )](x, y, n_tiles, CAPACITY=4, BLOCK=block, num_warps=4)
    torch.cuda.synchronize()
    scale = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda").repeat_interleave(block).repeat(n_tiles)
    torch.testing.assert_close(y, x * scale)
    ttgir = ck.asm["ttgir"]
    assert "tle.pipe" not in ttgir
    # The first pipe takes 8 hardware barriers; the remaining 5 ids cannot hold another pipe,
    # so the other three spin on shared-memory counters.
    assert "ttng.wait_barrier_named" in ttgir
    assert "scf.while" in ttgir
