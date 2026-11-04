# flagtree tle
"""FlagMega-on-Radeon runtime coverage for warp_specialize + pipe on RDNA.

These exercise the P1-B wave-id partition and P1-A LDS mbarrier handshake on a
real GPU -- the paths lit cannot check. The worker partition occupies waves
above the default group, so the CTA must be launched with (default + worker)
waves; add_convert_warp_specialize records that via ttg.total-num-warps and the
AMD backend launches it. Without that the producer wave never runs and the
consumer deadlocks on wait_barrier.
"""

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle


def _is_amd_rdna() -> bool:
    try:
        target = triton.runtime.driver.active.get_current_target()
    except Exception:
        return False
    arch = str(getattr(target, "arch", ""))
    return target.backend == "hip" and arch.startswith("gfx1")


pytestmark = pytest.mark.skipif(
    not _is_amd_rdna(),
    reason="warp_specialize wave-id partition + RDNA mbarrier handshake require an AMD RDNA GPU",
)


# --- fieldless control pipe: a pure acquire/commit/wait/release ping-pong ------

@triton.jit(noinline=True)
def _cyclic_consumer(handoff_writer, output):
    for iteration in tl.range(0, 4, loop_unroll_factor=1):
        tl.store(output + iteration, iteration + 1)
        handoff_writer.acquire(iteration)
        handoff_writer.commit(iteration)


@triton.jit(noinline=True)
def _cyclic_producer(handoff_reader, output):
    for iteration in tl.range(0, 4, loop_unroll_factor=1):
        handoff_reader.wait(iteration)
        tl.store(output + 4 + iteration, iteration + 5)
        handoff_reader.release(iteration)


@triton.jit
def _cyclic_kernel(output):
    handoff = tle.pipe(capacity=1, scope="cta", name="cyclic_handoff")
    tle.gpu.warp_specialize(
        [
            (_cyclic_consumer, (handoff.writer(), output)),
            (_cyclic_producer, (handoff.reader(), output)),
        ],
        [1],
        [32],
    )


@pytest.mark.require_tle("pipe", "gpu.warp_specialize")
def test_fieldless_cyclic_handoff_runs_on_rdna(with_allocator):
    output = torch.zeros(8, dtype=torch.int32, device="cuda")
    compiled = _cyclic_kernel.warmup(output, grid=(1,), num_warps=8)
    # 8 default waves + 1 worker wave must be launched.
    assert compiled.metadata.num_warps == 9

    _cyclic_kernel[(1,)](output, num_warps=8)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        output.cpu(), torch.arange(1, 9, dtype=torch.int32)
    )


# --- payload pipe: producer fills an LDS field, consumer transforms it --------

_CAP = tl.constexpr(2)
_ROWS = tl.constexpr(8)
_COLS = tl.constexpr(16)
_TILE = tl.constexpr(8 * 16)
_N_TILES = tl.constexpr(4)


@triton.jit(noinline=True)
def _payload_producer(writer, in_ptr):
    row = tl.broadcast_to(tl.arange(0, _ROWS)[:, None], (_ROWS, _COLS))
    col = tl.broadcast_to(tl.arange(0, _COLS)[None, :], (_ROWS, _COLS))
    lin = row * _COLS + col
    for t in tl.range(0, _N_TILES):
        slot = writer.acquire(t)
        g = tl.load(in_ptr + t * _TILE + lin)
        tl.store(tle.gpu.local_ptr(slot.buf, (row, col), [_ROWS, _COLS]), g)
        writer.commit(t)
    writer.close(_N_TILES)


@triton.jit(noinline=True)
def _payload_consumer(reader, out_ptr):
    row = tl.broadcast_to(tl.arange(0, _ROWS)[:, None], (_ROWS, _COLS))
    col = tl.broadcast_to(tl.arange(0, _COLS)[None, :], (_ROWS, _COLS))
    lin = row * _COLS + col
    for t in tl.range(0, _N_TILES):
        ready = reader.wait(t)
        v = tl.load(tle.gpu.local_ptr(ready.slot.buf, (row, col), [_ROWS, _COLS]))
        tl.store(out_ptr + t * _TILE + lin, v * 2.0 + 1.0)
        reader.release(t)


@triton.jit
def _payload_kernel(in_ptr, out_ptr):
    stages = tle.gpu.alloc(
        [_CAP, _ROWS, _COLS],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    p = tle.pipe(capacity=_CAP, scope="cta", name="pc", buf=stages)
    tle.gpu.warp_specialize(
        [
            (_payload_consumer, (p.reader(), out_ptr)),
            (_payload_producer, (p.writer(), in_ptr)),
        ],
        [1],
        [32],
    )


@pytest.mark.require_tle("pipe", "gpu.warp_specialize", "gpu.alloc", "gpu.local_ptr")
def test_pipe_payload_dataflow_runs_on_rdna(with_allocator):
    total = 8 * 16 * 4
    torch.manual_seed(0)
    inp = torch.randn(total, dtype=torch.float32, device="cuda")
    out = torch.full((total,), -1.0, dtype=torch.float32, device="cuda")

    _payload_kernel[(1,)](inp, out, num_warps=8)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, inp * 2.0 + 1.0)
