"""Explicit TLE warp_specialize + pipe tests for the GCU backend.

Covers three warp-specialization-focused scenarios:
  1. Basic producer/consumer correctness (SPSC, single pipe).
  2. 2D elementwise add — SPSC and 3-partition (load->compute->store).
  3. Multiple WarpSpecializeOp regions in one kernel.
"""
import importlib.util

import pytest
import torch
import triton
import triton.language as tl

if importlib.util.find_spec("triton.backends.enflame") is None:
    import triton_gcu.triton
from torch_gcu import transfer_to_gcu  # noqa: F401  (enables .gcu() on tensors)
import triton.experimental.tle.language as tle

BLOCK_SIZE = 64
NUM_STAGES = 1
DEVICE = triton.runtime.driver.active.get_active_torch_device()

# ---------------------------------------------------------------------------
# Partition functions — each recomputes offsets/mask from scalar pid+numel.
# The pipe_writer / pipe_reader objects are passed in to synchronize access
# to the shared-memory buffer across producer and consumer partitions.
# ---------------------------------------------------------------------------

@triton.jit
def _consumer(out_ptr, reader, pid, numel, BLOCK: tl.constexpr):
    """Default partition (consumer): pipe wait -> load smem -> add 1 -> store global -> pipe release."""
    result = reader.wait(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    smem_ptrs = tle.gpu.local_ptr(result.slot.a, (tl.arange(0, BLOCK),))
    vals = tl.load(smem_ptrs, mask=mask, other=0.0)
    out_vals = vals + 1.0
    tl.store(out_ptr + offsets, out_vals, mask=mask)
    reader.release(0)


@triton.jit
def _producer(x_ptr, writer, pid, numel, BLOCK: tl.constexpr):
    """Worker partition (producer): load global -> pipe acquire -> store smem -> pipe commit."""
    slot = writer.acquire(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    smem_ptrs = tle.gpu.local_ptr(slot.a, (tl.arange(0, BLOCK),))
    x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(smem_ptrs, x_vals, mask=mask)
    writer.commit(0)


@triton.jit
def _ws_pipe_kernel(x_ptr, out_ptr, numel, BLOCK: tl.constexpr, NUM_STAGES: tl.constexpr):
    pid = tl.program_id(0)

    smem = tle.gpu.alloc(
        (NUM_STAGES, BLOCK),
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )

    p = tle.pipe(capacity=NUM_STAGES, a=smem)
    writer = p.writer()
    reader = p.reader()

    tle.gpu.warp_specialize(
        [
            (_consumer, (out_ptr, reader, pid, numel, tl.constexpr(BLOCK))),
            (_producer, (x_ptr, writer, pid, numel, tl.constexpr(BLOCK))),
        ],
        [1], # consumer_num_warps
        [8], # consumer_num_regs
    )


# ===========================================================================
# Basic Tests
# ===========================================================================

class TestTLEExplicitWarpSpecialize:
    """Explicit warp_specialize + pipe end-to-end tests."""

    def test_ws_producer_consumer_correctness(self):
        """Verify that producer+consumer partitions produce x + 1."""
        numel = BLOCK_SIZE
        x = torch.randn(numel, device=DEVICE, dtype=torch.float32)
        out = torch.empty_like(x)

        grid = (1,)
        _ws_pipe_kernel[grid](x, out, numel, BLOCK_SIZE, NUM_STAGES, num_warps=4)

        torch.testing.assert_close(out, x + 1.0, atol=1e-6, rtol=1e-6)


# ===========================================================================
# 2D Elementwise Add - SPSC (Single-Producer Single-Consumer)
# Migrated from: python/tutorials/gluon/08-warp-specialization.py
# ===========================================================================

BLOCK_X = 32
BLOCK_Y = 64


@triton.jit
def _add_load_producer_spsc(
    a_ptr, b_ptr, writer, pid,
    xnumel, ynumel, stride_x, stride_y,
    XBLOCK: tl.constexpr, YBLOCK: tl.constexpr,
):
    """Producer: loads A and B tiles into shared memory via pipe."""
    xoff = pid * XBLOCK
    num_y_tiles = tl.cdiv(ynumel, YBLOCK)

    for i in range(num_y_tiles):
        slot = writer.acquire(i)
        yoff = i * YBLOCK
        offs_x = xoff + tl.arange(0, XBLOCK)
        offs_y = yoff + tl.arange(0, YBLOCK)
        mask = (offs_x[:, None] < xnumel) & (offs_y[None, :] < ynumel)
        flat_idx = tl.arange(0, XBLOCK)[:, None] * YBLOCK + tl.arange(0, YBLOCK)[None, :]

        a_vals = tl.load(
            a_ptr + offs_x[:, None] * stride_x + offs_y[None, :] * stride_y,
            mask=mask, other=0.0
        )
        b_vals = tl.load(
            b_ptr + offs_x[:, None] * stride_x + offs_y[None, :] * stride_y,
            mask=mask, other=0.0
        )

        a_ptrs = tle.gpu.local_ptr(slot.a_buf, (flat_idx,))
        b_ptrs = tle.gpu.local_ptr(slot.b_buf, (flat_idx,))
        tl.store(a_ptrs, a_vals)
        tl.store(b_ptrs, b_vals)
        writer.commit(i)


@triton.jit
def _add_compute_consumer_spsc(
    c_ptr, reader, pid,
    xnumel, ynumel, stride_x, stride_y,
    XBLOCK: tl.constexpr, YBLOCK: tl.constexpr,
):
    """Consumer: reads A and B from pipe, computes C = A + B, stores to global."""
    xoff = pid * XBLOCK
    num_y_tiles = tl.cdiv(ynumel, YBLOCK)

    for i in range(num_y_tiles):
        result = reader.wait(i)
        flat_idx = tl.arange(0, XBLOCK)[:, None] * YBLOCK + tl.arange(0, YBLOCK)[None, :]

        a_ptrs = tle.gpu.local_ptr(result.slot.a_buf, (flat_idx,))
        b_ptrs = tle.gpu.local_ptr(result.slot.b_buf, (flat_idx,))
        a_vals = tl.load(a_ptrs)
        b_vals = tl.load(b_ptrs)
        reader.release(i)

        c_vals = a_vals + b_vals

        yoff = i * YBLOCK
        offs_x = xoff + tl.arange(0, XBLOCK)
        offs_y = yoff + tl.arange(0, YBLOCK)
        mask = (offs_x[:, None] < xnumel) & (offs_y[None, :] < ynumel)
        tl.store(
            c_ptr + offs_x[:, None] * stride_x + offs_y[None, :] * stride_y,
            c_vals, mask=mask
        )


@triton.jit
def elementwise_add_ws_spsc_kernel(
    a_ptr, b_ptr, c_ptr,
    xnumel, ynumel, stride_x, stride_y,
    XBLOCK: tl.constexpr, YBLOCK: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """SPSC warp-specialized elementwise add kernel.
    Producer (1 warp) loads, consumer (4 warps) computes and stores.
    """
    pid = tl.program_id(0)
    BUF_SIZE: tl.constexpr = XBLOCK * YBLOCK

    a_buf = tle.gpu.alloc(
        (NUM_STAGES, BUF_SIZE), dtype=tl.float32, layout=None,
        scope=tle.gpu.smem, nv_mma_shared_layout=False,
    )
    b_buf = tle.gpu.alloc(
        (NUM_STAGES, BUF_SIZE), dtype=tl.float32, layout=None,
        scope=tle.gpu.smem, nv_mma_shared_layout=False,
    )

    p = tle.pipe(capacity=NUM_STAGES, a_buf=a_buf, b_buf=b_buf)
    writer = p.writer()
    reader = p.reader()

    tle.gpu.warp_specialize(
        [
            (_add_compute_consumer_spsc, (
                c_ptr, reader, pid,
                xnumel, ynumel, stride_x, stride_y,
                tl.constexpr(XBLOCK), tl.constexpr(YBLOCK),
            )),
            (_add_load_producer_spsc, (
                a_ptr, b_ptr, writer, pid,
                xnumel, ynumel, stride_x, stride_y,
                tl.constexpr(XBLOCK), tl.constexpr(YBLOCK),
            )),
        ],
        [1], # consumer_num_warps
        [8], # consumer_num_regs
    )


def _elementwise_add_ws_spsc(a, b, XBLOCK=BLOCK_X, YBLOCK=BLOCK_Y, num_stages=2):
    """SPSC: 1 producer warp + 4 consumer warps."""
    assert a.shape == b.shape
    xnumel, ynumel = a.shape
    c = torch.empty_like(a)
    grid = (triton.cdiv(xnumel, XBLOCK),)
    elementwise_add_ws_spsc_kernel[grid](
        a, b, c, xnumel, ynumel, a.stride(0), a.stride(1),
        XBLOCK=XBLOCK, YBLOCK=YBLOCK, NUM_STAGES=num_stages,
        num_warps=4,
    )
    return c


# ===========================================================================
# 3-Partition Implementation
# ===========================================================================
# Aligned with python/tutorials/gluon/08-warp-specialization.py:
#   default partition = compute
#   worker 0          = load A/B into input pipe
#   worker 1          = store C from output pipe
# Uses two pipes: one for A/B operands and one for C result.
# ===========================================================================

@triton.jit
def _add_load_producer_spmc(
    a_ptr, b_ptr, writer, pid,
    xnumel, ynumel, stride_x, stride_y,
    XBLOCK: tl.constexpr, YBLOCK: tl.constexpr,
):
    """Load partition: loads A and B tiles into the input pipe."""
    xoff = pid * XBLOCK
    num_y_tiles = tl.cdiv(ynumel, YBLOCK)

    for i in range(num_y_tiles):
        slot = writer.acquire(0)
        yoff = i * YBLOCK
        offs_x = xoff + tl.arange(0, XBLOCK)
        offs_y = yoff + tl.arange(0, YBLOCK)
        mask = (offs_x[:, None] < xnumel) & (offs_y[None, :] < ynumel)
        flat_idx = tl.arange(0, XBLOCK)[:, None] * YBLOCK + tl.arange(0, YBLOCK)[None, :]

        a_vals = tl.load(
            a_ptr + offs_x[:, None] * stride_x + offs_y[None, :] * stride_y,
            mask=mask, other=0.0
        )
        b_vals = tl.load(
            b_ptr + offs_x[:, None] * stride_x + offs_y[None, :] * stride_y,
            mask=mask, other=0.0
        )

        a_ptrs = tle.gpu.local_ptr(slot.a_buf, (flat_idx,))
        b_ptrs = tle.gpu.local_ptr(slot.b_buf, (flat_idx,))
        tl.store(a_ptrs, a_vals)
        tl.store(b_ptrs, b_vals)
        writer.commit(0)


@triton.jit
def _add_compute_partition_spmc(
    input_reader, output_writer, pid,
    xnumel, ynumel, stride_x, stride_y,
    XBLOCK: tl.constexpr, YBLOCK: tl.constexpr,
):
    """Compute partition (default): consume A/B, produce C into output pipe."""
    xoff = pid * XBLOCK
    num_y_tiles = tl.cdiv(ynumel, YBLOCK)
    flat_idx = tl.arange(0, XBLOCK)[:, None] * YBLOCK + tl.arange(0, YBLOCK)[None, :]

    for i in range(num_y_tiles):
        result = input_reader.wait(0)

        a_ptrs = tle.gpu.local_ptr(result.slot.a_buf, (flat_idx,))
        b_ptrs = tle.gpu.local_ptr(result.slot.b_buf, (flat_idx,))
        a_vals = tl.load(a_ptrs)
        b_vals = tl.load(b_ptrs)
        input_reader.release(0)

        c_vals = a_vals + b_vals

        out_slot = output_writer.acquire(0)
        c_ptrs = tle.gpu.local_ptr(out_slot.c_buf, (flat_idx,))
        tl.store(c_ptrs, c_vals)
        output_writer.commit(0)


@triton.jit
def _add_store_partition_spmc(
    c_ptr, output_reader, pid,
    xnumel, ynumel, stride_x, stride_y,
    XBLOCK: tl.constexpr, YBLOCK: tl.constexpr,
):
    """Store partition: consume C from output pipe and write to global memory."""
    xoff = pid * XBLOCK
    num_y_tiles = tl.cdiv(ynumel, YBLOCK)
    flat_idx = tl.arange(0, XBLOCK)[:, None] * YBLOCK + tl.arange(0, YBLOCK)[None, :]

    for i in range(num_y_tiles):
        result = output_reader.wait(0)

        c_ptrs = tle.gpu.local_ptr(result.slot.c_buf, (flat_idx,))
        c_vals = tl.load(c_ptrs)
        output_reader.release(0)

        yoff = i * YBLOCK
        offs_x = xoff + tl.arange(0, XBLOCK)
        offs_y = yoff + tl.arange(0, YBLOCK)
        mask = (offs_x[:, None] < xnumel) & (offs_y[None, :] < ynumel)
        tl.store(
            c_ptr + offs_x[:, None] * stride_x + offs_y[None, :] * stride_y,
            c_vals, mask=mask
        )


@triton.jit
def elementwise_add_ws_spmc_kernel(
    a_ptr, b_ptr, c_ptr,
    xnumel, ynumel, stride_x, stride_y,
    XBLOCK: tl.constexpr, YBLOCK: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """3-partition warp-specialized elementwise add kernel.
    load worker (1 warp) -> compute default -> store worker (1 warp).
    NOTE: Disabled because enflame backend does not support 3 partitions yet.
    """
    pid = tl.program_id(0)
    BUF_SIZE: tl.constexpr = XBLOCK * YBLOCK

    a_buf = tle.gpu.alloc(
        (NUM_STAGES, BUF_SIZE), dtype=tl.float32, layout=None,
        scope=tle.gpu.smem, nv_mma_shared_layout=False,
    )
    b_buf = tle.gpu.alloc(
        (NUM_STAGES, BUF_SIZE), dtype=tl.float32, layout=None,
        scope=tle.gpu.smem, nv_mma_shared_layout=False,
    )
    c_buf = tle.gpu.alloc(
        (NUM_STAGES, BUF_SIZE), dtype=tl.float32, layout=None,
        scope=tle.gpu.smem, nv_mma_shared_layout=False,
    )

    input_pipe = tle.pipe(capacity=NUM_STAGES, a_buf=a_buf, b_buf=b_buf)
    input_writer = input_pipe.writer()
    input_reader = input_pipe.reader()

    output_pipe = tle.pipe(capacity=NUM_STAGES, c_buf=c_buf)
    output_writer = output_pipe.writer()
    output_reader = output_pipe.reader()

    tle.gpu.warp_specialize(
        [
            (_add_compute_partition_spmc, (
                input_reader, output_writer, pid,
                xnumel, ynumel, stride_x, stride_y,
                tl.constexpr(XBLOCK), tl.constexpr(YBLOCK),
            )),
            (_add_load_producer_spmc, (
                a_ptr, b_ptr, input_writer, pid,
                xnumel, ynumel, stride_x, stride_y,
                tl.constexpr(XBLOCK), tl.constexpr(YBLOCK),
            )),
            (_add_store_partition_spmc, (
                c_ptr, output_reader, pid,
                xnumel, ynumel, stride_x, stride_y,
                tl.constexpr(XBLOCK), tl.constexpr(YBLOCK),
            )),
        ],
        [1, 1], # consumer_num_warps
        [8, 8], # consumer_num_regs
    )


def _elementwise_add_ws_spmc(a, b, XBLOCK=BLOCK_X, YBLOCK=BLOCK_Y, num_stages=2):
    """3-partition warp-specialized elementwise add."""
    assert a.shape == b.shape
    xnumel, ynumel = a.shape
    c = torch.empty_like(a)
    grid = (triton.cdiv(xnumel, XBLOCK),)
    elementwise_add_ws_spmc_kernel[grid](
        a, b, c, xnumel, ynumel, a.stride(0), a.stride(1),
        XBLOCK=XBLOCK, YBLOCK=YBLOCK, NUM_STAGES=num_stages,
        num_warps=4,
    )
    return c


class TestTLEElementwiseAddWarpSpecialize:
    """2D elementwise add with TLE explicit warp specialization."""

    @pytest.mark.parametrize("xnumel,ynumel", [(1000, 2000), (4000, 120)])
    @pytest.mark.parametrize("num_stages", [1, 2])
    def test_spsc(self, xnumel, ynumel, num_stages):
        """Test SPSC warp-specialized elementwise add."""
        a = torch.randn(xnumel, ynumel, device=DEVICE)
        b = torch.randn(xnumel, ynumel, device=DEVICE)
        c = _elementwise_add_ws_spsc(a, b, BLOCK_X, BLOCK_Y, num_stages)
        torch.testing.assert_close(a + b, c, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("xnumel,ynumel", [(1000, 2000), (4000, 120)])
    @pytest.mark.parametrize("num_stages", [1, 2])
    def test_two_pipeline_spsc(self, xnumel, ynumel, num_stages):
        """Test 3-partition two-pipeline SPSC: load->compute->store.
        Compute partition is consumer of input pipe and producer of
        output pipe — two different pipeline instances, which is allowed."""
        a = torch.randn(xnumel, ynumel, device=DEVICE)
        b = torch.randn(xnumel, ynumel, device=DEVICE)
        c = _elementwise_add_ws_spmc(a, b, BLOCK_X, BLOCK_Y, num_stages)
        torch.testing.assert_close(a + b, c, atol=1e-5, rtol=1e-5)


# ===========================================================================
# 3. Multiple WarpSpecializeOp in one kernel
# ===========================================================================

@triton.jit
def _ws1_producer(x_ptr, writer, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    slot = writer.acquire(0)
    smem_ptrs = tle.gpu.local_ptr(slot.data, (tl.arange(0, BLOCK),))
    tl.store(smem_ptrs, x, mask=mask)
    writer.commit(0)


@triton.jit
def _ws1_consumer(out_ptr, reader, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    result = reader.wait(0)
    smem_ptrs = tle.gpu.local_ptr(result.slot.data, (tl.arange(0, BLOCK),))
    x = tl.load(smem_ptrs)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    tl.store(out_ptr + offs, x, mask=mask)
    reader.release(0)


@triton.jit
def _ws2_producer(x_ptr, writer, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    slot = writer.acquire(0)
    smem_ptrs = tle.gpu.local_ptr(slot.data, (tl.arange(0, BLOCK),))
    tl.store(smem_ptrs, x * 2.0, mask=mask)
    writer.commit(0)


@triton.jit
def _ws2_consumer(out_ptr, reader, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    result = reader.wait(0)
    smem_ptrs = tle.gpu.local_ptr(result.slot.data, (tl.arange(0, BLOCK),))
    x = tl.load(smem_ptrs)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    tl.store(out_ptr + offs, x, mask=mask)
    reader.release(0)


@triton.jit
def _multi_ws_kernel(a_ptr, b_ptr, c_ptr, d_ptr, numel,
                     BLOCK: tl.constexpr, NUM_STAGES: tl.constexpr):
    pid = tl.program_id(0)

    # First WS: copy a -> c
    smem1 = tle.gpu.alloc(
        (NUM_STAGES, BLOCK), dtype=tl.float32, layout=None,
        scope=tle.gpu.smem, nv_mma_shared_layout=False)
    p1 = tle.pipe(capacity=NUM_STAGES, data=smem1)
    w1 = p1.writer()
    r1 = p1.reader()
    tle.gpu.warp_specialize(
        [
            (_ws1_consumer, (c_ptr, r1, numel, tl.constexpr(BLOCK))),
            (_ws1_producer, (a_ptr, w1, numel, tl.constexpr(BLOCK))),
        ],
        [1], [8],
    )

    # Second WS: compute d = b * 2
    smem2 = tle.gpu.alloc(
        (NUM_STAGES, BLOCK), dtype=tl.float32, layout=None,
        scope=tle.gpu.smem, nv_mma_shared_layout=False)
    p2 = tle.pipe(capacity=NUM_STAGES, data=smem2)
    w2 = p2.writer()
    r2 = p2.reader()
    tle.gpu.warp_specialize(
        [
            (_ws2_consumer, (d_ptr, r2, numel, tl.constexpr(BLOCK))),
            (_ws2_producer, (b_ptr, w2, numel, tl.constexpr(BLOCK))),
        ],
        [1], [8],
    )


class TestTLEMultiWarpSpecialize:
    """Multiple WarpSpecializeOp regions in one kernel."""

    def test_multi_ws(self):
        """Two sequential TLE WS regions in the same kernel both execute correctly."""
        numel = BLOCK_SIZE
        a = torch.randn(numel, device=DEVICE, dtype=torch.float32)
        b = torch.randn(numel, device=DEVICE, dtype=torch.float32)
        c = torch.zeros_like(a)
        d = torch.zeros_like(b)

        _multi_ws_kernel[(1,)](a, b, c, d, numel, BLOCK=BLOCK_SIZE,
                               NUM_STAGES=2, num_warps=4)

        torch.testing.assert_close(c, a, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(d, b * 2.0, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    TestTLEExplicitWarpSpecialize().test_ws_producer_consumer_correctness()

