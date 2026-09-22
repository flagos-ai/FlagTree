"""
Async Copy in Gluon (Iluvatar SME)
==================================

Modern GPUs provide asynchronous instructions for long-running operations like
global memory reads and writes. Asynchronous operations allow overlapping memory
transactions with compute, also known as "pipelining".

On Iluvatar, async global→shared copies use the Streaming Memory Engine (SME).
Each SME transfer is a 2D tile (16 rows × 64 bytes). Source pointers need an SME
``IluvatarBlockedLayout`` (``is_sme=True``), destinations need
``IluvatarSwizzledSharedLayout(use_tcu=True)``, and ``stride`` must be 64-byte aligned.
"""

import pytest
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from triton.experimental.gluon import iluvatar as gil
from triton.experimental.gluon.iluvatar.ivcore11 import async_copy as cp
from triton._internal_testing import is_corex

if __name__ == "__main__" and not is_corex():
    raise RuntimeError("This tutorial requires Iluvatar corex")

# %%
# SME is 2D-only, so the introductory copy uses a small 2D tile instead of a
# 1D memcpy. Shared memory is still a descriptor with a layout chosen to match
# the SME / TCU shared encoding.
# Warp size is 64; fp32 SME tiles are 16×16, and sme_warps_per_cta=[2, 2]
# covers a 32×32 footprint (block dims must be multiples of 32). The engine takes
# each tile's address out of the pointer tensor, so size_per_thread ×
# threads_per_warp must be exactly one tile: a warp can only transfer rows whose
# addresses it holds.


@gluon.jit
def memcpy_2d_sme_kernel(in_ptr, out_ptr, M, N, stride_m, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr):
    pid = gl.program_id(0)
    sme_layout: gl.constexpr = gil.IluvatarBlockedLayout([1, 4], [16, 4], [2, 2], [1, 0], is_sme=True,
                                                         sme_warps_per_cta=[2, 2])
    # Column-contiguous register layout: consecutive threads cover consecutive
    # columns, so the global store is coalesced.
    reg: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [4, 1], [1, 0])
    smem_layout: gl.constexpr = gil.IluvatarSwizzledSharedLayout(vec=16, per_phase=1, max_phase=1, order=[1, 0],
                                                                 use_tcu=True)

    row = (pid * BLOCK_M) + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, sme_layout))
    col = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, sme_layout))
    smem = gl.allocate_shared_memory(gl.float32, [BLOCK_M, BLOCK_N], layout=smem_layout)

    cp.async_copy_global_to_shared(smem, in_ptr + row[:, None] * stride_m + col[None, :], stride=stride_m)
    cp.commit_group()
    cp.wait_group(0)

    value = smem.load(reg)
    out_row = (pid * BLOCK_M) + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, reg))
    out_col = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, reg))
    gl.store(out_ptr + out_row[:, None] * stride_m + out_col[None, :], value)


def memcpy_2d_sme(input, output, BLOCK_M=32, BLOCK_N=32, num_warps=4):
    assert input.shape == output.shape and input.stride(1) == 1
    grid = (triton.cdiv(input.shape[0], BLOCK_M), )
    memcpy_2d_sme_kernel[grid](input, output, *input.shape, input.stride(0), BLOCK_M, BLOCK_N, num_warps=num_warps)


@pytest.mark.parametrize("m, n", [(64, 64), (128, 64)])
@pytest.mark.skipif(not is_corex(), reason="Requires Iluvatar corex")
def test_memcpy_2d_sme(m, n):
    input = torch.randn(m, n, device="cuda")
    output = torch.empty_like(input)
    memcpy_2d_sme(input, output, BLOCK_M=32, BLOCK_N=n)
    torch.testing.assert_close(input, output, atol=0, rtol=0)


# %%
# You can see that we will be able to overlap the async copy with compute by
# issuing the copy and performing compute before waiting on it. Let's use an
# elementwise addition kernel to explore pipelining.
#
# First, let's write the kernel such that each program performs additions for
# the whole row, one block at a time. For simplicity, we will assume all inputs
# have the same global memory layout.


@gluon.jit
def elementwise_add_kernel(  #
        a_ptr, b_ptr, c_ptr, xnumel, ynumel,  #
        xstride_a, ystride_a, xstride_b, ystride_b, xstride_c, ystride_c,  #
        XBLOCK: gl.constexpr, YBLOCK: gl.constexpr,  #
):
    pid = gl.program_id(0)

    # Compute the offset to the row this program will process.
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [1, 4], [1, 0])
    xoffs = pid * XBLOCK + gl.arange(0, XBLOCK, gl.SliceLayout(1, layout))

    a_ptrs = a_ptr + xstride_a * xoffs[:, None]
    b_ptrs = b_ptr + xstride_b * xoffs[:, None]
    c_ptrs = c_ptr + xstride_c * xoffs[:, None]

    for yoff in range(0, ynumel, YBLOCK):
        # Offset to the column block.
        yoffs = yoff + gl.arange(0, YBLOCK, gl.SliceLayout(0, layout))
        mask = (xoffs < xnumel)[:, None] & (yoffs < ynumel)[None, :]

        a_val = gl.load(a_ptrs + ystride_a * yoffs[None, :], mask=mask)
        b_val = gl.load(b_ptrs + ystride_b * yoffs[None, :], mask=mask)

        c_val = a_val + b_val

        gl.store(c_ptrs + ystride_c * yoffs[None, :], c_val, mask=mask)


def elementwise_add(A, B, C, XBLOCK=32, YBLOCK=64):
    assert A.shape == B.shape == C.shape
    xnumel, ynumel = A.shape
    grid = (triton.cdiv(xnumel, XBLOCK), )
    return elementwise_add_kernel[grid](
        A, B, C, xnumel, ynumel,  #
        *A.stride(), *B.stride(), *C.stride(),  #
        XBLOCK, YBLOCK, num_warps=4)


@pytest.mark.parametrize("xnumel, ynumel", [(1024, 2048)])
@pytest.mark.parametrize("XBLOCK, YBLOCK", [(32, 32), (128, 128)])
def test_elementwise_add(xnumel, ynumel, XBLOCK, YBLOCK):
    a = torch.randn(xnumel, ynumel, device="cuda")
    b = torch.randn(xnumel, ynumel, device="cuda")
    c = torch.empty_like(a, device="cuda")
    elementwise_add(a, b, c, XBLOCK, YBLOCK)
    torch.testing.assert_close(a + b, c, atol=0, rtol=0)


# %%
# Let's rewrite the kernel to use SME async copies without pipelining, which will
# make it more obvious how we will pipeline the inner loop. Let's parameterize
# the kernel over the shared memory layout to see how it can affect performance.


@gluon.jit
def elementwise_add_sme_kernel(  #
        a_ptr, b_ptr, c_ptr, xnumel, ynumel,  #
        xstride_a, ystride_a, xstride_b, ystride_b, xstride_c, ystride_c,  #
        XBLOCK: gl.constexpr, YBLOCK: gl.constexpr,  #
        smem_layout: gl.constexpr,  #
):
    pid = gl.program_id(0)
    sme_layout: gl.constexpr = gil.IluvatarBlockedLayout([1, 4], [16, 4], [2, 2], [1, 0], is_sme=True,
                                                         sme_warps_per_cta=[2, 2])
    # Column-contiguous register layout keeps the global store coalesced.
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [4, 1], [1, 0])
    xoffs = pid * XBLOCK + gl.arange(0, XBLOCK, gl.SliceLayout(1, sme_layout))
    a_ptrs = a_ptr + xstride_a * xoffs[:, None]
    b_ptrs = b_ptr + xstride_b * xoffs[:, None]

    # New: declare shared memory for the A tile and B tile.
    dtype: gl.constexpr = a_ptr.dtype.element_ty
    a_smem = gl.allocate_shared_memory(dtype, [XBLOCK, YBLOCK], layout=smem_layout)
    b_smem = gl.allocate_shared_memory(dtype, [XBLOCK, YBLOCK], layout=smem_layout)

    for yoff in range(0, ynumel, YBLOCK):
        yoffs = yoff + gl.arange(0, YBLOCK, gl.SliceLayout(0, sme_layout))

        # Issue loads for both A and B tiles. SME needs an explicit row stride.
        # This tutorial uses full tiles only: a dynamic per-element mask would
        # drain outstanding SME copies and patch shared memory, which prevents
        # useful overlap with compute.
        cp.async_copy_global_to_shared(a_smem, a_ptrs + ystride_a * yoffs[None, :], stride=xstride_a)
        cp.async_copy_global_to_shared(b_smem, b_ptrs + ystride_b * yoffs[None, :], stride=xstride_b)
        # Commit both loads to the same group.
        cp.commit_group()
        # Wait until both loads are complete!
        cp.wait_group(0)

        a_val = a_smem.load(layout)
        b_val = b_smem.load(layout)

        c_val = a_val + b_val

        out_x = pid * XBLOCK + gl.arange(0, XBLOCK, gl.SliceLayout(1, layout))
        out_y = yoff + gl.arange(0, YBLOCK, gl.SliceLayout(0, layout))
        gl.store(c_ptr + xstride_c * out_x[:, None] + ystride_c * out_y[None, :], c_val)


def elementwise_add_sme(A, B, C, smem_layout, XBLOCK=32, YBLOCK=64):
    assert A.shape == B.shape == C.shape
    xnumel, ynumel = A.shape
    # Full tiles only, so the SME copies don't need per-element masks.
    assert xnumel % XBLOCK == 0 and ynumel % YBLOCK == 0
    grid = (triton.cdiv(xnumel, XBLOCK), )
    return elementwise_add_sme_kernel[grid](
        A, B, C, xnumel, ynumel,  #
        *A.stride(), *B.stride(), *C.stride(),  #
        XBLOCK, YBLOCK, smem_layout, num_warps=4)


@pytest.mark.parametrize("xnumel, ynumel", [(1024, 2048)])
@pytest.mark.parametrize("XBLOCK, YBLOCK", [(32, 32), (128, 128)])
@pytest.mark.skipif(not is_corex(), reason="Requires Iluvatar corex")
def test_elementwise_add_sme(xnumel, ynumel, XBLOCK, YBLOCK):
    a = torch.randn(xnumel, ynumel, device="cuda")
    b = torch.randn(xnumel, ynumel, device="cuda")
    c = torch.empty_like(a, device="cuda")
    smem_layout = gil.IluvatarSwizzledSharedLayout(vec=16, per_phase=1, max_phase=1, order=[1, 0], use_tcu=True)
    elementwise_add_sme(a, b, c, smem_layout, XBLOCK, YBLOCK)
    torch.testing.assert_close(a + b, c, atol=0, rtol=0)


def get_throughput(ms, C):
    # Because this kernel is memory-bound, we will measure bandwidth.
    tbytes = 3 * C.numel() * C.element_size() / 1024**4
    return tbytes / (ms * 1e-3)


if __name__ == "__main__":
    print("Benchmarking elementwise_add")
    print("============================")
    xnumel, ynumel = 32 * 1024, 32 * 1024
    A = torch.randn(xnumel, ynumel, device="cuda")
    B = torch.randn(xnumel, ynumel, device="cuda")
    C = torch.empty_like(A, device="cuda")

    ms = triton.testing.do_bench(lambda: elementwise_add(A, B, C))
    print(f"elementwise_add: {get_throughput(ms, C):.2f} TB/s")

    smem_layout = gil.IluvatarSwizzledSharedLayout(vec=16, per_phase=1, max_phase=1, order=[1, 0], use_tcu=True)
    ms = triton.testing.do_bench(lambda: elementwise_add_sme(A, B, C, smem_layout))
    print(f"elementwise_add_sme: {get_throughput(ms, C):.2f} TB/s")

# %%
# ```
# elementwise_add: 0.27 TB/s
# elementwise_add_sme: 0.53 TB/s
# ```
#
# Even without software pipelining, the SME version is already much faster.
# The register layout used after ``smem.load`` matters as much as the copy
# itself: a column-contiguous blocked layout keeps the global store coalesced.
# ``use_tcu=True`` selects the shared encoding that matches SME's hardware write.

# %%
# Software pipelining is an optimization technique for hiding the latencies of
# operations that execute asynchronously with respect to each other. If we
# prefetch the loads of the next operands before the current add, we can overlap
# it with the add and store. This requires multi-buffering shared memory, so it
# can be used by both the load and the add at the same time.
#
# Based on the relative latencies of the operations, we can determine the
# "pipeline depth". This is the number of prefetched loads in-flight. For
# example, if a load takes 3 times as long as the add, we should pipeline with
# depth 3 so each load has time to complete before the operands are needed.


@gluon.jit
def issue_loads(copy_idx, a_smem, b_smem, a_ptrs, xstride_a, ystride_a, b_ptrs, xstride_b, y_idx, ystride_b,
                YBLOCK: gl.constexpr, num_buffers: gl.constexpr):
    # Full tiles only (see the non-pipelined kernel): avoid dynamic masks here.
    yoffs = copy_idx * YBLOCK + y_idx
    cp.async_copy_global_to_shared(a_smem.index(copy_idx % num_buffers),  #
                                   a_ptrs + ystride_a * yoffs[None, :], stride=xstride_a)
    cp.async_copy_global_to_shared(b_smem.index(copy_idx % num_buffers),  #
                                   b_ptrs + ystride_b * yoffs[None, :], stride=xstride_b)
    cp.commit_group()
    return copy_idx + 1


@gluon.jit
def perform_add(read_idx, a_smem, b_smem, c_ptr, xstride_c, ystride_c, y_idx, x_idx, YBLOCK: gl.constexpr,
                num_buffers: gl.constexpr, layout: gl.constexpr):
    a_val = a_smem.index(read_idx % num_buffers).load(layout)
    b_val = b_smem.index(read_idx % num_buffers).load(layout)
    c_val = a_val + b_val
    yoffs = read_idx * YBLOCK + y_idx
    gl.store(c_ptr + xstride_c * x_idx[:, None] + ystride_c * yoffs[None, :], c_val)
    return read_idx + 1


@gluon.jit
def elementwise_add_pipelined_kernel(  #
        a_ptr, b_ptr, c_ptr, xnumel, ynumel,  #
        xstride_a, ystride_a, xstride_b, ystride_b, xstride_c, ystride_c,  #
        XBLOCK: gl.constexpr, YBLOCK: gl.constexpr,  #
        smem_layout: gl.constexpr, num_buffers: gl.constexpr,  #
):
    pid = gl.program_id(0)
    sme_layout: gl.constexpr = gil.IluvatarBlockedLayout([1, 4], [16, 4], [2, 2], [1, 0], is_sme=True,
                                                         sme_warps_per_cta=[2, 2])
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [4, 1], [1, 0])
    xoffs = pid * XBLOCK + gl.arange(0, XBLOCK, gl.SliceLayout(1, sme_layout))
    a_ptrs = a_ptr + xstride_a * xoffs[:, None]
    b_ptrs = b_ptr + xstride_b * xoffs[:, None]

    y_idx = gl.arange(0, YBLOCK, gl.SliceLayout(0, sme_layout))
    x_idx = pid * XBLOCK + gl.arange(0, XBLOCK, gl.SliceLayout(1, layout))
    y_idx_reg = gl.arange(0, YBLOCK, gl.SliceLayout(0, layout))

    # New: declare multi-buffered shared memory by adding a pipelining dimension
    # to the descriptors.
    dtype: gl.constexpr = a_ptr.dtype.element_ty
    a_smem = gl.allocate_shared_memory(dtype, [num_buffers, XBLOCK, YBLOCK], layout=smem_layout)
    b_smem = gl.allocate_shared_memory(dtype, [num_buffers, XBLOCK, YBLOCK], layout=smem_layout)
    copy_idx = 0
    read_idx = 0

    # Peel the `num_buffers-1` iterations from the inner loop to prefetch the
    # first set of copies, filling our pipeline.
    for _ in gl.static_range(num_buffers - 1):
        copy_idx = issue_loads(copy_idx, a_smem, b_smem, a_ptrs, xstride_a, ystride_a, b_ptrs, xstride_b, y_idx,
                               ystride_b, YBLOCK, num_buffers)

    # Inner loop iterations with overlapped copies and compute. This is the
    # steady state of the pipeline.
    for _ in range(gl.cdiv(ynumel, YBLOCK) - (num_buffers - 1)):
        # Issue the overlapped copy.
        copy_idx = issue_loads(copy_idx, a_smem, b_smem, a_ptrs, xstride_a, ystride_a, b_ptrs, xstride_b, y_idx,
                               ystride_b, YBLOCK, num_buffers)

        # Wait for `num_buffers-1` copies to complete, which is the last issued
        # copy. We can process that buffer.
        cp.wait_group(num_buffers - 1)
        read_idx = perform_add(read_idx, a_smem, b_smem, c_ptr, xstride_c, ystride_c, y_idx_reg, x_idx, YBLOCK,
                               num_buffers, layout)

    # Peeled iterations to drain the pipeline.
    for i in gl.static_range(num_buffers - 1):
        cp.wait_group(num_buffers - 2 - i)
        read_idx = perform_add(read_idx, a_smem, b_smem, c_ptr, xstride_c, ystride_c, y_idx_reg, x_idx, YBLOCK,
                               num_buffers, layout)


def elementwise_add_pipelined(A, B, C, XBLOCK=32, YBLOCK=64, num_buffers=2):
    assert A.shape == B.shape == C.shape
    xnumel, ynumel = A.shape
    # Full tiles only, and enough y-blocks to fill the prologue's prefetches.
    assert xnumel % XBLOCK == 0 and ynumel % YBLOCK == 0
    assert triton.cdiv(ynumel, YBLOCK) >= num_buffers - 1
    grid = (triton.cdiv(xnumel, XBLOCK), )
    smem_layout = gil.IluvatarSwizzledSharedLayout(vec=16, per_phase=1, max_phase=1, order=[1, 0], use_tcu=True)
    return elementwise_add_pipelined_kernel[grid](
        A, B, C, xnumel, ynumel,  #
        *A.stride(), *B.stride(), *C.stride(),  #
        XBLOCK, YBLOCK, smem_layout, num_buffers, num_warps=4)


@pytest.mark.parametrize("xnumel, ynumel", [(1024, 2048), (4096, 128)])
@pytest.mark.parametrize("XBLOCK, YBLOCK", [(32, 64)])
@pytest.mark.parametrize("num_buffers", [1, 2, 3])
@pytest.mark.skipif(not is_corex(), reason="Requires Iluvatar corex")
def test_elementwise_add_pipelined(xnumel, ynumel, XBLOCK, YBLOCK, num_buffers):
    a = torch.randn(xnumel, ynumel, device="cuda")
    b = torch.randn(xnumel, ynumel, device="cuda")
    c = torch.empty_like(a, device="cuda")
    elementwise_add_pipelined(a, b, c, XBLOCK, YBLOCK, num_buffers)
    torch.testing.assert_close(a + b, c, atol=0, rtol=0)


if __name__ == "__main__":
    ms = triton.testing.do_bench(lambda: elementwise_add_pipelined(A, B, C, num_buffers=2))
    print(f"elementwise_add_pipelined (double buffer): {get_throughput(ms, C):.2f} TB/s")
    ms = triton.testing.do_bench(lambda: elementwise_add_pipelined(A, B, C, num_buffers=3))
    print(f"elementwise_add_pipelined (triple buffer): {get_throughput(ms, C):.2f} TB/s")

# %%
# ```
# elementwise_add_pipelined (double buffer): 0.53 TB/s
# elementwise_add_pipelined (triple buffer): 0.53 TB/s
# ```
#
# Pipelining with SME async copy overlaps the next tile's g2s with the current
# add/store. On this memory-bound kernel, double and triple buffering match the
# non-pipelined SME version: once the copy is asynchronous, extra depth does not
# buy more bandwidth.
#
# Main takeaways:
#
# - Asynchronous instructions allow overlapping memory operations with compute.
# - Iluvatar SME async copies are 2D, need ``is_sme`` / ``use_tcu`` layouts and a
#   64-byte-aligned stride, and are tracked with commit groups
#   (``wait_group`` → hardware G2S waitcnt).
# - Software pipelining overlaps async copies with later iterations; more buffers
#   help only when copy latency still exceeds the compute between waits.
# - Register and shared layouts matter as much as the async API: prefer a
#   coalesced register layout for the global store after ``smem.load``.
