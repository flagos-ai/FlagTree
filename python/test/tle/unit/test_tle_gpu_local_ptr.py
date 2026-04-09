# flagtree tle
"""
Unit test covering a minimal Triton kernel that stages data through
TLE local pointers before writing results back to global memory.
"""

import re

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

BLOCK_SIZE = 64


def _require_cuda():
    try:
        torch.cuda.init()
    except Exception as exc:
        pytest.skip(f"CUDA init failed: {exc}")


@pytest.fixture(scope="module", autouse=True)
def _cuda_guard():
    _require_cuda()


@triton.jit
def _local_pointer_axpy_kernel(x_ptr, y_ptr, out_ptr, numel, alpha, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel

    smem_tile = tle.gpu.alloc([BLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    smem_ptrs = tle.gpu.local_ptr(smem_tile, (tl.arange(0, BLOCK), ))

    x_tile = x_ptr + offsets
    y_tile = y_ptr + offsets
    out_tile = out_ptr + offsets

    x_vals = tl.load(x_tile, mask=mask, other=0.0)
    tl.store(smem_ptrs, x_vals, mask=mask)

    shared_values = tl.load(smem_ptrs, mask=mask, other=0.0)
    y_values = tl.load(y_tile, mask=mask, other=0.0)
    updated = shared_values * alpha + y_values

    tl.store(smem_ptrs, updated, mask=mask)
    out_vals = tl.load(smem_ptrs, mask=mask, other=0.0)
    tl.store(out_tile, out_vals, mask=mask)


@triton.jit
def _local_pointer_store_kernel(out_ptr, numel, value, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel

    smem_tile = tle.gpu.alloc([BLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    smem_ptrs = tle.gpu.local_ptr(smem_tile, (tl.arange(0, BLOCK), ))

    init = tl.full((BLOCK, ), value, tl.float32)
    tl.store(smem_ptrs, init, mask=mask)
    out_tile = out_ptr + offsets
    out_vals = tl.load(smem_ptrs, mask=mask, other=0.0)
    tl.store(out_tile, out_vals, mask=mask)


@triton.jit
def _local_pointer_full_view_store_kernel(out_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel

    smem_tile = tle.gpu.alloc([BLOCK], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    smem_ptrs = tle.gpu.local_ptr(smem_tile)

    vals = tl.arange(0, BLOCK)
    tl.store(smem_ptrs, vals)

    out_vals = tl.load(smem_ptrs, mask=mask, other=-1)
    tl.store(out_ptr + offsets, out_vals, mask=mask)


@triton.jit
def _local_pointer_local_load_none_kernel(out_ptr, BLOCK: tl.constexpr):
    idx = tl.arange(0, BLOCK)
    smem_tile = tle.gpu.alloc([BLOCK], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    smem_ptrs = tle.gpu.local_ptr(smem_tile)
    tl.store(smem_ptrs, idx + 3)
    vals = tl.load(smem_ptrs)
    tl.store(out_ptr + idx, vals)


@triton.jit
def _local_pointer_local_load_full_indices_kernel(out_ptr, BLOCK: tl.constexpr):
    idx = tl.arange(0, BLOCK)
    smem_tile = tle.gpu.alloc([BLOCK], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    smem_ptrs = tle.gpu.local_ptr(smem_tile, (idx, ))
    tl.store(smem_ptrs, idx + 5)
    vals = tl.load(smem_ptrs)
    tl.store(out_ptr + idx, vals)


@triton.jit
def _local_pointer_full_view_2d_copy_kernel(
    x_ptr,
    out_ptr,
    stride_xm,
    stride_xn,
    stride_om,
    stride_on,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    smem = tle.gpu.alloc([ROWS, COLS], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    rows = tl.arange(0, ROWS)[:, None]
    cols = tl.arange(0, COLS)[None, :]
    x_tile = x_ptr + rows * stride_xm + cols * stride_xn
    tle.gpu.copy(x_tile, smem, [ROWS, COLS])

    full_ptrs = tle.gpu.local_ptr(smem)
    vals = tl.load(full_ptrs)

    out_tile = out_ptr + rows * stride_om + cols * stride_on
    tl.store(out_tile, vals)


@triton.jit
def _local_pointer_conditional_mask_store_kernel(out_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    idx = tl.arange(0, BLOCK)
    mask = idx < numel

    smem = tle.gpu.alloc([BLOCK], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    ptrs = tle.gpu.local_ptr(smem, (idx, ))

    # Keep the masked store inside an scf.if region. This used to trigger a
    # verifier failure in `triton-tle-select-encodings` because the
    # store mask did not match the pointer layout.
    if pid == 0:
        tl.store(ptrs, idx, mask=mask)

    vals = tl.load(ptrs, mask=mask, other=-1)
    tl.store(out_ptr + idx, vals, mask=mask)


@triton.jit
def _local_pointer_looped_elementwise_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    numel,
    alpha,
    BLOCK: tl.constexpr,
    CHUNKS: tl.constexpr,
    SLICES: tl.constexpr,
    SLICE_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * BLOCK * CHUNKS

    smem_tile = tle.gpu.alloc([BLOCK], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    smem_ptrs = tle.gpu.local_ptr(smem_tile, (tl.arange(0, BLOCK), ))
    assert BLOCK % SLICE_SIZE == 0, "BLOCK must be divisible by SLICE_SIZE"
    slice_indices = tl.arange(0, SLICE_SIZE)

    for chunk in range(CHUNKS):
        offsets = base + chunk * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < numel
        x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        tl.store(smem_ptrs, x_vals, mask=mask)

        for slice_idx in range(SLICES):
            block_offset = slice_idx * SLICE_SIZE
            slice_ptr = tle.gpu.local_ptr(
                smem_tile,
                (block_offset + slice_indices, ),
            )
            slice_offsets = base + chunk * BLOCK + block_offset + slice_indices
            slice_mask = slice_offsets < numel
            shared_vals = tl.load(slice_ptr, mask=slice_mask, other=0.0)
            y_vals = tl.load(y_ptr + slice_offsets, mask=slice_mask, other=0.0)
            updated = shared_vals * alpha + y_vals
            tl.store(slice_ptr, updated, mask=slice_mask)

        out_vals = tl.load(smem_ptrs, mask=mask, other=0.0)
        tl.store(out_ptr + offsets, out_vals, mask=mask)


@triton.jit
def _local_pointer_tiled_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_K_TILES: tl.constexpr,
    SLICE_PARTS: tl.constexpr,
    SLICE_WIDTH: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    smem_a = tle.gpu.alloc([BLOCK_M, BLOCK_K], dtype=tl.float16, layout=None, scope=tle.gpu.smem,
                           nv_mma_shared_layout=False)
    smem_b = tle.gpu.alloc([BLOCK_K, BLOCK_N], dtype=tl.float16, layout=None, scope=tle.gpu.smem,
                           nv_mma_shared_layout=False)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    slice_parts = int(SLICE_PARTS)
    slice_width = int(SLICE_WIDTH)
    assert BLOCK_K % slice_parts == 0, "BLOCK_K must divide slice_parts"

    for k_tile in range(NUM_K_TILES):
        k_offsets = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)
        a_tile = a_ptr + offs_m[:, None] * stride_am + k_offsets[None, :] * stride_ak
        b_tile = b_ptr + k_offsets[:, None] * stride_bk + offs_n[None, :] * stride_bn
        tle.gpu.copy(a_tile, smem_a, [BLOCK_M, BLOCK_K])
        tle.gpu.copy(b_tile, smem_b, [BLOCK_K, BLOCK_N])

        for slice_idx in range(slice_parts):
            k_start = slice_idx * slice_width
            a_rows = tl.arange(0, BLOCK_M)[:, None]
            a_cols = tl.arange(0, SLICE_WIDTH)[None, :] + k_start
            a_rows = tl.broadcast_to(a_rows, (BLOCK_M, SLICE_WIDTH))
            a_cols = tl.broadcast_to(a_cols, (BLOCK_M, SLICE_WIDTH))
            a_slice = tle.gpu.local_ptr(smem_a, (a_rows, a_cols))

            b_rows = tl.arange(0, SLICE_WIDTH)[:, None] + k_start
            b_cols = tl.arange(0, BLOCK_N)[None, :]
            b_rows = tl.broadcast_to(b_rows, (SLICE_WIDTH, BLOCK_N))
            b_cols = tl.broadcast_to(b_cols, (SLICE_WIDTH, BLOCK_N))
            b_slice = tle.gpu.local_ptr(smem_b, (b_rows, b_cols))
            a_vals = tl.load(a_slice)
            b_vals = tl.load(b_slice)
            acc += tl.dot(a_vals, b_vals, out_dtype=tl.float32)

    c_tile = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_tile, acc)


@triton.jit
def _local_pointer_axis_gather_kernel(
    x_ptr,
    out_ptr,
    stride_xm,
    stride_xn,
    stride_om,
    stride_on,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    SLICE: tl.constexpr,
):
    smem = tle.gpu.alloc([ROWS, COLS], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    offs_m = tl.arange(0, ROWS)[:, None]
    offs_n = tl.arange(0, COLS)[None, :]
    x_tile = x_ptr + offs_m * stride_xm + offs_n * stride_xn
    tle.gpu.copy(x_tile, smem, [ROWS, COLS])

    row_ids = tl.broadcast_to(offs_m, (ROWS, SLICE))
    col_ids = tl.broadcast_to(1 + tl.arange(0, SLICE)[None, :], (ROWS, SLICE))
    smem_slice = tle.gpu.local_ptr(smem, (row_ids, col_ids))
    vals = tl.load(smem_slice)

    out_tile = out_ptr + offs_m * stride_om + tl.arange(0, SLICE)[None, :] * stride_on
    tl.store(out_tile, vals)


@triton.jit
def _local_pointer_dynamic_scalar_load_after_vector_store_kernel(
    out_ptr,
    BLOCK: tl.constexpr,
):
    smem = tle.gpu.alloc([BLOCK], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    vec_idx = tl.arange(0, BLOCK)
    vec_ptr = tle.gpu.local_ptr(smem, (vec_idx, ))
    tl.store(vec_ptr, vec_idx + 1)
    zero = tl.program_id(0) * 0

    for i in range(BLOCK):
        scalar_idx = zero + i
        scalar_ptr = tle.gpu.local_ptr(smem, (scalar_idx, ))
        scalar_val = tl.load(scalar_ptr)
        tl.store(out_ptr + i, scalar_val)


@triton.jit
def _local_pointer_full_view_dot_kernel(
    a_ptr,
    out_ptr,
    stride_ai,
    stride_aj,
    stride_oi,
    stride_oj,
    BLOCK: tl.constexpr,
):
    offs_i = tl.arange(0, BLOCK)[:, None]
    offs_j = tl.arange(0, BLOCK)[None, :]

    a_tile_ptr = a_ptr + offs_i * stride_ai + offs_j * stride_aj
    a_tile = tl.load(a_tile_ptr)

    smem = tle.gpu.alloc([BLOCK, BLOCK], dtype=tl.float16, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    smem_ptr = tle.gpu.local_ptr(smem)
    tl.store(smem_ptr, a_tile)

    staged = tl.load(smem_ptr)
    acc = tl.dot(staged, tl.trans(staged), out_dtype=tl.float32)

    out_ptrs = out_ptr + offs_i * stride_oi + offs_j * stride_oj
    tl.store(out_ptrs, acc.to(tl.float16))


class TestTLELocalPointerKernel:
    """Ensure kernels can perform load/compute/store entirely via local pointers."""

    def test_local_pointer_axpy_matches_torch(self):
        torch.manual_seed(0)
        numel = BLOCK_SIZE * 4
        alpha = 1.5

        x = torch.randn(numel, device="cuda", dtype=torch.float32)
        y = torch.randn_like(x)
        out = torch.empty_like(x)

        grid = (triton.cdiv(numel, BLOCK_SIZE), )
        _local_pointer_axpy_kernel[grid](x, y, out, numel, alpha, BLOCK_SIZE)

        expected = alpha * x + y
        torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)

    def test_local_pointer_store_populates_constant(self):
        numel = BLOCK_SIZE * 4
        value = 2.25
        out = torch.empty(numel, device="cuda", dtype=torch.float32)

        grid = (triton.cdiv(numel, BLOCK_SIZE), )
        _local_pointer_store_kernel[grid](out, numel, value, BLOCK_SIZE)

        expected = torch.full_like(out, value)
        torch.testing.assert_close(out, expected, atol=1e-7, rtol=0)

    def test_local_pointer_none_generates_full_view_1d(self):
        block = 128
        numel = block - 9
        out = torch.full((block, ), -1, device="cuda", dtype=torch.int32)

        compiled = _local_pointer_full_view_store_kernel.warmup(
            out,
            numel,
            BLOCK=block,
            grid=(1, ),
            num_warps=4,
        )
        ttgir = compiled.asm["ttgir"]
        assert "ttg.local_store" in ttgir
        line = next((line for line in ttgir.splitlines() if "tle.gpu.local_pointers" in line), None)
        if line is not None:
            line_lhs = line.split(":", 1)[0]
            assert "tle.gpu.local_pointers" in line_lhs
            assert "," not in line_lhs

        _local_pointer_full_view_store_kernel[(1, )](
            out,
            numel,
            BLOCK=block,
            num_warps=4,
        )
        expected = torch.arange(block, device="cuda", dtype=torch.int32)
        expected[numel:] = -1
        torch.testing.assert_close(out, expected, atol=0, rtol=0)

    def test_local_pointer_none_generates_full_view_2d(self):
        rows = 16
        cols = 32
        x = torch.randn((rows, cols), device="cuda", dtype=torch.float32)
        out = torch.empty_like(x)

        _local_pointer_full_view_2d_copy_kernel[(1, )](
            x,
            out,
            x.stride(0),
            x.stride(1),
            out.stride(0),
            out.stride(1),
            ROWS=rows,
            COLS=cols,
        )
        torch.testing.assert_close(out, x, atol=1e-6, rtol=1e-6)

    def test_local_pointer_none_load_rewrites_to_local_load(self):
        block = 64
        out = torch.empty((block, ), device="cuda", dtype=torch.int32)
        compiled = _local_pointer_local_load_none_kernel.warmup(
            out,
            BLOCK=block,
            grid=(1, ),
            num_warps=4,
        )
        ttgir = compiled.asm["ttgir"]
        assert "ttg.local_load" in ttgir

        _local_pointer_local_load_none_kernel[(1, )](out, BLOCK=block, num_warps=4)
        expected = torch.arange(block, device="cuda", dtype=torch.int32) + 3
        torch.testing.assert_close(out, expected, atol=0, rtol=0)

    def test_local_pointer_full_indices_load_rewrites_to_local_load(self):
        block = 64
        out = torch.empty((block, ), device="cuda", dtype=torch.int32)
        compiled = _local_pointer_local_load_full_indices_kernel.warmup(
            out,
            BLOCK=block,
            grid=(1, ),
            num_warps=4,
        )
        ttgir = compiled.asm["ttgir"]
        assert "ttg.local_load" in ttgir

        _local_pointer_local_load_full_indices_kernel[(1, )](out, BLOCK=block, num_warps=4)
        expected = torch.arange(block, device="cuda", dtype=torch.int32) + 5
        torch.testing.assert_close(out, expected, atol=0, rtol=0)

    def test_local_pointer_conditional_mask_store_compiles(self):
        block = 512
        numel = block - 7
        out = torch.full((block, ), -1, device="cuda", dtype=torch.int32)

        compiled = _local_pointer_conditional_mask_store_kernel.warmup(
            out,
            numel,
            BLOCK=block,
            grid=(1, ),
            num_warps=8,
        )
        assert compiled is not None

        _local_pointer_conditional_mask_store_kernel[(1, )](
            out,
            numel,
            BLOCK=block,
            num_warps=8,
        )
        expected = torch.arange(block, device="cuda", dtype=torch.int32)
        expected[numel:] = -1
        torch.testing.assert_close(out, expected, atol=0, rtol=0)

    def test_local_pointer_looped_elementwise_matches_torch(self):
        chunks = 4
        numel = BLOCK_SIZE * chunks * 3
        alpha = 0.75

        x = torch.randn(numel, device="cuda", dtype=torch.float32)
        y = torch.randn_like(x)
        out = torch.empty_like(x)

        slices = 4
        slice_size = BLOCK_SIZE // slices
        grid = (triton.cdiv(numel, BLOCK_SIZE * chunks), )
        _local_pointer_looped_elementwise_kernel[grid](x, y, out, numel, alpha, BLOCK_SIZE, chunks, slices, slice_size)

        expected = alpha * x + y
        torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)

    def test_local_pointer_tiled_matmul_matches_torch(self):
        block_m = 32
        block_n = 32
        block_k = 32
        num_k_tiles = 2
        m = block_m
        n = block_n
        k = block_k * num_k_tiles

        a = torch.randn((m, k), device="cuda", dtype=torch.float16)
        b = torch.randn((k, n), device="cuda", dtype=torch.float16)
        c = torch.empty((m, n), device="cuda", dtype=torch.float32)

        slice_parts = 2
        slice_width = block_k // slice_parts
        grid = (m // block_m, n // block_n)
        _local_pointer_tiled_matmul_kernel[grid](
            a,
            b,
            c,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            block_m,
            block_n,
            block_k,
            num_k_tiles,
            slice_parts,
            slice_width,
        )

        expected = (a.float()) @ (b.float())
        torch.testing.assert_close(c, expected, atol=5e-3, rtol=5e-3)

    def test_local_pointer_axis_gather_matches_torch(self):
        rows = 8
        cols = 8
        slice_width = 4
        x = torch.arange(rows * cols, device="cuda", dtype=torch.float32).reshape(rows, cols)
        out = torch.empty((rows, slice_width), device="cuda", dtype=torch.float32)

        grid = (1, )
        _local_pointer_axis_gather_kernel[grid](
            x,
            out,
            x.stride(0),
            x.stride(1),
            out.stride(0),
            out.stride(1),
            ROWS=rows,
            COLS=cols,
            SLICE=slice_width,
        )

        expected = x[:, 1:1 + slice_width]
        torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)

    def test_local_pointer_scalar_dynamic_index_inserts_barrier(self):
        block = 64
        out = torch.empty((block, ), device="cuda", dtype=torch.int32)
        compiled = _local_pointer_dynamic_scalar_load_after_vector_store_kernel.warmup(
            out,
            BLOCK=block,
            grid=(1, ),
            num_warps=4,
        )
        ttgir = compiled.asm["ttgir"]
        assert "gpu.barrier" in ttgir
        assert "\"tt.reduce\"" not in ttgir

        _local_pointer_dynamic_scalar_load_after_vector_store_kernel[(1, )](
            out,
            BLOCK=block,
            num_warps=4,
        )
        torch.testing.assert_close(
            out,
            torch.arange(1, block + 1, device="cuda", dtype=torch.int32),
            atol=0,
            rtol=0,
        )

    def test_local_pointer_full_view_dot_avoids_pointer_convert_layout(self):
        block = 32
        a = torch.randn((block, block), device="cuda", dtype=torch.float16)
        out = torch.empty_like(a)

        compiled = _local_pointer_full_view_dot_kernel.warmup(
            a,
            out,
            a.stride(0),
            a.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK=block,
            grid=(1, ),
            num_warps=4,
            num_stages=2,
        )
        ttgir = compiled.asm["ttgir"]
        assert "ttg.local_load" in ttgir
        assert re.search(r"ttg\\.convert_layout .*-> tensor<.*!tt\\.ptr", ttgir) is None

        _local_pointer_full_view_dot_kernel[(1, )](
            a,
            out,
            a.stride(0),
            a.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK=block,
            num_warps=4,
            num_stages=2,
        )
        expected = (a.float() @ a.float().T).to(torch.float16)
        torch.testing.assert_close(out, expected, atol=2e-1, rtol=2e-1)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
