# Copyright (c) 2025  XCoreSigma Inc. All rights reserved.
# flagtree tle
"""
Unit test covering a minimal Triton kernel that stages data through
TLE local pointers before writing results back to global memory.
"""

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language.gpu as tle


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

    smem_tile = tle.alloc([BLOCK], dtype=tl.float32, layout=None, scope=tle.smem, nv_mma_shared_layout=False)
    smem_ptrs = tle.local_ptr(smem_tile)

    x_tile = x_ptr + offsets
    y_tile = y_ptr + offsets
    out_tile = out_ptr + offsets

    tle.copy(x_tile, smem_tile, [BLOCK])

    shared_values = tl.load(smem_ptrs, mask=mask, other=0.0)
    y_values = tl.load(y_tile, mask=mask, other=0.0)
    updated = shared_values * alpha + y_values

    tl.store(smem_ptrs, updated, mask=mask)
    tle.copy(smem_tile, out_tile, [BLOCK])


@triton.jit
def _local_pointer_store_kernel(out_ptr, numel, value, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel

    smem_tile = tle.alloc([BLOCK], dtype=tl.float32, layout=None, scope=tle.smem, nv_mma_shared_layout=False)
    smem_ptrs = tle.local_ptr(smem_tile)

    init = tl.full((BLOCK,), value, tl.float32)
    tl.store(smem_ptrs, init, mask=mask)
    out_tile = out_ptr + offsets
    tle.copy(smem_tile, out_tile, [BLOCK])


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

    smem_tile = tle.alloc([BLOCK], dtype=tl.float32, layout=None, scope=tle.smem, nv_mma_shared_layout=False)
    assert BLOCK % SLICE_SIZE == 0, "BLOCK must be divisible by SLICE_SIZE"
    slice_indices = tl.arange(0, SLICE_SIZE)

    for chunk in range(CHUNKS):
        offsets = base + chunk * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < numel
        tle.copy(x_ptr + offsets, smem_tile, [BLOCK])

        for slice_idx in range(SLICES):
            block_offset = slice_idx * SLICE_SIZE
            smem_offsets = (block_offset + slice_indices).to(tl.int32)
            slice_ptr = tle.local_ptr(
                smem_tile,
                offsets=smem_offsets,
            )
            slice_offsets = base + chunk * BLOCK + smem_offsets
            slice_mask = slice_offsets < numel
            shared_vals = tl.load(slice_ptr, mask=slice_mask, other=0.0)
            y_vals = tl.load(y_ptr + slice_offsets, mask=slice_mask, other=0.0)
            updated = shared_vals * alpha + y_vals
            tl.store(slice_ptr, updated, mask=slice_mask)

        tle.copy(smem_tile, out_ptr + offsets, [BLOCK])


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

    smem_a = tle.alloc([BLOCK_M, BLOCK_K], dtype=tl.float16, layout=None, scope=tle.smem, nv_mma_shared_layout=False)
    smem_b = tle.alloc([BLOCK_K, BLOCK_N], dtype=tl.float16, layout=None, scope=tle.smem, nv_mma_shared_layout=False)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    slice_parts = int(SLICE_PARTS)
    slice_width = int(SLICE_WIDTH)
    assert BLOCK_K % slice_parts == 0, "BLOCK_K must divide slice_parts"

    for k_tile in range(NUM_K_TILES):
        k_offsets = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)
        a_tile = a_ptr + offs_m[:, None] * stride_am + k_offsets[None, :] * stride_ak
        b_tile = b_ptr + k_offsets[:, None] * stride_bk + offs_n[None, :] * stride_bn
        tle.copy(a_tile, smem_a, [BLOCK_M, BLOCK_K])
        tle.copy(b_tile, smem_b, [BLOCK_K, BLOCK_N])

        for slice_idx in range(slice_parts):
            k_start = slice_idx * slice_width
            a_rows = tl.arange(0, BLOCK_M)[:, None]
            a_cols = tl.arange(0, SLICE_WIDTH)[None, :]
            a_offsets = (a_rows * BLOCK_K + k_start + a_cols).to(tl.int32)
            a_slice = tle.local_ptr(
                smem_a,
                offsets=a_offsets,
            )

            b_rows = (k_start + tl.arange(0, SLICE_WIDTH))[:, None]
            b_cols = tl.arange(0, BLOCK_N)[None, :]
            b_offsets = (b_rows * BLOCK_N + b_cols).to(tl.int32)
            b_slice = tle.local_ptr(
                smem_b,
                offsets=b_offsets,
            )
            a_vals = tl.load(a_slice)
            b_vals = tl.load(b_slice)
            acc += tl.dot(a_vals, b_vals, out_dtype=tl.float32)

    c_tile = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_tile, acc)


class TestTLELocalPointerKernel:
    """Ensure kernels can perform load/compute/store entirely via local pointers."""

    def test_local_pointer_axpy_matches_torch(self):
        torch.manual_seed(0)
        numel = BLOCK_SIZE * 4
        alpha = 1.5

        x = torch.randn(numel, device="cuda", dtype=torch.float32)
        y = torch.randn_like(x)
        out = torch.empty_like(x)

        grid = (triton.cdiv(numel, BLOCK_SIZE),)
        _local_pointer_axpy_kernel[grid](x, y, out, numel, alpha, BLOCK_SIZE)

        expected = alpha * x + y
        torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)

    def test_local_pointer_store_populates_constant(self):
        numel = BLOCK_SIZE * 4
        value = 2.25
        out = torch.empty(numel, device="cuda", dtype=torch.float32)

        grid = (triton.cdiv(numel, BLOCK_SIZE),)
        _local_pointer_store_kernel[grid](out, numel, value, BLOCK_SIZE)

        expected = torch.full_like(out, value)
        torch.testing.assert_close(out, expected, atol=1e-7, rtol=0)

    def test_local_pointer_looped_elementwise_matches_torch(self):
        chunks = 4
        numel = BLOCK_SIZE * chunks * 3
        alpha = 0.75

        x = torch.randn(numel, device="cuda", dtype=torch.float32)
        y = torch.randn_like(x)
        out = torch.empty_like(x)

        slices = 4
        slice_size = BLOCK_SIZE // slices
        grid = (triton.cdiv(numel, BLOCK_SIZE * chunks),)
        _local_pointer_looped_elementwise_kernel[grid](
            x, y, out, numel, alpha, BLOCK_SIZE, chunks, slices, slice_size
        )

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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
