import re

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.compiler.errors import CompilationError

from test_tle_utils import compile_musa, compile_to_ttir, require_mthreads_libtriton

require_mthreads_libtriton()


@triton.jit
def _local_ptr_subview_kernel(out_ptr, BLOCK: tl.constexpr):
    init = tl.arange(0, 64).to(tl.float32) + 1.0
    smem = tle.gpu.alloc((64, ), dtype=tl.float32, init_value=init, nv_mma_shared_layout=False)
    offsets = tl.arange(0, BLOCK) * 2
    ptrs = tle.gpu.local_ptr(smem, (offsets, ))
    values = tl.load(ptrs)
    tl.store(out_ptr + tl.arange(0, BLOCK), values)


@triton.jit
def _local_ptr_scalar_kernel(out_ptr):
    init = tl.full((16, ), 0.0, tl.float32)
    smem = tle.gpu.alloc((16, ), dtype=tl.float32, init_value=init, nv_mma_shared_layout=False)
    ptr = tle.gpu.local_ptr(smem, (5, ))
    tl.store(ptr, 42.0)
    value = tl.load(ptr)
    tl.store(out_ptr, value)


@triton.jit
def _local_ptr_full_view_kernel(out_ptr):
    smem = tle.gpu.alloc((16, ), dtype=tl.float32, nv_mma_shared_layout=False)
    values = tl.arange(0, 16).to(tl.float32) + 7.0
    ptrs = tle.gpu.local_ptr(smem)
    tl.store(ptrs, values)
    loaded = tl.load(ptrs)
    tl.store(out_ptr + tl.arange(0, 16), loaded)


@triton.jit
def _local_ptr_axpy_kernel(x_ptr, y_ptr, out_ptr, numel, alpha, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel

    smem = tle.gpu.alloc((BLOCK, ), dtype=tl.float32, nv_mma_shared_layout=False)
    ptrs = tle.gpu.local_ptr(smem, (tl.arange(0, BLOCK), ))

    x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(ptrs, x_vals, mask=mask)

    shared_values = tl.load(ptrs, mask=mask, other=0.0)
    y_values = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    updated = shared_values * alpha + y_values

    tl.store(ptrs, updated, mask=mask)
    out_vals = tl.load(ptrs, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, out_vals, mask=mask)


@triton.jit
def _local_ptr_constant_store_kernel(out_ptr, numel, value, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel

    smem = tle.gpu.alloc((BLOCK, ), dtype=tl.float32, nv_mma_shared_layout=False)
    ptrs = tle.gpu.local_ptr(smem, (tl.arange(0, BLOCK), ))

    init = tl.full((BLOCK, ), value, tl.float32)
    tl.store(ptrs, init, mask=mask)
    out_vals = tl.load(ptrs, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, out_vals, mask=mask)


@triton.jit
def _local_ptr_full_view_tail_mask_kernel(out_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel

    smem = tle.gpu.alloc((BLOCK, ), dtype=tl.int32, nv_mma_shared_layout=False)
    ptrs = tle.gpu.local_ptr(smem)

    vals = tl.arange(0, BLOCK)
    tl.store(ptrs, vals)

    out_vals = tl.load(ptrs, mask=mask, other=-1)
    tl.store(out_ptr + offsets, out_vals, mask=mask)


@triton.jit
def _local_ptr_full_view_2d_copy_kernel(x_ptr, out_ptr, stride_xm, stride_xn, stride_om, stride_on, ROWS: tl.constexpr,
                                        COLS: tl.constexpr):
    smem = tle.gpu.alloc((ROWS, COLS), dtype=tl.float32, nv_mma_shared_layout=False)
    rows = tl.arange(0, ROWS)[:, None]
    cols = tl.arange(0, COLS)[None, :]
    x_tile = x_ptr + rows * stride_xm + cols * stride_xn
    tle.gpu.copy(x_tile, smem, (ROWS, COLS))

    full_ptrs = tle.gpu.local_ptr(smem)
    vals = tl.load(full_ptrs)

    out_tile = out_ptr + rows * stride_om + cols * stride_on
    tl.store(out_tile, vals)


@triton.jit
def _local_ptr_local_load_none_kernel(out_ptr, BLOCK: tl.constexpr):
    idx = tl.arange(0, BLOCK)
    smem = tle.gpu.alloc((BLOCK, ), dtype=tl.int32, nv_mma_shared_layout=False)
    ptrs = tle.gpu.local_ptr(smem)
    tl.store(ptrs, idx + 3)
    vals = tl.load(ptrs)
    tl.store(out_ptr + idx, vals)


@triton.jit
def _local_ptr_local_load_full_indices_kernel(out_ptr, BLOCK: tl.constexpr):
    idx = tl.arange(0, BLOCK)
    smem = tle.gpu.alloc((BLOCK, ), dtype=tl.int32, nv_mma_shared_layout=False)
    ptrs = tle.gpu.local_ptr(smem, (idx, ))
    tl.store(ptrs, idx + 5)
    vals = tl.load(ptrs)
    tl.store(out_ptr + idx, vals)


@triton.jit
def _local_ptr_conditional_mask_store_kernel(out_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    idx = tl.arange(0, BLOCK)
    mask = idx < numel

    smem = tle.gpu.alloc((BLOCK, ), dtype=tl.int32, nv_mma_shared_layout=False)
    ptrs = tle.gpu.local_ptr(smem, (idx, ))

    if pid == 0:
        tl.store(ptrs, idx, mask=mask)

    vals = tl.load(ptrs, mask=mask, other=-1)
    tl.store(out_ptr + idx, vals, mask=mask)


@triton.jit
def _local_ptr_looped_elementwise_kernel(x_ptr, y_ptr, out_ptr, numel, alpha, BLOCK: tl.constexpr, CHUNKS: tl.constexpr,
                                         SLICES: tl.constexpr, SLICE_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * BLOCK * CHUNKS

    smem = tle.gpu.alloc((BLOCK, ), dtype=tl.float32, nv_mma_shared_layout=False)
    ptrs = tle.gpu.local_ptr(smem, (tl.arange(0, BLOCK), ))
    assert BLOCK % SLICE_SIZE == 0, "BLOCK must be divisible by SLICE_SIZE"
    slice_indices = tl.arange(0, SLICE_SIZE)

    for chunk in range(CHUNKS):
        offsets = base + chunk * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < numel
        x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        tl.store(ptrs, x_vals, mask=mask)

        for slice_idx in range(SLICES):
            block_offset = slice_idx * SLICE_SIZE
            slice_ptr = tle.gpu.local_ptr(smem, (block_offset + slice_indices, ))
            slice_offsets = base + chunk * BLOCK + block_offset + slice_indices
            slice_mask = slice_offsets < numel
            shared_vals = tl.load(slice_ptr, mask=slice_mask, other=0.0)
            y_vals = tl.load(y_ptr + slice_offsets, mask=slice_mask, other=0.0)
            updated = shared_vals * alpha + y_vals
            tl.store(slice_ptr, updated, mask=slice_mask)

        out_vals = tl.load(ptrs, mask=mask, other=0.0)
        tl.store(out_ptr + offsets, out_vals, mask=mask)


@triton.jit
def _local_ptr_axis_gather_kernel(x_ptr, out_ptr, stride_xm, stride_xn, stride_om, stride_on, ROWS: tl.constexpr,
                                  COLS: tl.constexpr, SLICE: tl.constexpr):
    smem = tle.gpu.alloc((ROWS, COLS), dtype=tl.float32, nv_mma_shared_layout=False)
    offs_m = tl.arange(0, ROWS)[:, None]
    offs_n = tl.arange(0, COLS)[None, :]
    x_tile = x_ptr + offs_m * stride_xm + offs_n * stride_xn
    tle.gpu.copy(x_tile, smem, (ROWS, COLS))

    row_ids = tl.broadcast_to(offs_m, (ROWS, SLICE))
    col_ids = tl.broadcast_to(1 + tl.arange(0, SLICE)[None, :], (ROWS, SLICE))
    slice_ptrs = tle.gpu.local_ptr(smem, (row_ids, col_ids))
    vals = tl.load(slice_ptrs)

    out_tile = out_ptr + offs_m * stride_om + tl.arange(0, SLICE)[None, :] * stride_on
    tl.store(out_tile, vals)


@triton.jit
def _local_ptr_dynamic_scalar_load_after_vector_store_kernel(out_ptr, BLOCK: tl.constexpr):
    smem = tle.gpu.alloc((BLOCK, ), dtype=tl.int32, nv_mma_shared_layout=False)
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
def _local_ptr_tiled_matmul_kernel(a_ptr, b_ptr, c_ptr, stride_am, stride_ak, stride_bk, stride_bn, stride_cm,
                                   stride_cn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                                   NUM_K_TILES: tl.constexpr, SLICE_PARTS: tl.constexpr, SLICE_WIDTH: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    smem_a = tle.gpu.alloc((BLOCK_M, BLOCK_K), dtype=tl.float16, nv_mma_shared_layout=False)
    smem_b = tle.gpu.alloc((BLOCK_K, BLOCK_N), dtype=tl.float16, nv_mma_shared_layout=False)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    slice_parts = int(SLICE_PARTS)
    slice_width = int(SLICE_WIDTH)
    assert BLOCK_K % slice_parts == 0, "BLOCK_K must divide slice_parts"

    for k_tile in range(NUM_K_TILES):
        k_offsets = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)
        a_tile = a_ptr + offs_m[:, None] * stride_am + k_offsets[None, :] * stride_ak
        b_tile = b_ptr + k_offsets[:, None] * stride_bk + offs_n[None, :] * stride_bn
        tle.gpu.copy(a_tile, smem_a, (BLOCK_M, BLOCK_K))
        tle.gpu.copy(b_tile, smem_b, (BLOCK_K, BLOCK_N))

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
def _local_ptr_full_view_dot_kernel(a_ptr, out_ptr, stride_ai, stride_aj, stride_oi, stride_oj, BLOCK: tl.constexpr):
    offs_i = tl.arange(0, BLOCK)[:, None]
    offs_j = tl.arange(0, BLOCK)[None, :]

    a_tile_ptr = a_ptr + offs_i * stride_ai + offs_j * stride_aj
    a_tile = tl.load(a_tile_ptr)

    smem = tle.gpu.alloc((BLOCK, BLOCK), dtype=tl.float16, nv_mma_shared_layout=False)
    smem_ptr = tle.gpu.local_ptr(smem)
    tl.store(smem_ptr, a_tile)

    staged = tl.load(smem_ptr)
    acc = tl.dot(staged, tl.trans(staged), out_dtype=tl.float32)

    out_ptrs = out_ptr + offs_i * stride_oi + offs_j * stride_oj
    tl.store(out_ptrs, acc.to(tl.float16))


@triton.jit
def _local_ptr_atomic_add_kernel(out_ptr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    init = tl.full((BLOCK, ), 0, tl.int32)
    smem = tle.gpu.alloc((BLOCK, ), dtype=tl.int32, init_value=init, nv_mma_shared_layout=False)
    ptrs = tle.gpu.local_ptr(smem, (offsets, ))
    increments = offsets.to(tl.int32) + 1
    old = tl.atomic_add(ptrs, increments, sem="relaxed", scope="cta")
    after = tl.load(ptrs)
    tl.store(out_ptr + offsets, old)
    tl.store(out_ptr + BLOCK + offsets, after)


@triton.jit
def _local_ptr_atomic_cas_kernel(out_ptr):
    init = tl.full((1, ), 3, tl.int32)
    smem = tle.gpu.alloc((1, ), dtype=tl.int32, init_value=init, nv_mma_shared_layout=False)
    ptr = tle.gpu.local_ptr(smem, (0, ))
    old = tl.atomic_cas(ptr, 3, 9, sem="relaxed", scope="cta")
    after = tl.load(ptr)
    tl.store(out_ptr, old)
    tl.store(out_ptr + 1, after)


@triton.jit
def _local_ptr_atomic_cas_update_kernel(out_ptr):
    init = tl.full((1, ), 3, tl.int32)
    smem = tle.gpu.alloc((1, ), dtype=tl.int32, init_value=init, nv_mma_shared_layout=False)
    ptr = tle.gpu.local_ptr(smem, (0, ))
    tl.atomic_cas(ptr, 3, 9, sem="relaxed", scope="cta")
    after = tl.load(ptr)
    tl.store(out_ptr, after)


@triton.jit
def _local_ptr_non_integer_index_kernel(out_ptr):
    smem = tle.gpu.alloc((16, ), dtype=tl.float32, nv_mma_shared_layout=False)
    idx = tl.arange(0, 16).to(tl.float32)
    ptrs = tle.gpu.local_ptr(smem, (idx, ))
    values = tl.load(ptrs)
    tl.store(out_ptr + tl.arange(0, 16), values)


@triton.jit
def _local_ptr_mixed_scalar_tensor_index_kernel(out_ptr):
    smem = tle.gpu.alloc((4, 4), dtype=tl.float32, nv_mma_shared_layout=False)
    cols = tl.arange(0, 4)
    ptrs = tle.gpu.local_ptr(smem, (0, cols))
    values = tl.load(ptrs)
    tl.store(out_ptr + cols, values)


@triton.jit
def _local_ptr_wrong_rank_index_kernel(out_ptr):
    smem = tle.gpu.alloc((4, 4), dtype=tl.float32, nv_mma_shared_layout=False)
    rows = tl.arange(0, 4)
    ptrs = tle.gpu.local_ptr(smem, (rows, ))
    values = tl.load(ptrs)
    tl.store(out_ptr + rows, values)


@triton.jit
def _local_ptr_tmem_kernel(out_ptr):
    smem = tle.gpu.alloc((16, 16), dtype=tl.float32, scope=tle.gpu.tmem)
    idx = tl.arange(0, 16)
    ptrs = tle.gpu.local_ptr(smem, (idx, idx))
    values = tl.load(ptrs)
    tl.store(out_ptr + idx, values)


def test_tle_local_ptr_subview_lowers_through_mthreads_llvm():
    compiled = compile_musa(
        _local_ptr_subview_kernel,
        signature={"out_ptr": "*fp32", "BLOCK": "constexpr"},
        constexprs={"BLOCK": 16},
    )

    ttgir = compiled.asm["ttgir"]
    llir = compiled.asm["llir"]
    assert "musa_tle.local_pointers" in ttgir, ttgir
    assert "tensor<16x!tt.ptr<f32, 3>" in ttgir, ttgir
    assert "musa_tle.local_pointers" not in llir, llir


def test_tle_local_ptr_scalar_lowers_through_mthreads_llvm():
    compiled = compile_musa(_local_ptr_scalar_kernel, signature={"out_ptr": "*fp32"})

    ttgir = compiled.asm["ttgir"]
    llir = compiled.asm["llir"]
    assert "musa_tle.local_pointers" in ttgir, ttgir
    assert "-> !tt.ptr<f32, 3>" in ttgir, ttgir
    assert "musa_tle.local_pointers" not in llir, llir


def test_tle_local_ptr_full_view_store_load_rewrites_to_memdesc_ops():
    compiled = compile_musa(_local_ptr_full_view_kernel, signature={"out_ptr": "*fp32"})

    ttgir = compiled.asm["ttgir"]
    llir = compiled.asm["llir"]
    assert "ttg.local_store" in ttgir, ttgir
    assert "ttg.local_load" in ttgir, ttgir
    assert "musa_tle.local_pointers" not in llir, llir


def test_tle_local_ptr_full_view_tail_mask_lowers_through_mthreads_llvm():
    compiled = compile_musa(
        _local_ptr_full_view_tail_mask_kernel,
        signature={"out_ptr": "*i32", "numel": "i32", "BLOCK": "constexpr"},
        constexprs={"BLOCK": 128},
    )

    ttgir = compiled.asm["ttgir"]
    llir = compiled.asm["llir"]
    assert "ttg.local_store" in ttgir, ttgir
    assert "musa_tle.local_pointers" not in llir, llir


def test_tle_local_ptr_full_view_and_indices_load_rewrite_to_local_load():
    none_compiled = compile_musa(
        _local_ptr_local_load_none_kernel,
        signature={"out_ptr": "*i32", "BLOCK": "constexpr"},
        constexprs={"BLOCK": 64},
    )
    indices_compiled = compile_musa(
        _local_ptr_local_load_full_indices_kernel,
        signature={"out_ptr": "*i32", "BLOCK": "constexpr"},
        constexprs={"BLOCK": 64},
    )

    none_ttgir = none_compiled.asm["ttgir"]
    indices_ttgir = indices_compiled.asm["ttgir"]
    assert "ttg.local_load" in none_ttgir, none_ttgir
    assert "ttg.local_load" in indices_ttgir, indices_ttgir
    assert "musa_tle.local_pointers" not in none_compiled.asm["llir"], none_compiled.asm["llir"]
    assert "musa_tle.local_pointers" not in indices_compiled.asm["llir"], indices_compiled.asm["llir"]


def test_tle_local_ptr_2d_copy_and_axis_gather_lower_through_mthreads_llvm():
    copy_compiled = compile_musa(
        _local_ptr_full_view_2d_copy_kernel,
        signature={
            "x_ptr": "*fp32",
            "out_ptr": "*fp32",
            "stride_xm": "i32",
            "stride_xn": "i32",
            "stride_om": "i32",
            "stride_on": "i32",
            "ROWS": "constexpr",
            "COLS": "constexpr",
        },
        constexprs={"ROWS": 16, "COLS": 32},
    )
    gather_compiled = compile_musa(
        _local_ptr_axis_gather_kernel,
        signature={
            "x_ptr": "*fp32",
            "out_ptr": "*fp32",
            "stride_xm": "i32",
            "stride_xn": "i32",
            "stride_om": "i32",
            "stride_on": "i32",
            "ROWS": "constexpr",
            "COLS": "constexpr",
            "SLICE": "constexpr",
        },
        constexprs={"ROWS": 8, "COLS": 8, "SLICE": 4},
    )

    copy_ttgir = copy_compiled.asm["ttgir"]
    gather_ttgir = gather_compiled.asm["ttgir"]
    assert "ttg.async_copy_global_to_local" in copy_ttgir, copy_ttgir
    assert "ttg.local_load" in copy_ttgir, copy_ttgir
    assert "ttg.async_copy_global_to_local" in gather_ttgir, gather_ttgir
    assert "musa_tle.local_pointers" not in copy_compiled.asm["llir"], copy_compiled.asm["llir"]
    assert "musa_tle.local_pointers" not in gather_compiled.asm["llir"], gather_compiled.asm["llir"]


def test_tle_local_ptr_conditional_mask_store_compiles():
    compiled = compile_musa(
        _local_ptr_conditional_mask_store_kernel,
        signature={"out_ptr": "*i32", "numel": "i32", "BLOCK": "constexpr"},
        constexprs={"BLOCK": 512},
    )

    assert compiled.asm["llir"], compiled.asm
    assert "musa_tle.local_pointers" not in compiled.asm["llir"], compiled.asm["llir"]


def test_tle_local_ptr_dynamic_scalar_index_inserts_barrier():
    compiled = compile_musa(
        _local_ptr_dynamic_scalar_load_after_vector_store_kernel,
        signature={"out_ptr": "*i32", "BLOCK": "constexpr"},
        constexprs={"BLOCK": 64},
    )

    ttgir = compiled.asm["ttgir"]
    assert "ttg.barrier local" in ttgir, ttgir
    assert '"tt.reduce"' not in ttgir, ttgir
    assert "musa_tle.local_pointers" not in compiled.asm["llir"], compiled.asm["llir"]


def test_tle_local_ptr_tiled_matmul_matches_torch():
    block_m = 32
    block_n = 32
    block_k = 32
    num_k_tiles = 2
    m = block_m
    n = block_n
    k = block_k * num_k_tiles

    a = torch.randn((m, k), device="musa", dtype=torch.float16)
    b = torch.randn((k, n), device="musa", dtype=torch.float16)
    c = torch.empty((m, n), device="musa", dtype=torch.float32)

    slice_parts = 2
    slice_width = block_k // slice_parts
    grid = (m // block_m, n // block_n)
    _local_ptr_tiled_matmul_kernel[grid](
        a,
        b,
        c,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        NUM_K_TILES=num_k_tiles,
        SLICE_PARTS=slice_parts,
        SLICE_WIDTH=slice_width,
        num_warps=4,
    )

    expected = a.float() @ b.float()
    torch.testing.assert_close(c.cpu(), expected.cpu(), rtol=5e-3, atol=5e-3)


def test_tle_local_ptr_full_view_dot_avoids_pointer_convert_layout():
    block = 32
    a = torch.randn((block, block), device="musa", dtype=torch.float16)
    out = torch.empty_like(a)

    compiled = compile_musa(
        _local_ptr_full_view_dot_kernel,
        signature={
            "a_ptr": "*fp16",
            "out_ptr": "*fp16",
            "stride_ai": "i32",
            "stride_aj": "i32",
            "stride_oi": "i32",
            "stride_oj": "i32",
            "BLOCK": "constexpr",
        },
        constexprs={"BLOCK": block},
    )
    ttgir = compiled.asm["ttgir"]
    assert "ttg.local_load" in ttgir, ttgir
    assert re.search(r"ttg\.convert_layout .*-> tensor<.*!tt\.ptr", ttgir) is None

    _local_ptr_full_view_dot_kernel[(1, )](
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
    torch.testing.assert_close(out.cpu(), expected.cpu(), rtol=2e-1, atol=2e-1)


def test_tle_local_ptr_atomic_ops_accept_addrspace3_ttir():
    add_ttir = compile_to_ttir(
        _local_ptr_atomic_add_kernel,
        signature={"out_ptr": "*i32", "BLOCK": "constexpr"},
        constexprs={"BLOCK": 16},
    )
    cas_ttir = compile_to_ttir(_local_ptr_atomic_cas_kernel, signature={"out_ptr": "*i32"})

    assert "tt.atomic_rmw add, relaxed, cta" in add_ttir, add_ttir
    assert ("(tensor<16x!tt.ptr<i32, 3>>, tensor<16xi32>, tensor<16xi1>) -> tensor<16xi32>" in add_ttir), add_ttir
    assert "tt.atomic_cas relaxed, cta" in cas_ttir, cas_ttir
    assert "(!tt.ptr<i32, 3>, i32, i32) -> i32" in cas_ttir, cas_ttir


def test_tle_local_ptr_atomic_add_lowers_through_mthreads_llvm():
    compiled = compile_musa(
        _local_ptr_atomic_add_kernel,
        signature={"out_ptr": "*i32", "BLOCK": "constexpr"},
        constexprs={"BLOCK": 16},
    )

    ttgir = compiled.asm["ttgir"]
    llir = compiled.asm["llir"]
    assert "tt.atomic_rmw" in ttgir, ttgir
    assert "tensor<16x!tt.ptr<i32, 3>" in ttgir, ttgir
    assert "musa_tle.local_pointers" not in llir, llir


def test_tle_local_ptr_atomic_cas_lowers_through_mthreads_llvm():
    compiled = compile_musa(_local_ptr_atomic_cas_kernel, signature={"out_ptr": "*i32"})

    ttgir = compiled.asm["ttgir"]
    llir = compiled.asm["llir"]
    assert "tt.atomic_cas" in ttgir, ttgir
    assert "-> !tt.ptr<i32, 3>" in ttgir, ttgir
    assert "musa_tle.local_pointers" not in llir, llir


def test_tle_local_ptr_rejects_non_integer_indices():
    with pytest.raises(CompilationError, match="local_ptr indices must use integer dtypes"):
        compile_musa(_local_ptr_non_integer_index_kernel, signature={"out_ptr": "*fp32"})


def test_tle_local_ptr_rejects_mixed_scalar_tensor_indices():
    with pytest.raises(CompilationError, match="local_ptr indices must be either all scalar or all tensors"):
        compile_musa(_local_ptr_mixed_scalar_tensor_index_kernel, signature={"out_ptr": "*fp32"})


def test_tle_local_ptr_rejects_wrong_index_rank():
    with pytest.raises(CompilationError, match="local_ptr indices must provide 2 tensors, got 1"):
        compile_musa(_local_ptr_wrong_rank_index_kernel, signature={"out_ptr": "*fp32"})


def test_tle_local_ptr_unsupported_storage_keeps_mthreads_error():
    with pytest.raises(CompilationError, match="mthreads TLE alloc does not support tmem storage"):
        compile_musa(_local_ptr_tmem_kernel, signature={"out_ptr": "*fp32"})


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
def test_tle_local_ptr_subview_runtime_loads_shared_values():
    block = 16
    out = torch.empty((block, ), device="musa", dtype=torch.float32)

    _local_ptr_subview_kernel[(1, )](out, BLOCK=block, num_warps=1)

    ref = torch.arange(0, block * 2, 2, dtype=torch.float32) + 1.0
    torch.testing.assert_close(out.cpu(), ref, rtol=0, atol=0)


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
def test_tle_local_ptr_scalar_runtime_store_load():
    out = torch.empty((1, ), device="musa", dtype=torch.float32)

    _local_ptr_scalar_kernel[(1, )](out, num_warps=1)

    torch.testing.assert_close(out.cpu(), torch.tensor([42.0], dtype=torch.float32), rtol=0, atol=0)


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
def test_tle_local_ptr_full_view_runtime_round_trip():
    out = torch.empty((16, ), device="musa", dtype=torch.float32)

    _local_ptr_full_view_kernel[(1, )](out, num_warps=1)

    ref = torch.arange(0, 16, dtype=torch.float32) + 7.0
    torch.testing.assert_close(out.cpu(), ref, rtol=0, atol=0)


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
def test_tle_local_ptr_axpy_runtime_matches_torch():
    torch.manual_seed(0)
    block = 64
    numel = block * 4
    alpha = 1.5
    x = torch.randn((numel, ), device="musa", dtype=torch.float32)
    y = torch.randn_like(x)
    out = torch.empty_like(x)

    grid = (triton.cdiv(numel, block), )
    _local_ptr_axpy_kernel[grid](x, y, out, numel, alpha, BLOCK=block, num_warps=1)

    expected = (alpha * x + y).cpu()
    torch.testing.assert_close(out.cpu(), expected, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
def test_tle_local_ptr_constant_store_runtime_populates_value():
    block = 64
    numel = block * 4
    value = 2.25
    out = torch.empty((numel, ), device="musa", dtype=torch.float32)

    grid = (triton.cdiv(numel, block), )
    _local_ptr_constant_store_kernel[grid](out, numel, value, BLOCK=block, num_warps=1)

    expected = torch.full((numel, ), value, dtype=torch.float32)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=1e-7)


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
def test_tle_local_ptr_full_view_tail_mask_runtime_preserves_tail():
    block = 128
    numel = block - 9
    out = torch.full((block, ), -1, device="musa", dtype=torch.int32)

    _local_ptr_full_view_tail_mask_kernel[(1, )](out, numel, BLOCK=block, num_warps=4)

    expected = torch.arange(0, block, dtype=torch.int32)
    expected[numel:] = -1
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
def test_tle_local_ptr_full_view_2d_copy_runtime_matches_input():
    rows = 16
    cols = 32
    x = torch.randn((rows, cols), device="musa", dtype=torch.float32)
    out = torch.empty_like(x)

    _local_ptr_full_view_2d_copy_kernel[(1, )](
        x,
        out,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        ROWS=rows,
        COLS=cols,
        num_warps=4,
    )

    torch.testing.assert_close(out.cpu(), x.cpu(), rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
def test_tle_local_ptr_full_view_load_runtime_uses_local_load():
    block = 64
    out = torch.empty((block, ), device="musa", dtype=torch.int32)

    _local_ptr_local_load_none_kernel[(1, )](out, BLOCK=block, num_warps=4)

    expected = torch.arange(0, block, dtype=torch.int32) + 3
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
def test_tle_local_ptr_full_indices_load_runtime_uses_local_load():
    block = 64
    out = torch.empty((block, ), device="musa", dtype=torch.int32)

    _local_ptr_local_load_full_indices_kernel[(1, )](out, BLOCK=block, num_warps=4)

    expected = torch.arange(0, block, dtype=torch.int32) + 5
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
def test_tle_local_ptr_conditional_mask_store_runtime_preserves_tail():
    block = 512
    numel = block - 7
    out = torch.full((block, ), -1, device="musa", dtype=torch.int32)

    _local_ptr_conditional_mask_store_kernel[(1, )](out, numel, BLOCK=block, num_warps=8)

    expected = torch.arange(0, block, dtype=torch.int32)
    expected[numel:] = -1
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
def test_tle_local_ptr_looped_elementwise_runtime_matches_torch():
    torch.manual_seed(0)
    block = 64
    chunks = 4
    numel = block * chunks * 3
    alpha = 0.75
    slices = 4
    slice_size = block // slices
    x = torch.randn((numel, ), device="musa", dtype=torch.float32)
    y = torch.randn_like(x)
    out = torch.empty_like(x)

    grid = (triton.cdiv(numel, block * chunks), )
    _local_ptr_looped_elementwise_kernel[grid](
        x,
        y,
        out,
        numel,
        alpha,
        BLOCK=block,
        CHUNKS=chunks,
        SLICES=slices,
        SLICE_SIZE=slice_size,
        num_warps=1,
    )

    expected = (alpha * x + y).cpu()
    torch.testing.assert_close(out.cpu(), expected, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
def test_tle_local_ptr_axis_gather_runtime_matches_slice():
    rows = 8
    cols = 8
    slice_width = 4
    x = torch.arange(0, rows * cols, device="musa", dtype=torch.float32).reshape(rows, cols)
    out = torch.empty((rows, slice_width), device="musa", dtype=torch.float32)

    _local_ptr_axis_gather_kernel[(1, )](
        x,
        out,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        ROWS=rows,
        COLS=cols,
        SLICE=slice_width,
        num_warps=4,
    )

    torch.testing.assert_close(out.cpu(), x[:, 1:1 + slice_width].cpu(), rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
def test_tle_local_ptr_dynamic_scalar_index_runtime_matches_vector_store():
    block = 64
    out = torch.empty((block, ), device="musa", dtype=torch.int32)

    _local_ptr_dynamic_scalar_load_after_vector_store_kernel[(1, )](out, BLOCK=block, num_warps=4)

    expected = torch.arange(1, block + 1, dtype=torch.int32)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
def test_tle_local_ptr_atomic_add_runtime_round_trip():
    block = 16
    out = torch.empty((block * 2, ), device="musa", dtype=torch.int32)

    _local_ptr_atomic_add_kernel[(1, )](out, BLOCK=block, num_warps=1)

    ref_old = torch.zeros((block, ), dtype=torch.int32)
    ref_after = torch.arange(1, block + 1, dtype=torch.int32)
    torch.testing.assert_close(out[:block].cpu(), ref_old, rtol=0, atol=0)
    torch.testing.assert_close(out[block:].cpu(), ref_after, rtol=0, atol=0)


@pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA device is not available")
def test_tle_local_ptr_atomic_cas_runtime_round_trip():
    out = torch.empty((1, ), device="musa", dtype=torch.int32)

    _local_ptr_atomic_cas_update_kernel[(1, )](out, num_warps=1)

    torch.testing.assert_close(out.cpu(), torch.tensor([9], dtype=torch.int32), rtol=0, atol=0)
