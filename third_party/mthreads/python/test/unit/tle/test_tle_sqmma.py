"""Compile and runtime coverage for the mthreads TLE SQMMA contract."""

import os
import pytest
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.compiler.errors import CompilationError

from test_tle_utils import (compile_musa, compile_to_ttir, mthreads_backend,
                             require_mthreads_libtriton)

require_mthreads_libtriton()

_SQMMA_MN_SHAPES = (
    (16, 64),
    (32, 32),
    (32, 64),
    (32, 128),
    (64, 16),
    (64, 32),
    (64, 64),
    (64, 128),
    (128, 32),
    (128, 64),
    (128, 128),
)
_SQMMA_DTYPE_CASES = (
    ("float16", 0, 2, "fmma", (16, 32, 64), 1.0, 1.0e-1, 5.0e-2),
    ("bfloat16", 1, 2, "bfmma", (16, 32, 64), 1.0, 3.0e-1, 1.0e-1),
    ("float8_e4m3fn", 2, 1, "e4m3", (32, 64, 128), 0.5, 5.0e-1, 1.5e-1),
)
_SQMMA_SHAPE_CASES = tuple(
    pytest.param(
        torch_dtype_name,
        dtype_kind,
        input_bytes,
        intrinsic_tag,
        m,
        n,
        k,
        scale,
        atol,
        rtol,
        id=f"{intrinsic_tag}-m{m}-n{n}-k{k}",
    )
    for torch_dtype_name, dtype_kind, input_bytes, intrinsic_tag, k_shapes, scale, atol, rtol in _SQMMA_DTYPE_CASES
    for m, n in _SQMMA_MN_SHAPES
    for k in k_shapes)

# PH1 (S5000) exposes only the two narrow FP16/BF16 K=128 forms.  Keep these
# cases separate from the architecture-independent shape matrix above so the
# test remains valid on backends that only implement the common K<=64 set.
_PH1_K128_CASES = (
    pytest.param("float16", 0, 2, "fmma", 16, 64, 128, 1.0, 1.0e-1, 5.0e-2,
                 id="fmma-m16-n64-k128"),
    pytest.param("float16", 0, 2, "fmma", 64, 16, 128, 1.0, 1.0e-1, 5.0e-2,
                 id="fmma-m64-n16-k128"),
    pytest.param("bfloat16", 1, 2, "bfmma", 16, 64, 128, 1.0, 3.0e-1, 1.0e-1,
                 id="bfmma-m16-n64-k128"),
    pytest.param("bfloat16", 1, 2, "bfmma", 64, 16, 128, 1.0, 3.0e-1, 1.0e-1,
                 id="bfmma-m64-n16-k128"),
)

_SQMMA_TRANSPOSE_CASES = (
    pytest.param(False, False, 0, 0, id="nn"),
    pytest.param(True, False, 1, 0, id="tn"),
    pytest.param(False, True, 0, 1, id="nt"),
    pytest.param(True, True, 1, 1, id="tt"),
)


@triton.jit
def _tle_sqmma_kernel(out):
    a = tle.gpu.alloc((128, 64), dtype=tl.float16, layout=None)
    b = tle.gpu.alloc((64, 128), dtype=tl.float16, layout=None)
    acc = tl.zeros((128, 128), dtype=tl.float32)
    acc = tle.gpu.wgmma(a, b, acc)
    acc = tle.gpu.wgmma_wait(0, acc)
    offsets = tl.arange(0, 128)[:, None] * 128 + tl.arange(0, 128)[None, :]
    tl.store(out + offsets, acc)


@triton.jit
def _tle_sqmma_adjacent_waits_kernel(out):
    a = tle.gpu.alloc((128, 64), dtype=tl.float16, layout=None)
    b = tle.gpu.alloc((64, 128), dtype=tl.float16, layout=None)
    zero = tl.zeros((128, 128), dtype=tl.float32)
    acc0 = tle.gpu.wgmma(a, b, zero)
    acc1 = tle.gpu.wgmma(a, b, zero)
    acc0 = tle.gpu.wgmma_wait(0, acc0)
    acc1 = tle.gpu.wgmma_wait(0, acc1)
    offsets = tl.arange(0, 128)[:, None] * 128 + tl.arange(0, 128)[None, :]
    tl.store(out + offsets, acc0 + acc1)


@triton.jit
def _tle_sqmma_wait_reuse_kernel(out):
    a = tle.gpu.alloc((128, 64), dtype=tl.float16, layout=None)
    b = tle.gpu.alloc((64, 128), dtype=tl.float16, layout=None)
    acc = tle.gpu.wgmma(a, b, tl.zeros((128, 128), dtype=tl.float32))
    acc = tle.gpu.wgmma_wait(0, acc)
    # Feed a released accumulator into the next async dot.  The lowering can
    # reuse the native wait result instead of converting it back to SQMMA
    # layout before issuing the second dot.
    acc = tle.gpu.wgmma(a, b, acc)
    acc = tle.gpu.wgmma_wait(0, acc)
    offsets = tl.arange(0, 128)[:, None] * 128 + tl.arange(0, 128)[None, :]
    tl.store(out + offsets, acc)


@triton.jit
def _tle_sqmma_loop_direct_yield_kernel(out, k_tiles: tl.constexpr):
    # A loop-carried asynchronous accumulator is completed by one wait after
    # the loop, avoiding a wait in every K iteration.
    a = tle.gpu.alloc((128, 64), dtype=tl.float16, layout=None,
                      nv_mma_shared_layout=True)
    b = tle.gpu.alloc((64, 128), dtype=tl.float16, layout=None,
                      nv_mma_shared_layout=True)
    acc = tl.zeros((128, 128), dtype=tl.float32)
    for _ in range(0, k_tiles):
        acc = tle.gpu.wgmma(a, b, acc)
    offsets = tl.arange(0, 128)[:, None] * 128 + tl.arange(0, 128)[None, :]
    tl.store(out + offsets, acc)


@triton.jit
def _tle_sqmma_loop_if_yield_kernel(out, k_tiles: tl.constexpr):
    # The dynamic loop predicate leaves the SQMMA result in an scf.if result:
    # the then branch issues the dot, while the implicit else branch carries
    # the previous accumulator.  This is the pattern used by a partial final
    # MMA group in the non-WS MM kernel.
    a = tle.gpu.alloc((128, 64), dtype=tl.float16, layout=None,
                      nv_mma_shared_layout=True)
    b = tle.gpu.alloc((64, 128), dtype=tl.float16, layout=None,
                      nv_mma_shared_layout=True)
    acc = tl.zeros((128, 128), dtype=tl.float32)
    for k_iter in tl.range(0, k_tiles, num_stages=1):
        if k_iter < k_tiles:
            acc = tle.gpu.wgmma(a, b, acc)
    offsets = tl.arange(0, 128)[:, None] * 128 + tl.arange(0, 128)[None, :]
    tl.store(out + offsets, acc)


@triton.jit
def _tle_sqmma_loop_wait_chain_kernel(out, k_tiles: tl.constexpr):
    # The wait result feeds a second dot in the same loop iteration.  This is
    # a useful regression shape for the MTT async lifetime rule: removing the
    # wait across the loop body can silently corrupt the accumulator.
    a = tle.gpu.alloc((128, 64), dtype=tl.float16, layout=None,
                      nv_mma_shared_layout=True)
    b = tle.gpu.alloc((64, 128), dtype=tl.float16, layout=None,
                      nv_mma_shared_layout=True)
    acc = tl.zeros((128, 128), dtype=tl.float32)
    for _ in range(0, k_tiles):
        acc = tle.gpu.wgmma(a, b, acc)
        acc = tle.gpu.wgmma_wait(0, acc)
        acc = tle.gpu.wgmma(a, b, acc)
        acc = tle.gpu.wgmma_wait(0, acc)
    offsets = tl.arange(0, 128)[:, None] * 128 + tl.arange(0, 128)[None, :]
    tl.store(out + offsets, acc)


@triton.jit
def _tle_sqmma_nonzero_wait_kernel(out):
    a = tle.gpu.alloc((128, 64), dtype=tl.float16, layout=None)
    b = tle.gpu.alloc((64, 128), dtype=tl.float16, layout=None)
    acc = tle.gpu.wgmma(a, b, tl.zeros((128, 128), dtype=tl.float32))
    acc = tle.gpu.wgmma_wait(1, acc)
    offsets = tl.arange(0, 128)[:, None] * 128 + tl.arange(0, 128)[None, :]
    tl.store(out + offsets, acc)


@triton.jit
def _tle_sqmma_ph1_k128_static_kernel(out):
    # PH1's narrow FP16 K=128 compatibility form is lowered to two K=64
    # intrinsics by FlagTree when explicitly enabled.  Keep this kernel free
    # of TMA descriptors so the test isolates SQMMA lowering and can inspect
    # the generated IR without a runtime input setup.
    a = tle.gpu.alloc(
        (16, 128),
        dtype=tl.float16,
        layout=None,
        nv_mma_shared_layout=True,
    )
    b = tle.gpu.alloc(
        (128, 64),
        dtype=tl.float16,
        layout=None,
        nv_mma_shared_layout=True,
    )
    acc = tle.gpu.wgmma(a, b, tl.zeros((16, 64), dtype=tl.float32))
    acc = tle.gpu.wgmma_wait(0, acc)
    offsets = tl.arange(0, 16)[:, None] * 64 + tl.arange(0, 64)[None, :]
    tl.store(out + offsets, acc)


@triton.jit
def _tle_sqmma_unknown_arch_k128_kernel(out):
    a = tle.gpu.alloc(
        (32, 128),
        dtype=tl.float16,
        layout=None,
        nv_mma_shared_layout=True,
    )
    b = tle.gpu.alloc(
        (128, 32),
        dtype=tl.float16,
        layout=None,
        nv_mma_shared_layout=True,
    )
    acc = tle.gpu.wgmma(a, b, tl.zeros((32, 32), dtype=tl.float32))
    acc = tle.gpu.wgmma_wait(0, acc)
    offsets = tl.arange(0, 32)[:, None] * 32 + tl.arange(0, 32)[None, :]
    tl.store(out + offsets, acc)


@triton.jit
def _tle_sqmma_non_auto_layout_kernel(out):
    a = tle.gpu.alloc((128, 64), dtype=tl.float16, layout=None, nv_mma_shared_layout=False)
    b = tle.gpu.alloc((64, 128), dtype=tl.float16, layout=None)
    acc = tle.gpu.wgmma(a, b, tl.zeros((128, 128), dtype=tl.float32))
    acc = tle.gpu.wgmma_wait(0, acc)
    offsets = tl.arange(0, 128)[:, None] * 128 + tl.arange(0, 128)[None, :]
    tl.store(out + offsets, acc)


@triton.jit
def _tle_sqmma_runtime_kernel(a_desc, b_desc, out):
    block_m: tl.constexpr = 128
    block_n: tl.constexpr = 128
    block_k: tl.constexpr = 64
    a = tle.gpu.alloc(
        (block_m, block_k),
        dtype=tl.float16,
        layout=None,
        nv_mma_shared_layout=True,
    )
    b = tle.gpu.alloc(
        (block_k, block_n),
        dtype=tl.float16,
        layout=None,
        nv_mma_shared_layout=True,
    )
    a_full = tle.gpu.alloc_barrier(expect_bytes=block_m * block_k * 2)
    b_full = tle.gpu.alloc_barrier(expect_bytes=block_k * block_n * 2)

    tle.gpu.copy(a_desc, a, (block_m, block_k), (0, 0), barrier=a_full)
    tle.gpu.copy(b_desc, b, (block_k, block_n), (0, 0), barrier=b_full)
    tle.gpu.barrier_wait(a_full, phaseIdx=0)
    tle.gpu.barrier_wait(b_full, phaseIdx=0)

    acc = tle.gpu.wgmma(a, b, tl.zeros((block_m, block_n), dtype=tl.float32))
    acc = tle.gpu.wgmma_wait(0, acc)
    offsets = tl.arange(0, block_m)[:, None] * block_n + tl.arange(0, block_n)[None, :]
    tl.store(out + offsets, acc)


@triton.jit
def _tle_sqmma_trans_compile_kernel(
    out,
    TRANS_A: tl.constexpr,
    TRANS_B: tl.constexpr,
):
    block_m: tl.constexpr = 128
    block_n: tl.constexpr = 128
    block_k: tl.constexpr = 64
    a_rows: tl.constexpr = block_k if TRANS_A else block_m
    a_cols: tl.constexpr = block_m if TRANS_A else block_k
    b_rows: tl.constexpr = block_n if TRANS_B else block_k
    b_cols: tl.constexpr = block_k if TRANS_B else block_n
    a = tle.gpu.alloc(
        (a_rows, a_cols),
        dtype=tl.float16,
        layout=None,
        nv_mma_shared_layout=True,
    )
    b = tle.gpu.alloc(
        (b_rows, b_cols),
        dtype=tl.float16,
        layout=None,
        nv_mma_shared_layout=True,
    )
    acc = tle.gpu.wgmma(
        a,
        b,
        tl.zeros((block_m, block_n), dtype=tl.float32),
        trans_a=TRANS_A,
        trans_b=TRANS_B,
    )
    acc = tle.gpu.wgmma_wait(0, acc)
    offsets = tl.arange(0, block_m)[:, None] * block_n + tl.arange(0, block_n)[None, :]
    tl.store(out + offsets, acc)


@triton.jit
def _tle_sqmma_staged_trans_compile_kernel(
    out,
    STAGES: tl.constexpr,
    TRANS_A: tl.constexpr,
    TRANS_B: tl.constexpr,
):
    block_m: tl.constexpr = 128
    block_n: tl.constexpr = 128
    block_k: tl.constexpr = 64
    a_rows: tl.constexpr = block_k if TRANS_A else block_m
    a_cols: tl.constexpr = block_m if TRANS_A else block_k
    b_rows: tl.constexpr = block_n if TRANS_B else block_k
    b_cols: tl.constexpr = block_k if TRANS_B else block_n
    a_staged = tle.gpu.alloc(
        (STAGES, a_rows, a_cols),
        dtype=tl.float16,
        layout=None,
        nv_mma_shared_layout=True,
    )
    b_staged = tle.gpu.alloc(
        (STAGES, b_rows, b_cols),
        dtype=tl.float16,
        layout=None,
        nv_mma_shared_layout=True,
    )
    a = a_staged.slot(0)
    b = b_staged.slot(0)
    acc = tle.gpu.wgmma(
        a,
        b,
        tl.zeros((block_m, block_n), dtype=tl.float32),
        trans_a=TRANS_A,
        trans_b=TRANS_B,
    )
    acc = tle.gpu.wgmma_wait(0, acc)
    offsets = tl.arange(0, block_m)[:, None] * block_n + tl.arange(0, block_n)[None, :]
    tl.store(out + offsets, acc)


@triton.jit
def _tle_sqmma_trans_invalid_rank_kernel(out):
    a = tle.gpu.alloc(
        (2, 128, 64),
        dtype=tl.float16,
        layout=None,
        nv_mma_shared_layout=True,
    )
    b = tle.gpu.alloc(
        (64, 128),
        dtype=tl.float16,
        layout=None,
        nv_mma_shared_layout=True,
    )
    acc = tle.gpu.wgmma(
        a,
        b,
        tl.zeros((128, 128), dtype=tl.float32),
        trans_a=True,
    )
    acc = tle.gpu.wgmma_wait(0, acc)
    offsets = tl.arange(0, 128)[:, None] * 128 + tl.arange(0, 128)[None, :]
    tl.store(out + offsets, acc)


@triton.jit
def _tle_sqmma_trans_runtime_kernel(
    a_desc,
    b_desc,
    out,
    dtype_kind: tl.constexpr,
    input_bytes: tl.constexpr,
    TRANS_A: tl.constexpr,
    TRANS_B: tl.constexpr,
):
    block_m: tl.constexpr = 128
    block_n: tl.constexpr = 128
    block_k: tl.constexpr = 64
    a_rows: tl.constexpr = block_k if TRANS_A else block_m
    a_cols: tl.constexpr = block_m if TRANS_A else block_k
    b_rows: tl.constexpr = block_n if TRANS_B else block_k
    b_cols: tl.constexpr = block_k if TRANS_B else block_n
    if dtype_kind == 0:
        a = tle.gpu.alloc(
            (a_rows, a_cols),
            dtype=tl.float16,
            layout=None,
            nv_mma_shared_layout=True,
        )
        b = tle.gpu.alloc(
            (b_rows, b_cols),
            dtype=tl.float16,
            layout=None,
            nv_mma_shared_layout=True,
        )
    elif dtype_kind == 1:
        a = tle.gpu.alloc(
            (a_rows, a_cols),
            dtype=tl.bfloat16,
            layout=None,
            nv_mma_shared_layout=True,
        )
        b = tle.gpu.alloc(
            (b_rows, b_cols),
            dtype=tl.bfloat16,
            layout=None,
            nv_mma_shared_layout=True,
        )
    else:
        a = tle.gpu.alloc(
            (a_rows, a_cols),
            dtype=tl.float8e4nv,
            layout=None,
            nv_mma_shared_layout=True,
        )
        b = tle.gpu.alloc(
            (b_rows, b_cols),
            dtype=tl.float8e4nv,
            layout=None,
            nv_mma_shared_layout=True,
        )
    a_full = tle.gpu.alloc_barrier(expect_bytes=block_m * block_k * input_bytes)
    b_full = tle.gpu.alloc_barrier(expect_bytes=block_k * block_n * input_bytes)
    tle.gpu.copy(a_desc, a, (a_rows, a_cols), (0, 0), barrier=a_full)
    tle.gpu.copy(b_desc, b, (b_rows, b_cols), (0, 0), barrier=b_full)
    tle.gpu.barrier_wait(a_full, phaseIdx=0)
    tle.gpu.barrier_wait(b_full, phaseIdx=0)
    acc = tle.gpu.wgmma(
        a,
        b,
        tl.zeros((block_m, block_n), dtype=tl.float32),
        trans_a=TRANS_A,
        trans_b=TRANS_B,
    )
    acc = tle.gpu.wgmma_wait(0, acc)
    offsets = tl.arange(0, block_m)[:, None] * block_n + tl.arange(0, block_n)[None, :]
    tl.store(out + offsets, acc)


@triton.jit
def _tle_sqmma_trans_for_loop_runtime_kernel(a_desc, b_desc, out, k_tiles: tl.constexpr):
    block_m: tl.constexpr = 128
    block_n: tl.constexpr = 128
    block_k: tl.constexpr = 64
    a = tle.gpu.alloc(
        (block_k, block_m),
        dtype=tl.float16,
        layout=None,
        nv_mma_shared_layout=True,
    )
    b = tle.gpu.alloc(
        (block_n, block_k),
        dtype=tl.float16,
        layout=None,
        nv_mma_shared_layout=True,
    )
    a_full = tle.gpu.alloc_barrier(expect_bytes=block_m * block_k * 2)
    b_full = tle.gpu.alloc_barrier(expect_bytes=block_k * block_n * 2)
    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    for k_iter in range(0, k_tiles):
        k_offset = k_iter * block_k
        tle.gpu.copy(a_desc, a, (block_k, block_m), (k_offset, 0), barrier=a_full)
        tle.gpu.copy(b_desc, b, (block_n, block_k), (0, k_offset), barrier=b_full)
        tle.gpu.barrier_wait(a_full, phaseIdx=k_iter)
        tle.gpu.barrier_wait(b_full, phaseIdx=k_iter)
        acc = tle.gpu.wgmma(a, b, acc, trans_a=True, trans_b=True)
        acc = tle.gpu.wgmma_wait(0, acc)
    offsets = tl.arange(0, block_m)[:, None] * block_n + tl.arange(0, block_n)[None, :]
    tl.store(out + offsets, acc)


@triton.jit
def _tle_sqmma_staged_trans_runtime_kernel(
    a_desc,
    b_desc,
    out,
    STAGES: tl.constexpr,
):
    block_m: tl.constexpr = 128
    block_n: tl.constexpr = 128
    block_k: tl.constexpr = 64
    a_staged = tle.gpu.alloc(
        (STAGES, block_k, block_m),
        dtype=tl.float16,
        layout=None,
        nv_mma_shared_layout=True,
    )
    b_staged = tle.gpu.alloc(
        (STAGES, block_n, block_k),
        dtype=tl.float16,
        layout=None,
        nv_mma_shared_layout=True,
    )
    a = a_staged.slot(0)
    b = b_staged.slot(0)
    a_full = tle.gpu.alloc_barrier(expect_bytes=block_m * block_k * 2)
    b_full = tle.gpu.alloc_barrier(expect_bytes=block_k * block_n * 2)
    tle.gpu.copy(a_desc, a, (block_k, block_m), (0, 0), barrier=a_full)
    tle.gpu.copy(b_desc, b, (block_n, block_k), (0, 0), barrier=b_full)
    tle.gpu.barrier_wait(a_full, phaseIdx=0)
    tle.gpu.barrier_wait(b_full, phaseIdx=0)
    acc = tle.gpu.wgmma(
        a,
        b,
        tl.zeros((block_m, block_n), dtype=tl.float32),
        trans_a=True,
        trans_b=True,
    )
    acc = tle.gpu.wgmma_wait(0, acc)
    offsets = tl.arange(0, block_m)[:, None] * block_n + tl.arange(0, block_n)[None, :]
    tl.store(out + offsets, acc)


@triton.jit
def _tle_sqmma_for_loop_runtime_kernel(a_desc, b_desc, out, k_tiles: tl.constexpr):
    block_m: tl.constexpr = 128
    block_n: tl.constexpr = 128
    block_k: tl.constexpr = 64
    a = tle.gpu.alloc(
        (block_m, block_k),
        dtype=tl.float16,
        layout=None,
        nv_mma_shared_layout=True,
    )
    b = tle.gpu.alloc(
        (block_k, block_n),
        dtype=tl.float16,
        layout=None,
        nv_mma_shared_layout=True,
    )
    a_full = tle.gpu.alloc_barrier(expect_bytes=block_m * block_k * 2)
    b_full = tle.gpu.alloc_barrier(expect_bytes=block_k * block_n * 2)

    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    for k_iter in range(0, k_tiles):
        k_offset = k_iter * block_k
        tle.gpu.copy(a_desc, a, (block_m, block_k), (0, k_offset), barrier=a_full)
        tle.gpu.copy(b_desc, b, (block_k, block_n), (k_offset, 0), barrier=b_full)
        tle.gpu.barrier_wait(a_full, phaseIdx=k_iter)
        tle.gpu.barrier_wait(b_full, phaseIdx=k_iter)
        acc = tle.gpu.wgmma(a, b, acc)
        acc = tle.gpu.wgmma_wait(0, acc)

    offsets = tl.arange(0, block_m)[:, None] * block_n + tl.arange(0, block_n)[None, :]
    tl.store(out + offsets, acc)


@triton.jit
def _tle_sqmma_dtype_runtime_kernel(
    a_desc,
    b_desc,
    out,
    input_bytes: tl.constexpr,
):
    block_m: tl.constexpr = 128
    block_n: tl.constexpr = 128
    block_k: tl.constexpr = 64
    if input_bytes == 1:
        a = tle.gpu.alloc(
            (block_m, block_k),
            dtype=tl.float8e4nv,
            layout=None,
            nv_mma_shared_layout=True,
        )
        b = tle.gpu.alloc(
            (block_k, block_n),
            dtype=tl.float8e4nv,
            layout=None,
            nv_mma_shared_layout=True,
        )
    else:
        a = tle.gpu.alloc(
            (block_m, block_k),
            dtype=tl.bfloat16,
            layout=None,
            nv_mma_shared_layout=True,
        )
        b = tle.gpu.alloc(
            (block_k, block_n),
            dtype=tl.bfloat16,
            layout=None,
            nv_mma_shared_layout=True,
        )
    a_full = tle.gpu.alloc_barrier(expect_bytes=block_m * block_k * input_bytes, )
    b_full = tle.gpu.alloc_barrier(expect_bytes=block_k * block_n * input_bytes, )

    tle.gpu.copy(a_desc, a, (block_m, block_k), (0, 0), barrier=a_full)
    tle.gpu.copy(b_desc, b, (block_k, block_n), (0, 0), barrier=b_full)
    tle.gpu.barrier_wait(a_full, phaseIdx=0)
    tle.gpu.barrier_wait(b_full, phaseIdx=0)
    acc = tle.gpu.wgmma(a, b, tl.zeros((block_m, block_n), dtype=tl.float32))
    acc = tle.gpu.wgmma_wait(0, acc)
    offsets = tl.arange(0, block_m)[:, None] * block_n + tl.arange(0, block_n)[None, :]
    tl.store(out + offsets, acc)


@triton.jit
def _tle_sqmma_all_shapes_runtime_kernel(
    a_desc,
    b_desc,
    out,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    dtype_kind: tl.constexpr,
    input_bytes: tl.constexpr,
):
    if dtype_kind == 0:
        a = tle.gpu.alloc(
            (block_m, block_k),
            dtype=tl.float16,
            layout=None,
            nv_mma_shared_layout=True,
        )
        b = tle.gpu.alloc(
            (block_k, block_n),
            dtype=tl.float16,
            layout=None,
            nv_mma_shared_layout=True,
        )
    elif dtype_kind == 1:
        a = tle.gpu.alloc(
            (block_m, block_k),
            dtype=tl.bfloat16,
            layout=None,
            nv_mma_shared_layout=True,
        )
        b = tle.gpu.alloc(
            (block_k, block_n),
            dtype=tl.bfloat16,
            layout=None,
            nv_mma_shared_layout=True,
        )
    else:
        a = tle.gpu.alloc(
            (block_m, block_k),
            dtype=tl.float8e4nv,
            layout=None,
            nv_mma_shared_layout=True,
        )
        b = tle.gpu.alloc(
            (block_k, block_n),
            dtype=tl.float8e4nv,
            layout=None,
            nv_mma_shared_layout=True,
        )

    a_full = tle.gpu.alloc_barrier(expect_bytes=block_m * block_k * input_bytes, )
    b_full = tle.gpu.alloc_barrier(expect_bytes=block_k * block_n * input_bytes, )
    tle.gpu.copy(a_desc, a, (block_m, block_k), (0, 0), barrier=a_full)
    tle.gpu.copy(b_desc, b, (block_k, block_n), (0, 0), barrier=b_full)
    tle.gpu.barrier_wait(a_full, phaseIdx=0)
    tle.gpu.barrier_wait(b_full, phaseIdx=0)
    acc = tle.gpu.wgmma(a, b, tl.zeros((block_m, block_n), dtype=tl.float32))
    acc = tle.gpu.wgmma_wait(0, acc)
    offsets = tl.arange(0, block_m)[:, None] * block_n + tl.arange(0, block_n)[None, :]
    tl.store(out + offsets, acc)


def test_mthreads_tle_sqmma_ttir_uses_backend_local_names():
    ttir = compile_to_ttir(_tle_sqmma_kernel, {"out": "*fp32"})
    assert "musa_tle.sqmma" in ttir, ttir
    assert "musa_tle.sqmma_wait" in ttir, ttir
    assert "musa_tle.wgmma" not in ttir, ttir
    assert ttir.count("musa_tle.auto_shared_layout") == 2, ttir


def test_mthreads_tle_sqmma_rejects_nonzero_pending_groups():
    with pytest.raises(CompilationError, match="requires pendings=0"):
        compile_to_ttir(_tle_sqmma_nonzero_wait_kernel, {"out": "*fp32"})


def test_mthreads_ph1_k128_opt_in_lowers_to_two_k64_intrinsics(monkeypatch):
    monkeypatch.delenv("TRITON_OVERRIDE_ARCH", raising=False)
    monkeypatch.setenv("TRITON_MUSA_ARCH", "ph1")
    monkeypatch.setenv("TRITON_MUSA_ENABLE_K128_SQMMA", "1")
    compiled = compile_musa(_tle_sqmma_ph1_k128_static_kernel, {"out": "*fp32"})
    llir = compiled.asm["llir"]
    assert "llvm.musa.sqmma.fmma.m16n64k128.mma" not in llir, llir
    assert llir.count("@llvm.musa.sqmma.fmma.m16n64k64.mma(") >= 2, llir
    assert "call void @llvm.musa.sqmma.wait()" in llir, llir


def test_mthreads_ph1_k128_requires_exact_opt_in(monkeypatch):
    monkeypatch.delenv("TRITON_OVERRIDE_ARCH", raising=False)
    monkeypatch.setenv("TRITON_MUSA_ARCH", "ph1")
    # Force a fresh lowering for each value so a prior opt-in specialization
    # cannot mask the fail-closed capability check.
    monkeypatch.setattr(triton.knobs.compilation, "always_compile", True)
    for value in ("10", "1debug"):
        monkeypatch.setenv("TRITON_MUSA_ENABLE_K128_SQMMA", value)
        # JITFunction keeps per-device specializations independently of the
        # global always_compile knob; clear them so each environment value
        # exercises the backend capability verifier.
        _tle_sqmma_ph1_k128_static_kernel.device_caches.clear()
        # The logical K=128 compatibility form is intentionally opt-in.
        # Similar-looking values must not silently enable it; the selector
        # should instead choose the ordinary legal K=64 configuration and
        # keep the source usable with older FlagTree installations.
        compiled = compile_musa(
            _tle_sqmma_ph1_k128_static_kernel, {"out": "*fp32"}
        )
        llir = compiled.asm["llir"]
        assert "llvm.musa.sqmma.fmma.m16n64k128.mma" not in llir, llir
        assert llir.count("@llvm.musa.sqmma.fmma.m16n64k64.mma(") >= 2, llir


def test_mthreads_tle_sqmma_rejects_unvalidated_k128_arch(monkeypatch):
    # Numeric capability ordering is not a support contract: a future MUSA
    # architecture must be rejected before llc sees an unknown -mcpu.
    monkeypatch.delenv("TRITON_OVERRIDE_ARCH", raising=False)
    monkeypatch.setenv("TRITON_MUSA_ARCH", "33")
    monkeypatch.setenv("TRITON_MUSA_ENABLE_K128_SQMMA", "1")
    with pytest.raises(ValueError, match="Unsupported MUSA arch"):
        compile_musa(_tle_sqmma_unknown_arch_k128_kernel, {"out": "*fp32"})


def test_mthreads_ph1s_arch_maps_to_capability_32(monkeypatch):
    monkeypatch.delenv("TRITON_OVERRIDE_ARCH", raising=False)
    monkeypatch.setenv("TRITON_MUSA_ARCH", "ph1s")
    target, backend = mthreads_backend()
    assert target.arch == "ph1s"
    options = backend.parse_options({})
    assert options.arch == "ph1s"
    from triton.backends.mthreads.driver import _arch_to_musa_capability
    assert _arch_to_musa_capability("ph1s") == 32
    assert _arch_to_musa_capability("ph1") == 31


def test_mthreads_tle_sqmma_requires_auto_shared_layout(capfd):
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        compile_musa(_tle_sqmma_non_auto_layout_kernel, {"out": "*fp32"})
    captured = capfd.readouterr()
    assert "requires layout=None and nv_mma_shared_layout=True" in captured.err


def test_mthreads_tle_sqmma_lowers_to_parameterless_wait():
    compiled = compile_musa(_tle_sqmma_kernel, {"out": "*fp32"})
    ttgir = compiled.asm["ttgir"]
    llir = compiled.asm["llir"]
    assert "musa_tle.sqmma" not in ttgir, ttgir
    assert "ttmg.squad_dot" in ttgir, ttgir
    assert ttgir.count("ttmg.squad_dot_wait") == 1, ttgir
    assert "ttg.local_load" not in ttgir, ttgir
    assert "llvm.musa.sqmma" in llir, llir
    assert "call void @llvm.musa.sqmma.wait()" in llir, llir


def test_mthreads_tle_sqmma_coalesces_adjacent_waits():
    compiled = compile_musa(_tle_sqmma_adjacent_waits_kernel, {"out": "*fp32"})
    ttir = compiled.asm["ttir"]
    ttgir = compiled.asm["ttgir"]
    llir = compiled.asm["llir"]
    assert ttir.count("musa_tle.sqmma_wait") == 2, ttir
    assert ttgir.count("ttmg.squad_dot_wait") == 1, ttgir
    assert llir.count("call void @llvm.musa.sqmma.wait()") == 1, llir


def test_mthreads_tle_sqmma_reuses_native_wait_result():
    compiled = compile_musa(_tle_sqmma_wait_reuse_kernel, {"out": "*fp32"})
    ttgir = compiled.asm["ttgir"]
    # The intermediate wait is elided because its result is consumed only as
    # the next dot's accumulator. The native accumulator chain now also keeps
    # the final result in the SQMMA layout, so no redundant layout conversion
    # is emitted before the ordinary store consumer.
    assert ttgir.count("ttg.convert_layout") == 0, ttgir
    assert ttgir.count("ttmg.squad_dot_wait") == 1, ttgir


def test_mthreads_tle_sqmma_lowers_direct_loop_yield():
    compiled = compile_musa(
        _tle_sqmma_loop_direct_yield_kernel,
        {"out": "*fp32", "k_tiles": "constexpr"},
        {"k_tiles": 2},
    )
    ttgir = compiled.asm["ttgir"]
    llir = compiled.asm["llir"]
    # The loop is carried as an opaque native SQMMA accumulator and completed
    # once after the loop rather than after every dot.
    assert "mtgpu.sqmma" in ttgir, ttgir
    assert "mtgpu.sqmma_wait" in ttgir, ttgir
    assert llir.count("call void @llvm.musa.sqmma.wait()") == 1, llir


def test_mthreads_tle_sqmma_lowers_if_carried_loop_yield():
    compiled = compile_musa(
        _tle_sqmma_loop_if_yield_kernel,
        {"out": "*fp32", "k_tiles": "constexpr"},
        {"k_tiles": 2},
    )
    ttgir = compiled.asm["ttgir"]
    llir = compiled.asm["llir"]
    # A structured conditional on the loop-carried accumulator is supported
    # by the path-aware carrier conversion.  Both the if result and the loop
    # iter/result should remain opaque until the final unpack after the loop.
    assert "mtgpu.sqmma_accumulator" in ttgir, ttgir
    assert "mtgpu.sqmma" in ttgir, ttgir
    assert "mtgpu.sqmma_wait" in ttgir, ttgir
    assert "ttg.convert_layout" not in ttgir, ttgir
    assert llir.count("call void @llvm.musa.sqmma.wait()") == 1, llir


def test_mthreads_tle_sqmma_keeps_wait_inside_loop_chain():
    compiled = compile_musa(
        _tle_sqmma_loop_wait_chain_kernel,
        {"out": "*fp32", "k_tiles": "constexpr"},
        {"k_tiles": 2},
    )
    ttgir = compiled.asm["ttgir"]
    # The loop-body wait is a required completion boundary.  Depending on
    # later canonicalization it may be represented by either the generic MUSA
    # or backend-private spelling, but it must not disappear entirely.
    # There are two completion boundaries in each loop body (one between the
    # dots and one before the loop yield); the first is the one that used to be
    # incorrectly elided.
    # Native carrier lowering spells the completion op `mtgpu.sqmma_wait`;
    # older generic lowering used `squad_dot_wait`.
    assert ttgir.count("sqmma_wait") >= 2, ttgir
    assert "mtgpu.sqmma_accumulator" in ttgir, ttgir
    # The carrier stays native across the two dots; only the final unpack is
    # needed for the ordinary tensor store.
    assert ttgir.count("ttg.convert_layout") == 0, ttgir


@pytest.mark.parametrize("trans_a,trans_b,layout_a,layout_b", _SQMMA_TRANSPOSE_CASES)
def test_mthreads_tle_sqmma_transpose_ir(trans_a, trans_b, layout_a, layout_b):
    signature = {
        "out": "*fp32",
        "TRANS_A": "constexpr",
        "TRANS_B": "constexpr",
    }
    constexprs = {"TRANS_A": trans_a, "TRANS_B": trans_b}
    ttir = compile_to_ttir(_tle_sqmma_trans_compile_kernel, signature, constexprs)
    expected_views = int(trans_a) + int(trans_b)
    assert ttir.count("ttg.memdesc_trans") == expected_views, ttir
    assert "tt.trans" not in ttir, ttir

    compiled = compile_musa(_tle_sqmma_trans_compile_kernel, signature, constexprs)
    ttgir = compiled.asm["ttgir"]
    llir = compiled.asm["llir"]
    assert "musa_tle.sqmma" not in ttgir, ttgir
    assert ttgir.count("ttg.memdesc_trans") == expected_views, ttgir
    assert f"layoutA = {layout_a} : i32, layoutB = {layout_b} : i32" in ttgir, ttgir
    mma_calls = [
        line for line in llir.splitlines() if "call" in line and "@llvm.musa.sqmma." in line and ".mma" in line
    ]
    assert mma_calls, llir
    assert any(f"i32 {layout_a}, i32 {layout_b}," in line for line in mma_calls), mma_calls


def test_mthreads_tle_sqmma_transpose_adds_no_shared_memory():
    signature = {
        "out": "*fp32",
        "TRANS_A": "constexpr",
        "TRANS_B": "constexpr",
    }
    shared_bytes = set()
    for trans_a, trans_b, _, _ in ((False, False, 0, 0), (True, False, 1, 0), (False, True, 0, 1), (True, True, 1, 1)):
        compiled = compile_musa(
            _tle_sqmma_trans_compile_kernel,
            signature,
            {"TRANS_A": trans_a, "TRANS_B": trans_b},
        )
        ttgir = compiled.asm["ttgir"]
        shared_bytes.add(compiled.metadata.shared)
        assert ttgir.count("ttg.local_alloc") == 2, ttgir
        assert ttgir.count("ttg.memdesc_trans") == int(trans_a) + int(trans_b), ttgir
        assert "ttg.local_load" not in ttgir, ttgir
    assert len(shared_bytes) == 1, shared_bytes


def test_mthreads_tle_sqmma_staged_transpose_ir():
    compiled = compile_musa(
        _tle_sqmma_staged_trans_compile_kernel,
        {
            "out": "*fp32",
            "STAGES": "constexpr",
            "TRANS_A": "constexpr",
            "TRANS_B": "constexpr",
        },
        {"STAGES": 3, "TRANS_A": True, "TRANS_B": True},
    )
    ttgir = compiled.asm["ttgir"]
    assert ttgir.count("ttg.memdesc_index") >= 2, ttgir
    assert ttgir.count("ttg.memdesc_trans") == 2, ttgir
    assert "layoutA = 1 : i32, layoutB = 1 : i32" in ttgir, ttgir
    assert "ttg.local_load" not in ttgir, ttgir


def test_mthreads_tle_sqmma_transpose_validates_before_building_view():
    with pytest.raises(CompilationError, match="requires rank-2 a"):
        compile_to_ttir(_tle_sqmma_trans_invalid_rank_kernel, {"out": "*fp32"})


def test_mthreads_tle_sqmma_runtime_precision():
    import torch
    from triton.tools.tensor_descriptor import TensorDescriptor

    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("MUSA device is not available")

    torch.manual_seed(1234)
    a_cpu = torch.randn((128, 64), dtype=torch.float16)
    b_cpu = torch.randn((64, 128), dtype=torch.float16)
    a = a_cpu.to("musa")
    b = b_cpu.to("musa")
    out = torch.empty((128, 128), device="musa", dtype=torch.float32)
    a_desc = TensorDescriptor.from_tensor(a, block_shape=[128, 64])
    b_desc = TensorDescriptor.from_tensor(b, block_shape=[64, 128])

    kernel = _tle_sqmma_runtime_kernel[(1, )](a_desc, b_desc, out, num_warps=4, num_stages=1)
    torch.musa.synchronize()

    expected = torch.matmul(a_cpu.float(), b_cpu.float())
    torch.testing.assert_close(out.cpu(), expected, atol=5e-2, rtol=5e-2)
    assert "llvm.musa.tme.ld.tile.2d" in kernel.asm["llir"]
    assert "llvm.musa.sqmma" in kernel.asm["llir"]
    assert "call void @llvm.musa.sqmma.wait()" in kernel.asm["llir"]


@pytest.mark.parametrize("trans_a,trans_b,layout_a,layout_b", _SQMMA_TRANSPOSE_CASES)
@pytest.mark.parametrize(
    "torch_dtype_name,dtype_kind,input_bytes,intrinsic_tag,scale,atol,rtol",
    [
        ("float16", 0, 2, "fmma", 1.0, 1.0e-1, 5.0e-2),
        ("bfloat16", 1, 2, "bfmma", 1.0, 3.0e-1, 1.0e-1),
        ("float8_e4m3fn", 2, 1, "e4m3", 0.5, 5.0e-1, 1.5e-1),
    ],
)
def test_mthreads_tle_sqmma_transpose_runtime_precision(
    trans_a,
    trans_b,
    layout_a,
    layout_b,
    torch_dtype_name,
    dtype_kind,
    input_bytes,
    intrinsic_tag,
    scale,
    atol,
    rtol,
):
    import torch
    from triton.tools.tensor_descriptor import TensorDescriptor

    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("MUSA device is not available")

    block_m, block_n, block_k = 128, 128, 64
    a_shape = (block_k, block_m) if trans_a else (block_m, block_k)
    b_shape = (block_n, block_k) if trans_b else (block_k, block_n)
    torch.manual_seed(20260810)
    torch_dtype = getattr(torch, torch_dtype_name)
    a_cpu = (torch.randn(a_shape, dtype=torch.float32) * scale).to(torch_dtype)
    b_cpu = (torch.randn(b_shape, dtype=torch.float32) * scale).to(torch_dtype)
    a = a_cpu.to("musa")
    b = b_cpu.to("musa")
    out = torch.empty((block_m, block_n), device="musa", dtype=torch.float32)
    a_desc = TensorDescriptor.from_tensor(a, block_shape=list(a_shape))
    b_desc = TensorDescriptor.from_tensor(b, block_shape=list(b_shape))

    kernel = _tle_sqmma_trans_runtime_kernel[(1, )](
        a_desc,
        b_desc,
        out,
        dtype_kind,
        input_bytes,
        trans_a,
        trans_b,
        num_warps=4,
        num_stages=1,
    )
    torch.musa.synchronize()

    a_ref = a_cpu.T if trans_a else a_cpu
    b_ref = b_cpu.T if trans_b else b_cpu
    expected = torch.matmul(a_ref.float(), b_ref.float())
    torch.testing.assert_close(out.cpu(), expected, atol=atol, rtol=rtol)
    assert f"layoutA = {layout_a} : i32, layoutB = {layout_b} : i32" in kernel.asm["ttgir"]
    assert f"llvm.musa.sqmma.{intrinsic_tag}" in kernel.asm["llir"]
    assert "call void @llvm.musa.sqmma.wait()" in kernel.asm["llir"]


def test_mthreads_tle_sqmma_transpose_for_loop_runtime_precision():
    import torch
    from triton.tools.tensor_descriptor import TensorDescriptor

    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("MUSA device is not available")

    torch.manual_seed(20260811)
    a_cpu = torch.randn((128, 128), dtype=torch.float16)
    b_cpu = torch.randn((128, 128), dtype=torch.float16)
    a = a_cpu.to("musa")
    b = b_cpu.to("musa")
    out = torch.empty((128, 128), device="musa", dtype=torch.float32)
    a_desc = TensorDescriptor.from_tensor(a, block_shape=[64, 128])
    b_desc = TensorDescriptor.from_tensor(b, block_shape=[128, 64])
    kernel = _tle_sqmma_trans_for_loop_runtime_kernel[(1, )](
        a_desc,
        b_desc,
        out,
        2,
        num_warps=4,
        num_stages=1,
    )
    torch.musa.synchronize()

    expected = torch.matmul(a_cpu.T.float(), b_cpu.T.float())
    torch.testing.assert_close(out.cpu(), expected, atol=7.0e-2, rtol=5.0e-2)
    assert "scf.for" in kernel.asm["ttgir"]
    assert "layoutA = 1 : i32, layoutB = 1 : i32" in kernel.asm["ttgir"]
    assert "llvm.musa.sqmma" in kernel.asm["llir"]


@pytest.mark.parametrize("stages", [1, 2, 3])
def test_mthreads_tle_sqmma_staged_transpose_runtime_precision(stages):
    import torch
    from triton.tools.tensor_descriptor import TensorDescriptor

    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("MUSA device is not available")

    torch.manual_seed(20260812 + stages)
    a_cpu = torch.randn((64, 128), dtype=torch.float16)
    b_cpu = torch.randn((128, 64), dtype=torch.float16)
    a = a_cpu.to("musa")
    b = b_cpu.to("musa")
    out = torch.empty((128, 128), device="musa", dtype=torch.float32)
    a_desc = TensorDescriptor.from_tensor(a, block_shape=[64, 128])
    b_desc = TensorDescriptor.from_tensor(b, block_shape=[128, 64])
    kernel = _tle_sqmma_staged_trans_runtime_kernel[(1, )](
        a_desc,
        b_desc,
        out,
        stages,
        num_warps=4,
        num_stages=stages,
    )
    torch.musa.synchronize()

    expected = torch.matmul(a_cpu.T.float(), b_cpu.T.float())
    torch.testing.assert_close(out.cpu(), expected, atol=7.0e-2, rtol=5.0e-2)
    assert kernel.asm["ttgir"].count("ttg.memdesc_trans") == 2
    assert "layoutA = 1 : i32, layoutB = 1 : i32" in kernel.asm["ttgir"]


def test_mthreads_tle_sqmma_for_loop_runtime_precision():
    import torch
    from triton.tools.tensor_descriptor import TensorDescriptor

    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("MUSA device is not available")

    torch.manual_seed(1234)
    a_cpu = torch.randn((128, 128), dtype=torch.float16)
    b_cpu = torch.randn((128, 128), dtype=torch.float16)
    a = a_cpu.to("musa")
    b = b_cpu.to("musa")
    out = torch.empty((128, 128), device="musa", dtype=torch.float32)
    a_desc = TensorDescriptor.from_tensor(a, block_shape=[128, 64])
    b_desc = TensorDescriptor.from_tensor(b, block_shape=[64, 128])

    kernel = _tle_sqmma_for_loop_runtime_kernel[(1, )](
        a_desc,
        b_desc,
        out,
        2,
        num_warps=4,
        num_stages=1,
    )
    torch.musa.synchronize()

    expected = torch.matmul(a_cpu.float(), b_cpu.float())
    torch.testing.assert_close(out.cpu(), expected, atol=7e-2, rtol=5e-2)
    assert "scf.for" in kernel.asm["ttgir"]
    assert "llvm.musa.tme.ld.tile.2d" in kernel.asm["llir"]
    assert "llvm.musa.sqmma" in kernel.asm["llir"]
    assert "call void @llvm.musa.sqmma.wait()" in kernel.asm["llir"]

@pytest.mark.parametrize(
    "torch_dtype_name,input_bytes,intrinsic_tag,atol,rtol",
    [
        ("bfloat16", 2, "bfmma", 1.5e-1, 5e-2),
        ("float8_e4m3fn", 1, "e4m3", 2.5e-1, 1e-1),
    ],
)
def test_mthreads_tle_sqmma_dtype_runtime_precision(
    torch_dtype_name,
    input_bytes,
    intrinsic_tag,
    atol,
    rtol,
):
    import torch
    from triton.tools.tensor_descriptor import TensorDescriptor

    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("MUSA device is not available")

    torch.manual_seed(1234)
    torch_dtype = getattr(torch, torch_dtype_name)
    scale = 0.5 if input_bytes == 1 else 1.0
    a_cpu = (torch.randn((128, 64), dtype=torch.float32) * scale).to(torch_dtype)
    b_cpu = (torch.randn((64, 128), dtype=torch.float32) * scale).to(torch_dtype)
    a = a_cpu.to("musa")
    b = b_cpu.to("musa")
    out = torch.empty((128, 128), device="musa", dtype=torch.float32)
    a_desc = TensorDescriptor.from_tensor(a, block_shape=[128, 64])
    b_desc = TensorDescriptor.from_tensor(b, block_shape=[64, 128])

    kernel = _tle_sqmma_dtype_runtime_kernel[(1, )](
        a_desc,
        b_desc,
        out,
        input_bytes,
        num_warps=4,
        num_stages=1,
    )
    torch.musa.synchronize()

    expected = torch.matmul(a_cpu.float(), b_cpu.float())
    torch.testing.assert_close(out.cpu(), expected, atol=atol, rtol=rtol)
    assert "llvm.musa.tme.ld.tile.2d" in kernel.asm["llir"]
    assert f"llvm.musa.sqmma.{intrinsic_tag}" in kernel.asm["llir"]
    assert "call void @llvm.musa.sqmma.wait()" in kernel.asm["llir"]


@pytest.mark.parametrize(
    "torch_dtype_name,dtype_kind,input_bytes,intrinsic_tag,m,n,k,scale,atol,rtol",
    _SQMMA_SHAPE_CASES,
)
def test_mthreads_tle_sqmma_all_supported_shapes_runtime(
    torch_dtype_name,
    dtype_kind,
    input_bytes,
    intrinsic_tag,
    m,
    n,
    k,
    scale,
    atol,
    rtol,
):
    import torch
    from triton.tools.tensor_descriptor import TensorDescriptor

    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("MUSA device is not available")

    torch.manual_seed(1234)
    torch_dtype = getattr(torch, torch_dtype_name)
    a_cpu = (torch.randn((m, k), dtype=torch.float32) * scale).to(torch_dtype)
    b_cpu = (torch.randn((k, n), dtype=torch.float32) * scale).to(torch_dtype)
    a = a_cpu.to("musa")
    b = b_cpu.to("musa")
    out = torch.empty((m, n), device="musa", dtype=torch.float32)
    a_desc = TensorDescriptor.from_tensor(a, block_shape=[m, k])
    b_desc = TensorDescriptor.from_tensor(b, block_shape=[k, n])

    kernel = _tle_sqmma_all_shapes_runtime_kernel[(1, )](
        a_desc,
        b_desc,
        out,
        m,
        n,
        k,
        dtype_kind,
        input_bytes,
        num_warps=4,
        num_stages=1,
    )
    torch.musa.synchronize()

    expected = torch.matmul(a_cpu.float(), b_cpu.float())
    torch.testing.assert_close(out.cpu(), expected, atol=atol, rtol=rtol)
    intrinsic = f"llvm.musa.sqmma.{intrinsic_tag}.m{m}n{n}k{k}.mma"
    assert intrinsic in kernel.asm["llir"]
    assert "call void @llvm.musa.sqmma.wait()" in kernel.asm["llir"]


@pytest.mark.skipif(
    os.getenv("TRITON_MUSA_ENABLE_K128_SQMMA") != "1",
    reason="bundled mthreads llc does not yet lower SQMMA K=128",
)
@pytest.mark.parametrize(
    "torch_dtype_name,dtype_kind,input_bytes,intrinsic_tag,m,n,k,scale,atol,rtol",
    _PH1_K128_CASES,
)
def test_mthreads_tle_sqmma_ph1_narrow_k128_runtime(
    torch_dtype_name,
    dtype_kind,
    input_bytes,
    intrinsic_tag,
    m,
    n,
    k,
    scale,
    atol,
    rtol,
):
    import torch
    from triton.tools.tensor_descriptor import TensorDescriptor

    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("MUSA device is not available")

    torch.manual_seed(4321)
    torch_dtype = getattr(torch, torch_dtype_name)
    a_cpu = (torch.randn((m, k), dtype=torch.float32) * scale).to(torch_dtype)
    b_cpu = (torch.randn((k, n), dtype=torch.float32) * scale).to(torch_dtype)
    a = a_cpu.to("musa")
    b = b_cpu.to("musa")
    out = torch.empty((m, n), device="musa", dtype=torch.float32)
    a_desc = TensorDescriptor.from_tensor(a, block_shape=[m, k])
    b_desc = TensorDescriptor.from_tensor(b, block_shape=[k, n])

    kernel = _tle_sqmma_all_shapes_runtime_kernel[(1,)](
        a_desc,
        b_desc,
        out,
        m,
        n,
        k,
        dtype_kind,
        input_bytes,
        num_warps=4,
        num_stages=1,
    )
    torch.musa.synchronize()

    expected = torch.matmul(a_cpu.float(), b_cpu.float())
    torch.testing.assert_close(out.cpu(), expected, atol=atol, rtol=rtol)
    # The logical PH1 K=128 contract is lowered to two stable K=64 LLVM
    # intrinsics because bundled llc crashes while expanding the native K=128
    # form. Keep this test tied to the compatibility lowering instead of
    # requiring the unavailable K=128 symbol in generated LLIR.
    intrinsic = f"llvm.musa.sqmma.{intrinsic_tag}.m{m}n{n}k64.mma"
    assert kernel.asm["llir"].count(f"@{intrinsic}(") >= 3
    assert f"llvm.musa.sqmma.{intrinsic_tag}.m{m}n{n}k128.mma" not in kernel.asm["llir"]
    assert "call void @llvm.musa.sqmma.wait()" in kernel.asm["llir"]


def test_mthreads_tle_sqmma_macro_tile_auto_decomposition_runtime():
    import torch
    from triton.tools.tensor_descriptor import TensorDescriptor

    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("MUSA device is not available")

    m, n, k = 256, 256, 64
    torch.manual_seed(5256)
    a_cpu = torch.randn((m, k), dtype=torch.float16)
    b_cpu = torch.randn((k, n), dtype=torch.float16)
    a = a_cpu.to("musa")
    b = b_cpu.to("musa")
    out = torch.empty((m, n), device="musa", dtype=torch.float32)
    a_desc = TensorDescriptor.from_tensor(a, block_shape=[m, k])
    b_desc = TensorDescriptor.from_tensor(b, block_shape=[k, n])

    kernel = _tle_sqmma_all_shapes_runtime_kernel[(1, )](
        a_desc,
        b_desc,
        out,
        m,
        n,
        k,
        0,
        2,
        num_warps=16,
        num_stages=1,
    )
    torch.musa.synchronize()

    expected = torch.matmul(a_cpu.float(), b_cpu.float())
    torch.testing.assert_close(out.cpu(), expected, atol=1.0e-1, rtol=5.0e-2)

    ttir = kernel.asm["ttir"]
    ttgir = kernel.asm["ttgir"]
    llir = kernel.asm["llir"]
    assert ttir.count("musa_tle.sqmma ") == 1
    assert ttir.count("musa_tle.sqmma_wait ") == 1
    assert "warpsPerCTA = [8, 2]" in ttgir
    assert "instrShape = [128, 128, 64]" in ttgir
    assert ttgir.count("ttmg.squad_dot ") == 1
    intrinsic = "llvm.musa.sqmma.fmma.m128n128k64.mma"
    assert llir.count(f"@{intrinsic}(") == 2  # One call plus one declaration.
    assert "call void @llvm.musa.sqmma.wait()" in llir
