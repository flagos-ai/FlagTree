"""Compile and runtime coverage for the mthreads TLE SQMMA contract."""

import os
from pathlib import Path
import re
import subprocess
import sys

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton._C import libtriton
from triton._C.libtriton import ir
from triton.compiler.errors import CompilationError

from test_tle_utils import compile_musa, compile_to_ttir, mthreads_backend, musa_target, require_mthreads_libtriton

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
def _tle_sqmma_nonzero_wait_kernel(out):
    a = tle.gpu.alloc((128, 64), dtype=tl.float16, layout=None)
    b = tle.gpu.alloc((64, 128), dtype=tl.float16, layout=None)
    acc = tle.gpu.wgmma(a, b, tl.zeros((128, 128), dtype=tl.float32))
    acc = tle.gpu.wgmma_wait(1, acc)
    offsets = tl.arange(0, 128)[:, None] * 128 + tl.arange(0, 128)[None, :]
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
    assert "musa_tle.auto_shared_layout" in ttir


def test_mthreads_tle_sqmma_rejects_nonzero_pending_groups():
    with pytest.raises(CompilationError, match="requires pendings=0"):
        compile_to_ttir(_tle_sqmma_nonzero_wait_kernel, {"out": "*fp32"})


def test_mthreads_tle_sqmma_requires_auto_shared_layout(capfd):
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        compile_musa(_tle_sqmma_non_auto_layout_kernel, {"out": "*fp32"})
    captured = capfd.readouterr()
    assert "requires layout=None and nv_mma_shared_layout=True" in captured.err


def test_mthreads_tle_sqmma_lowers_to_parameterless_wait():
    compiled = compile_musa(_tle_sqmma_kernel, {"out": "*fp32"})
    assert "call void @llvm.musa.sqmma.wait()" in compiled.asm["llir"]


@pytest.mark.parametrize("trans_a,trans_b,layout_a,layout_b", _SQMMA_TRANSPOSE_CASES)
def test_mthreads_tle_sqmma_transpose_ir(trans_a, trans_b, layout_a, layout_b):
    signature = {
        "out": "*fp32",
        "TRANS_A": "constexpr",
        "TRANS_B": "constexpr",
    }
    constexprs = {"TRANS_A": trans_a, "TRANS_B": trans_b}
    ttir = compile_to_ttir(_tle_sqmma_trans_compile_kernel, signature, constexprs)
    assert ("ttg.memdesc_trans" in ttir) == (trans_a or trans_b)
    assert "tt.trans" not in ttir
    compiled = compile_musa(_tle_sqmma_trans_compile_kernel, signature, constexprs)
    # Operand orientation is the contract; SSA names and view counts are not.
    assert f"layoutA = {layout_a} : i32, layoutB = {layout_b} : i32" in compiled.asm["ttgir"]


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
        shared_bytes.add(compiled.metadata.shared)
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
    assert "layoutA = 1 : i32, layoutB = 1 : i32" in compiled.asm["ttgir"]


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
    assert "llvm.musa.sqmma.fmma." in kernel.asm["llir"]


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
    assert f"llvm.musa.sqmma.{intrinsic_tag}" in kernel.asm["llir"]


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
    assert "llvm.musa.sqmma.fmma." in kernel.asm["llir"]


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
    assert "llvm.musa.sqmma.fmma." in kernel.asm["llir"]


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
    assert "llvm.musa.sqmma.fmma." in kernel.asm["llir"]


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
    assert f"llvm.musa.sqmma.{intrinsic_tag}" in kernel.asm["llir"]


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

    assert "llvm.musa.sqmma.fmma." in kernel.asm["llir"]


@triton.jit
def _sqmma_partition_dot_role(a, b, out, N: tl.constexpr):
    r = tl.arange(0, 16)
    k = tl.arange(0, 32)
    c = tl.arange(0, 64)
    sx = tle.gpu.alloc((16, 32), dtype=a.dtype.element_ty)
    sy = tle.gpu.alloc((32, 64), dtype=b.dtype.element_ty)
    for i in tl.range(0, N, num_stages=1):
        x = tl.load(a + i * 512 + r[:, None] * 32 + k[None, :])
        y = tl.load(b + i * 2048 + k[:, None] * 64 + c[None, :])
        tl.store(tle.gpu.local_ptr(sx), x)
        tl.store(tle.gpu.local_ptr(sy), y)
        x = tl.load(tle.gpu.local_ptr(sx))
        y = tl.load(tle.gpu.local_ptr(sy))
        z = tl.dot(x, y)
        tl.store(out + i * 1024 + r[:, None] * 64 + c[None, :], z)


@triton.jit
def _sqmma_partition_idle_role():
    pass


@triton.jit
def _sqmma_partition_dot_kernel(a, b, out, N: tl.constexpr, MODE: tl.constexpr):
    if MODE == 0:
        tle.gpu.warp_specialize([(_sqmma_partition_dot_role, (a, b, out, N)), (_sqmma_partition_idle_role, ())], [4],
                                [24])
    elif MODE == 1:
        tle.gpu.warp_specialize([(_sqmma_partition_idle_role, ()), (_sqmma_partition_dot_role, (a, b, out, N))], [4],
                                [24])
    elif MODE == 2:
        tle.gpu.warp_specialize([(_sqmma_partition_idle_role, ()), (_sqmma_partition_dot_role, (a, b, out, N)),
                                 (_sqmma_partition_idle_role, ())], [4, 4], [240, 168])
    else:
        _sqmma_partition_dot_role(a, b, out, N)


def _run_sqmma_partition_dot(mode, dtype_name, stages):
    torch.manual_seed(20260905)
    dtype = getattr(torch, dtype_name)
    n = 9
    # Independent inputs on every iteration detect stale shared operands.
    a_cpu = torch.randn((n, 16, 32)).to(dtype)
    b_cpu = torch.randn((n, 32, 64)).to(dtype)
    expected = a_cpu.double() @ b_cpu.double()
    a, b = a_cpu.to("musa"), b_cpu.to("musa")
    out = torch.empty((n, 16, 64), device="musa", dtype=torch.float32)
    compiled = _sqmma_partition_dot_kernel.warmup(a, b, out, n, mode, grid=(1, ), num_warps=4, num_stages=stages)
    llir = compiled.asm["llir"]
    intrinsic = "bfmma" if dtype_name == "bfloat16" else "fmma"
    assert f"llvm.musa.sqmma.{intrinsic}." in llir
    if mode != 3:
        # A CTA-wide rendezvous inside a static partition can deadlock.
        assert "llvm.musa.barrier0" not in llir
    for _ in range(3):
        out.fill_(float("nan"))
        compiled[(1, 1, 1)](a, b, out, n, mode)
        torch.musa.synchronize()
        torch.testing.assert_close(out.cpu().double(), expected, atol=0.002, rtol=0.002)
    print("SQMMA_PARTITION_PASS", mode, dtype_name, stages, flush=True)


@pytest.mark.parametrize("mode", [0, 1, 2, 3], ids=["default", "worker", "three_roles", "no_ws"])
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
@pytest.mark.parametrize("stages", [1, 3])
def test_sqmma_partition_runtime(mode, dtype, stages):
    completed = subprocess.run(
        [sys.executable,
         str(Path(__file__).resolve()), "--partition-dot",
         str(mode), dtype,
         str(stages)], capture_output=True, text=True, timeout=120, env={**os.environ, "TRITON_ALWAYS_COMPILE": "1"})
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "SQMMA_PARTITION_PASS" in completed.stdout


_SQMMA_ALIAS_WAITS = ("ttmg.squad_dot_wait", "mtgpu.sqmma_wait")
_SQMMA_ALIAS_CASES = ("single", "multiple", "tensor_first", "duplicate", "chain", "swapped", "view")


def _sqmma_alias_fixture(wait, case, reuse=False):
    n = 64
    storage = 128 if case == "view" else n
    mem = f"!ttg.memdesc<{n}xf32, #shared, #smem, mutable>"
    base = f"!ttg.memdesc<{storage}xf32, #shared, #smem, mutable>"
    tensor = f"tensor<{n}xf32, #blocked>"
    base_tensor = f"tensor<{storage}xf32, #blocked>"
    lines = [
        f"%va = arith.constant dense<11.0> : {base_tensor}",
        f"%vb = arith.constant dense<23.0> : {base_tensor}",
        f"%vc = arith.constant dense<37.0> : {tensor}",
        f'%a = ttg.local_alloc %va {{test.alias_id = "a"}} : ({base_tensor}) -> {base}',
        f'%b = ttg.local_alloc %vb {{test.alias_id = "b"}} : ({base_tensor}) -> {base}',
    ]
    a, b = "%a", "%b"
    if case == "view":
        lines += [
            f"%av = ttg.memdesc_subslice %a [64] : {base} -> {mem}",
            f"%bv = ttg.memdesc_subslice %b [0] : {base} -> {mem}",
        ]
        a, b = "%av", "%bv"
    inputs, types = [a, b], [mem, mem]
    ai, bi = 0, 1
    if case == "single":
        inputs, types = [a], [mem]
    elif case == "tensor_first":
        inputs, types = ["%vc", a, b], [tensor, mem, mem]
        ai, bi = 1, 2
    elif case == "carrier_first":
        acc = "tensor<64x64xf32, #mma>"
        carrier = f"!mtgpu.sqmma_accumulator<{acc}>"
        lines += [
            f"%zero = arith.constant dense<0.0> : {acc}",
            f"%acc = mtgpu.pack_sqmma_accumulator %zero : {acc} -> {carrier}",
        ]
        inputs, types = ["%acc", a, b], [carrier, mem, mem]
        ai, bi = 1, 2
    elif case == "duplicate":
        inputs, types = [a, a, b], [mem, mem, mem]
        ai, bi = 0, 2
    elif case == "swapped":
        inputs, types = [b, a], [mem, mem]
        ai, bi = 1, 0
    lines += [f"%ready:{len(inputs)} = {wait} {', '.join(inputs)} : {', '.join(types)}"]
    a_ready = f"%ready#{ai}"
    b_ready = b if case == "single" else f"%ready#{bi}"
    if case == "chain":
        lines += [f"%again:2 = {wait} {b_ready}, {a_ready} : {mem}, {mem}"]
        a_ready, b_ready = "%again#1", "%again#0"
    if reuse:
        lines += [f"%la = ttg.local_load {a_ready} : {mem} -> {tensor}"]
    lines += [f'%c = ttg.local_alloc %vc {{test.alias_id = "c"}} : ({tensor}) -> {mem}']
    if not reuse:
        lines += [f"%la = ttg.local_load {a_ready} : {mem} -> {tensor}"]
    lines += [
        f"%lb = ttg.local_load {b_ready} : {mem} -> {tensor}",
        f"%lc = ttg.local_load %c : {mem} -> {tensor}",
        "%r = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32, #blocked>",
        "%c64 = arith.constant dense<64> : tensor<64xi32, #blocked>",
        "%c128 = arith.constant dense<128> : tensor<64xi32, #blocked>",
        "%r1 = arith.addi %r, %c64 : tensor<64xi32, #blocked>",
        "%r2 = arith.addi %r, %c128 : tensor<64xi32, #blocked>",
        "%p = tt.splat %out : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked>",
    ]
    for index, value in enumerate(("%la", "%lb", "%lc")):
        offset = "%r" if index == 0 else f"%r{index}"
        lines += [
            f"%p{index} = tt.addptr %p, {offset} : tensor<64x!tt.ptr<f32>, #blocked>, tensor<64xi32, #blocked>",
            f"tt.store %p{index}, {value} : tensor<64x!tt.ptr<f32>, #blocked>",
        ]
    return """#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#mma = #ttg.musa_sqmma<{versionMajor = 3, versionMinor = 1, warpsPerCTA = [4, 1], instrShape = [64, 64, 64]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "musa:31", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @alias_probe(%out: !tt.ptr<f32>) {
""" + "\n".join(lines) + "\ntt.return\n}\n}\n"


def _allocate_sqmma_alias(path):
    _, backend = mthreads_backend()
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    module = ir.parse_mlir_module(str(path), context)
    pm = ir.pass_manager(context)
    # Deliberately no canonicalizer: wait forwarding itself must be understood.
    libtriton.mthreads.passes.ttgpuir.add_allocate_shared_memory(pm, 31)
    pm.run(module, "sqmma_wait_alias")
    return module.str_nodebug()


def _assert_sqmma_alias_allocation(text, case, reuse):
    buffers = {}
    for line in text.splitlines():
        tag = re.search(r'test.alias_id = "([abc])"', line)
        if not tag:
            continue
        offset = int(re.search(r"allocation.offset = (\d+) : i32", line)[1])
        elements = int(re.search(r"!ttg.memdesc<(\d+)xf32", line)[1])
        buffers[tag[1]] = (offset, offset + 4 * elements)
    assert set(buffers) == {"a", "b", "c"}, text

    def separate(x, y):
        return x[1] <= y[0] or y[1] <= x[0]

    assert separate(buffers["a"], buffers["b"]), text
    assert separate(buffers["b"], buffers["c"]), text
    if reuse:
        # A ends before C starts; B remains live. Unioning all input aliases
        # into every wait result would incorrectly keep A live with B.
        assert buffers["a"][0] <= buffers["c"][0] < buffers["c"][1] <= buffers["a"][1], text
        expected = 2 * (128 if case == "view" else 64) * 4
        assert f"ttg.shared = {expected} : i32" in text, text
    else:
        assert separate(buffers["a"], buffers["c"]), text


@pytest.mark.parametrize("wait", _SQMMA_ALIAS_WAITS)
@pytest.mark.parametrize("case", _SQMMA_ALIAS_CASES)
@pytest.mark.parametrize("reuse", [False, True], ids=["overlap", "reuse"])
def test_wait_alias_allocation(tmp_path, wait, case, reuse):
    path = tmp_path / "alias.ttgir"
    path.write_text(_sqmma_alias_fixture(wait, case, reuse))
    _assert_sqmma_alias_allocation(_allocate_sqmma_alias(path), case, reuse)


@pytest.mark.parametrize("reuse", [False, True])
def test_carrier_first_wait_alias_allocation(tmp_path, reuse):
    path = tmp_path / "carrier.ttgir"
    path.write_text(_sqmma_alias_fixture("mtgpu.sqmma_wait", "carrier_first", reuse))
    _assert_sqmma_alias_allocation(_allocate_sqmma_alias(path), "carrier_first", reuse)


def _run_sqmma_alias_runtime(path):
    compiled = triton.compile(str(path), target=musa_target(), options={"num_warps": 4, "num_stages": 1})
    assert "llvm.musa.sqmma.wait" in compiled.asm["llir"]
    out = torch.empty(3 * 64, dtype=torch.float32, device="musa")
    expected = torch.tensor([11., 23., 37.]).reshape(3, 1).expand(3, 64)
    for _ in range(3):
        out.fill_(float("nan"))
        compiled[(1, 1, 1)](out)
        torch.musa.synchronize()
        torch.testing.assert_close(out.cpu().reshape(3, 64), expected, atol=0, rtol=0)
    print("ALIAS_RUNTIME_PASS", flush=True)


@pytest.mark.parametrize("wait", _SQMMA_ALIAS_WAITS)
@pytest.mark.parametrize("case", ["multiple", "tensor_first", "chain", "swapped", "view"])
@pytest.mark.parametrize("reuse", [False, True])
def test_wait_alias_runtime(tmp_path, wait, case, reuse):
    path = tmp_path / "runtime.ttgir"
    path.write_text(_sqmma_alias_fixture(wait, case, reuse))
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--wait-alias-runtime",
         str(path)], capture_output=True, text=True, timeout=120, env={**os.environ, "TRITON_ALWAYS_COMPILE": "1"})
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "ALIAS_RUNTIME_PASS" in completed.stdout


def test_unknown_memdesc_operation_still_rejected(tmp_path):
    mem = "!ttg.memdesc<64xf32, #shared, #smem, mutable>"
    source = _sqmma_alias_fixture("ttmg.squad_dot_wait", "multiple")
    source = source.replace(f"%ready:2 = ttmg.squad_dot_wait %a, %b : {mem}, {mem}",
                            f"%ready:2 = builtin.unrealized_conversion_cast %a, %b : {mem}, {mem} to {mem}, {mem}")
    path = tmp_path / "unknown.ttgir"
    path.write_text(source)
    # The fix must not turn arbitrary descriptor-producing operations into
    # guessed aliases. Isolate the intentionally failing compiler invocation.
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--wait-alias-allocate",
         str(path)], capture_output=True, text=True, timeout=120)
    assert completed.returncode != 0
    assert "unknown operation creating memory descriptor" in completed.stderr


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--partition-dot":
        _run_sqmma_partition_dot(int(sys.argv[2]), sys.argv[3], int(sys.argv[4]))
    elif len(sys.argv) > 1 and sys.argv[1] == "--wait-alias-runtime":
        _run_sqmma_alias_runtime(Path(sys.argv[2]))
    elif len(sys.argv) > 1 and sys.argv[1] == "--wait-alias-allocate":
        import resource
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        print(_allocate_sqmma_alias(Path(sys.argv[2])))
    else:
        raise SystemExit("Use pytest to run SQMMA tests.")
