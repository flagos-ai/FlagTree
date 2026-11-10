import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback
import warnings
from types import SimpleNamespace

import pytest
import torch

import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton._C.libtriton import ir
from triton.backends.compiler import DotCap, DotSupport, GPUTarget
from triton.compiler import ASTSource
from triton.language.semantic import TritonSemantic
from triton.runtime.interpreter import _encode_fp64_to_float


@triton.jit
def precision_dot(A, B, ACC, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, B_COLUMN: tl.constexpr = False):
    rows = tl.arange(0, M)
    cols = tl.arange(0, N)
    inner = tl.arange(0, K)
    a = tl.load(A + rows[:, None] * K + inner[None, :])
    if B_COLUMN:
        b = tl.load(B + inner[:, None] + cols[None, :] * K)
    else:
        b = tl.load(B + inner[:, None] * N + cols[None, :])
    acc = None
    if ACC is not None:
        acc = tl.load(ACC + rows[:, None] * N + cols[None, :])
    result = tl.dot(a, b, acc=acc, out_dtype=C.dtype.element_ty)
    tl.store(C + rows[:, None] * N + cols[None, :], result)


@triton.jit
def precision_cast(X, Y, N: tl.constexpr, BLOCK: tl.constexpr, ROUND: tl.constexpr = None):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + offsets, offsets < N, other=0)
    tl.store(Y + offsets, value.to(Y.dtype.element_ty, fp_downcast_rounding=ROUND), offsets < N)


@triton.jit
def precision_fp4_encode(X, Y, N: tl.constexpr, ROUND: tl.constexpr, BLOCK: tl.constexpr = 512):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(X + offsets, offsets < N, other=0)
    packed = tl.fp_to_fp4(values, fp_downcast_rounding=ROUND)
    out_offsets = tl.program_id(0) * (BLOCK // 2) + tl.arange(0, BLOCK // 2)
    tl.store(Y + out_offsets, packed, out_offsets * 2 < N)


@triton.jit
def precision_fp4_decode(X, S, Y, N: tl.constexpr, BLOCK: tl.constexpr = 256):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    packed = tl.load(X + offsets, offsets < N, other=0)
    scale = None
    if S is not None:
        scale = tl.load(S + offsets, offsets < N, other=127)
    values = tl.fp4_to_fp_scaled(packed, scale, Y.dtype.element_ty, axis=-1)
    out_offsets = tl.program_id(0) * (BLOCK * 2) + tl.arange(0, BLOCK * 2)
    tl.store(Y + out_offsets, values, out_offsets < N * 2)


@triton.jit
def _pack_int4(values):
    low, high = tl.split(tl.reshape(values, (values.shape[0] // 2, 2)))
    return ((low & 15) | ((high & 15) << 4)).to(tl.uint8)


@triton.jit
def precision_int4_roundtrip(X, U, Y, N: tl.constexpr, BLOCK: tl.constexpr = 256):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    packed = tl.load(X + offsets, offsets < N, other=0)
    low = (packed << 4).to(tl.int8, bitcast=True) >> 4
    high = packed.to(tl.int8, bitcast=True) >> 4
    values = tl.interleave(low, high)
    repacked = _pack_int4(values)
    logical = tl.program_id(0) * BLOCK * 2 + tl.arange(0, BLOCK * 2)
    tl.store(U + logical, values, logical < N * 2)
    tl.store(Y + offsets, repacked, offsets < N)


@triton.jit
def precision_int4_quantize(X, Y, N: tl.constexpr, BLOCK: tl.constexpr = 256):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(X + offsets, offsets < N, other=0).to(tl.int32)
    packed = _pack_int4(values)
    output_offsets = tl.program_id(0) * (BLOCK // 2) + tl.arange(0, BLOCK // 2)
    tl.store(Y + output_offsets, packed, output_offsets * 2 < N)


@triton.jit
def precision_int4_dot(A, B, ACC, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, AXIS_A: tl.constexpr,
                       AXIS_B: tl.constexpr):
    ma = tl.arange(0, M // (2 if AXIS_A == 0 else 1))
    ka = tl.arange(0, K // (2 if AXIS_A == 1 else 1))
    kb = tl.arange(0, K // (2 if AXIS_B == 0 else 1))
    nb = tl.arange(0, N // (2 if AXIS_B == 1 else 1))
    a = tl.load(A + ma[:, None] * ka.shape[0] + ka[None, :])
    b = tl.load(B + kb[:, None] * nb.shape[0] + nb[None, :])
    rows = tl.arange(0, M)
    cols = tl.arange(0, N)
    acc = None
    if ACC is not None:
        acc = tl.load(ACC + rows[:, None] * N + cols[None, :])
    result = tl.dot_scaled(a, None, "int4", b, None, "int4", acc=acc, lhs_k_pack=AXIS_A == 1, rhs_k_pack=AXIS_B == 0,
                           out_dtype=tl.int32)
    tl.store(C + rows[:, None] * N + cols[None, :], result)


@triton.jit
def precision_tle_copy(X, Y, N: tl.constexpr, STRIDE: tl.constexpr, ASYNC: tl.constexpr, BLOCK: tl.constexpr = 1024):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tle.load(X + offsets * STRIDE, mask=offsets < N, other=0, is_async=ASYNC)
    tl.store(Y + offsets, value, offsets < N)


@triton.jit
def precision_tle_copy_loop(X, Y, N: tl.constexpr, BLOCK: tl.constexpr = 1024):
    local = tl.arange(0, BLOCK)
    for chunk in range(4):
        offsets = (tl.program_id(0) * 4 + chunk) * BLOCK + local
        value = tle.load(X + offsets, mask=offsets < N, other=0, is_async=True)
        tl.store(Y + offsets, value, offsets < N)


@triton.jit
def precision_shared_copy(X, Y, N: tl.constexpr, COPY: tl.constexpr, BLOCK: tl.constexpr = 1024):
    local = tl.arange(0, BLOCK)
    offsets = tl.program_id(0) * BLOCK + local
    if COPY:
        shared = tle.gpu.alloc((BLOCK, ), X.dtype.element_ty, nv_mma_shared_layout=False)
        tle.gpu.copy(X + offsets, shared, (BLOCK, ), mask=offsets < N, other=0)
    else:
        value = tl.load(X + offsets, mask=offsets < N, other=0)
        shared = tle.gpu.alloc((BLOCK, ), X.dtype.element_ty, init_value=value, nv_mma_shared_layout=False)
    tl.debug_barrier()
    # Both exchanges cross warp boundaries and must retain every stored bit.
    permuted = tl.load(tle.gpu.local_ptr(shared, (local ^ 64, )), volatile=True)
    tl.debug_barrier()
    tl.store(tle.gpu.local_ptr(shared, (local, )), permuted)
    tl.debug_barrier()
    restored = tl.load(tle.gpu.local_ptr(shared, (local ^ 64, )), volatile=True)
    if COPY:
        tl.debug_barrier()
        tl.store(tle.gpu.local_ptr(shared, (local, )), restored)
        tl.debug_barrier()
        tle.gpu.copy(shared, Y + offsets, (BLOCK, ), mask=offsets < N)
    else:
        tl.store(Y + offsets, restored, mask=offsets < N)


@triton.jit
def precision_reverse_copy(X, Y, N: tl.constexpr, STRIDE: tl.constexpr = 1, OFFSET: tl.constexpr = 0,
                           BLOCK: tl.constexpr = 1024, OVERWRITE: tl.constexpr = False):
    local = tl.arange(0, BLOCK)
    offsets = tl.program_id(0) * BLOCK + local
    shared = tle.gpu.alloc((BLOCK, ), X.dtype.element_ty, nv_mma_shared_layout=False)
    # Two different tiles exercise immediate LDS reuse after the first copy.
    for phase in tl.static_range(2):
        permuted = tl.program_id(0) * BLOCK + (local ^ 64)
        value = tl.load(X + phase * N + permuted, permuted < N, other=0)
        tl.store(tle.gpu.local_ptr(shared, (local ^ 64, )), value)
        tl.debug_barrier()
        destination = Y + OFFSET + (phase * N + offsets) * STRIDE
        if OVERWRITE:
            saved = tl.load(tle.gpu.local_ptr(shared, (local, )))
            tl.store(tle.gpu.local_ptr(shared, (local, )), tl.full((BLOCK, ), 0, X.dtype.element_ty))
            tl.store(destination, saved, offsets < N)
        elif N % BLOCK == 0:
            tle.gpu.copy(shared, destination, (BLOCK, ))
        else:
            tle.gpu.copy(shared, destination, (BLOCK, ), mask=offsets < N)
        tl.debug_barrier()


@triton.jit
def precision_reverse_subview(X, Y, SOURCE_OFFSET: tl.constexpr, BLOCK: tl.constexpr = 1024):
    full = tl.arange(0, 2 * BLOCK)
    value = tl.load(X + full)
    shared = tle.gpu.alloc((2 * BLOCK, ), X.dtype.element_ty, init_value=value, nv_mma_shared_layout=False)
    tl.debug_barrier()
    local = tl.arange(0, BLOCK)
    selected = tl.load(tle.gpu.local_ptr(shared, (local + SOURCE_OFFSET, )))
    tl.store(Y + local, selected)


@triton.jit
def precision_reverse_alias(X, Y, OFFSET_BYTES: tl.constexpr, BLOCK: tl.constexpr = 1024):
    full = tl.arange(0, 2 * BLOCK)
    shared = tle.gpu.alloc((2 * BLOCK, ), X.dtype.element_ty, init_value=tl.load(X + full), nv_mma_shared_layout=False)
    alias = tle.gpu.alloc((BLOCK, ), X.dtype.element_ty, alias=shared, alias_offset_bytes=OFFSET_BYTES,
                          nv_mma_shared_layout=False)
    tl.debug_barrier()
    tle.gpu.copy(alias, Y + tl.arange(0, BLOCK), (BLOCK, ))


@triton.jit
def precision_tle_local_load(X, Y, ASYNC: tl.constexpr, BLOCK: tl.constexpr = 1024):
    local = tl.arange(0, BLOCK)
    shared = tle.gpu.alloc((BLOCK, ), X.dtype.element_ty, init_value=tl.load(X + local), nv_mma_shared_layout=False)
    tl.debug_barrier()
    value = tle.load(tle.gpu.local_ptr(shared, (local, )), mask=local < BLOCK - 1, other=0, is_async=ASYNC)
    tl.store(Y + local, value)


@triton.jit
def precision_integer_scaled(A, SA, B, SB, ACC, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                             FORMAT_A: tl.constexpr, FORMAT_B: tl.constexpr, OUT_DTYPE: tl.constexpr = tl.float32):
    rows = tl.arange(0, M)
    cols = tl.arange(0, N)
    ka = tl.arange(0, K // (2 if FORMAT_A == "int4" else 1))
    kb = tl.arange(0, K // (2 if FORMAT_B == "int4" else 1))
    a = tl.load(A + rows[:, None] * ka.shape[0] + ka[None, :])
    b = tl.load(B + kb[:, None] * N + cols[None, :])
    scale_a = None
    scale_b = None
    groups = tl.arange(0, K // 32)
    if SA is not None:
        scale_a = tl.load(SA + rows[:, None] * (K // 32) + groups[None, :])
    if SB is not None:
        scale_b = tl.load(SB + cols[:, None] * (K // 32) + groups[None, :])
    acc = tl.load(ACC + rows[:, None] * N + cols[None, :])
    result = tl.dot_scaled(a, scale_a, FORMAT_A, b, scale_b, FORMAT_B, acc=acc, out_dtype=OUT_DTYPE)
    tl.store(C + rows[:, None] * N + cols[None, :], result)


@triton.jit
def precision_scaled_dot(A, sa0, sa1, SA, B, sb0, sb1, SB, ACC, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                         FORMAT_A: tl.constexpr, FORMAT_B: tl.constexpr, LHS_K_PACK: tl.constexpr = True,
                         RHS_K_PACK: tl.constexpr = True):
    rows = tl.arange(0, M)
    cols = tl.arange(0, N)
    input_rows = tl.arange(0, M // (2 if FORMAT_A == "e2m1" and not LHS_K_PACK else 1))
    input_cols = tl.arange(0, N // (2 if FORMAT_B == "e2m1" and not RHS_K_PACK else 1))
    ka = tl.arange(0, K // (2 if FORMAT_A == "e2m1" and LHS_K_PACK else 1))
    kb = tl.arange(0, K // (2 if FORMAT_B == "e2m1" and RHS_K_PACK else 1))
    a = tl.load(A + input_rows[:, None] * sa0 + ka[None, :] * sa1)
    b = tl.load(B + kb[:, None] * sb0 + input_cols[None, :] * sb1)
    scale_a = None
    scale_b = None
    if SA is not None:
        groups = tl.arange(0, K // 32)
        scale_a = tl.load(SA + rows[:, None] * (K // 32) + groups[None, :])
    if SB is not None:
        groups = tl.arange(0, K // 32)
        scale_b = tl.load(SB + cols[:, None] * (K // 32) + groups[None, :])
    acc = None
    if ACC is not None:
        acc = tl.load(ACC + rows[:, None] * N + cols[None, :])
    result = tl.dot_scaled(a, scale_a, FORMAT_A, b, scale_b, FORMAT_B, acc=acc, lhs_k_pack=LHS_K_PACK,
                           rhs_k_pack=RHS_K_PACK)
    tl.store(C + rows[:, None] * N + cols[None, :], result)


@triton.jit
def precision_scaled_dot_loop(A, sa0, sa1, SA, B, sb0, sb1, SB, ACC, C, M: tl.constexpr, N: tl.constexpr,
                              K: tl.constexpr, FORMAT_A: tl.constexpr, FORMAT_B: tl.constexpr, BM: tl.constexpr = 16,
                              BN: tl.constexpr = 16, BK: tl.constexpr = 32, NATIVE: tl.constexpr = False,
                              VOLATILE: tl.constexpr = False, SINK=None, INDICES=None):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    ka = tl.arange(0, BK // (2 if FORMAT_A == "e2m1" else 1))
    kb = tl.arange(0, BK // (2 if FORMAT_B == "e2m1" else 1))
    acc = tl.full((BM, BN), 0, tl.float32)
    if ACC is not None:
        acc = tl.load(ACC + rows[:, None] * N + cols[None, :], mask=(rows[:, None] < M) & (cols[None, :] < N), other=0)
    for chunk in range(tl.cdiv(K, BK)):
        ak = chunk * (BK // (2 if FORMAT_A == "e2m1" else 1)) + ka
        if INDICES is not None:
            ak = chunk * (BK // 2) + tl.load(INDICES + chunk * (BK // 2) + ka)
        bk = chunk * (BK // (2 if FORMAT_B == "e2m1" else 1)) + kb
        a = tl.load(A + rows[:, None] * sa0 + ak[None, :] * sa1,
                    mask=(rows[:, None] < M) & (ak[None, :] < tl.cdiv(K, 2 if FORMAT_A == "e2m1" else 1)), other=0,
                    volatile=VOLATILE)
        b = tl.load(B + bk[:, None] * sb0 + cols[None, :] * sb1,
                    mask=(cols[None, :] < N) & (bk[:, None] < tl.cdiv(K, 2 if FORMAT_B == "e2m1" else 1)), other=0,
                    volatile=VOLATILE)
        if SINK is not None:
            tl.store(SINK + rows[:, None] * tl.cdiv(K, 2) + ak[None, :], a,
                     mask=(rows[:, None] < M) & (ak[None, :] < tl.cdiv(K, 2)))
        if FORMAT_A == "e2m1":
            a = tl.where(ak[None, :] * 2 + 1 < K, a, a & 15)
        if FORMAT_B == "e2m1":
            b = tl.where(bk[:, None] * 2 + 1 < K, b, b & 15)
        if NATIVE:
            acc = tl.dot(a, b, acc)
        else:
            scale_a = None
            scale_b = None
            groups = chunk * (BK // 32) + tl.arange(0, BK // 32)
            if SA is not None:
                scale_a = tl.load(SA + rows[:, None] * tl.cdiv(K, 32) + groups[None, :],
                                  mask=(rows[:, None] < M) & (groups[None, :] < tl.cdiv(K, 32)), other=127)
            if SB is not None:
                scale_b = tl.load(SB + cols[:, None] * tl.cdiv(K, 32) + groups[None, :],
                                  mask=(cols[:, None] < N) & (groups[None, :] < tl.cdiv(K, 32)), other=127)
            acc = tl.dot_scaled(a, scale_a, FORMAT_A, b, scale_b, FORMAT_B, acc=acc)
    tl.store(C + rows[:, None] * N + cols[None, :], acc, mask=(rows[:, None] < M) & (cols[None, :] < N))


@triton.jit
def precision_loaded_bf16_dot(A, B, ACC, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    rows = tl.arange(0, M)
    cols = tl.arange(0, N)
    inner = tl.arange(0, 32)
    acc = tl.load(ACC + rows[:, None] * N + cols[None, :])
    for start in range(0, K, 32):
        a = tl.load(A + rows[:, None] * K + start + inner[None, :])
        b = tl.load(B + (start + inner[:, None]) * N + cols[None, :])
        acc = tl.dot(a, b, acc)
    tl.store(C + rows[:, None] * N + cols[None, :], acc)


FP4_PIPELINE_CASES = [(a, b, k, scaled, 2)
                      for a, b in [("e2m1", "e2m1"), ("e2m1", "fp16"), ("e4m3", "e2m1")]
                      for k in [96, 160]
                      for scaled in [False, True]] + [("e2m1", "e2m1", k, scaled, stages)
                                                      for k in [96, 160]
                                                      for scaled in [False, True]
                                                      for stages in [3, 4]]
BF16_PROLOGUE_CASES = [(8, 3, 32, 32, 96), (8, 4, 32, 32, 96), (8, 3, 64, 64, 128), (8, 3, 64, 32, 128),
                       (4, 3, 64, 64, 128)]


def assert_fp4_pipeline(compiled, stages, formats, block=(32, 32, 32)):
    # Check the actual FP4 operand paths, not [M,1]/[N,1] scale buffers. The
    # MMA shared layout can hold packed bytes or already decoded FP16/BF16.
    buffers = re.findall(r"ttg.local_alloc\s*:\s*\(\)\s*-> !ttg.memdesc<(\d+)x(\d+)x(\d+)x(i8|f16|bf16),[^\n]*mutable>",
                         compiled.asm["ttgir"])
    buffers = [(int(count), int(rows), int(cols), dtype) for count, rows, cols, dtype in buffers]
    bm, bn, bk = block
    for operand, fmt in enumerate(formats):
        if fmt != "e2m1":
            continue
        packed = (bm, bk // 2) if operand == 0 else (bk // 2, bn)
        decoded = (bm, bk) if operand == 0 else (bk, bn)
        match = next((i for i, (count, rows, cols, dtype) in enumerate(buffers) if count >= stages - 1 and (
            (dtype == "i8" and (rows, cols) == packed) or (dtype in ("f16", "bf16") and (rows, cols) == decoded))),
                     None)
        assert match is not None, f"FP4 operand {operand} stage buffers missing: num_stages did not enable pipelining"
        buffers.pop(match)
    assert "ttg.memdesc_index" in compiled.asm["ttgir"]
    assert re.search(r"\bv_mmac_", compiled.asm["amdgcn"])


def assert_bf16_prologue(compiled):
    assert re.search(r"memdesc<[23]x\d+x\d+xbf16,", compiled.asm["ttgir"])
    lines = compiled.asm["llir"].splitlines()
    # The loaded FP32 accumulator's layout conversion ends in LDS reads.
    # Order those reads before the first BF16 stage-buffer write, even when
    # the allocator reuses its scratch for the stage buffers.
    # BF16 pairs may remain packed in i32 after buffer loads. Accumulator
    # scratch is FP32, so the first non-FP32 LDS store writes a stage buffer.
    first_store = next(i for i, line in enumerate(lines)
                       if re.search(r"\bstore\b.*ptr addrspace\(3\)", line) and not re.search(r"\bfloat\b", line))
    last_read = max(i for i, line in enumerate(lines[:first_store]) if re.search(r"\bload\b.*ptr addrspace\(3\)", line))
    assert any("@llvm.amdgcn.s.barrier" in line for line in lines[last_read + 1:first_store])


@pytest.mark.parametrize("format_a,format_b,inner,scaled,stages", FP4_PIPELINE_CASES)
def test_hcu_fp4_pipeline_compile(format_a, format_b, inner, scaled, stages, fresh_triton_cache):
    types = {"e2m1": "u8", "e4m3": "fp8e4nv", "fp16": "fp16"}
    signature = {"A": "*" + types[format_a], "B": "*" + types[format_b], "ACC": "*fp32", "C": "*fp32"}
    constants = dict(sa0=inner // (2 if format_a == "e2m1" else 1), sa1=1, sb0=32, sb1=1, M=32, N=32, K=inner,
                     FORMAT_A=format_a, FORMAT_B=format_b, BM=32, BN=32, BK=32, NATIVE=False, VOLATILE=False, SINK=None,
                     INDICES=None)
    for name in ["SA", "SB"]:
        if scaled:
            signature[name] = "*u8"
        else:
            constants[name] = None
    fn = precision_scaled_dot_loop
    attrs = {(fn.arg_names.index(name), ): [("tt.divisibility", 16)] for name in signature}
    compiled = triton.compile(ASTSource(fn, signature, constexprs=constants, attrs=attrs),
                              target=GPUTarget("hip", "gfx936", 64), options={"num_warps": 8, "num_stages": stages})
    assert_fp4_pipeline(compiled, stages, (format_a, format_b))
    if format_a == format_b == "e2m1":
        assert not re.search(r"ttg.local_alloc\s+%",
                             compiled.asm["ttgir"]), "redundant per-iteration MMA buffer remains"


@pytest.mark.parametrize("volatile,side_effect,dependent", [(True, False, False), (False, True, False),
                                                            (False, False, True)])
def test_hcu_fp4_pipeline_unsafe_graph(volatile, side_effect, dependent, fresh_triton_cache):
    signature = {"A": "*u8", "B": "*u8", "ACC": "*fp32", "C": "*fp32"}
    constants = dict(sa0=48, sa1=1, sb0=32, sb1=1, SA=None, SB=None, M=32, N=32, K=96, FORMAT_A="e2m1", FORMAT_B="e2m1",
                     BM=32, BN=32, BK=32, NATIVE=False, VOLATILE=volatile)
    if side_effect:
        signature["SINK"] = "*u8"
    else:
        constants["SINK"] = None
    if dependent:
        signature["INDICES"] = "*i32"
    else:
        constants["INDICES"] = None
    fn = precision_scaled_dot_loop
    attrs = {(fn.arg_names.index(name), ): [("tt.divisibility", 16)] for name in signature}
    compiled = triton.compile(ASTSource(fn, signature, constexprs=constants, attrs=attrs),
                              target=GPUTarget("hip", "gfx936", 64), options={"num_warps": 8, "num_stages": 3})
    assert not re.search(r"ttg.local_alloc\s*:\s*\(\)\s*-> !ttg.memdesc<\d+x\d+x\d+xi8,", compiled.asm["ttgir"])


@pytest.mark.parametrize("warps,stages,m,n,k", BF16_PROLOGUE_CASES)
def test_hcu_bf16_loaded_accumulator_compile(warps, stages, m, n, k, fresh_triton_cache):
    fn = precision_loaded_bf16_dot
    signature = {"A": "*bf16", "B": "*bf16", "ACC": "*fp32", "C": "*fp32"}
    attrs = {(fn.arg_names.index(name), ): [("tt.divisibility", 16)] for name in signature}
    compiled = triton.compile(ASTSource(fn, signature, constexprs=dict(M=m, N=n, K=k), attrs=attrs),
                              target=GPUTarget("hip", "gfx936", 64), options={"num_warps": warps, "num_stages": stages})
    assert_bf16_prologue(compiled)


@pytest.mark.parametrize("warps,stages,m,n,k", BF16_PROLOGUE_CASES)
def test_hcu_bf16_loaded_accumulator_values(warps, stages, m, n, k, device):
    generator = torch.Generator().manual_seed(1493)
    a = torch.randint(-4, 5, (m, k), generator=generator).to(torch.bfloat16)
    b = torch.randint(-4, 5, (k, n), generator=generator).to(torch.bfloat16)
    acc = (torch.arange(m * n).reshape(m, n) % 13).float() / 8
    expected = (a.double() @ b.double() + acc.double()).float()
    a, b, acc = a.to(device), b.to(device), acc.to(device)
    output = torch.full((m * n + 32, ), 1234.5, device=device)
    # Compile once and repeatedly execute exactly the same cached kernel.
    compiled = precision_loaded_bf16_dot[(1, )](a, b, acc, output[16:-16], m, n, k, num_warps=warps, num_stages=stages)
    assert_bf16_prologue(compiled)
    for iteration in range(500):
        precision_loaded_bf16_dot[(1, )](a, b, acc, output[16:-16], m, n, k, num_warps=warps, num_stages=stages)
        torch.testing.assert_close(output[16:-16].reshape(m, n).cpu(), expected, rtol=0, atol=0,
                                   msg=f"iteration={iteration}")
        assert torch.all(output[:16] == 1234.5) and torch.all(output[-16:] == 1234.5)


@pytest.mark.parametrize("format_a,format_b,inner,scaled,stages", FP4_PIPELINE_CASES)
def test_hcu_fp4_pipeline_repeated_values(format_a, format_b, inner, scaled, stages, device):
    m = n = 32
    generator = torch.Generator().manual_seed(1493)
    a, ra = precision_operand(format_a, (m, inner), 1, generator, device)
    b, rb = precision_operand(format_b, (inner, n), 0, generator, device)
    sa = (126 + torch.arange(m * (inner // 32)).reshape(m, inner // 32) % 3).to(torch.uint8) if scaled else None
    sb = (126 + (torch.arange(n * (inner // 32)).reshape(n, inner // 32) + 1) % 3).to(torch.uint8) if scaled else None
    if scaled:
        ra *= torch.exp2(sa.double() - 127).repeat_interleave(32, 1)
        rb *= torch.exp2(sb.double() - 127).repeat_interleave(32, 1).T
    acc = (torch.arange(m * n).reshape(m, n) % 13).float() / 8
    expected = (ra @ rb + acc.double()).float()
    output = torch.full((m * n + 32, ), 1234.5, device=device)
    args = (a, *a.stride(), sa.to(device) if scaled else None, b, *b.stride(), sb.to(device) if scaled else None,
            acc.to(device), output[16:-16], m, n, inner, format_a, format_b)
    compiled = precision_scaled_dot_loop[(1, )](*args, BM=32, BN=32, BK=32, num_warps=8, num_stages=stages)
    assert_fp4_pipeline(compiled, stages, (format_a, format_b))
    for iteration in range(500):
        precision_scaled_dot_loop[(1, )](*args, BM=32, BN=32, BK=32, num_warps=8, num_stages=stages)
        torch.testing.assert_close(output[16:-16].reshape(m, n).cpu(), expected, rtol=0, atol=0,
                                   msg=f"iteration={iteration}")
        assert torch.all(output[:16] == 1234.5) and torch.all(output[-16:] == 1234.5)


def compile_scaled(format_a, format_b, scale_mode, with_acc=True, geometry=(16, 16, 32), k_pack=True):
    types = {"e2m1": "u8", "e4m3": "fp8e4nv", "bf16": "bf16", "fp16": "fp16"}
    signature = {
        "A": "*" + types[format_a], "B": "*" + types[format_b], "C": "*fp32", "sa0": "i32", "sa1": "i32", "sb0": "i32",
        "sb1": "i32"
    }
    m, n, k = geometry
    constants = {
        "M": m, "N": n, "K": k, "FORMAT_A": format_a, "FORMAT_B": format_b, "LHS_K_PACK": k_pack or format_a != "e2m1",
        "RHS_K_PACK": k_pack or format_b != "e2m1"
    }
    for name, present in [("SA", scale_mode in ("lhs", "both")), ("SB", scale_mode in ("rhs", "both")),
                          ("ACC", with_acc)]:
        if present:
            signature[name] = "*fp32" if name == "ACC" else "*u8"
        else:
            constants[name] = None
    return triton.compile(ASTSource(precision_scaled_dot, signature, constexprs=constants),
                          target=GPUTarget("hip", "gfx936", 64), options={"num_warps": 4, "num_stages": 1})


FORMAT_PAIRS = [("e2m1", "e2m1"), ("e2m1", "e4m3"), ("e4m3", "e2m1"), ("e2m1", "bf16"), ("bf16", "e2m1"),
                ("e2m1", "fp16"), ("fp16", "e2m1"), ("e4m3", "e4m3")]


@pytest.mark.parametrize("format_a,format_b", FORMAT_PAIRS)
@pytest.mark.parametrize("scale_mode", ["none", "lhs", "rhs", "both"])
@pytest.mark.parametrize("with_acc", [False, True])
def test_hcu_scaled_precision_compile(format_a, format_b, scale_mode, with_acc, fresh_triton_cache):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compiled = compile_scaled(format_a, format_b, scale_mode, with_acc)
    assert any("dot_scaled" in str(item.message) and "native=false" in str(item.message) for item in caught)
    assert not re.search(r"\btt\.dot_scaled\s", compiled.asm["ttgir"])
    assert re.search(r"\bv_mmac_", compiled.asm["amdgcn"])
    if "e2m1" in (format_a, format_b):
        assert "cvt_pk_f32_fp8" not in compiled.asm["amdgcn"]


@pytest.mark.parametrize("formats", [("e2m1", "e2m1"), ("e4m3", "e4m3"), ("e2m1", "fp16")])
@pytest.mark.parametrize("geometry,k_pack", [((32, 32, 64), True), ((64, 64, 128), True), ((32, 32, 64), False)])
@pytest.mark.parametrize("scale_mode", ["lhs", "both"])
def test_hcu_scaled_group_layout_compile(formats, geometry, k_pack, scale_mode, fresh_triton_cache):
    with pytest.warns(UserWarning, match="native=false"):
        compiled = compile_scaled(*formats, scale_mode, geometry=geometry, k_pack=k_pack)
    assert re.search(r"\bv_mmac_", compiled.asm["amdgcn"])
    assert not re.search(r"\btt\.dot_scaled\s", compiled.asm["ttgir"])


@pytest.mark.parametrize("inner", [1, 4, 16, 32])
def test_hcu_integer_dot_compile(inner, fresh_triton_cache):
    signature = {"A": "*i8", "B": "*i8", "ACC": "*i32", "C": "*i32"}
    compiled = triton.compile(
        ASTSource(precision_dot, signature, constexprs={"M": 16, "N": 16, "K": inner, "B_COLUMN": False}),
        target=GPUTarget("hip", "gfx936", 64), options={"num_warps": 4, "num_stages": 1})
    assert not re.search(r"\b(sitofp|uitofp|fptosi|fptoui)\b", compiled.asm["llir"])


@pytest.mark.parametrize("dtype,inner,matrix,software", [
    ("fp64", 4, False, True),
    ("fp32", 4, False, True),
    ("fp32", 8, True, False),
    ("fp16", 8, False, True),
    ("fp16", 16, True, False),
    ("fp8e4nv", 32, True, True),
    ("i8", 4, False, True),
    ("i8", 32, True, False),
])
def test_hcu_dot_path_diagnostics(dtype, inner, matrix, software, fresh_triton_cache):
    output = "i32" if dtype == "i8" else "fp32" if dtype == "fp8e4nv" else dtype
    signature = {"A": "*" + dtype, "B": "*" + dtype, "ACC": "*" + output, "C": "*" + output}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compiled = triton.compile(
            ASTSource(precision_dot, signature, constexprs={"M": 16, "N": 16, "K": inner, "B_COLUMN": False}),
            target=GPUTarget("hip", "gfx936", 64), options={"num_warps": 4, "num_stages": 1})
    assert bool(re.search(r"\bv_mmac_", compiled.asm["amdgcn"])) == matrix
    assert any("native=false" in str(item.message) for item in caught) == software


@pytest.mark.parametrize("dtype,accumulator", [("i8", "fp32"), ("fp32", "i32")])
def test_hcu_dot_accumulator_dtype(dtype, accumulator, fresh_triton_cache):
    output = "i32" if dtype == "i8" else dtype
    signature = {"A": "*" + dtype, "B": "*" + dtype, "ACC": "*" + accumulator, "C": "*" + output}
    with pytest.raises(triton.CompilationError, match="expected accumulator dtype"):
        triton.compile(ASTSource(precision_dot, signature, constexprs={"M": 16, "N": 16, "K": 16, "B_COLUMN": False}),
                       target=GPUTarget("hip", "gfx936", 64), options={"num_warps": 4})


@pytest.mark.parametrize("arch", ["gfx936", "gfx938"])
@pytest.mark.parametrize("a,b", [("fp8e4nv", "fp8e4nv"), ("fp8e4nv", "fp8e5"), ("fp8e5", "fp8e4nv"),
                                 ("fp8e5", "fp8e5")])
def test_hcu_fp8_dot_target_paths(arch, a, b, fresh_triton_cache):
    signature = {"A": "*" + a, "B": "*" + b, "C": "*fp32"}
    constants = {"ACC": None, "M": 16, "N": 16, "K": 32, "B_COLUMN": False}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compiled = triton.compile(ASTSource(precision_dot, signature, constexprs=constants),
                                  target=GPUTarget("hip", arch, 64), options={"num_warps": 4, "num_stages": 1})
    software = arch == "gfx936"
    instruction_types = "f16" if software else "(?:fp8|bf8)"
    assert re.search(r"\bv_mmac_.*" + instruction_types + r"\b", compiled.asm["amdgcn"])
    assert any(re.search(r"tl.dot.*native=false", str(item.message)) for item in caught) == software


@pytest.mark.parametrize("a,b", [("u8", "u8"), ("i16", "i16"), ("i32", "i8"), ("i8", "i32"), ("i32", "i32"),
                                 ("fp16", "bf16")])
def test_hcu_dot_format_rejection(a, b, fresh_triton_cache):
    with pytest.raises(triton.CompilationError, match="is not supported on hip/gfx936"):
        triton.compile(
            ASTSource(precision_dot, {"A": "*" + a, "B": "*" + b, "C": "*fp32"},
                      constexprs={"ACC": None, "M": 16, "N": 16, "K": 32, "B_COLUMN": False}),
            target=GPUTarget("hip", "gfx936", 64), options={"num_warps": 4})


@pytest.mark.parametrize("dtype,block,message", [("i32", 512, "requires FP16/BF16/FP32/FP64"),
                                                 ("fp32", 1, "packing dimension must be even")])
def test_hcu_fp4_encoding_rejection(dtype, block, message, fresh_triton_cache):
    with pytest.raises(triton.CompilationError, match=message):
        triton.compile(
            ASTSource(precision_fp4_encode, {"X": "*" + dtype, "Y": "*u8"},
                      constexprs={"N": block, "ROUND": "rtne", "BLOCK": block}), target=GPUTarget("hip", "gfx936", 64),
            options={"num_warps": 4})


def test_hcu_int4_container_rejection(fresh_triton_cache):
    with pytest.raises(triton.CompilationError) as caught:
        triton.compile(
            ASTSource(precision_int4_dot, {"A": "*i8", "B": "*u8", "C": "*i32"},
                      constexprs={"ACC": None, "M": 16, "N": 16, "K": 32, "AXIS_A": 1, "AXIS_B": 0}),
            target=GPUTarget("hip", "gfx936", 64), options={"num_warps": 4})
    assert "INT4 operands require packed uint8" in ''.join(traceback.format_exception(caught.value))


@pytest.mark.parametrize("operation", ["dot", "scaled"])
@pytest.mark.parametrize("support", [None, DotSupport.NATIVE, DotSupport.NON_NATIVE, DotSupport.UNSUPPORTED])
def test_precision_capability_semantics(operation, support):

    class ReachedLowering(Exception):
        pass

    def stop(*args):
        raise ReachedLowering()

    cap = None if support is None else DotCap(support, "precision capability diagnostic")
    builder = SimpleNamespace(
        codegen_fns={"resolve_dot": lambda *args: cap, "resolve_dot_scaled": lambda *args: cap, "min_dot_size": stop},
        options=SimpleNamespace(allowed_dot_input_precisions=("ieee", )), get_fp32=stop)
    semantic = TritonSemantic(builder)
    dtype = tl.float16 if operation == "dot" else tl.bfloat16
    operand = tl.tensor(None, tl.block_type(dtype, [16, 16]))
    expected = ValueError if support == DotSupport.UNSUPPORTED else ReachedLowering
    message = "precision capability diagnostic" if support == DotSupport.UNSUPPORTED else None
    with warnings.catch_warnings(record=True) as caught, pytest.raises(expected, match=message):
        warnings.simplefilter("always")
        if operation == "dot":
            semantic.dot(operand, operand, None, "ieee", 0, tl.float32)
        else:
            semantic.dot_scaled(operand, None, tl.constexpr("bf16"), operand, None, tl.constexpr("bf16"), None, False,
                                True, True, tl.float32)
    assert bool(caught) == (support == DotSupport.NON_NATIVE)


def test_precision_legacy_int32_dot_rejected():
    semantic = TritonSemantic(SimpleNamespace(codegen_fns={}))
    operand = tl.tensor(None, tl.block_type(tl.int32, [16, 16]))
    with pytest.raises(AssertionError, match="Unsupported lhs dtype int32"):
        semantic.dot(operand, operand, None, "ieee", 0, tl.int32)


def test_hcu_i32_bf16_compile(fresh_triton_cache):
    compiled = triton.compile(
        ASTSource(precision_cast, {"X": "*i32", "Y": "*bf16"}, constexprs={"N": 513, "BLOCK": 256, "ROUND": None}),
        target=GPUTarget("hip", "gfx936", 64), options={"num_warps": 4})
    assert not re.search(r"\bsitofp\b", compiled.asm["llir"])


@pytest.mark.parametrize("dtype", ["fp16", "bf16", "fp32", "fp64"])
@pytest.mark.parametrize("rounding", ["rtne", "rtz"])
def test_hcu_fp4_encode_compile(dtype, rounding, fresh_triton_cache):
    with pytest.warns(UserWarning, match="software encoding"):
        triton.compile(
            ASTSource(precision_fp4_encode, {"X": "*" + dtype, "Y": "*u8"},
                      constexprs={"N": 513, "ROUND": rounding, "BLOCK": 512}), target=GPUTarget("hip", "gfx936", 64),
            options={"num_warps": 4})


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("with_scale", [False, True])
def test_hcu_fp4_decode_compile(dtype, with_scale, fresh_triton_cache):
    signature = {"X": "*u8", "Y": "*" + dtype}
    constants = {"N": 513, "BLOCK": 256}
    if with_scale:
        signature["S"] = "*u8"
    else:
        constants["S"] = None
    with pytest.warns(UserWarning, match="software decoding"):
        triton.compile(ASTSource(precision_fp4_decode, signature, constexprs=constants),
                       target=GPUTarget("hip", "gfx936", 64), options={"num_warps": 4})


def test_hcu_int4_roundtrip_compile(fresh_triton_cache):
    triton.compile(
        ASTSource(precision_int4_roundtrip, {"X": "*u8", "U": "*i8", "Y": "*u8"}, constexprs={"N": 513, "BLOCK": 256}),
        target=GPUTarget("hip", "gfx936", 64), options={"num_warps": 4})


@pytest.mark.parametrize("format_a", ["int4", "int8"])
@pytest.mark.parametrize("format_b", ["int4", "int8"])
def test_hcu_integer_scaled_compile(format_a, format_b, fresh_triton_cache):
    types = {"int4": "u8", "int8": "i8"}
    signature = {
        "A": "*" + types[format_a], "B": "*" + types[format_b], "SA": "*fp32", "SB": "*fp32", "ACC": "*fp32", "C":
        "*fp32"
    }
    with pytest.warns(UserWarning, match="groupwise integer dot"):
        triton.compile(
            ASTSource(
                precision_integer_scaled, signature, constexprs={
                    "M": 16, "N": 16, "K": 64, "FORMAT_A": format_a, "FORMAT_B": format_b, "OUT_DTYPE": tl.float32
                }), target=GPUTarget("hip", "gfx936", 64), options={"num_warps": 4, "enable_fp_fusion": False})


@pytest.mark.parametrize("format_a,format_b", [("int32", "int8"), ("int8", "int32"), ("int32", "int32")])
def test_hcu_integer_scaled_int32_operands_rejected(format_a, format_b, fresh_triton_cache):
    signature = {
        "A": "*i32" if format_a == "int32" else "*i8", "B": "*i32" if format_b == "int32" else "*i8", "ACC": "*fp32",
        "C": "*fp32"
    }
    with pytest.raises(triton.CompilationError) as caught:
        triton.compile(
            ASTSource(
                precision_integer_scaled, signature, constexprs={
                    "SA": None, "SB": None, "M": 16, "N": 16, "K": 64, "FORMAT_A": format_a, "FORMAT_B": format_b,
                    "OUT_DTYPE": tl.float32
                }), target=GPUTarget("hip", "gfx936", 64), options={"num_warps": 4})
    assert "integer operand formats must be int4/int8" in ''.join(traceback.format_exception(caught.value))


def test_hcu_integer_scaled_rejects_int32_output_with_scale(fresh_triton_cache):
    signature = {"A": "*i8", "B": "*i8", "SA": "*fp32", "ACC": "*i32", "C": "*i32"}
    constants = {"SB": None, "M": 16, "N": 16, "K": 64, "FORMAT_A": "int8", "FORMAT_B": "int8", "OUT_DTYPE": tl.int32}
    with pytest.raises(triton.CompilationError) as caught:
        triton.compile(ASTSource(precision_integer_scaled, signature, constexprs=constants),
                       target=GPUTarget("hip", "gfx936", 64), options={"num_warps": 4})
    assert "INT32 output requires both integer scales to be None" in ''.join(traceback.format_exception(caught.value))


@pytest.mark.parametrize("dtype", ["i32", "i8", "u8", "fp8e4nv"])
@pytest.mark.parametrize("async_request,stride,n", [(False, 1, 4096), (True, 1, 4096), (True, 2, 4093)])
def test_hcu_tle_copy_compile(dtype, async_request, stride, n, fresh_triton_cache):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compiled = triton.compile(
            ASTSource(
                precision_tle_copy, {"X": "*" + dtype, "Y": "*" + dtype},
                constexprs={"N": n, "STRIDE":
                            stride, "ASYNC": async_request, "BLOCK": 1024}, attrs={(0, ): [("tt.divisibility", 16),
                                                                                           ("tt.pointer_range", 32)],
                                                                                   (1, ): [("tt.divisibility", 16),
                                                                                           ("tt.pointer_range", 32)]}),
            target=GPUTarget("hip", "gfx936", 64), options={"num_warps": 4, "num_stages": 1})
    native = async_request and (stride == 1 or dtype == "i32")
    assert ("async_copy_global_to_local" in compiled.asm["ttgir"]
            or "buffer_load_to_local" in compiled.asm["ttgir"]) == native
    assert bool(re.search(r"(?:global|buffer)_load\w*[^\n]*\blds\b", compiled.asm["amdgcn"])) == native
    assert any("synchronous load" in str(item.message) for item in caught) == (async_request and not native)
    if native:
        assert "async_wait" in compiled.asm["ttgir"]
        assert "s_waitcnt" in compiled.asm["amdgcn"] and "s_barrier" in compiled.asm["amdgcn"]


@pytest.mark.parametrize("dtype", ["i32", "i8", "u8", "fp8e4nv"])
@pytest.mark.parametrize("copy", [False, True])
def test_hcu_shared_copy_compile(dtype, copy, fresh_triton_cache):
    compiled = triton.compile(
        ASTSource(precision_shared_copy, {"X": "*" + dtype, "Y": "*" + dtype},
                  constexprs={"N": 1027, "COPY": copy, "BLOCK": 1024}), target=GPUTarget("hip", "gfx936", 64),
        options={"num_warps": 4, "num_stages": 1})
    assert compiled.metadata.shared > 0
    assert re.search(r"\bds_(?:read|load)\w*", compiled.asm["amdgcn"])
    assert re.search(r"\bds_(?:write|store)\w*", compiled.asm["amdgcn"])


@pytest.mark.parametrize("dtype", ["u8", "i8", "fp8e4nv", "fp16", "bf16", "i32", "fp32"])
@pytest.mark.parametrize("warps", [1, 4, 8])
def test_hcu_reverse_copy_compile(dtype, warps, fresh_triton_cache):
    compiled = triton.compile(
        ASTSource(precision_reverse_copy, {"X": "*" + dtype, "Y": "*" + dtype},
                  constexprs={"N": 4096, "BLOCK": 2048, "STRIDE": 1, "OFFSET": 0, "OVERWRITE":
                              False}, attrs={(1, ): [("tt.divisibility", 16)]}), target=GPUTarget("hip", "gfx936", 64),
        options={"num_warps": warps, "num_stages": 1})
    assert compiled.asm["ttgir"].count("hcug.buffer_store_lds") == 2
    assert "ttg.local_load" not in compiled.asm["ttgir"]
    assert "buffer_store_lds_dword" in compiled.asm["amdgcn"]
    # Producer permutations may use LDS layout scratch. The two copies must
    # consume the memdesc directly, without loading their payload into tensors.
    assert "s_waitcnt vmcnt(0)" in compiled.asm["amdgcn"]


@pytest.mark.parametrize("constants,arch,dtype", [
    ({"STRIDE": 2}, "gfx936", "u8"),
    ({"OFFSET": 1}, "gfx936", "u8"),
    ({"N": 4093}, "gfx936", "u8"),
    ({"BLOCK": 256}, "gfx936", "u8"),
    ({"OVERWRITE": True}, "gfx936", "i32"),
    ({}, "gfx936", "fp64"),
    ({}, "gfx938", "u8"),
])
def test_hcu_reverse_copy_fallback_compile(constants, arch, dtype, fresh_triton_cache):
    constants = dict({"N": 4096, "BLOCK": 1024, "STRIDE": 1, "OFFSET": 0, "OVERWRITE": False}, **constants)
    compiled = triton.compile(
        ASTSource(precision_reverse_copy, {"X": "*" + dtype, "Y": "*" + dtype}, constexprs=constants,
                  attrs={(1, ): [("tt.divisibility", 16)]}), target=GPUTarget("hip", arch, 64),
        options={"num_warps": 4, "num_stages": 1})
    assert "buffer_store_lds" not in compiled.asm["ttgir"]
    assert "buffer_store_lds_dword" not in compiled.asm["amdgcn"]


@pytest.mark.parametrize("dtype", ["u8", "fp16", "i32"])
@pytest.mark.parametrize("offset", [0, 1, 4, 1024])
def test_hcu_reverse_subview_compile(dtype, offset, fresh_triton_cache):
    compiled = triton.compile(
        ASTSource(precision_reverse_subview, {"X": "*" + dtype, "Y": "*" + dtype},
                  constexprs={"SOURCE_OFFSET": offset, "BLOCK": 1024}, attrs={(1, ): [("tt.divisibility", 16)]}),
        target=GPUTarget("hip", "gfx936", 64), options={"num_warps": 4, "num_stages": 1})
    assert ("buffer_store_lds_dword" in compiled.asm["amdgcn"]) == (offset % 1024 == 0)


@pytest.mark.parametrize("dtype,offset", [("u8", 1), ("fp16", 2), ("i32", 16), ("u8", 0)])
def test_hcu_reverse_alias_compile(dtype, offset, fresh_triton_cache):
    compiled = triton.compile(
        ASTSource(precision_reverse_alias, {"X": "*" + dtype, "Y": "*" + dtype},
                  constexprs={"OFFSET_BYTES": offset, "BLOCK": 1024}, attrs={(1, ): [("tt.divisibility", 16)]}),
        target=GPUTarget("hip", "gfx936", 64), options={"num_warps": 4, "num_stages": 1})
    assert "buffer_store_lds_dword" not in compiled.asm["amdgcn"]


@pytest.mark.parametrize("source,destination", [("fp8e4nv", "fp8e5"), ("fp8e5", "fp8e4nv")])
@pytest.mark.parametrize("arch", ["gfx936", "gfx938"])
def test_hcu_fp8_cross_cast_compile(source, destination, arch, fresh_triton_cache):
    triton.compile(
        ASTSource(precision_cast, {"X": "*" + source, "Y": "*" + destination},
                  constexprs={"N": 256, "BLOCK": 256, "ROUND": None}), target=GPUTarget("hip", arch, 64),
        options={"num_warps": 4})


@pytest.mark.parametrize("async_request", [False, True])
def test_hcu_tle_local_load_compile(async_request, fresh_triton_cache):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compiled = triton.compile(
            ASTSource(precision_tle_local_load, {"X": "*i32", "Y": "*i32"},
                      constexprs={"ASYNC": async_request, "BLOCK": 1024}), target=GPUTarget("hip", "gfx936", 64),
            options={"num_warps": 4, "num_stages": 1})
    assert "async_copy_global_to_local" not in compiled.asm["ttgir"]
    assert any("synchronous load" in str(item.message) for item in caught) == async_request


def test_hcu_unbounded_async_diagnostic(fresh_triton_cache, capfd):
    # A pointer range is deliberately absent, so buffer conversion is illegal.
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        triton.compile(
            ASTSource(precision_tle_copy, {"X": "*i32", "Y": "*i32"},
                      constexprs={"N": 4096, "STRIDE": 1, "ASYNC": True, "BLOCK": 1024}),
            target=GPUTarget("hip", "gfx936", 64), options={"num_warps": 4, "num_stages": 1})
    assert "gfx936 native async copies require buffer-compatible pointers" in capfd.readouterr().err


@pytest.mark.parametrize("inner", [1, 4, 16, 32])
@pytest.mark.parametrize("with_acc", [False, True])
@pytest.mark.parametrize("overflow", [False, True])
def test_hcu_integer_dot_values(inner, with_acc, overflow, device):
    generator = torch.Generator().manual_seed(1493)
    bound = 16
    a = torch.randint(-bound, bound, (16, inner), generator=generator, dtype=torch.int64)
    b = torch.randint(-bound, bound, (inner, 16), generator=generator, dtype=torch.int64)
    if overflow:
        a.fill_(127)
        b.fill_(127)
        b[:, 1::2] = -128
    acc = torch.full((16, 16), 2**31 - 2 if overflow else 123, dtype=torch.int64)
    if overflow:
        acc[:, 1::2] = -(2**31) + 2
    expected = (a @ b + (acc if with_acc else 0)).to(torch.int32)
    output = torch.full((16 * 16 + 32, ), 314159, dtype=torch.int32, device=device)
    precision_dot[(1, )](a.to(dtype=torch.int8, device=device), b.to(dtype=torch.int8, device=device),
                         acc.to(dtype=torch.int32, device=device) if with_acc else None, output[16:-16], 16, 16, inner,
                         num_warps=4, num_stages=1)
    torch.testing.assert_close(output[16:-16].reshape(16, 16).cpu(), expected, rtol=0, atol=0)
    assert torch.all(output[:16] == 314159) and torch.all(output[-16:] == 314159)


@pytest.mark.parametrize("dtype_a,dtype_b", [(torch.float8_e4m3fn, torch.float8_e4m3fn),
                                             (torch.float8_e4m3fn, torch.float8_e5m2),
                                             (torch.float8_e5m2, torch.float8_e4m3fn),
                                             (torch.float8_e5m2, torch.float8_e5m2)])
@pytest.mark.parametrize("inner", [32, 64])
@pytest.mark.parametrize("with_acc", [False, True])
@pytest.mark.parametrize("column_b", [False, True])
def test_hcu_fp8_dot_values(dtype_a, dtype_b, inner, with_acc, column_b, device, fresh_triton_cache):
    generator = torch.Generator().manual_seed(1493)
    a = (torch.randint(-8, 9, (16, inner), generator=generator).double() / 2).to(dtype_a)
    b = (torch.randint(-8, 9, (inner, 16), generator=generator).double() / 2).to(dtype_b)
    acc = ((torch.arange(256).reshape(16, 16) % 7) - 3).float() / 4
    expected = (a.double() @ b.double() + (acc.double() if with_acc else 0)).float()
    physical_b = b.T.contiguous() if column_b else b
    output = torch.full((256 + 32, ), 1234.5, device=device)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compiled = precision_dot[(1, )](a.to(device), physical_b.to(device), acc.to(device) if with_acc else None,
                                        output[16:-16], 16, 16, inner, column_b, num_warps=4, num_stages=1)
    arch = triton.runtime.driver.active.get_current_target().arch.split(":")[0].lower()
    # gfx936 decodes to FP16; BW1100/gfx938 can execute FP8 MMAC directly.
    software = arch not in ("gfx938", "gfx92a", "gfx946")
    instruction_types = "f16" if software else "(?:fp8|bf8)"
    assert re.search(r"\bv_mmac_.*" + instruction_types + r"\b", compiled.asm["amdgcn"])
    assert any(re.search(r"tl.dot.*native=false", str(item.message)) for item in caught) == software
    torch.testing.assert_close(output[16:-16].reshape(16, 16).cpu(), expected, rtol=0, atol=0)
    assert torch.all(output[:16] == 1234.5) and torch.all(output[-16:] == 1234.5)


def test_hcu_i32_bf16_values(device):
    # Every integral BF16 rounding midpoint, plus adjacent integers and signs.
    values = {0, 1, -1, -(2**31), 2**31 - 1, 16842753, -16842753}
    for exponent in range(8, 31):
        step = 1 << (exponent - 7)
        for significand in range(128, 256):
            midpoint = significand * step + step // 2
            for delta in (-1, 0, 1):
                value = midpoint + delta
                if value < 2**31:
                    values.update((value, -value))
    source = torch.tensor(sorted(values), dtype=torch.int32)
    bits = []
    for value in source.tolist():
        magnitude = abs(value)
        exponent = magnitude.bit_length() - 1
        shift = max(exponent - 7, 0)
        significand = magnitude >> shift
        remainder = magnitude - (significand << shift)
        if shift and (remainder > 1 << (shift - 1) or (remainder == 1 << (shift - 1) and significand & 1)):
            significand += 1
        encoded = 0 if magnitude == 0 else ((exponent + 126) << 7) + (significand if shift else magnitude <<
                                                                      (7 - exponent))
        encoded |= 0x8000 if value < 0 else 0
        bits.append(encoded if encoded < 0x8000 else encoded - 0x10000)
    expected = torch.tensor(bits, dtype=torch.int16)
    output = torch.full((source.numel() + 32, ), 123, dtype=torch.bfloat16, device=device)
    precision_cast[(triton.cdiv(source.numel(), 256), )](source.to(device), output[16:-16], source.numel(), 256)
    torch.testing.assert_close(output[16:-16].cpu().view(torch.int16), expected, rtol=0, atol=0)
    assert torch.all(output[:16] == 123) and torch.all(output[-16:] == 123)


def conversion_source_values(dtype, boundaries):
    generator = torch.Generator().manual_seed(1493)
    if dtype in (torch.float16, torch.bfloat16):
        source = torch.arange(65536).to(torch.int16).view(dtype)
    else:
        middle = torch.tensor(boundaries, dtype=dtype)
        adjacent = torch.cat((middle, torch.nextafter(middle, torch.full_like(middle, float('inf'))),
                              torch.nextafter(middle, torch.zeros_like(middle))))
        if dtype == torch.float32:
            random = torch.randint(0, 2**32, (4096, ), generator=generator).to(torch.int32).view(dtype)
        else:
            random = torch.randint(0, 2**63 - 1, (4096, ), generator=generator).view(dtype)
        source = torch.cat((adjacent, -adjacent, random, -random,
                            torch.tensor([0., -0., float('inf'), -float('inf'),
                                          float('nan'), -float('nan')], dtype=dtype)))
    source = torch.cat((source, torch.zeros(1, dtype=dtype)))  # odd logical tail
    return source


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("rounding", ["rtne", "rtz"])
def test_hcu_fp4_encode_values(dtype, rounding, device):
    source = conversion_source_values(dtype, [.25, .75, 1.25, 1.75, 2.5, 3.5, 5., .5, 1., 1.5, 2., 3., 4., 6.])
    magnitude = source.to(torch.float64).abs()
    levels = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], dtype=torch.float64)
    if rounding == "rtz":
        codes = (torch.bucketize(magnitude.clamp(max=6), levels, right=True) - 1).clamp(0, 7)
    else:
        distances = (magnitude.clamp(max=6)[:, None] - levels[None, :]).abs()
        codes = distances.argmin(dim=1)
        # Resolve equal nearest distances using code parity, independently of
        # the device encoder's boundary comparisons.
        indices = torch.arange(8)
        even = torch.where((distances == distances.min(dim=1, keepdim=True).values) & (indices % 2 == 0), indices,
                           8).min(dim=1).values
        codes = torch.where(even < 8, even, codes)
    codes = codes.to(torch.uint8)
    codes[magnitude.isnan()] = 7
    codes |= torch.signbit(source).to(torch.uint8) << 3
    codes = torch.cat((codes, torch.zeros(1, dtype=torch.uint8)))
    expected = codes[::2] | (codes[1::2] << 4)
    output = torch.full((expected.numel() + 32, ), 165, dtype=torch.uint8, device=device)
    precision_fp4_encode[(triton.cdiv(source.numel(), 512), )](source.to(device), output[16:-16], source.numel(),
                                                               rounding)
    torch.testing.assert_close(output[16:-16].cpu(), expected, rtol=0, atol=0)
    assert torch.all(output[:16] == 165) and torch.all(output[-16:] == 165)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("with_scale", [False, True])
def test_hcu_fp4_decode_values(dtype, with_scale, device):
    n = 65536 if with_scale else 513
    source = (torch.arange(n) % 256).to(torch.uint8)
    scales = (torch.arange(n) // 256).to(torch.uint8) if with_scale else None
    codes = torch.stack((source & 15, source >> 4), dim=-1).reshape(-1).long()
    table = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.])
    expected = table[codes]
    if with_scale:
        factors = torch.exp2(scales.to(torch.float32) - 127).repeat_interleave(2)
        expected *= factors
        expected[scales.repeat_interleave(2) == 255] = float('nan')
    expected = expected.to(dtype)
    output = torch.full((n * 2 + 32, ), 123, dtype=dtype, device=device)
    precision_fp4_decode[(triton.cdiv(n, 256), )](source.to(device), scales.to(device) if with_scale else None,
                                                  output[16:-16], n)
    actual = output[16:-16].cpu()
    assert torch.equal(actual.isnan(), expected.isnan())
    finite_or_inf = ~expected.isnan()
    torch.testing.assert_close(
        actual.view(torch.int16)[finite_or_inf],
        expected.view(torch.int16)[finite_or_inf], rtol=0, atol=0)
    assert torch.all(output[:16] == 123) and torch.all(output[-16:] == 123)


@pytest.mark.parametrize("n", [255, 256, 513])
def test_hcu_int4_roundtrip_values(n, device):
    source = (torch.arange(n) % 256).to(torch.uint8)
    low = ((source.to(torch.int8) << 4) >> 4)
    high = source.to(torch.int8) >> 4
    expected = torch.stack((low, high), dim=-1).reshape(-1)
    unpacked = torch.full((n * 2 + 32, ), 42, dtype=torch.int8, device=device)
    repacked = torch.full((n + 32, ), 165, dtype=torch.uint8, device=device)
    precision_int4_roundtrip[(triton.cdiv(n, 256), )](source.to(device), unpacked[16:-16], repacked[16:-16], n)
    torch.testing.assert_close(unpacked[16:-16].cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(repacked[16:-16].cpu(), source, rtol=0, atol=0)
    assert torch.all(unpacked[:16] == 42) and torch.all(unpacked[-16:] == 42)
    assert torch.all(repacked[:16] == 165) and torch.all(repacked[-16:] == 165)


@pytest.mark.parametrize("dtype",
                         [torch.int8, torch.int32, torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_hcu_int4_quantize_values(dtype, device):
    if dtype == torch.int8:
        source = torch.arange(-128, 128).to(dtype)
    elif dtype == torch.int32:
        source = torch.tensor([-(2**31), 2**31 - 1] + list(range(-512, 513)), dtype=dtype)
    else:
        source = (torch.arange(-511, 512).double() / 4).to(dtype)
    if source.numel() % 2 == 0:
        source = torch.cat((source, torch.zeros(1, dtype=dtype)))
    codes = source.double().trunc().long() & 15
    codes = torch.cat((codes, torch.zeros(1, dtype=torch.int64)))
    expected = (codes[::2] | (codes[1::2] << 4)).to(torch.uint8)
    output = torch.full((expected.numel() + 32, ), 165, dtype=torch.uint8, device=device)
    precision_int4_quantize[(triton.cdiv(source.numel(), 256), )](source.to(device), output[16:-16], source.numel())
    torch.testing.assert_close(output[16:-16].cpu(), expected, rtol=0, atol=0)
    assert torch.all(output[:16] == 165) and torch.all(output[-16:] == 165)


@pytest.mark.parametrize("axis_a,axis_b", [(1, 0), (1, 1), (0, 0), (0, 1)])
@pytest.mark.parametrize("with_acc", [False, True])
def test_hcu_int4_dot_values(axis_a, axis_b, with_acc, device):
    generator = torch.Generator().manual_seed(1493)
    a = torch.randint(-8, 8, (16, 32), generator=generator)
    b = torch.randint(-8, 8, (32, 16), generator=generator)
    a[0, :16] = torch.arange(-8, 8)
    b[:16, 0] = torch.arange(-8, 8).flip(0)
    data = []
    for value, axis in [(a, axis_a), (b, axis_b)]:
        packed = (((value[:, ::2] & 15) | ((value[:, 1::2] & 15) << 4)) if axis == 1 else
                  ((value[::2, :] & 15) | ((value[1::2, :] & 15) << 4)))
        data.append(packed.to(dtype=torch.uint8, device=device))
    acc = torch.randint(-128, 128, (16, 16), generator=generator, dtype=torch.int32)
    acc[0, 0], acc[0, 1] = 2**31 - 1, -(2**31)
    expected = (a @ b + (acc.long() if with_acc else 0)).to(torch.int32)
    output = torch.full((16 * 16 + 32, ), 42, dtype=torch.int32, device=device)
    with pytest.warns(UserWarning, match="software unpacking"):
        compiled = precision_int4_dot[(1, )](data[0], data[1], acc.to(device) if with_acc else None, output[16:-16], 16,
                                             16, 32, axis_a, axis_b, num_warps=4, num_stages=1)
    assert "v_mmac_i32_16x16x32_i8" in compiled.asm["amdgcn"]
    assert not re.search(r"\b(sitofp|uitofp|fptosi|fptoui)\b", compiled.asm["llir"])
    torch.testing.assert_close(output[16:-16].reshape(16, 16).cpu(), expected, rtol=0, atol=0)
    assert torch.all(output[:16] == 42) and torch.all(output[-16:] == 42)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int8, torch.uint8, torch.float8_e4m3fn])
@pytest.mark.parametrize("async_request,stride,n", [(False, 1, 4096), (True, 1, 4096), (True, 2, 4093)])
@pytest.mark.parametrize("offset", [0, 1])
def test_hcu_tle_copy_values(dtype, async_request, stride, n, offset, device):
    generator = torch.Generator().manual_seed(1493)
    storage = torch.int32 if dtype == torch.int32 else torch.uint8
    host = (torch.randint(-(2**31), 2**31,
                          (n * stride + 64, ), generator=generator).to(storage) if storage == torch.int32 else
            (torch.arange(n * stride + 64) * 37).to(storage))
    source = host.to(device).view(dtype)
    output = torch.full((n + 32, ), 42, dtype=storage, device=device).view(dtype)
    expected = host[16 + offset:16 + offset + n * stride:stride]
    kernel = precision_tle_copy[(triton.cdiv(n, 1024), )](source[16 + offset:], output[16:-16], n, stride,
                                                          async_request, num_warps=4, num_stages=1)
    torch.testing.assert_close(output[16:-16].cpu().view(storage), expected, rtol=0, atol=0)
    assert torch.all(output[:16].view(storage) == 42) and torch.all(output[-16:].view(storage) == 42)
    torch.testing.assert_close(source.cpu().view(storage), host, rtol=0, atol=0)
    if async_request and stride == 1 and offset == 0:
        assert re.search(r"(?:global|buffer)_load\w*[^\n]*\blds\b", kernel.asm["amdgcn"])


@pytest.mark.parametrize("dtype", [torch.int32, torch.uint8, torch.float8_e4m3fn])
@pytest.mark.parametrize("tail", [0, 13])
@pytest.mark.parametrize("stages", [1, 2])
def test_hcu_tle_copy_pipeline_values(dtype, tail, stages, device):
    n = 16384 + tail
    storage = torch.int32 if dtype == torch.int32 else torch.uint8
    host = (torch.arange(n + 32) * 37).to(storage)
    source = host.to(device).view(dtype)
    output = torch.full((n + 32, ), 42, dtype=storage, device=device).view(dtype)
    for repeat in range(100 if stages == 2 else 3):
        compiled = precision_tle_copy_loop[(triton.cdiv(n, 4096), )](source[16:-16], output[16:-16], n, num_warps=4,
                                                                     num_stages=stages)
        torch.testing.assert_close(output[16:-16].cpu().view(storage), host[16:-16], rtol=0, atol=0)
        assert torch.all(output[:16].view(storage) == 42) and torch.all(output[-16:].view(storage) == 42)
    if tail == 0:
        assert re.search(r"(?:global|buffer)_load\w*[^\n]*\blds\b", compiled.asm["amdgcn"])


@pytest.mark.parametrize("dtype", [torch.int32, torch.int8, torch.uint8, torch.float8_e4m3fn])
@pytest.mark.parametrize("copy", [False, True])
def test_hcu_shared_copy_values(dtype, copy, device):
    generator = torch.Generator().manual_seed(1493)
    storage = torch.int32 if dtype == torch.int32 else torch.uint8
    n = 1027
    host = (torch.randint(-(2**31), 2**31, (n + 32, ), generator=generator).to(storage) if storage == torch.int32 else
            (torch.arange(n + 32) * 37).to(storage))
    source = host.to(device).view(dtype)
    output = torch.full((n + 32, ), 42, dtype=storage, device=device).view(dtype)
    for repeat in range(10):
        precision_shared_copy[(triton.cdiv(n, 1024), )](source[16:-16], output[16:-16], n, copy, num_warps=4,
                                                        num_stages=1)
        torch.testing.assert_close(output[16:-16].cpu().view(storage), host[16:-16], rtol=0, atol=0)
        assert torch.all(output[:16].view(storage) == 42) and torch.all(output[-16:].view(storage) == 42)


@pytest.mark.parametrize("async_request", [False, True])
def test_hcu_tle_local_load_values(async_request, device):
    host = torch.arange(1024, dtype=torch.int32) * 37 + 13
    expected = host.clone()
    expected[-1] = 0
    output = torch.full((1056, ), 42, dtype=torch.int32, device=device)
    compiled = precision_tle_local_load[(1, )](host.to(device), output[16:-16], async_request, num_warps=4,
                                               num_stages=1)
    torch.testing.assert_close(output[16:-16].cpu(), expected, rtol=0, atol=0)
    assert torch.all(output[:16] == 42) and torch.all(output[-16:] == 42)
    assert not re.search(r"(?:global|buffer)_load\w*[^\n]*\blds\b", compiled.asm["amdgcn"])


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float16, torch.int32])
@pytest.mark.parametrize("offset", [0, 1, 4, 1024])
def test_hcu_reverse_subview_values(dtype, offset, device):
    block, repeats = 1024, 500
    width = torch.empty((), dtype=dtype).element_size()
    host = torch.randint(0, 256, (2 * block * width, ), dtype=torch.uint8,
                         generator=torch.Generator().manual_seed(1493))
    source = host.to(device).view(dtype)
    span = block + 32
    output = torch.full((repeats * span * width, ), 42, dtype=torch.uint8, device=device)
    for repeat in range(repeats):
        compiled = precision_reverse_subview[(1, )](source, output.view(dtype)[repeat * span + 16:], offset,
                                                    num_warps=4, num_stages=1)
    expected = torch.full((repeats, span * width), 42, dtype=torch.uint8)
    expected[:, 16 * width:(16 + block) * width] = host[offset * width:(offset + block) * width]
    torch.testing.assert_close(output.cpu().reshape_as(expected), expected, rtol=0, atol=0)
    arch = triton.runtime.driver.active.get_current_target().arch.split(":")[0].lower()
    assert ("buffer_store_lds_dword" in compiled.asm["amdgcn"]) == (arch == "gfx936" and offset % block == 0)


@pytest.mark.parametrize("dtype,offset", [(torch.uint8, 1), (torch.float16, 2), (torch.int32, 16), (torch.uint8, 0)])
def test_hcu_reverse_alias_values(dtype, offset, device):
    block = 1024
    width = torch.empty((), dtype=dtype).element_size()
    host = torch.randint(0, 256, (2 * block * width, ), dtype=torch.uint8,
                         generator=torch.Generator().manual_seed(1493))
    output = torch.full((block * width + 32, ), 42, dtype=torch.uint8, device=device)
    compiled = precision_reverse_alias[(1, )](host.to(device).view(dtype), output[16:-16].view(dtype), offset,
                                              num_warps=4, num_stages=1)
    torch.testing.assert_close(output[16:-16].cpu(), host[offset:offset + block * width], rtol=0, atol=0)
    assert torch.all(output[:16] == 42) and torch.all(output[-16:] == 42)
    assert "buffer_store_lds_dword" not in compiled.asm["amdgcn"]


@pytest.mark.parametrize("source,destination", [(torch.float8_e4m3fn, torch.float8_e5m2),
                                                (torch.float8_e5m2, torch.float8_e4m3fn)])
def test_hcu_fp8_cross_cast_values(source, destination, device):
    host = torch.arange(256, dtype=torch.int32).to(torch.uint8).view(source)
    limit = torch.finfo(destination).max
    expected = host.float().clamp(-limit, limit).to(destination)
    output = torch.full((288, ), 42, dtype=torch.uint8, device=device).view(destination)
    precision_cast[(1, )](host.to(device), output[16:-16], 256, 256)
    actual = output[16:-16].cpu().float()
    assert torch.equal(actual.isnan(), expected.float().isnan())
    valid = ~expected.float().isnan()
    torch.testing.assert_close(output[16:-16].cpu().view(torch.uint8)[valid],
                               expected.view(torch.uint8)[valid], rtol=0, atol=0)
    assert torch.all(output[:16].view(torch.uint8) == 42) and torch.all(output[-16:].view(torch.uint8) == 42)


@pytest.mark.parametrize(
    "dtype", [torch.uint8, torch.int8, torch.float8_e4m3fn, torch.float16, torch.bfloat16, torch.int32, torch.float32])
@pytest.mark.parametrize("warps", [1, 4, 8])
def test_hcu_reverse_copy_values(dtype, warps, device):
    n, repeats = 4096, 500
    width = torch.empty((), dtype=dtype).element_size()
    host = torch.randint(0, 256, (2 * n * width, ), dtype=torch.uint8, generator=torch.Generator().manual_seed(1493))
    source = host.to(device).view(dtype)
    span = 2 * n + 32
    output = torch.full((repeats * span * width, ), 42, dtype=torch.uint8, device=device)
    for repeat in range(repeats):
        compiled = precision_reverse_copy[(2, )](source, output.view(dtype)[repeat * span + 16:], n, BLOCK=2048,
                                                 num_warps=warps, num_stages=1)
    expected = torch.full((repeats, span * width), 42, dtype=torch.uint8)
    expected[:, 16 * width:(16 + 2 * n) * width] = host
    torch.testing.assert_close(output.cpu().reshape_as(expected), expected, rtol=0, atol=0)
    arch = triton.runtime.driver.active.get_current_target().arch.split(":")[0].lower()
    assert ("buffer_store_lds_dword" in compiled.asm["amdgcn"]) == (arch == "gfx936")


@pytest.mark.parametrize("stride,offset,n,block,overwrite", [(2, 0, 4096, 1024, False), (1, 1, 4096, 1024, False),
                                                             (1, 0, 4093, 1024, False), (1, 0, 4096, 256, False),
                                                             (1, 0, 4096, 1024, True)])
def test_hcu_reverse_copy_fallback_values(stride, offset, n, block, overwrite, device):
    host = torch.randint(0, 256, (2 * n, ), dtype=torch.uint8, generator=torch.Generator().manual_seed(1493))
    output = torch.full((2 * n * stride + 64, ), 42, dtype=torch.uint8, device=device)
    compiled = precision_reverse_copy[(triton.cdiv(n, block), )](host.to(device), output[16:], n, stride, offset, block,
                                                                 overwrite, num_warps=4, num_stages=1)
    expected = torch.full_like(output.cpu(), 42)
    expected[16 + offset:16 + offset + 2 * n * stride:stride] = host
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    assert "buffer_store_lds_dword" not in compiled.asm["amdgcn"]


@pytest.mark.parametrize("format_a", ["int4", "int8"])
@pytest.mark.parametrize("format_b", ["int4", "int8"])
@pytest.mark.parametrize("scale_mode", ["none", "lhs", "rhs", "both"])
def test_hcu_integer_scaled_values(format_a, format_b, scale_mode, device):
    generator = torch.Generator().manual_seed(1493)
    m, n, k = 16, 16, 64
    data, reference = [], []
    for fmt, shape, axis in [(format_a, (m, k), 1), (format_b, (k, n), 0)]:
        bound = {"int4": 8, "int8": 128}[fmt]
        value = torch.randint(-bound, bound, shape, generator=generator)
        reference.append(value)
        if fmt == "int4":
            value = ((value[:, ::2] & 15) |
                     ((value[:, 1::2] & 15) << 4)) if axis == 1 else ((value[::2, :] & 15) |
                                                                      ((value[1::2, :] & 15) << 4))
        data.append(value.to(dtype={"int4": torch.uint8, "int8": torch.int8}[fmt], device=device))
    scale_a = ((torch.arange(m * 2).reshape(m, 2) % 5) - 2).float() / 2
    scale_b = ((torch.arange(n * 2).reshape(n, 2) % 7) - 3).float() / 4
    acc = ((torch.arange(m * n).reshape(m, n) % 7) - 3).float() / 4
    expected = acc.clone()
    for group in range(2):
        partial = (reference[0][:, group * 32:(group + 1) * 32] @ reference[1][group * 32:(group + 1) * 32, :]).to(
            torch.int32).float()
        if scale_mode in ("lhs", "both"):
            partial *= scale_a[:, group, None]
        if scale_mode in ("rhs", "both"):
            partial *= scale_b[None, :, group]
        expected += partial
    output = torch.full((m * n + 32, ), 1234.5, dtype=torch.float32, device=device)
    precision_integer_scaled[(1, )](data[0], scale_a.to(device) if scale_mode in ("lhs", "both") else None, data[1],
                                    scale_b.to(device) if scale_mode in ("rhs", "both") else None, acc.to(device),
                                    output[16:-16], m, n, k, format_a, format_b, num_warps=4, num_stages=1,
                                    enable_fp_fusion=False)
    torch.testing.assert_close(output[16:-16].reshape(m, n).cpu(), expected, rtol=0, atol=0)
    assert torch.all(output[:16] == 1234.5) and torch.all(output[-16:] == 1234.5)


def precision_operand(fmt, shape, packing_axis, generator, device):
    if fmt == "e2m1":
        codes = torch.randint(0, 16, shape, generator=generator)
        table = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.],
                             dtype=torch.float64)
        values = table[codes]
        if shape[packing_axis] % 2:
            padding_shape = list(shape)
            padding_shape[packing_axis] = 1
            # The kernel must ignore the unused high nibble without altering
            # the stored byte. A nonzero padding code detects accidental use.
            codes = torch.cat((codes, torch.full(padding_shape, 13)), dim=packing_axis)
        if packing_axis == 1:
            data = codes[:, ::2] | (codes[:, 1::2] << 4)
        else:
            data = codes[::2, :] | (codes[1::2, :] << 4)
        data = data.to(torch.uint8)
    else:
        values = torch.randint(-8, 9, shape, generator=generator).to(torch.float64) / 2
        dtype = {"e4m3": torch.float8_e4m3fn, "bf16": torch.bfloat16, "fp16": torch.float16}[fmt]
        data = values.to(dtype)
    return data.to(device), values


@pytest.mark.parametrize("format_a,format_b", [("e2m1", "e2m1"), ("e2m1", "fp16"), ("e4m3", "e4m3")])
@pytest.mark.parametrize("inner", [81, 96])
@pytest.mark.parametrize("scaled", [False, True])
@pytest.mark.parametrize("warps", [4, 8])
@pytest.mark.parametrize("stages", [1, 2])
def test_hcu_scaled_pipeline_values(format_a, format_b, inner, scaled, warps, stages, device):
    m, n, k = 24, 20, inner
    generator = torch.Generator().manual_seed(1493)
    a, ref_a = precision_operand(format_a, (m, k), 1, generator, device)
    b, ref_b = precision_operand(format_b, (k, n), 0, generator, device)
    if warps == 8:
        b = b.T.contiguous().T
    groups = triton.cdiv(k, 32)
    sa = (126 + torch.arange(m * groups).reshape(m, groups) % 3).to(torch.uint8) if scaled else None
    sb = (126 + (torch.arange(n * groups).reshape(n, groups) + 1) % 3).to(torch.uint8) if scaled else None
    if scaled:
        ref_a *= torch.exp2(sa.double() - 127).repeat_interleave(32, 1)[:, :k]
        ref_b *= torch.exp2(sb.double() - 127).repeat_interleave(32, 1)[:, :k].T
    acc = ((torch.arange(m * n).reshape(m, n) % 7) - 3).float() / 4
    expected = (ref_a @ ref_b + acc.double()).float()
    output = torch.full((m * n + 32, ), 1234.5, device=device)
    args = (a, *a.stride(), sa.to(device) if scaled else None, b, *b.stride(), sb.to(device) if scaled else None,
            acc.to(device), output[16:-16], m, n, k, format_a, format_b)
    for repeat in range(100 if stages == 2 else 3):
        compiled = precision_scaled_dot_loop[(triton.cdiv(m, 16), triton.cdiv(n, 16))](*args, num_warps=warps,
                                                                                       num_stages=stages)
        if repeat == 0 and stages == 2 and "e2m1" in (format_a, format_b):
            assert_fp4_pipeline(compiled, stages, (format_a, format_b), (16, 16, 32))
        torch.testing.assert_close(output[16:-16].reshape(m, n).cpu(), expected, rtol=0, atol=0)
        assert torch.all(output[:16] == 1234.5) and torch.all(output[-16:] == 1234.5)


@pytest.mark.parametrize("fmt,code,scale_a,scale_b,groups", [
    ("e2m1", 1, 127, 127, 1),
    ("e2m1", 1, 0, 254, 1),
    ("e2m1", 1, 254, 0, 1),
    ("e2m1", 1, 255, 127, 1),
    ("e2m1", 7, 254, 0, 1),
    ("e2m1", 1, 0, 254, 2),
    ("e4m3", 1, 0, 127, 1),
    ("e4m3", 126, 254, 0, 1),
    ("e4m3", 127, 127, 127, 1),
])
def test_hcu_scaled_extremes_values(fmt, code, scale_a, scale_b, groups, device):
    k = 32 * groups
    packed = fmt == "e2m1"
    storage_code = code | (code << 4) if packed else code
    a = torch.full((16, k // (2 if packed else 1)), storage_code, dtype=torch.uint8, device=device)
    b = torch.full((k // (2 if packed else 1), 16), storage_code, dtype=torch.uint8, device=device)
    if packed:
        value = [0., .5, 1., 1.5, 2., 3., 4., 6.][code]
    else:
        value = torch.tensor([code], dtype=torch.uint8).view(torch.float8_e4m3fn).double().item()
        a, b = a.view(torch.float8_e4m3fn), b.view(torch.float8_e4m3fn)
    sa = torch.tensor([scale_a, scale_b] if groups == 2 else [scale_a], dtype=torch.uint8).repeat(16, 1).to(device)
    sb = torch.tensor([scale_b, scale_a] if groups == 2 else [scale_b], dtype=torch.uint8).repeat(16, 1).to(device)
    acc = torch.full((16, 16), .25, device=device) if groups == 2 else None
    output = torch.full((16 * 16 + 32, ), 1234.5, device=device)
    precision_scaled_dot[(1, )](a, *a.stride(), sa, b, *b.stride(), sb, acc, output[16:-16], 16, 16, k, fmt, fmt,
                                num_warps=4, num_stages=1)
    actual = output[16:-16].cpu()
    if scale_a == 255 or scale_b == 255 or math.isnan(value):
        assert actual.isnan().all()
    else:
        expected = math.ldexp(32 * value * value, scale_a + scale_b - 254) * groups + (.25 if groups == 2 else 0)
        expected = torch.tensor(expected, dtype=torch.float64).float().expand_as(actual)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.all(output[:16] == 1234.5) and torch.all(output[-16:] == 1234.5)


@pytest.mark.parametrize("format_a,format_b", FORMAT_PAIRS)
@pytest.mark.parametrize("scale_mode", ["none", "lhs", "rhs", "both"])
@pytest.mark.parametrize("col_b", [False, True])
@pytest.mark.parametrize("geometry", [(16, 16, 32, True), (32, 16, 64, True), (32, 32, 64, False)])
def test_hcu_scaled_precision_values(format_a, format_b, scale_mode, col_b, geometry, device):
    m, n, k, k_pack = geometry
    lhs_k_pack = k_pack or format_a != "e2m1"
    rhs_k_pack = k_pack or format_b != "e2m1"
    generator = torch.Generator().manual_seed(1493)
    a, ref_a = precision_operand(format_a, (m, k), 1 if lhs_k_pack else 0, generator, device)
    b, ref_b = precision_operand(format_b, (k, n), 0 if rhs_k_pack else 1, generator, device)
    if col_b:
        b = b.T.contiguous().T
    scales = []
    for present, length in [(scale_mode in ("lhs", "both"), m), (scale_mode in ("rhs", "both"), n)]:
        codes = (126 + torch.arange(length * (k // 32)).reshape(length, k // 32) % 3).to(torch.uint8)
        scales.append(codes.to(device) if present else None)
    for index, value in enumerate([ref_a, ref_b]):
        if scales[index] is not None:
            factors = torch.exp2(scales[index].cpu().to(torch.float64) - 127).repeat_interleave(32, dim=1)
            value.mul_(factors if index == 0 else factors.T)
    compute_dtype = torch.float16 if "fp16" in (format_a, format_b) else torch.bfloat16
    ref_a, ref_b = ref_a.to(compute_dtype).to(torch.float64), ref_b.to(compute_dtype).to(torch.float64)
    acc = (torch.arange(m * n).reshape(m, n) % 7 - 3).to(torch.float32) / 4
    expected = (ref_a @ ref_b + acc.to(torch.float64)).to(torch.float32)
    output = torch.full((m * n + 32, ), 1234.5, dtype=torch.float32, device=device)
    precision_scaled_dot[(1, )](a, *a.stride(), scales[0], b, *b.stride(), scales[1], acc.to(device), output[16:-16], m,
                                n, k, format_a, format_b, lhs_k_pack, rhs_k_pack, num_warps=4, num_stages=1)
    torch.testing.assert_close(output[16:-16].reshape(m, n).cpu(), expected, rtol=0, atol=0)
    assert torch.all(output[:16] == 1234.5) and torch.all(output[-16:] == 1234.5)


@pytest.mark.parametrize("integer", [torch.int8, torch.int32])
@pytest.mark.parametrize("floating", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("to_float", [False, True])
def test_hcu_standard_integer_cast_values(integer, floating, to_float, device):
    if to_float:
        source = (torch.arange(-128, 128).to(integer) if integer == torch.int8 else torch.tensor(
            [-(2**31), 2**31 - 1, -16777217, 16777217, -65521, 65521, -128, 127, -1, 0, 1], dtype=integer))
        expected = source.to(floating)
        output = torch.full((source.numel() + 32, ), 42, dtype=floating, device=device)
    else:
        source = torch.tensor([
            -2147483520., -65504., -128.75, -128., -127.5, -1.5, -.75, -0., 0., .75, 1.5, 126.75, 127.5, 128., 65504.,
            2147483520.
        ], dtype=floating)
        truncated = source.double().trunc()
        valid = source.isfinite() & (truncated >= torch.iinfo(integer).min) & (truncated <= torch.iinfo(integer).max)
        source = source[valid]
        expected = source.to(integer)
        output = torch.full((source.numel() + 32, ), 42, dtype=integer, device=device)
    precision_cast[(triton.cdiv(source.numel(), 256), )](source.to(device), output[16:-16], source.numel(), 256)
    torch.testing.assert_close(output[16:-16].cpu(), expected, rtol=0, atol=0)
    assert torch.all(output[:16] == 42) and torch.all(output[-16:] == 42)


@pytest.mark.parametrize("to_int8", [False, True])
def test_hcu_integer_width_cast_values(to_int8, device):
    source = (torch.tensor([-(2**31), 2**31 - 1, -257, -256, -129, -128, 127, 128, 255, 256], dtype=torch.int32)
              if to_int8 else torch.arange(-128, 128, dtype=torch.int8))
    destination = torch.int8 if to_int8 else torch.int32
    expected = source.to(destination)
    output = torch.full((source.numel() + 32, ), 42, dtype=destination, device=device)
    precision_cast[(triton.cdiv(source.numel(), 256), )](source.to(device), output[16:-16], source.numel(), 256)
    torch.testing.assert_close(output[16:-16].cpu(), expected, rtol=0, atol=0)
    assert torch.all(output[:16] == 42) and torch.all(output[-16:] == 42)


@pytest.mark.parametrize("source", ["fp16", "bf16", "fp32", "fp64"])
@pytest.mark.parametrize("destination", ["fp8e4nv", "fp8e5"])
@pytest.mark.parametrize("rounding", [None, "rtne", "rtz"])
def test_hcu_fp8_downcast_compile(source, destination, rounding, fresh_triton_cache):
    triton.compile(
        ASTSource(precision_cast, {"X": "*" + source, "Y": "*" + destination},
                  constexprs={"N": 513, "ROUND": rounding, "BLOCK": 256}), target=GPUTarget("hip", "gfx936", 64),
        options={"num_warps": 4})


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("destination", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("rounding", [None, "rtne", "rtz"])
def test_hcu_fp8_downcast_values(dtype, destination, rounding, device):
    levels = torch.arange(128).to(torch.uint8).view(destination).double()
    levels = levels[levels.isfinite()]
    boundaries = ((levels[:-1] + levels[1:]) / 2).tolist() + [450., 464., 465., 57344., 61440., 65504.]
    source = conversion_source_values(dtype, boundaries)
    limit = 448 if destination == torch.float8_e4m3fn else 57344
    clipped = source.double().clamp(-limit, limit)
    fmt = tl.float8e4nv if destination == torch.float8_e4m3fn else tl.float8e5
    mode = ir.ROUNDING_MODE.RTZ if rounding == "rtz" else ir.ROUNDING_MODE.RTNE
    expected = torch.from_numpy(_encode_fp64_to_float(clipped.numpy(), fmt, mode))
    output = torch.full((source.numel() + 32, ), 165, dtype=torch.uint8, device=device).view(destination)
    precision_cast[(triton.cdiv(source.numel(), 256), )](source.to(device), output[16:-16], source.numel(), 256,
                                                         rounding)
    actual = output[16:-16].cpu()
    assert torch.equal(actual.isnan(), source.isnan())
    not_nan = ~source.isnan()
    torch.testing.assert_close(actual.view(torch.uint8)[not_nan], expected[not_nan], rtol=0, atol=0)
    assert torch.all(output[:16].view(torch.uint8) == 165) and torch.all(output[-16:].view(torch.uint8) == 165)


@pytest.mark.parametrize("source_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("destination", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_hcu_fp8_upcast_values(source_dtype, destination, device):
    source = torch.arange(256).to(torch.uint8).view(source_dtype)
    expected = source.to(destination)
    output = torch.full((256 + 32, ), 42, dtype=destination, device=device)
    precision_cast[(1, )](source.to(device), output[16:-16], 256, 256)
    actual = output[16:-16].cpu()
    assert torch.equal(actual.isnan(), expected.isnan())
    torch.testing.assert_close(actual[~expected.isnan()], expected[~expected.isnan()], rtol=0, atol=0)
    assert torch.equal(torch.signbit(actual[~expected.isnan()]), torch.signbit(expected[~expected.isnan()]))
    assert torch.all(output[:16] == 42) and torch.all(output[-16:] == 42)


@pytest.mark.parametrize("integer", [torch.int8, torch.int32])
@pytest.mark.parametrize("fp8", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("to_fp8", [False, True])
def test_hcu_fp8_integer_values(integer, fp8, to_fp8, device):
    if to_fp8:
        source = (torch.arange(-128, 128).to(integer) if integer == torch.int8 else torch.tensor(
            [-(2**31), 2**31 - 1, -16842753, 16842753, -57345, 57345, -449, 449, -1, 0, 1], dtype=integer))
        limit = 448 if fp8 == torch.float8_e4m3fn else 57344
        expected = source.float().clamp(-limit, limit).to(fp8).view(torch.uint8)
        output = torch.full((source.numel() + 32, ), 165, dtype=torch.uint8, device=device).view(fp8)
    else:
        source = torch.arange(256).to(torch.uint8).view(fp8)
        values = source.double()
        valid = values.isfinite() & (values >= torch.iinfo(integer).min) & (values <= torch.iinfo(integer).max)
        # Index raw bytes: PyTorch 2.4 has no CPU FP8 masked-index kernel.
        source = source.view(torch.uint8)[valid].view(fp8)
        expected = source.double().trunc().to(integer)
        output = torch.full((source.numel() + 32, ), 42, dtype=integer, device=device)
    precision_cast[(triton.cdiv(source.numel(), 256), )](source.to(device), output[16:-16], source.numel(), 256)
    torch.testing.assert_close(output[16:-16].cpu().view(expected.dtype), expected, rtol=0, atol=0)
    guard = output.view(torch.uint8) if to_fp8 else output
    assert torch.all(guard[:16] == (165 if to_fp8 else 42)) and torch.all(guard[-16:] == (165 if to_fp8 else 42))


@pytest.mark.parametrize("source", ["core", "semantic", "standard", "driver", "hip_utils", "tle_core"])
def test_hcu_provider_source_invalidates_cache(source, tmp_path):
    package = Path(triton.__file__).parent
    runtime = tmp_path / "runtime/triton"
    shutil.copytree(package, runtime, ignore=shutil.ignore_patterns("_C", "__pycache__"), symlinks=True)
    (runtime / "_C").symlink_to(package / "_C", target_is_directory=True)
    if source == "hip_utils":
        provider = runtime / "language/extra/hip/utils.py"
        old = "(.25, .75, 1.25, 1.75, 2.5, 3.5, 5.)"
        new = "(.25, 1., 1.25, 1.75, 2.5, 3.5, 5.)"
    elif source == "tle_core":
        provider = runtime / "experimental/tle/language/core.py"
        old = 'x.handle.set_attr("tt.load.async", _semantic.builder.get_bool_attr(is_async))\n    return x'
        new = old.replace("return x", "return tl.add(x, 1, _semantic=_semantic)")
    elif source == "driver":
        provider = runtime / "runtime/driver.py"
        old = "def reset_active(self) -> None:\n        self._active = self.default"
        new = old.replace("self.default", "None")
    else:
        provider = runtime / "language" / (source + ".py")
        if source == "core":
            old = "return _semantic.full(shape, value, dtype)"
            new = "return _semantic.full(shape, value + 1, dtype)"
        elif source == "semantic":
            old = "return self.splat(self.make_scalar(value, dtype), shape)"
            new = "return self.splat(self.make_scalar(value + 1, dtype), shape)"
        else:
            old = "return (x + (div - 1)) // div"
            new = old + " + 1"
    assert provider.resolve().is_relative_to(runtime.resolve())
    original = provider.read_text()
    assert original.count(old) == 1
    probe = tmp_path / "probe.py"
    probe.write_text('''import hashlib, json
from pathlib import Path
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.backends.compiler import GPUTarget
from triton.backends.hcu.compiler_hcu import HIPBackend
from triton.compiler import ASTSource
assert Path(triton.__file__).is_relative_to(Path(__file__).parent / "runtime")
@triton.jit
def encode(X, Y, Z):
    offsets = tl.arange(0, 8)
    packed = tl.fp_to_fp4(tl.full((16,), .75, tl.float32))
    tl.store(Y + offsets + tl.cdiv(0, 1), packed)
    tl.store(Z + offsets, tle.load(X + offsets))
target = GPUTarget("hip", "gfx936", 64)
kernel = triton.compile(ASTSource(encode, {"X": "*u8", "Y": "*u8", "Z": "*u8"}), target=target)
print(json.dumps([HIPBackend(target).hash(), kernel.hash, hashlib.sha256(kernel.asm["ttir"].encode()).hexdigest()]))
''')
    env = dict(os.environ, PYTHONPATH=str(runtime.parent), PYTHONDONTWRITEBYTECODE="1",
               TRITON_CACHE_DIR=str(tmp_path / "cache"), HIP_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="")
    before = json.loads(subprocess.check_output([sys.executable, str(probe)], env=env, text=True, timeout=120))
    provider.write_text(original.replace(old, new))
    after = json.loads(subprocess.check_output([sys.executable, str(probe)], env=env, text=True, timeout=120))
    assert before[0] != after[0]
    assert before[1] != after[1]
    assert (before[2] == after[2]) == (source == "driver")
