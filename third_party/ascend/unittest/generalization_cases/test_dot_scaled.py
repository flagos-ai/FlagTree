# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import contextlib
import itertools
import re
import math
import textwrap
import os
import inspect
import pathlib
import test_common
import numpy as np
import pytest
import torch
import torch_npu
import triton
import triton.language as tl

from numpy.random import RandomState
from triton.language.extra import libdevice
from triton.tools.get_ascend_devices import is_compile_on_910_95


@triton.jit
def dot_scale_kernel(a_base, stride_a0: tl.constexpr, stride_a1: tl.constexpr, a_scale, b_base, stride_b0: tl.constexpr,
                     stride_b1: tl.constexpr, b_scale, out, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                     BLOCK_K: tl.constexpr, type_a: tl.constexpr, type_b: tl.constexpr, acc_num: tl.constexpr):
    PACKED_BLOCK_K_A: tl.constexpr = BLOCK_K
    PACKED_BLOCK_K_B: tl.constexpr = BLOCK_K
    str_a0: tl.constexpr = stride_a0
    a_ptr = a_base + tl.arange(0, BLOCK_M)[:, None] * stride_a0 + tl.arange(0, str_a0)[None, :] * stride_a1
    b_ptr = b_base + tl.arange(0, PACKED_BLOCK_K_B)[:, None] * stride_b0 + tl.arange(0, BLOCK_N)[None, :] * stride_b1

    a = tl.load(a_ptr)
    b = tl.load(b_ptr)
    SCALE_BLOCK_K: tl.constexpr = BLOCK_K // 32
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    if a_scale is not None:
        scale_a_ptr = a_scale + tl.arange(0, BLOCK_M)[:, None] * SCALE_BLOCK_K + tl.arange(0, SCALE_BLOCK_K)[None, :]
        a_scale = tl.load(scale_a_ptr)
    if b_scale is not None:
        scale_b_ptr = b_scale + tl.arange(0, BLOCK_N)[:, None] * SCALE_BLOCK_K + tl.arange(0, SCALE_BLOCK_K)[None, :]
        b_scale = tl.load(scale_b_ptr)
    accumulator = tl.dot_scaled(a, a_scale, type_a, b, b_scale, type_b, acc=accumulator, out_dtype=tl.float32)
    if acc_num is not None:
        for _ in range(acc_num):
            accumulator = tl.dot_scaled(a, a_scale, type_a, b, b_scale, type_b, acc=accumulator, out_dtype=tl.float32)

    out_ptr = out + tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    tl.store(out_ptr, accumulator.to(a.dtype))


def golden_ref(x, scale_x, y, scale_y):
    shape_expand_x = x.shape[-1] // scale_x.shape[-1]
    if x.dtype == torch.bfloat16:
        upscale_x = scale_x.repeat_interleave(shape_expand_x, dim=1).to(torch.int16)
        upscale_x = (upscale_x + 127 << 7).view(torch.bfloat16)
    else:
        scale_fp32 = scale_x.repeat_interleave(shape_expand_x, dim=1).to(torch.int32)
        scale_fp32 = (scale_fp32 + 127 << 23).view(torch.float32)
        upscale_x = scale_fp32.to(torch.float16)
    upscale_y = None
    if scale_y is None:
        upscale_y = torch.ones_like(y)
    else:
        scale_y = scale_y.T
        shape_expand_y = y.shape[0] // scale_y.shape[0]
        if y.dtype == torch.bfloat16:
            upscale_y = scale_y.repeat_interleave(shape_expand_y, dim=0).to(torch.int16)
            upscale_y = (upscale_y + 127 << 7).view(torch.bfloat16)
        else:
            scale_fp32 = scale_y.repeat_interleave(shape_expand_y, dim=0).to(torch.int32)
            scale_fp32 = (scale_fp32 + 127 << 23).view(torch.float32)
            upscale_y = scale_fp32.to(torch.float16)
    ret = torch.matmul(x * upscale_x, y * upscale_y)
    return ret


@pytest.mark.parametrize("M, N, K, rhs_scale, normal_type, acc_num, num_warps",
                         [(M, N, K, rhs_scale, normal_type, acc_num, 4)
                          for M, N, K in itertools.product([16, 32, 64, 128], [16, 32, 64, 128], [32, 64])
                          for rhs_scale in [False, True]
                          for normal_type in ["bf16", "fp16"]
                          for acc_num in [None, 1, 2]])
def test_scaled_dot(M, N, K, rhs_scale, normal_type, num_warps, acc_num):
    device = "npu"

    # The max exponent we use to initialize data in the x/y and associated scale tensor to avoid
    # overflow when scaling.
    comp_dtype_max_exp = 6 if normal_type == "fp16" else 15

    torch.manual_seed(0)

    def make_arg(shape, ty):
        if ty == "bf16" or ty == "fp16":
            comp_dtype = torch.float16 if ty == "fp16" else torch.bfloat16
            ret = torch.randn(shape, dtype=comp_dtype, device=device)
            # Clamp to avoid relative error issues
            ret.clamp_(-2**comp_dtype_max_exp, 2**comp_dtype_max_exp - 1)
        else:
            ret = torch.randint(256, shape, dtype=torch.int8, device=device)
        return ret

    type_a = normal_type
    type_b = type_a

    x = make_arg((M, K), type_a)
    y = make_arg((K, N), type_b)

    min_scale, max_scale = (0, 142) if type_a == torch.bfloat16 else (124, 131)
    scale_x = torch.randint(min_scale - 128, max_scale - 127, (M, K // 32), dtype=torch.int8, device=device)
    min_scale, max_scale = (0, 142) if type_b == torch.bfloat16 else (124, 131)
    scale_y = torch.randint(min_scale - 128, max_scale - 127, (N, K // 32), dtype=torch.int8, device=device)

    if not rhs_scale:
        scale_y = None

    kernel_kwargs = {"num_warps": num_warps}
    z = x.new_empty((M, N), dtype=x.dtype)
    pgm = dot_scale_kernel[(1, )](x, *x.stride(), scale_x, y, *y.stride(), scale_y, z, M, N, K, type_a, type_b, acc_num,
                                  **kernel_kwargs)
    z_ref = golden_ref(x, scale_x, y, scale_y)
    if acc_num is not None:
        z_ref = z_ref * (acc_num + 1)

    atol = 1e-5
    rtol = 1e-2
    torch.testing.assert_close(z, z_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("B, M, N, K", [(1, 32, 64, 64)])
def test_4d_dot(B, M, N, K):
    device = "npu"
    torch.manual_seed(0)

    x4d = torch.randn((B, B, M, N), dtype=torch.float16, device=device)
    y4d = torch.randn((B, B, N, K), dtype=torch.float16, device=device)

    x2d = x4d.view(-1, N)  # shape (B*B*M, N)
    y2d = y4d.view(-1, K)  # shape (B*B*N, K)
    scale_x = torch.randint(-10, 10, (x2d.shape[0], N // 32), dtype=torch.int8, device=device)
    scale_y = torch.randint(-10, 10, (y2d.shape[1], N // 32), dtype=torch.int8, device=device)

    z = torch.empty((x2d.shape[0], y2d.shape[0]), dtype=x2d.dtype, device=device)
    acc_num = None
    dot_scale_kernel[(1, )](x2d, *x2d.stride(), scale_x, y2d, *y2d.stride(), None, z, x2d.shape[0], y2d.shape[0], K,
                            "fp16", "fp16", None, num_warps=4)
    z_ref = golden_ref(x2d, scale_x, y2d, None)
    if acc_num is not None:
        z_ref = z_ref * (acc_num + 1)

    atol = 1e-5
    rtol = 1e-2
    torch.testing.assert_close(z, z_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("B, M, N, K", [(2, 16, 16, 32)])
@test_common.raises_with_match(triton.compiler.errors.CompilationError,
                               r"lhs last dimension .* must equal rhs penultimate dimension")
def test_2d_dot_invaild_shape(B, M, N, K):
    device = "npu"
    torch.manual_seed(0)

    x4d = torch.randn((B, B, M, N), dtype=torch.float16, device=device)
    y4d = torch.randn((B, B, N, K), dtype=torch.float16, device=device)

    x2d = x4d.view(-1, N)  # shape (B*B*M, N)
    y2d = y4d.view(-1, K)  # shape (B*B*N, K)
    scale_x = torch.randint(-10, 10, (x2d.shape[0], N // 32), dtype=torch.int8, device=device)
    scale_y = torch.randint(-10, 10, (y2d.shape[1], N // 32), dtype=torch.int8, device=device)

    z = torch.empty((x2d.shape[0], y2d.shape[0]), dtype=x2d.dtype, device=device)
    acc_num = None
    dot_scale_kernel[(1, )](x2d, *x2d.stride(), scale_x, y2d, *y2d.stride(), None, z, x2d.shape[0], y2d.shape[0], K,
                            "fp16", "fp16", None, num_warps=4)


VALID_MAIN_DTYPES = {
    torch.float16,  # fp16
    torch.bfloat16,  # bf16
}

ALL_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float32,  # fp32
    torch.bool,
}
ILLEGAL_MAIN_DTYPES = ALL_DTYPES - VALID_MAIN_DTYPES

ILLEGAL_SCALE_DTYPES = {
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.float32,
    torch.bfloat16,
    torch.bool,
}

from itertools import product


def is_legal_dtype(lhs_dtype, rhs_dtype, lhs_scale_dtype, rhs_scale_dtype):
    return (lhs_dtype in VALID_MAIN_DTYPES and rhs_dtype in VALID_MAIN_DTYPES and lhs_scale_dtype is torch.int8
            and rhs_scale_dtype is torch.int8)


illegal_cases = []
for lhs, rhs, lhs_s, rhs_s in product(
        VALID_MAIN_DTYPES | ILLEGAL_MAIN_DTYPES,
        VALID_MAIN_DTYPES | ILLEGAL_MAIN_DTYPES,
    {torch.int8} | ILLEGAL_SCALE_DTYPES,
    {torch.int8} | ILLEGAL_SCALE_DTYPES,
):

    if not is_legal_dtype(lhs, rhs, lhs_s, rhs_s):
        illegal_cases.append((lhs, rhs, lhs_s, rhs_s))

illegal_cases = sorted(set(illegal_cases), key=lambda t: tuple(str(i) for i in t))


@pytest.mark.parametrize(
    "lhs_dtype, rhs_dtype, lhs_scale_dtype, rhs_scale_dtype",
    illegal_cases,
)
@test_common.raises_with_match(Exception, r"(?i)invalid|unsupported|dtype")
def test_invalid_dtype_should_fail(lhs_dtype, rhs_dtype, lhs_scale_dtype, rhs_scale_dtype):
    device = "npu"
    M, N, K = 32, 32, 64
    num_warps = 4

    def make_tensor(shape, dtype):
        return torch.randn(shape, dtype=dtype, device=device) \
            if dtype.is_floating_point else \
            torch.randint(-10, 10, shape, dtype=dtype, device=device)

    def make_scale(shape, dtype):
        return torch.randint(-10, 10, shape, dtype=dtype, device=device)

    x = make_tensor((M, K), lhs_dtype)
    y = make_tensor((K, N), rhs_dtype)
    lhs_scale = make_scale((M, K // 32), lhs_scale_dtype)
    rhs_scale = make_scale((N, K // 32), rhs_scale_dtype)
    z = torch.empty((M, N), dtype=lhs_dtype, device=device)

    dot_scale_kernel[(1, )](
        x,
        *x.stride(),
        lhs_scale,
        y,
        *y.stride(),
        rhs_scale,
        z,
        M,
        N,
        K,
        str(lhs_dtype).split('.')[-1],
        str(rhs_dtype).split('.')[-1],
        None,
        num_warps=num_warps,
    )


@pytest.mark.parametrize(
    "M, N, K, col_a, col_b, type_a, type_b, num_warps",
    list(
        itertools.product([32, 64, 128],  # M
                          [32, 64, 128],  # N
                          [64, 128],  # K
                          [True, False],  # col_a
                          [True, False],  # col_b
                          ["e4m3", "e5m2"],  # type_a
                          ["e4m3", "e5m2"],  # type_b
                          [4]  # num_warps
                          )))
def test_scaled_dot_fp8(M, N, K, col_a, col_b, type_a, type_b, num_warps):
    device = "npu"
    if not is_compile_on_910_95:
        pytest.skip("Skipping dot_scaled on A2/A3 case")

    @triton.jit
    def dot_scale_fp8_kernel(a_base, stride_a0, stride_a1, a_scale, b_base, stride_b0, stride_b1, out,
                             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, type_a: tl.constexpr,
                             type_b: tl.constexpr):
        tl.static_assert(type_b == "e4m3" or type_b == "e5m2", "type_b must be fp8")
        IS_FP8: tl.constexpr = type_a == "e4m3" or type_a == "e5m2"
        DIV_FACTOR: tl.constexpr = 1 if IS_FP8 else 2
        PACKED_BLOCK_K_A: tl.constexpr = BLOCK_K // DIV_FACTOR
        PACKED_BLOCK_K_B: tl.constexpr = BLOCK_K
        a_ptr = a_base + tl.arange(0, BLOCK_M)[:, None] * stride_a0 + tl.arange(0,
                                                                                PACKED_BLOCK_K_A)[None, :] * stride_a1
        b_ptr = b_base + tl.arange(0, PACKED_BLOCK_K_B)[:, None] * stride_b0 + tl.arange(0,
                                                                                         BLOCK_N)[None, :] * stride_b1

        SCALE_BLOCK_K: tl.constexpr = BLOCK_K // 32
        scale_a_ptr = a_scale + tl.arange(0, BLOCK_M)[:, None] * SCALE_BLOCK_K + tl.arange(0, SCALE_BLOCK_K)[None, :]

        a = tl.load(a_ptr)
        b = tl.load(b_ptr)
        a_scale = tl.load(scale_a_ptr)
        c = tl.dot_scaled(a, a_scale, type_a, b, None, type_b)
        out_ptr = out + tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
        tl.store(out_ptr, c.to(tl.bfloat16))

    @triton.jit
    def mxfp_to_bf16_kernel(
        x_ptr,
        scale_ptr,
        mxfp_ptr,
        N,
        e_bits: tl.constexpr,
        m_bits: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        is_fp8: tl.constexpr = e_bits + m_bits == 7
        # fp8: BLOCK_SIZE -> BLOCK_SIZE // 32, 32
        # fp4: BLOCK_SIZE // 2 -> BLOCK_SIZE // 32 , 16
        PARALLEL_DIM: tl.constexpr = BLOCK_SIZE // 32
        LAST_DIM: tl.constexpr = 32 if is_fp8 else 16
        LOAD_SIZE: tl.constexpr = LAST_DIM * PARALLEL_DIM

        offsets = (tl.program_id(0) * LOAD_SIZE + tl.arange(0, PARALLEL_DIM)[:, None] * LAST_DIM +
                   tl.arange(0, LAST_DIM)[None, :])
        x = tl.load(x_ptr + offsets, mask=offsets < N * LAST_DIM)

        offsets = tl.program_id(0) * PARALLEL_DIM + tl.arange(0, PARALLEL_DIM)[:, None]
        scale = tl.load(scale_ptr + offsets, mask=offsets < N)
        tl.static_assert(scale.dtype == tl.uint8)
        tl.static_assert(x.dtype == tl.uint8)

        scale_bf16 = (scale.to(tl.uint16) << 7).to(tl.bfloat16, bitcast=True)
        if is_fp8:
            if e_bits == 5 and m_bits == 2:
                x_f8 = x.to(tl.float8e5, bitcast=True)
                x_bf16 = x_f8.to(tl.bfloat16)
                # Preserve infs and nans. FIXME Fp8E5M2_to_Bf16 doesn't preserve them!
                non_finite_mask: tl.constexpr = ((1 << e_bits) - 1) << m_bits
                non_finite_mask_bf16: tl.constexpr = ((1 << 8) - 1) << 7
                x_bf16 = tl.where(
                    x & non_finite_mask == non_finite_mask,
                    (x_bf16.to(tl.uint16, bitcast=True) | non_finite_mask_bf16).to(tl.bfloat16, bitcast=True),
                    x_bf16,
                )
            else:
                tl.static_assert(e_bits == 4 and m_bits == 3)
                x_f8 = x.to(tl.float8e4nv, bitcast=True)
                x_bf16 = x_f8.to(tl.bfloat16)
        else:
            # e2m1
            em0 = x & 0x70
            em1 = x & 0x7
            x0 = (em0.to(tl.uint16) << 2) | ((x & 0x80).to(tl.uint16) << 8)
            x1 = (em1.to(tl.uint16) << (2 + 4)) | ((x & 0x8).to(tl.uint16) << (8 + 4))
            # Three cases:
            # 1) x is normal and non-zero: Correct bias
            x0 = tl.where((em0 & 0x60) != 0, x0 + ((127 - 1) << 7), x0)
            x1 = tl.where((em1 & 0x6) != 0, x1 + ((127 - 1) << 7), x1)
            # 2) x is subnormal (x == 0bs001 where s is the sign): Map to +-0.5 in bf16
            x0 = tl.where(em0 == 0x10, 16128 | (x0 & 0x8000), x0)
            x1 = tl.where(em1 == 0x1, 16128 | (x1 & 0x8000), x1)
            # 3) x is zero, do nothing
            x_bf16 = tl.interleave(x0, x1).to(tl.bfloat16, bitcast=True)
        # Multiplication preserves infs and NaNs in x_bf16
        mxfp = x_bf16 * scale_bf16
        # If scale is NaN, we encode it as an bf16 inf, so we need to correct for that
        mxfp = tl.where(scale == 0xFF, float("nan"), mxfp)

        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        tl.store(mxfp_ptr + offsets, tl.ravel(mxfp), mask=offsets < N * 32)

    def dot_scale_ref(x, scale, y, type_x, type_y):
        e_bits, m_bits = {"e2m1": (2, 1), "e4m3": (4, 3), "e5m2": (5, 2)}[type_x]
        type_fp8_y = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}[type_y]

        comp_dtype = torch.bfloat16

        x = x.contiguous()
        x_upcast = x.new_empty(scale.shape[:-1] + (32 * scale.shape[-1], ), dtype=comp_dtype)

        N = x_upcast.numel()
        BLOCK_SIZE = 512
        grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE, )
        mxfp_to_bf16_kernel[grid](x, scale, x_upcast, scale.numel(), e_bits, m_bits, BLOCK_SIZE, num_warps=num_warps)
        assert x_upcast.isfinite().all()

        y_upcast = y.view(type_fp8_y).to(comp_dtype)

        class AccumulateInFp32:

            def __enter__(self):
                self.prev_value = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

            def __exit__(self, exc_type, exc_val, exc_tb):
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = self.prev_value

        with AccumulateInFp32():
            return torch.matmul(x_upcast.to(comp_dtype), y_upcast.to(comp_dtype))

    torch.manual_seed(0)

    def create_uint8(shape, col_major=False, max_val=255):
        if col_major:
            shape = shape[:-2] + (shape[-1], shape[-2])
        ret = torch.randint(max_val + 1, shape, dtype=torch.uint8, device=device)
        if col_major:
            ret = ret.mT
        return ret

    DIV_FACTOR = 2 if type_a == "e2m1" else 1
    x = create_uint8((M, K // DIV_FACTOR), col_major=col_a)
    y = create_uint8((K, N), col_major=col_b)

    # sample scales that don't overflow as otherwise it's implementation defined (underflowing is alright)
    # We substract a reasonably high number (64) so that the sum of all the mxfp elements does not overflow
    m_bytes = int(type_a[1])
    bias_type_a = 1 << (m_bytes - 1) - 1
    max_exponent_type_a = (1 << m_bytes) - 1 - bias_type_a
    scale_x = create_uint8((M, K // 32), max_val=255 - max_exponent_type_a - 64)

    def make_finite(x, dtype):
        # e5m2 has too many non-finite values when sampled uniformly (1 / 32) and
        # Fp8E5M2_to_Bf16 doesn't preserve NaNs (fixme)
        if dtype not in ("e5m2", "e4m3"):
            return x
        mask = 0x7C if dtype == "e5m2" else 0x7F
        finite = torch.arange(x.numel(), device=device, dtype=torch.int32).reshape_as(x) % mask
        x_finite = torch.where(x & mask == mask, finite | (0x80 & x), x)
        x.copy_(x_finite)
        return x

    x = make_finite(x, type_a)
    y = make_finite(y, type_b)

    z = x.new_empty((M, N), dtype=torch.bfloat16)
    pgm = dot_scale_fp8_kernel[(1, )](
        x, *x.stride(), scale_x, y, *y.stride(), z, M, N, K, type_a, type_b,
        num_warps=num_warps)  # to compare with "dot_scale_ref(x, scale_x, y, type_a, type_b)"
