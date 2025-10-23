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


@triton.jit
def dot_scale_kernel(a_base, stride_a0: tl.constexpr, stride_a1: tl.constexpr, a_scale, b_base, stride_b0: tl.constexpr,
                     stride_b1: tl.constexpr, b_scale, out,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, type_a: tl.constexpr,
                     type_b: tl.constexpr, acc_num: tl.constexpr):
    PACKED_BLOCK_K_A: tl.constexpr = BLOCK_K
    PACKED_BLOCK_K_B: tl.constexpr = BLOCK_K
    str_a0: tl.constexpr = stride_a0
    a_ptr = a_base + tl.arange(0, BLOCK_M)[:, None] * stride_a0 + tl.arange(0,
                                                                            str_a0)[None, :] * stride_a1
    b_ptr = b_base + tl.arange(0, PACKED_BLOCK_K_B)[:, None] * stride_b0 + tl.arange(0,
                                                                                     BLOCK_N)[None, :] * stride_b1

    a = tl.load(a_ptr)
    b = tl.load(b_ptr)
    SCALE_BLOCK_K: tl.constexpr = BLOCK_K // 32
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    if a_scale is not None:
        scale_a_ptr = a_scale + tl.arange(0, BLOCK_M)[:, None] * SCALE_BLOCK_K + tl.arange(0,
                                                                                           SCALE_BLOCK_K)[None, :]
        a_scale = tl.load(scale_a_ptr)
    if b_scale is not None:
        scale_b_ptr = b_scale + tl.arange(0, BLOCK_N)[:, None] * SCALE_BLOCK_K + tl.arange(0,
                                                                                           SCALE_BLOCK_K)[None, :]
        b_scale = tl.load(scale_b_ptr)
    accumulator = tl.dot_scaled(a, a_scale, type_a, b, b_scale, type_b, acc=accumulator, out_dtype=tl.float32)
    if acc_num is not None:
        for _ in range(acc_num):
            accumulator = tl.dot_scaled(a, a_scale, type_a, b, b_scale, type_b, acc=accumulator,
                                        out_dtype=tl.float32)

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
            ret.clamp_(-2 ** comp_dtype_max_exp, 2 ** comp_dtype_max_exp - 1)
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
    pgm = dot_scale_kernel[(1,)](x, *x.stride(), scale_x, y, *y.stride(), scale_y, z, M, N, K, type_a, type_b,
                                 acc_num, **kernel_kwargs)
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
    scale_x = torch.randint(-10, 10, (x2d.shape[0], N // 32),
                            dtype=torch.int8, device=device)
    scale_y = torch.randint(-10, 10, (y2d.shape[1], N // 32),
                            dtype=torch.int8, device=device)

    z = torch.empty((x2d.shape[0], y2d.shape[0]),
                    dtype=x2d.dtype, device=device)
    acc_num = None
    dot_scale_kernel[(1,)](
        x2d, *x2d.stride(), scale_x,
        y2d, *y2d.stride(), None,
        z,
        x2d.shape[0], y2d.shape[0], K,
        "fp16", "fp16", None,
        num_warps=4
    )
    z_ref = golden_ref(x2d, scale_x, y2d, None)
    if acc_num is not None:
        z_ref = z_ref * (acc_num + 1)

    atol = 1e-5
    rtol = 1e-2
    torch.testing.assert_close(z, z_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("B, M, N, K", [(2, 16, 16, 32)])
@test_common.raises_with_match(triton.compiler.errors.CompilationError,
                               r"lhs last dimension .* must equal rhs penultimate dimension"
                               )
def test_2d_dot_invaild_shape(B, M, N, K):
    device = "npu"
    torch.manual_seed(0)

    x4d = torch.randn((B, B, M, N), dtype=torch.float16, device=device)
    y4d = torch.randn((B, B, N, K), dtype=torch.float16, device=device)

    x2d = x4d.view(-1, N)  # shape (B*B*M, N)
    y2d = y4d.view(-1, K)  # shape (B*B*N, K)
    scale_x = torch.randint(-10, 10, (x2d.shape[0], N // 32),
                            dtype=torch.int8, device=device)
    scale_y = torch.randint(-10, 10, (y2d.shape[1], N // 32),
                            dtype=torch.int8, device=device)

    z = torch.empty((x2d.shape[0], y2d.shape[0]),
                    dtype=x2d.dtype, device=device)
    acc_num = None
    dot_scale_kernel[(1,)](
        x2d, *x2d.stride(), scale_x,
        y2d, *y2d.stride(), None,
        z,
        x2d.shape[0], y2d.shape[0], K,
        "fp16", "fp16", None,
        num_warps=4
    )


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
    return (
            lhs_dtype in VALID_MAIN_DTYPES and
            rhs_dtype in VALID_MAIN_DTYPES and
            lhs_scale_dtype is torch.int8 and
            rhs_scale_dtype is torch.int8
    )


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
def test_invalid_dtype_should_fail(lhs_dtype, rhs_dtype,
                                   lhs_scale_dtype, rhs_scale_dtype):
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

    dot_scale_kernel[(1,)](
        x, *x.stride(), lhs_scale,
        y, *y.stride(), rhs_scale,
        z,
        M, N, K,
        str(lhs_dtype).split('.')[-1],
        str(rhs_dtype).split('.')[-1],
        None,
        num_warps=num_warps,
    )
