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

import numpy as np
import pytest
import torch
import torch_npu
import triton
import triton.language as tl

from numpy.random import RandomState
from triton.language.extra import libdevice


@pytest.mark.parametrize("M, N, K, rhs_scale, normal_type, acc_num, num_warps",
                         [(M, N, K, rhs_scale, normal_type, acc_num, 4)
                          for M, N, K in itertools.product([32, 64], [32, 64], [32])
                          for rhs_scale in [False, True]
                          for normal_type in ["bf16", "fp16"]
                          for acc_num in [None, 1, 2]])
def test_scaled_dot(M, N, K, rhs_scale, normal_type, num_warps, acc_num):
    device = "npu"

    @triton.jit
    def dot_scale_kernel(a_base, stride_a0, stride_a1, a_scale, b_base, stride_b0, stride_b1, b_scale, out,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, type_a: tl.constexpr,
                         type_b: tl.constexpr, acc_num: tl.constexpr):
                         
        PACKED_BLOCK_K_A: tl.constexpr = BLOCK_K
        PACKED_BLOCK_K_B: tl.constexpr = BLOCK_K
        a_ptr = a_base + tl.arange(0, BLOCK_M)[:, None] * stride_a0 + tl.arange(0,
                                                                                PACKED_BLOCK_K_A)[None, :] * stride_a1
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

    kernel_kwargs = {"num_warps": num_warps}
    z = x.new_empty((M, N), dtype=x.dtype)
    pgm = dot_scale_kernel[(1, )](x, *x.stride(), scale_x, y, *y.stride(), scale_y, z, M, N, K, type_a, type_b,
                                  acc_num, **kernel_kwargs)
    z_ref = golden_ref(x, scale_x, y, scale_y)
    if acc_num is not None:
        z_ref = z_ref * (acc_num + 1)

    atol = 1e-5
    rtol = 1e-2
    torch.testing.assert_close(z, z_ref, atol=atol, rtol=rtol)
