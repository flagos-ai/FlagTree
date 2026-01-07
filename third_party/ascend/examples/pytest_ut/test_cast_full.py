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


import triton
import triton.language as tl

import torch
import torch_npu
import pytest
import test_common

Dimensions = tuple[int, int, int]


@triton.jit
def cast_to_bool(output_ptr, x_ptr, dims: Dimensions, overflow_mode: tl.constexpr):
    xidx = tl.arange(0, dims.XB)
    yidx = tl.arange(0, dims.YB)
    zidx = tl.arange(0, dims.ZB)
    idx = xidx[:, None, None] * YB * ZB + yidx[None, :, None] * ZB + zidx[None, None, :]

    X = tl.load(x_ptr + idx)
    overflow_mode = "trunc" if overflow_mode == 0 else "saturate"
    ret = tl.cast(X, dtype=tl.int1, overflow_mode=overflow_mode)

    tl.store(output_ptr + idx, ret)


@triton.jit
def cast_to_i8(output_ptr, x_ptr, dims: Dimensions, overflow_mode: tl.constexpr):
    xidx = tl.arange(0, dims.XB)
    yidx = tl.arange(0, dims.YB)
    zidx = tl.arange(0, dims.ZB)
    idx = xidx[:, None, None] * YB * ZB + yidx[None, :, None] * ZB + zidx[None, None, :]

    X = tl.load(x_ptr + idx)
    overflow_mode = "trunc" if overflow_mode == 0 else "saturate"
    ret = tl.cast(X, dtype=tl.int8, overflow_mode=overflow_mode)

    tl.store(output_ptr + idx, ret)


def cast_to_i16(output_ptr, x_ptr, dims: Dimensions, overflow_mode: tl.constexpr):
    xidx = tl.arange(0, dims.XB)
    yidx = tl.arange(0, dims.YB)
    zidx = tl.arange(0, dims.ZB)
    idx = xidx[:, None, None] * YB * ZB + yidx[None, :, None] * ZB + zidx[None, None, :]

    X = tl.load(x_ptr + idx)
    overflow_mode = "trunc" if overflow_mode == 0 else "saturate"
    ret = tl.cast(X, dtype=tl.int16, overflow_mode=overflow_mode)

    tl.store(output_ptr + idx, ret)


@triton.jit
def cast_to_i32(output_ptr, x_ptr, dims: Dimensions, overflow_mode: tl.constexpr):
    xidx = tl.arange(0, dims.XB)
    yidx = tl.arange(0, dims.YB)
    zidx = tl.arange(0, dims.ZB)
    idx = xidx[:, None, None] * YB * ZB + yidx[None, :, None] * ZB + zidx[None, None, :]

    X = tl.load(x_ptr + idx)
    overflow_mode = "trunc" if overflow_mode == 0 else "saturate"
    ret = tl.cast(X, dtype=tl.int32, overflow_mode=overflow_mode)

    tl.store(output_ptr + idx, ret)


@triton.jit
def cast_to_i64(output_ptr, x_ptr, dims: Dimensions, overflow_mode: tl.constexpr):
    xidx = tl.arange(0, dims.XB)
    yidx = tl.arange(0, dims.YB)
    zidx = tl.arange(0, dims.ZB)
    idx = xidx[:, None, None] * YB * ZB + yidx[None, :, None] * ZB + zidx[None, None, :]

    X = tl.load(x_ptr + idx)
    overflow_mode = "trunc" if overflow_mode == 0 else "saturate"
    ret = tl.cast(X, dtype=tl.int64, overflow_mode=overflow_mode)

    tl.store(output_ptr + idx, ret)


@triton.jit
def cast_to_fp32(output_ptr, x_ptr, dims: Dimensions, overflow_mode: tl.constexpr):
    xidx = tl.arange(0, dims.XB)
    yidx = tl.arange(0, dims.YB)
    zidx = tl.arange(0, dims.ZB)
    idx = xidx[:, None, None] * YB * ZB + yidx[None, :, None] * ZB + zidx[None, None, :]

    X = tl.load(x_ptr + idx)
    overflow_mode = "trunc" if overflow_mode == 0 else "saturate"
    ret = tl.cast(X, dtype=tl.float32, overflow_mode=overflow_mode)

    tl.store(output_ptr + idx, ret)


@triton.jit
def cast_to_fp16(output_ptr, x_ptr, dims: Dimensions, overflow_mode: tl.constexpr):
    xidx = tl.arange(0, dims.XB)
    yidx = tl.arange(0, dims.YB)
    zidx = tl.arange(0, dims.ZB)
    idx = xidx[:, None, None] * YB * ZB + yidx[None, :, None] * ZB + zidx[None, None, :]

    X = tl.load(x_ptr + idx)
    overflow_mode = "trunc" if overflow_mode == 0 else "saturate"
    ret = tl.cast(X, dtype=tl.float16, overflow_mode=overflow_mode)

    tl.store(output_ptr + idx, ret)


@triton.jit
def cast_to_bf16(output_ptr, x_ptr, dims: Dimensions, overflow_mode: tl.constexpr):
    xidx = tl.arange(0, dims.XB)
    yidx = tl.arange(0, dims.YB)
    zidx = tl.arange(0, dims.ZB)
    idx = xidx[:, None, None] * YB * ZB + yidx[None, :, None] * ZB + zidx[None, None, :]

    X = tl.load(x_ptr + idx)
    overflow_mode = "trunc" if overflow_mode == 0 else "saturate"
    ret = tl.cast(X, dtype=tl.bfloat16, overflow_mode=overflow_mode)

    tl.store(output_ptr + idx, ret)


import numpy as np


def cast_npu(para_type, data_type, to_para, to_dtype, XB, YB, ZB, overflow_mode):

    print(f"TESTING: cast from {para_type} to {to_para} in shape ({XB}, {YB}, {ZB})")

    if para_type == "*i1":
        x = torch.randint(low=0, high=2, size=(XB, YB, ZB), dtype=data_type).npu()
    elif (
        para_type == "*i8"
        or para_type == "*i16"
        or para_type == "*i32"
        or para_type == "*64"
    ):
        x = torch.randint(low=-128, high=128, size=(XB, YB, ZB), dtype=data_type).npu()
    elif para_type == "*i16":
        x = torch.randint(
            low=-32768, high=32768, size=(XB, YB, ZB), dtype=data_type
        ).npu()
    elif para_type == "*i32":
        x = torch.randint(
            low=-65536, high=65536, size=(XB, YB, ZB), dtype=data_type
        ).npu()
    elif para_type == "*i64":
        x = torch.randint(
            low=-65536, high=65536, size=(XB, YB, ZB), dtype=data_type
        ).npu()
    else:  # float
        x = torch.randn((XB, YB, ZB), dtype=data_type).npu()

    if to_para == "*i1":
        triton_func = cast_to_bool
        cmp_type = "bool"
    elif to_para == "*i8":
        triton_func = cast_to_i8
        cmp_type = "int8"
    elif to_para == "*i16":
        triton_func = cast_to_i16
        cmp_type = "int16"
    elif to_para == "*i32":
        triton_func = cast_to_i32
        cmp_type = "int32"
    elif to_para == "*i64":
        triton_func = cast_to_i64
        cmp_type = "int64"
    elif to_para == "*fp16":
        triton_func = cast_to_fp16
        cmp_type = "float16"
    elif to_para == "*fp32":
        triton_func = cast_to_fp32
        cmp_type = "float32"
    elif to_para == "*bf16":
        triton_func = cast_to_bf16
        cmp_type = "bfloat16"

    output = torch.randint(1, (XB, YB, ZB), dtype=to_dtype).npu()

    a = x.to(to_dtype)
    dims = Dimensions(XB=XB, YB=YB, ZB=ZB)

    triton_func[1, 1, 1](output, x, dims, overflow_mode)

    test_common.validate_cmp(cmp_type, a, output, overflow_mode)


def test_cast_high_priority_dtype():

    typelist = [
        (torch.int8, "*i8"),
        (torch.float32, "*fp32"),
        (torch.float16, "*fp16"),
    ]

    overflow_mode = [
        0,  # "trunc",
        1,  # "saturate",
    ]

    shapes = [(8, 32, 32)]
    ContinueList = []
    for src in typelist:
        for dst in typelist:
            if src != dst and (src[1], dst[1]) not in ContinueList:
                for shape in shapes:
                    for mode in overflow_mode:
                        (
                            src[1],
                            src[0],
                            dst[1],
                            dst[0],
                            shape[0],
                            shape[1],
                            shape[2],
                            mode,
                        )

    print("test_cast_full passed")
