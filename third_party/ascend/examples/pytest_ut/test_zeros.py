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


@triton.jit
def fn_npu_f32(output_ptr, x_ptr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    xidx = tl.arange(0, XB)
    yidx = tl.arange(0, YB)
    zidx = tl.arange(0, ZB)

    idx = xidx[:, None, None] * YB * ZB + yidx[None, :, None] * ZB + zidx[None, None, :]

    X = tl.load(x_ptr + idx)

    ret = tl.zeros((XB, YB, ZB), dtype=tl.float32)

    oidx = xidx[:, None, None] * YB * ZB + yidx[None, :, None] * ZB + zidx[None, None, :]

    tl.store(output_ptr + oidx, ret)


@triton.jit
def fn_npu_f16(output_ptr, x_ptr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    xidx = tl.arange(0, XB)
    yidx = tl.arange(0, YB)
    zidx = tl.arange(0, ZB)

    idx = xidx[:, None, None] * YB * ZB + yidx[None, :, None] * ZB + zidx[None, None, :]

    X = tl.load(x_ptr + idx)

    ret = tl.zeros((XB, YB, ZB), dtype=tl.float16)

    oidx = xidx[:, None, None] * YB * ZB + yidx[None, :, None] * ZB + zidx[None, None, :]

    tl.store(output_ptr + oidx, ret)


@triton.jit
def fn_npu_i8(output_ptr, x_ptr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    xidx = tl.arange(0, XB)
    yidx = tl.arange(0, YB)
    zidx = tl.arange(0, ZB)

    idx = xidx[:, None, None] * YB * ZB + yidx[None, :, None] * ZB + zidx[None, None, :]

    X = tl.load(x_ptr + idx)

    ret = tl.zeros((XB, YB, ZB), dtype=tl.int8)

    oidx = xidx[:, None, None] * YB * ZB + yidx[None, :, None] * ZB + zidx[None, None, :]

    tl.store(output_ptr + oidx, ret)


@pytest.mark.parametrize('param_list',
                         [
                             ['float32', (2, 256, 16), 1, 2, 256, 16],
                             ['float32', (8, 8, 4), 1, 8, 8, 4],
                             ['float16', (2, 256, 16), 1, 2, 256, 16],
                             ['float16', (8, 8, 4), 1, 8, 8, 4],
                             ['int8', (2, 256, 16), 1, 2, 256, 16],
                             ['int8', (8, 8, 4), 1, 8, 8, 4],
                         ]
                         )
def test_case(param_list):
    dtype, shape, ncore, XB, YB, ZB = param_list
    x0 = test_common.generate_tensor(shape, dtype)

    y_ref = torch.full((XB, YB, ZB), 0, dtype=eval('torch.' + dtype)).npu()
    print(f"y_ref = {y_ref[0, 0, 0:4]}")

    y_cal = torch.randint(1, (XB, YB, ZB), dtype=eval('torch.' + dtype)).npu()
    if dtype == "float32":
        fn_npu_f32[ncore, 1, 1](y_cal, x0, XB, YB, ZB)
    elif dtype == "float16":
        fn_npu_f16[ncore, 1, 1](y_cal, x0, XB, YB, ZB)
    else:
        fn_npu_i8[ncore, 1, 1](y_cal, x0, XB, YB, ZB)
    print(f"y_cal = {y_cal[0, 0, 0:4]}")
    test_common.validate_cmp(dtype, y_cal, y_ref)
