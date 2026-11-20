# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

import pytest

import triton
import triton.language as tl
import test_common

import torch
import torch_npu


@triton.jit
def kernel_load_diff_axis_broadcast(
    X_ptr,
    Out_ptr,
    M,
    XBLOCK: tl.constexpr,
    YBLOCK: tl.constexpr,
):
    offsetx = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)  # [XBLOCK]
    offsety = tl.program_id(1) * YBLOCK + tl.arange(0, YBLOCK)  # [YBLOCK]

    ptr = X_ptr + offsety[None, :]
    mask = (offsetx < M)[:, None]

    val = tl.load(ptr, mask, other=0.0)
    offsets = offsetx[:, None] * YBLOCK + offsety[None, :]
    tl.store(Out_ptr + offsets, val)


def torch_load_diff_axis_broadcast(X, M, XBLOCK, YBLOCK):
    out = torch.zeros((XBLOCK, YBLOCK), dtype=X.dtype)
    if M > 0:
        out[:min(M, XBLOCK), :] = X[0, :]
    return out


@pytest.mark.parametrize('param_list', [
    ['float32', 140, 256, 64],
    ['float32', 1, 256, 64],
    ['float32', 256, 256, 64],
])
def test_load_diff_axis_broadcast_other(param_list):
    dtype, M, XBLOCK, YBLOCK = param_list
    x = test_common.generate_tensor((1, YBLOCK), dtype).npu()
    out = torch.zeros((XBLOCK, YBLOCK), dtype=eval('torch.' + dtype)).npu()
    ref = torch_load_diff_axis_broadcast(x.cpu(), M, XBLOCK, YBLOCK)

    kernel_load_diff_axis_broadcast[(1, 1)](x, out, M, XBLOCK, YBLOCK)

    test_common.validate_cmp(dtype, out, ref)


@triton.jit
def kernel_load_diff_axis_broadcast_no_other(
    X_ptr,
    Out_ptr,
    M,
    XBLOCK: tl.constexpr,
    YBLOCK: tl.constexpr,
):
    offsetx = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)  # [XBLOCK]
    offsety = tl.program_id(1) * YBLOCK + tl.arange(0, YBLOCK)  # [YBLOCK]

    ptr = X_ptr + offsety[None, :]
    mask = (offsetx < M)[:, None]

    val = tl.load(ptr, mask)
    offsets = offsetx[:, None] * YBLOCK + offsety[None, :]
    tl.store(Out_ptr + offsets, val)


@pytest.mark.parametrize('param_list', [
    ['float32', 140, 256, 64],
    ['float32', 256, 256, 64],
])
def test_load_diff_axis_broadcast_no_other(param_list):
    dtype, M, XBLOCK, YBLOCK = param_list
    x = test_common.generate_tensor((1, YBLOCK), dtype).npu()
    out = torch.zeros((XBLOCK, YBLOCK), dtype=eval('torch.' + dtype)).npu()
    ref = torch_load_diff_axis_broadcast(x.cpu(), M, XBLOCK, YBLOCK)

    kernel_load_diff_axis_broadcast_no_other[(1, 1)](x, out, M, XBLOCK, YBLOCK)

    test_common.validate_cmp(dtype, out, ref)


@triton.jit
def kernel_load_same_axis_broadcast(
    X_ptr,
    Out_ptr,
    YM,
    XBLOCK: tl.constexpr,
    YBLOCK: tl.constexpr,
):
    offsetx = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)  # [XBLOCK]
    offsety = tl.program_id(1) * YBLOCK + tl.arange(0, YBLOCK)  # [YBLOCK]

    ptr = X_ptr + offsety[None, :]
    mask = (offsety < YM)[None, :]

    val = tl.load(ptr, mask, other=0.0)
    offsets = offsetx[:, None] * YBLOCK + offsety[None, :]
    tl.store(Out_ptr + offsets, val)


def torch_load_same_axis_broadcast(X, YM, XBLOCK, YBLOCK):
    out = torch.zeros((XBLOCK, YBLOCK), dtype=X.dtype)
    out[:, :min(YM, YBLOCK)] = X[0, :min(YM, YBLOCK)]
    return out


@pytest.mark.parametrize('param_list', [
    ['float32', 40, 256, 64],
    ['float32', 64, 256, 64],
])
def test_load_same_axis_broadcast(param_list):
    dtype, YM, XBLOCK, YBLOCK = param_list
    x = test_common.generate_tensor((1, YBLOCK), dtype).npu()
    out = torch.zeros((XBLOCK, YBLOCK), dtype=eval('torch.' + dtype)).npu()
    ref = torch_load_same_axis_broadcast(x.cpu(), YM, XBLOCK, YBLOCK)

    kernel_load_same_axis_broadcast[(1, 1)](x, out, YM, XBLOCK, YBLOCK)

    test_common.validate_cmp(dtype, out, ref)
