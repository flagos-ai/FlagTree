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

import triton
import triton.language as tl
import torch
import torch_npu
import pytest
import test_common


@triton.jit
def kernel_where_blocked_iter_arg(
    X_ptr,
    Other_ptr,
    Out_ptr,
    N,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    range_val = offs

    for start in range(0, N, BLOCK):
        x = tl.load(X_ptr + range_val)
        other = tl.load(Other_ptr + range_val)
        mask = range_val < N
        res = tl.where(mask, x, other)
        tl.store(Out_ptr + range_val, res)
        range_val = range_val + BLOCK


def torch_where_blocked_iter_arg(X, Other, N, BLOCK):
    Out = torch.zeros_like(X)
    range_val = 0
    for start in range(0, N, BLOCK):
        end = min(range_val + BLOCK, N)
        Out[range_val:end] = X[range_val:end]
        if end < range_val + BLOCK:
            Out[end:range_val + BLOCK] = Other[end:range_val + BLOCK]
        range_val = range_val + BLOCK
    return Out


@pytest.mark.parametrize('param_list', [
    ['float32', 300, 64],
    ['float32', 50, 64],
    ['float32', 192, 64],
])
def test_select_analysis_where_iterarg(param_list):
    dtype, N, BLOCK = param_list
    x = torch.rand(N + BLOCK, dtype=eval('torch.' + dtype), device='npu')
    other = torch.rand(N + BLOCK, dtype=eval('torch.' + dtype), device='npu')
    out = torch.zeros(N + BLOCK, dtype=eval('torch.' + dtype), device='npu')
    ref = torch_where_blocked_iter_arg(x, other, N, BLOCK)

    kernel_where_blocked_iter_arg[(1, )](x, other, out, N, BLOCK)

    test_common.validate_cmp(dtype, out, ref)


@triton.jit
def kernel_where_blocked_iter_arg_neg_step(
    X_ptr,
    Other_ptr,
    Out_ptr,
    N,
    LB: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    range_val = offs

    for start in range(LB, N, -BLOCK):
        x = tl.load(X_ptr + range_val)
        other = tl.load(Other_ptr + range_val)
        mask = range_val < N
        res = tl.where(mask, x, other)
        tl.store(Out_ptr + range_val, res)
        range_val = range_val + BLOCK


def torch_where_blocked_iter_arg_neg_step(X, Other, N, LB, BLOCK):
    Out = torch.zeros_like(X)
    range_val = 0
    for start in range(LB, N, -BLOCK):
        end = min(range_val + BLOCK, N)
        Out[range_val:end] = X[range_val:end]
        if end < range_val + BLOCK:
            Out[end:range_val + BLOCK] = Other[end:range_val + BLOCK]
        range_val = range_val + BLOCK
    return Out


@pytest.mark.parametrize('param_list', [
    ['float32', 140, 320, 64],
    ['float32', 192, 320, 64],
])
def test_select_analysis_where_iterarg_neg_step(param_list):
    dtype, N, LB, BLOCK = param_list
    x = torch.rand(N + BLOCK, dtype=eval('torch.' + dtype), device='npu')
    other = torch.rand(N + BLOCK, dtype=eval('torch.' + dtype), device='npu')
    out = torch.zeros(N + BLOCK, dtype=eval('torch.' + dtype), device='npu')
    ref = torch_where_blocked_iter_arg_neg_step(x, other, N, LB, BLOCK)

    kernel_where_blocked_iter_arg_neg_step[(1, )](x, other, out, N, LB, BLOCK)

    test_common.validate_cmp(dtype, out, ref)


@triton.jit
def kernel_where_blocked_iter_arg_nonzero_lb(
    X_ptr,
    Other_ptr,
    Out_ptr,
    N,
    START: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    range_val = offs

    for start in range(START, N, BLOCK):
        x = tl.load(X_ptr + range_val)
        other = tl.load(Other_ptr + range_val)
        mask = range_val < N
        res = tl.where(mask, x, other)
        tl.store(Out_ptr + range_val, res)
        range_val = range_val + BLOCK


def torch_where_blocked_iter_arg_nonzero_lb(X, Other, N, START, BLOCK):
    Out = torch.zeros_like(X)
    range_val = 0
    for start in range(START, N, BLOCK):
        end = min(range_val + BLOCK, N)
        Out[range_val:end] = X[range_val:end]
        if end < range_val + BLOCK:
            Out[end:range_val + BLOCK] = Other[end:range_val + BLOCK]
        range_val = range_val + BLOCK
    return Out


@pytest.mark.parametrize('param_list', [
    ['float32', 300, 16, 64],
    ['float32', 256, 128, 64],
])
def test_select_analysis_where_iterarg_nonzero_lb(param_list):
    dtype, N, START, BLOCK = param_list
    x = torch.rand(N + BLOCK, dtype=eval('torch.' + dtype), device='npu')
    other = torch.rand(N + BLOCK, dtype=eval('torch.' + dtype), device='npu')
    out = torch.zeros(N + BLOCK, dtype=eval('torch.' + dtype), device='npu')
    ref = torch_where_blocked_iter_arg_nonzero_lb(x, other, N, START, BLOCK)

    kernel_where_blocked_iter_arg_nonzero_lb[(1, )](x, other, out, N, START, BLOCK)

    test_common.validate_cmp(dtype, out, ref)
