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

import pytest
import torch
import torch_npu
import triton
import triton.language as tl
import test_common


@pytest.fixture(scope="function")
def restore_npu_hf32_setting():
    original_allow_hf32 = torch_npu.npu.matmul.allow_hf32
    try:
        torch_npu.npu.matmul.allow_hf32 = True
        yield
    finally:
        torch_npu.npu.matmul.allow_hf32 = original_allow_hf32


def torch_dot_None(x0, x1):
    res = torch.matmul(x0, x1)
    return res


@triton.jit
def triton_dot_2_None(output_ptr, x_ptr, y_ptr, B: tl.constexpr, C: tl.constexpr, D: tl.constexpr):
    bidx = tl.arange(0, B)
    cidx = tl.arange(0, C)
    didx = tl.arange(0, D)

    x_mask = (bidx[:, None] < B) & (cidx[None, :] < C)
    y_mask = (cidx[:, None] < C) & (didx[None, :] < D)
    out_mask = (bidx[:, None] < B) & (didx[None, :] < D)
    Xidx = bidx[:, None] * C + cidx[None, :]
    Yidx = cidx[:, None] * D + didx[None, :]
    X = tl.load(x_ptr + Xidx, mask=x_mask, other=0.0)
    Y = tl.load(y_ptr + Yidx, mask=y_mask, other=0.0)
    ret = tl.dot(X, Y, input_precision="hf32")
    oidx = bidx[:, None] * D + didx[None, :]
    tl.store(output_ptr + oidx, ret, mask=out_mask)


@triton.jit
def triton_dot_2_allow_tf32(output_ptr, x_ptr, y_ptr, B: tl.constexpr, C: tl.constexpr, D: tl.constexpr):
    bidx = tl.arange(0, B)
    cidx = tl.arange(0, C)
    didx = tl.arange(0, D)

    x_mask = (bidx[:, None] < B) & (cidx[None, :] < C)
    y_mask = (cidx[:, None] < C) & (didx[None, :] < D)
    out_mask = (bidx[:, None] < B) & (didx[None, :] < D)
    Xidx = bidx[:, None] * C + cidx[None, :]
    Yidx = cidx[:, None] * D + didx[None, :]
    X = tl.load(x_ptr + Xidx, mask=x_mask, other=0.0)
    Y = tl.load(y_ptr + Yidx, mask=y_mask, other=0.0)
    ret = tl.dot(X, Y, allow_tf32=True)
    oidx = bidx[:, None] * D + didx[None, :]
    tl.store(output_ptr + oidx, ret, mask=out_mask)


@triton.jit
def triton_dot_2_input_tf32(output_ptr, x_ptr, y_ptr, B: tl.constexpr, C: tl.constexpr, D: tl.constexpr):
    bidx = tl.arange(0, B)
    cidx = tl.arange(0, C)
    didx = tl.arange(0, D)

    x_mask = (bidx[:, None] < B) & (cidx[None, :] < C)
    y_mask = (cidx[:, None] < C) & (didx[None, :] < D)
    out_mask = (bidx[:, None] < B) & (didx[None, :] < D)
    Xidx = bidx[:, None] * C + cidx[None, :]
    Yidx = cidx[:, None] * D + didx[None, :]
    X = tl.load(x_ptr + Xidx, mask=x_mask, other=0.0)
    Y = tl.load(y_ptr + Yidx, mask=y_mask, other=0.0)
    ret = tl.dot(X, Y, input_precision="tf32")
    oidx = bidx[:, None] * D + didx[None, :]
    tl.store(output_ptr + oidx, ret, mask=out_mask)


@triton.jit
def triton_dot_2_ignore_tf32(output_ptr, x_ptr, y_ptr, B: tl.constexpr, C: tl.constexpr, D: tl.constexpr):
    bidx = tl.arange(0, B)
    cidx = tl.arange(0, C)
    didx = tl.arange(0, D)

    x_mask = (bidx[:, None] < B) & (cidx[None, :] < C)
    y_mask = (cidx[:, None] < C) & (didx[None, :] < D)
    out_mask = (bidx[:, None] < B) & (didx[None, :] < D)
    Xidx = bidx[:, None] * C + cidx[None, :]
    Yidx = cidx[:, None] * D + didx[None, :]
    X = tl.load(x_ptr + Xidx, mask=x_mask, other=0.0)
    Y = tl.load(y_ptr + Yidx, mask=y_mask, other=0.0)
    ret = tl.dot(X, Y, input_precision="hf32")
    oidx = bidx[:, None] * D + didx[None, :]
    tl.store(output_ptr + oidx, ret, mask=out_mask)


testlist1 = [
    (10, 13, 35, 39),
]

testlist2 = [(16, 32, 16)]

typelist = [
    'float32',
]


@pytest.mark.skip(reason="not supported after the NPUIR is updated in April, and will be fixed later")
@pytest.mark.parametrize("B, C, D", testlist2)
@pytest.mark.parametrize("sigtype", typelist)
def test_dot_2(restore_npu_hf32_setting, sigtype, B, C, D):
    x = test_common.generate_tensor((B, C), sigtype).npu()
    y = test_common.generate_tensor((C, D), sigtype).npu()
    z_ref = torch_dot_None(x, y).to(torch.float32)
    z = torch.zeros((B, D), dtype=torch.float32).npu()
    triton_dot_2_None[1, 1, 1](z, x, y, B, C, D)
    test_common.validate_cmp(sigtype, z, z_ref)


@pytest.mark.xfail(
    reason="Temporarily disabled: TA backend does not support allow_tf32 yet. Will be fixed in follow-up.")
@pytest.mark.parametrize("B, C, D", testlist2)
@pytest.mark.parametrize("sigtype", typelist)
def test_dot_2_allow_tf32(restore_npu_hf32_setting, sigtype, B, C, D):
    x = test_common.generate_tensor((B, C), sigtype).npu()
    y = test_common.generate_tensor((C, D), sigtype).npu()
    z_ref = torch_dot_None(x, y).to(torch.float32)
    z = torch.zeros((B, D), dtype=torch.float32).npu()
    triton_dot_2_allow_tf32[1, 1, 1](z, x, y, B, C, D)
    test_common.validate_cmp(sigtype, z, z_ref)


@pytest.mark.skip(reason="not supported after the NPUIR is updated in April, and will be fixed later")
@pytest.mark.parametrize("B, C, D", testlist2)
@pytest.mark.parametrize("sigtype", typelist)
def test_dot_2_input_tf32(restore_npu_hf32_setting, sigtype, B, C, D):
    x = test_common.generate_tensor((B, C), sigtype).npu()
    y = test_common.generate_tensor((C, D), sigtype).npu()
    z_ref = torch_dot_None(x, y).to(torch.float32)
    z = torch.zeros((B, D), dtype=torch.float32).npu()
    triton_dot_2_input_tf32[1, 1, 1](z, x, y, B, C, D)
    test_common.validate_cmp(sigtype, z, z_ref)


@pytest.mark.parametrize("B, C, D", testlist2)
@pytest.mark.parametrize("sigtype", typelist)
def test_dot_2_ignore_tf32(sigtype, B, C, D):
    input_type = "bfloat16"
    x = test_common.generate_tensor((B, C), input_type).npu()
    y = test_common.generate_tensor((C, D), input_type).npu()
    z = torch.zeros((B, D), dtype=torch.float32).npu()

    original_allow_hf32 = torch_npu.npu.matmul.allow_hf32
    try:
        torch_npu.npu.matmul.allow_hf32 = False
        z_ref = torch_dot_None(x.to(torch.float32), y.to(torch.float32)).to(torch.float32)

    finally:
        torch_npu.npu.matmul.allow_hf32 = original_allow_hf32

    triton_dot_2_ignore_tf32[1, 1, 1](z, x, y, B, C, D)
    test_common.validate_cmp(sigtype, z, z_ref)
