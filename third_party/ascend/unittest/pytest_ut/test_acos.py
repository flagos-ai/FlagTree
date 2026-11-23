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

import triton
import triton.language as tl
import triton.language.extra.cann.libdevice as libdevice
import test_common

import torch
import torch_npu


@triton.jit
def triton_acos(in_ptr0, out_ptr0, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base1 = tl.arange(0, XBLOCK_SUB)
    loops1: tl.constexpr = XBLOCK // XBLOCK_SUB
    for loop1 in range(loops1):
        x0 = offset + (loop1 * XBLOCK_SUB) + base1
        tmp0 = tl.load(in_ptr0 + (x0), None)
        tmp1 = libdevice.acos(tmp0)
        tl.store(out_ptr0 + (x0), tmp1, None)


@pytest.mark.parametrize('param_list', ['float32', 'float16', 'bfloat16'])
def test_asinh_special(param_list):
    dtype = param_list
    x0 = torch.linspace(-1.0 + 1e-6, 1.0 - 1e-6, 256, dtype=eval("torch." + dtype)).npu()

    y_ref = torch.acos(x0)
    y_cal = torch.zeros_like(y_ref)
    triton_acos[1, 1, 1](x0, y_cal, x0.shape[0], x0.shape[0])
    bf16_tolerance = 1.0 / 128
    if dtype == 'bfloat16':
        torch.testing.assert_close(y_ref, y_cal, rtol=bf16_tolerance, atol=bf16_tolerance)
    else:
        torch.testing.assert_close(y_ref, y_cal, rtol=1e-3, atol=1e-3)


@triton.jit
def acos_kernel(in_ptr0, out_ptr0, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr0 + offsets, mask=mask)
    x = x.to(tl.float32)
    y = libdevice.acos(x)
    tl.store(out_ptr0 + offsets, y, mask=mask)


@pytest.mark.acos
@pytest.mark.parametrize('shape', [(), (1, ), (1024, 1024), (20, 320, 15), (16, 128, 64, 60), (16, 7, 57, 32, 29)])
@pytest.mark.parametrize('dtype', [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_acos(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device='npu')
    ref_out = torch.acos(inp)
    res_out = torch.zeros_like(ref_out)
    n_elements = inp.numel()
    grid = (triton.cdiv(n_elements, 1024), )
    acos_kernel[grid](inp, res_out, n_elements, BLOCK_SIZE=1024)
    print("ref_out:", ref_out)
    print("res_out:", res_out)
    if dtype == torch.bfloat16:
        bf16_tolerance = 1.0 / 128
        torch.testing.assert_close(res_out, ref_out, rtol=bf16_tolerance, atol=bf16_tolerance, equal_nan=True)
    elif dtype == torch.float16:
        torch.testing.assert_close(res_out, ref_out, rtol=1e-3, atol=1e-4, equal_nan=True)
    else:
        torch.testing.assert_close(res_out, ref_out, rtol=1e-6, atol=1e-4, equal_nan=True)
