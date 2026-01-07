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
# from .utils import calculate_settings
def standard_binary(e,g):
    ee = e.to(torch.float32)
    f = ee* torch.sigmoid(ee)
    h = (f * g).to(g.dtype)
    return h
@triton.jit
def _fg_kernel(e, g, h, n_elements, BLOCK_SIZE : tl.constexpr,):
    block_idx = tl.program_id(0)
    offsets = block_idx*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    e_row = tl.load(e + offsets, mask = mask, other = 0).to(tl.float32)
    g_row = tl.load(g + offsets, mask = mask, other = 0)#.to(tl.float32)

    # f = e * sigmoid(e)
    f_row = e_row * tl.sigmoid(e_row) # e_row / (1 + tl.exp(-e_row))
    # f_row = f_row.to(g_row.dtype) # bf16 should always cast to fp32 when calculating
    # h = f * g
    h_row = (f_row * g_row).to(g_row.dtype)
    # Store h
    tl.store(h + offsets, h_row, mask = mask)
pass

def swiglu_fg_kernel(e, g):
    batch, seq_len, hd = e.shape
    n_elements = e.numel()
    h = torch.empty((batch, seq_len, hd), dtype = e.dtype, device = "npu")
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    kk=_fg_kernel[grid](e, g, h, n_elements, BLOCK_SIZE = 1024,)
    print(kk.asm['ttir'])
    return h
pass

@pytest.mark.parametrize('param_list',
                         [
                            ['float32', (2, 128, 128)],
                            ['float16', (2, 128, 128)],
                            ['bfloat16', (2, 128, 128)],
                         ]
                         )

def test_case(param_list):
    dtype, size = param_list
    torch.manual_seed(0)
    x = torch.rand(size, device='npu', dtype = eval('torch.' + dtype))
    y = torch.rand(size, device='npu', dtype = eval('torch.' + dtype))
    std_ret = standard_binary(x,y)
    print(f"std_ret= {std_ret}")
    ret = swiglu_fg_kernel(x,y)
    print(f"ret= {ret}")
    test_common.validate_cmp(dtype,std_ret,ret)
pass
