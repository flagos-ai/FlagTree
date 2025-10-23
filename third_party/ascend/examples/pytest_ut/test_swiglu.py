# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
