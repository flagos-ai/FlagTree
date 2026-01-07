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
import torch
import triton.language as tl
import triton.language.extra.ascend.libdevice as libdevice
import test_common

@triton.jit
def log10_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    
    y = libdevice.log10(x)
    
    tl.store(y_ptr + offsets, y, mask=mask)

@pytest.mark.parametrize('shape', [(12,16),])
@pytest.mark.parametrize('dtype', ['float32'])
def test_log10(shape, dtype):   
    x = test_common.generate_tensor(shape, dtype).npu()
    
    x[0, 0] = 1.0      # log10(1) = 0
    x[0, 1] = 10.0     # log10(10) = 1
    x[0, 2] = 100.0    # log10(100) = 2
    x[0, 3] = 0.1      # log10(0.1) = -1
    x[0, 4] = 0.0      # log10(0) = -inf
    x[0, 5] = -1.0     # log10(-1) = NaN
    x[0, 6] = 2.0      # log10(2) ≈ 0.3010
    
    y = torch.empty_like(x)
    
    BLOCK_SIZE = 192
    grid = lambda meta: (triton.cdiv(192, meta['BLOCK_SIZE']),)
    
    log10_kernel[grid](x, y, 192, BLOCK_SIZE=BLOCK_SIZE)
    
    expected = torch.log10(x)
    print(f"triton_ret = {y}")
    print(f"triton_ret = {expected}")
    
    valid_mask = (x > 0)
    
    if torch.any(valid_mask):
        valid_y = y[valid_mask]
        valid_expected = expected[valid_mask]
        
        torch.testing.assert_close(valid_y, valid_expected, rtol=1e-3, atol=1e-3)
    
    # Check if negative values return NaN
    negative_mask = (x < 0)
    if torch.any(negative_mask):
        negative_y = y[negative_mask]
        assert torch.all(torch.isnan(negative_y)), "Negative inputs should return NaN"
    
    # Check if zero value returns -inf
    zero_mask = (x == 0)
    if torch.any(zero_mask):
        zero_y = y[zero_mask]
        assert torch.all(torch.isinf(zero_y) & (zero_y < 0)), "Zero inputs should return -inf"
    
    print("✓ LOG10 test PASSED!")