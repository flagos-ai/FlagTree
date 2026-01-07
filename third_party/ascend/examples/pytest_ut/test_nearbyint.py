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
def nearbyint_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    
    y = libdevice.nearbyint(x)
    
    tl.store(y_ptr + offsets, y, mask=mask)


@pytest.mark.parametrize('shape', [(12,16),])
@pytest.mark.parametrize('dtype', ['float32'])
def test_nearbyint(shape, dtype):   
    n_elements = shape[0] * shape[1]
    x = test_common.generate_tensor(shape, dtype).npu()
    
    # Ensure some boundary cases are included
    x[0, 0] = 0.0
    x[0, 1] = 3.14 # Should be rounded to 3.0
    x[0, 2] = -2.71 # Should be rounded to -3.0
    x[0, 3] = 5.0 # Integer, should remain unchanged
    x[0, 4] = -3.0 # Negative integer, should remain unchanged
    x[0, 5] = 2.5 # Should be rounded to 3.0
    x[0, 6] = -1.5 # Should be rounded to -2.0
    
    y = torch.empty_like(x)
    
    BLOCK_SIZE = 192
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    nearbyint_kernel[grid](x, y, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    expected = torch.round(x)
    test_common.validate_cmp('float32', y, expected)