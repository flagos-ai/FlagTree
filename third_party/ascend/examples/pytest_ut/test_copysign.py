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
def copysign_kernel(x_ptr, y_ptr, z_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    z = libdevice.copysign(x, y)
    
    tl.store(z_ptr + offsets, z, mask=mask)

@pytest.mark.parametrize('shape', [(12,16),])
@pytest.mark.parametrize('dtype', ['float32'])
def test_copysign(shape, dtype):   
    n_elements = shape[0] * shape[1]
    x = test_common.generate_tensor(shape, dtype).npu()
    y = test_common.generate_tensor(shape, dtype).npu()
    
    # Ensure to include some boundary cases
    x[0, 0] = 3.14
    y[0, 0] = 1.0 # The result should be 3.14

    x[0, 1] = 3.14
    y[0, 1] = -1.0 # The result should be -3.14

    x[0, 2] = -3.14
    y[0, 2] = 1.0 # The result should be 3.14

    x[0, 3] = -3.14
    y[0, 3] = -1.0 # The result should be -3.14

    x[0, 4] = 0.0
    y[0, 4] = -1.0 # The result should be -0.0

    x[0, 5] = 0.0
    y[0, 5] = 1.0 # The result should be 0.0
    
    x[0, 6] = 3.14
    y[0, 6] = 0.0   # The result should be 3.14

    x[0, 7] = 3.14
    y[0, 7] = -0.0  # The result should be -3.14

    x[0, 8] = -3.14
    y[0, 8] = 0.0   # The result should be 3.14

    x[0, 9] = -3.14
    y[0, 9] = -0.0  # The result should be -3.14

    x[0, 10] = 0.0
    y[0, 10] = 0.0   # The result should be 0.0

    x[0, 11] = 0.0
    y[0, 11] = -0.0  # The result should be -0.0

    x[0, 12] = -0.0
    y[0, 12] = 0.0   # The result should be 0.0

    x[0, 13] = -0.0
    y[0, 13] = -0.0  # The result should be -0.0
    
    z = torch.empty_like(x)
    
    BLOCK_SIZE = 192
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    copysign_kernel[grid](x, y, z, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    expected = torch.copysign(x, y)
    
    torch.testing.assert_close(z, expected, rtol=1e-3, atol=1e-3)
    
    print("✓ COPYSIGN test PASSED!")