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
"""
Unit test for gather_2d_simd kernel.
"""

import torch
import torch_npu
import triton
import triton.language as tl
import pytest

from triton.language.extra.kernels import gather_2d_simd


@pytest.mark.parametrize("M,N,K", [
    (32, 128, 64),
])
def test_gather_2d_simd(M, N, K):
    """Test gather_2d_simd with various tensor sizes."""
    src = torch.randn(M, N, dtype=torch.float32, device='npu')
    indices = torch.randint(0, N, (M, K), dtype=torch.int32, device='npu')
    output = torch.empty((M, K), dtype=src.dtype, device='npu')

    grid = (triton.cdiv(M, 32), )
    gather_2d_simd[grid](src, indices, output, M, N, K, XBLOCK=32, XBLOCK_SUB=4)

    ref = torch.gather(src, 1, indices.long())
    assert torch.allclose(output, ref, rtol=1e-5, atol=1e-5)
