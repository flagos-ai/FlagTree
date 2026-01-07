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

import torch
import torch_npu

import triton
import triton.language as tl

import pytest

@triton.jit
def triton_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr, SLICE_OFFSET: tl.constexpr, SLICE_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    x_sub = tl.extract_slice(x, [block_start+SLICE_OFFSET], [SLICE_SIZE], [1])
    y_sub = tl.extract_slice(y, [block_start+SLICE_OFFSET], [SLICE_SIZE], [1])
    output_sub = x_sub + y_sub
    output = tl.load(output_ptr + offsets, mask=mask)
    output = tl.insert_slice(output, output_sub, [block_start+SLICE_OFFSET], [SLICE_SIZE], [1])
    tl.store(output_ptr + offsets, output, mask=mask)

def triton_func(x: torch.Tensor, y: torch.Tensor, slice_offset: int, slice_size: int):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    triton_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024, SLICE_OFFSET=0, SLICE_SIZE=32)
    return output

def test_extract_slice():
    size = 1024
    slice_offset = 0
    slice_size = 32
    x = torch.rand(size, device='npu')
    y = torch.rand(size, device='npu')
    torch_ref = x + y
    triton_cal = triton_func(x, y, slice_offset, slice_size)
    torch.testing.assert_close(triton_cal[slice_offset:slice_offset+slice_size],
                               torch_ref[slice_offset:slice_offset+slice_size])
