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
import pytest
import test_common
import torch
import torch_npu


@triton.jit
def atomic_cas(in_ptr0, in_ptr1, out_ptr0, out_ptr1, n_elements, BLOCK_SIZE: tl.constexpr):
    xoffset = tl.program_id(0) * BLOCK_SIZE
    xindex = xoffset + tl.arange(0, BLOCK_SIZE)[:]
    yindex = tl.arange(0, BLOCK_SIZE)[:]
    xmask = xindex < n_elements
    x0 = xindex
    x1 = yindex
    val = tl.load(in_ptr0 + (x0), xmask)
    cmp = tl.load(in_ptr1 + (x0), xmask)
    tmp1 = tl.atomic_cas(out_ptr0 + (x1), cmp, val)
    tl.store(out_ptr1 + (x1), tmp1, xmask)


@pytest.mark.parametrize('param_list',
                         [
                             ['int16', (8, 8), 2],
                             ['int32', (32, 32), 6],
                             ['int64', (32, 32), 2],
                             ['float32', (32, 32), 2],
                             ['float16', (64, 64), 4],
                             ['float32', (128, 128), 8],
                             ['float16', (128, 128), 16],
                         ]
                         )
def test_atomic_cas(param_list):
    dtype, shape, ncore = param_list
    block_size = shape[0] * shape[1] // ncore
    split_size = shape[0] // ncore

    import random
    cmp_val = [random.randint(0, 10) for _ in range(ncore)]
    
    cmp = torch.ones(split_size, shape[1], dtype=eval(f'torch.{dtype}')).to().npu() * cmp_val[0]
    for i in range(1, ncore):
        append = torch.ones(split_size, shape[1], dtype=eval(f'torch.{dtype}')).to().npu() * cmp_val[i]
        cmp = torch.cat([cmp, append], dim=0)

    val = torch.randint(low=0, high=10, size=shape, dtype=eval(f'torch.{dtype}')).npu()

    pointer = torch.randint(low=0, high=10, size=(split_size, shape[1]), dtype=eval(f'torch.{dtype}')).npu()
    pointer_old = torch.full_like(pointer, -10).npu()
    pointer_ref = pointer.clone()
    
    for i in range(ncore):
        val_subview = val[(i * split_size):((i + 1) * split_size)]
        pointer_ref = torch.where(pointer_ref == cmp_val[i], val_subview, pointer_ref)
    
    n_elements = shape[0] * shape[1]
    atomic_cas[ncore, 1, 1](val, cmp, pointer, pointer_old, n_elements, BLOCK_SIZE=split_size * shape[1])
    test_common.validate_cmp(dtype, pointer, pointer_ref)
