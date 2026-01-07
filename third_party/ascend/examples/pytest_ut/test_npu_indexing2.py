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
import time


def foo(a, b, c):
    y = a + b + c
    y = y.sum(dim = 1)
    return y

@triton.jit
def triton_codegen2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, XBLOCK : tl.constexpr, XBLOCK_SUB : tl.constexpr, RBLOCK : tl.constexpr):
    ynumel = 8
    rnumel = 2048
    xnumel = 1024
    offset = tl.program_id(0) * XBLOCK
    base1 = tl.arange(0, XBLOCK_SUB)
    loops1: tl.constexpr = XBLOCK // XBLOCK_SUB
    base2 = tl.arange(0, RBLOCK)
    loops2: tl.constexpr = rnumel // RBLOCK
    for y in range(ynumel):
        y0 = y
        for loop1 in range(loops1):
            x = offset + (loop1 * XBLOCK_SUB) + base1
            x1 = offset + (loop1 * XBLOCK_SUB) + base1[None, :]
            _tmp6 = tl.full([XBLOCK_SUB, RBLOCK], 0, tl.float32)
            for loop2 in range(loops2):
                r2 = loop2 * RBLOCK + base2[:, None]
                tmp0 = tl.load(in_ptr0 + (x1 + (1024*r2) + (2097152*y0)), None, eviction_policy='evict_last')
                tmp1 = tl.load(in_ptr1 + (x1 + (1024*r2) + (2097152*y0)), None, eviction_policy='evict_last')
                tmp3 = tl.load(in_ptr2 + (x1 + (1024*r2) + (2097152*y0)), None, eviction_policy='evict_last')
                tmp2 = tmp0 + tmp1
                tmp4 = tmp2 + tmp3
                tmp5 = tl.reshape(tmp4, [RBLOCK, XBLOCK_SUB])
                tmp7 = _tmp6 + tmp5
                _tmp6 = tmp7
            tmp6 = tl.sum(_tmp6, 0).reshape(XBLOCK_SUB)

            tl.store(out_ptr0 + (x + (1024*y0)), tmp6, None)

def foo_triton_wrapper(a, b, c):
    NBLOCKS = 8
    BLOCK1 = a.shape[2] // NBLOCKS
    BLOCK1_SUB = 64
    BLOCK2 = 64

    value = torch.empty_strided((c.shape[0], c.shape[2]),
                                ( c.shape[2], 1), dtype=torch.float32).npu()
    
    triton_codegen2[NBLOCKS, 1, 1](a, b, c, value, BLOCK1, BLOCK1_SUB, BLOCK2)

    return value


def test_npu_indexing2():

    Y, X, R = ( 8, 2048, 1024)
    a = torch.randn(( Y, X, R), dtype=torch.float32).npu()
    b = torch.randn(( Y, X, R), dtype=torch.float32).npu()
    c = torch.randn(( Y, X, R), dtype=torch.float32).npu()
    r = foo_triton_wrapper(a, b, c)
    r1 = foo(a, b, c)
    print(r[ 0:8, 0:8, ])
    print(r1[ 0:8, 0:8])
    torch.testing.assert_close(r, r1, rtol=1e-3, atol=1e-3)
