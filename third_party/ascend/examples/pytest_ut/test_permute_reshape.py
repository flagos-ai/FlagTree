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


@triton.jit
def triton_foo(in_ptr0, in_ptr1, in_ptr2, out_ptr0, BLOCK1: tl.constexpr, BLOCK1_SUB: tl.constexpr,
               BLOCK2: tl.constexpr,  S: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
                ):
    offset: tl.constexpr = tl.program_id(0) * BLOCK1
    base1 = tl.arange(0, BLOCK1_SUB)
    base2 = tl.arange(0, BLOCK2)
    loops1: tl.constexpr = BLOCK1 // BLOCK1_SUB
    loops2: tl.constexpr = D // BLOCK2

    for loop1 in range(loops1):
        off1 = loop1
        s = offset + (off1 * BLOCK1_SUB) + base1[:, None]
        for n in range(N) :
            for loop2 in range(loops2):
                d = loop2 * BLOCK2 + base2[None, :]
                tmp0 = tl.load(in_ptr0 + ((32768*n) + (8*s) + d), None)
                tmp1 = tl.load(in_ptr1 + ((32768*n) + (8*s) + d), None)
                tmp2 = tmp0 + tmp1

                tmp3 = tl.load(in_ptr2 + ((8*n) + d + (96*s)), None)
                tmp9 =  tmp3 + tmp2
                tl.store(out_ptr0 +  ((8*n) + d + (96*s)), tmp9,  None)

def foo_triton_wrapper(a, b, c):
    NBLOCKS = 32 if a.shape[2] >=256 else 1
    BLOCK1 = a.shape[2] // NBLOCKS
    BLOCK1_SUB = BLOCK1 if BLOCK1 < 64 else 64
    BLOCK2 = a.shape[3] if a.shape[3] < 64 else 64

    value = torch.empty_strided((c.shape[0], c.shape[1], c.shape[2]),
                                (c.stride()[0], c.stride()[1], c.stride()[2]), dtype=torch.float32).npu()
    triton_foo[NBLOCKS, 1, 1](a, b, c, value, BLOCK1, BLOCK1_SUB, BLOCK2, a.shape[2], a.shape[1], a.shape[3])

    return value

def foo(a, b, c):
    B, N, S, D = (1, 12, 4096, 8)
    y = a + b
    y = c + y.permute(2, 0, 1, 3 ).reshape(S,B, N*D)
    return y

def test_permute_reshape():
    B, N, S, D = (1, 12, 4096, 8)
    a = torch.randn((B, N, S, D), dtype=torch.float32).npu()
    b = torch.randn((B, N, S, D), dtype=torch.float32).npu()
    c = torch.randn((S, B, N*D), dtype=torch.float32).npu()
    r = foo_triton_wrapper(a, b, c)
    r1 = foo(a, b, c)
    print(r[0:8, 0, 0:8])
    print(r1[0:8,0, 0:8])
    torch.testing.assert_close(r, r1, rtol=1e-3, atol=1e-3)