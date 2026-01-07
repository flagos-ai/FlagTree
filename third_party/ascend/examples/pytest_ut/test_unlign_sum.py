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
import triton
import triton.language as tl

# [128,65] -> [128,128]
# [128,128] -> [128,65] mask 作用
# [128,65] -> [128,64]   [128,1] mask 作用

@triton.jit
def triton_unlign(in_ptr0, out_ptr0, x0_numel, r1_numel, XBLOCK : tl.constexpr, XBLOCK_SUB : tl.constexpr, RBLOCK : tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base1 = tl.arange(0, XBLOCK_SUB)
    loops1: tl.constexpr = (XBLOCK + XBLOCK_SUB - 1) // XBLOCK_SUB
    base2 = tl.arange(0, RBLOCK)
    loops2: tl.constexpr = (r1_numel + RBLOCK - 1)  // RBLOCK
    for loop1 in range(loops1):
        x = offset + (loop1 * XBLOCK_SUB) + base1
        x0 = offset + (loop1 * XBLOCK_SUB) + base1[:, None]
        xmask = x0 < x0_numel
        _tmp2 = tl.full([XBLOCK_SUB, RBLOCK], 0, tl.float32)
        for loop2 in range(loops2):
            r1_prime = loop2 * RBLOCK + base2[:, None]
            r1 = loop2 * RBLOCK + base2[None, :]
            rmask = r1 < r1_numel
            tmp0 = tl.load(in_ptr0 + (r1 + (65*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
            tmp1 = tl.reshape(tmp0, [XBLOCK_SUB, RBLOCK])
            tmp3 = _tmp2 + tmp1
            _tmp2 = tmp3
        tmp2 = tl.sum(_tmp2, 1).reshape(XBLOCK_SUB,1)
        tl.store(out_ptr0 + (x0), tmp2, xmask)

def test_cases():
    size = (128, 65)
    b = weights = torch.randn((size), dtype=torch.float32).npu()
    c = torch.sum(b, dim=1)
    ret = torch.randn((size[0]), device='npu', dtype=torch.float32).npu().reshape(size[0])
    triton_unlign[1, 1, 1](b, ret, size[0], size[1], size[0], size[0], 32)
    assert torch.allclose(c, ret, rtol=1e-03, atol=1e-03, equal_nan=True)

