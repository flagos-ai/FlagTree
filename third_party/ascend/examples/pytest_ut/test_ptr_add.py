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


def ptr_add(device):

    @triton.jit
    def wrap_stacked(a_ptr, c_ptr,
                     M: tl.constexpr,
                     N: tl.constexpr,
                     stride_am: tl.constexpr,
                     stride_an: tl.constexpr,
                     stride_cm: tl.constexpr,
                     stride_cn: tl.constexpr,
                     BLOCK_SIZE_K: tl.constexpr):
        offs_am = (2 + tl.arange(0, 4)) % M
        offs_an = tl.arange(0, 4)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am +
                          offs_an[None, :] * stride_an)

        offs_cm = tl.arange(0, 4)
        offs_cn = tl.arange(0, 4)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[
            None, :]
        for k in range(0, 2):
            a = tl.load(a_ptrs)
            tl.store(c_ptrs, a)
            a_ptrs += BLOCK_SIZE_K * stride_an
            c_ptrs += BLOCK_SIZE_K * stride_an

    M = 4
    N = 8
    AA = torch.arange(0, M * N, dtype=torch.float32).npu()
    A = AA.reshape((M, N))
    out = torch.full((M, N), 88888,  dtype=torch.float32).npu()
    grid = lambda meta: (1, )

    wrap_stacked[grid](A,
                       out,
                       M,
                       N,
                       A.stride(0),
                       A.stride(1),
                       out.stride(0),
                       out.stride(1),
                       BLOCK_SIZE_K=4)

    # Expected output copied from running triton on NPU
    expected_out = torch.tensor(
        [[16, 17, 18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29, 30, 31],
         [0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]],
        ).npu()

    assert torch.equal(expected_out.int(), out.int())


def test_ptr_add():
    device = 'npu'
    ptr_add(device)