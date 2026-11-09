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
import torch
import torch_npu
from torch.testing import assert_close

import triton
import triton.language as tl
import triton.language.extra.cann.extension as al


@triton.jit
def parallel_kernel(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr):
    # Load the full [M, N] block
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    block = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :])

    SUB_M: tl.constexpr = M // 2
    for s in al.parallel(0, 2):
        sub = al.extract_slice(block, (s * SUB_M, 0), (SUB_M, N), (1, 1))
        sub = sub * 2.0
        offs_sub_m = s * SUB_M + tl.arange(0, SUB_M)
        out_ptrs = out_ptr + offs_sub_m[:, None] * N + offs_n[None, :]
        tl.store(out_ptrs, sub)


@pytest.mark.parametrize("M, N", [(128, 128)])
def test_parallel(M, N):
    x = torch.randn((M, N), device="npu", dtype=torch.float32)
    out = torch.empty((M, N), device="npu", dtype=torch.float32)
    h = parallel_kernel[(1, )](x, out, M=M, N=N)
    torch.npu.synchronize()
    assert_close(out, x * 2, rtol=1e-3, atol=1e-3)
    code_str = h.asm["ttadapter"]
    count = code_str.count("hivm.parallel_loop")
    assert count == 1


if __name__ == "__main__":
    test_parallel(128, 128)
    print("test_parallel PASSED!")
