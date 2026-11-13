# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
import triton
import triton.language as tl
import torch_npu
import pytest


@triton.jit
def single_disc_mask_atomic_add_kernel(
    in_ptr,
    BLOCK_N: tl.constexpr,
):
    col_offs = tl.arange(0, BLOCK_N)
    disc_mask = (col_offs * 2) < BLOCK_N
    ptr_in = in_ptr + col_offs
    tl.atomic_add(ptr_in, 1, mask=disc_mask)


@pytest.mark.parametrize("BLOCK_N", [8])
def test_single_discrete_mask_atomic_add(BLOCK_N):
    in_tensor = torch.arange(BLOCK_N, dtype=torch.float16, device='npu')
    expected = in_tensor.clone()
    single_disc_mask_atomic_add_kernel[(1, )](in_tensor, BLOCK_N=BLOCK_N)

    half = BLOCK_N // 2
    expected[:half] += 1
    assert torch.allclose(in_tensor, expected), \
        f"Expected:\n{expected.cpu()}\nGot:\n{in_tensor.cpu()}"
