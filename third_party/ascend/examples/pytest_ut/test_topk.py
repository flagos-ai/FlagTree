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


import logging
import pytest

import triton
import triton.language as tl

import torch
import torch_npu
import test_common


@pytest.mark.interpreter
@pytest.mark.parametrize("shape", [
    (16, 64),
    (16, 8, 8),
    (2, 8, 8, 8),
])
@pytest.mark.parametrize("k", [4])
@pytest.mark.parametrize("dtype", ["float32"])
def test_topk_nd(shape, k, dtype):

    numel = 1
    for d in shape:
        numel *= d
    base = torch.arange(numel, dtype=torch.float32).view(shape) / 10.0
    if dtype == "float16":
        base = base.half()
    elif dtype == "int32":
        base = torch.arange(numel, dtype=torch.int32).view(shape)
    x = base.npu()

    y = torch.topk(x.cpu(), k=k, dim=-1).values

    M = int(torch.tensor(shape[:-1]).prod().item()) if len(shape) > 1 else 1
    N = shape[-1]

    x_2d = x.view(M, N)
    z = torch.empty((M, k), dtype=x_2d.dtype, device=x_2d.device)

    @triton.jit
    def topk_kernel_nd(X, stride_xm, Z, stride_zm, M: tl.constexpr, N: tl.constexpr, k: tl.constexpr):
        offs_m = tl.arange(0, M)
        offs_x_n = tl.arange(0, N)
        offs_z_n = tl.arange(0, k)
        offs_x = offs_m[:, None] * stride_xm + offs_x_n[None, :]
        x_val = tl.load(X + offs_x)
        z_val = tl.topk(x_val, k)
        offs_z = offs_m[:, None] * stride_zm + offs_z_n[None, :]
        tl.store(Z + offs_z, z_val)

    topk_kernel_nd[(1,)](x_2d, x_2d.stride(0), z, z.stride(0), M, N, k, num_warps=8)

    z_view = z.view(*shape[:-1], k) if len(shape) > 1 else z.view(k)
    test_common.validate_cmp(dtype, z_view.cpu(), y)
