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
import pytest
import numpy as np
import torch
import triton.language as tl
import test_common

# ---------------
# test sort op
# ---------------


@triton.jit
def sort_kernel_2d(X, Z, N: tl.constexpr, M: tl.constexpr, descending: tl.constexpr):
    offx = tl.arange(0, M)
    offy = tl.arange(0, N) * M
    off2d = offx[None, :] + offy[:, None]
    x = tl.load(X + off2d)
    x = tl.sort(x, descending=descending, dim=1)
    tl.store(Z + off2d, x)


@pytest.mark.interpreter
@pytest.mark.parametrize("shape", [(256, 16)])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("dtype", ['int8', 'int16', 'float16', 'float32', 'bfloat16', 'bool'])
def test_sort_2d(shape, descending, dtype):
    if dtype == "bool":
        x = test_common.generate_tensor(shape, dtype)
        np_sorted = np.sort(x)
        if descending:
            np_sorted = np_sorted[:, ::-1].copy()
        torch_res = torch.from_numpy(np_sorted).npu()
    else:
        x = test_common.generate_tensor(shape, dtype).npu()
        torch_res = torch.sort(x, descending=descending)[0]

    x = x.npu()
    triton_res = torch.zeros(shape, dtype=eval('torch.' + dtype)).npu()
    N = x.shape[0]
    M = x.shape[1]
    sort_kernel_2d[(1, )](x, triton_res, N, M, descending)
    assert (torch_res == triton_res).all(), (torch_res, triton_res)


@triton.jit
def sort_kernel_3d(X, Z, D0: tl.constexpr, D1: tl.constexpr, D2: tl.constexpr, descending: tl.constexpr):
    off2 = tl.arange(0, D2)
    off1 = tl.arange(0, D1) * D2
    off0 = tl.arange(0, D0) * D1 * D2

    off = off2[None, None, :] + off1[None, :, None] + off0[:, None, None]
    x = tl.load(X + off)

    x = tl.sort(x, descending=descending, dim=2)

    tl.store(Z + off, x)


@pytest.mark.interpreter
@pytest.mark.parametrize("shape", [(8, 4, 16)])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("dtype", ['int8', 'int16', 'float16', 'float32', 'bfloat16', 'bool'])
def test_sort_3d(shape, descending, dtype):
    if dtype == "bool":
        x = test_common.generate_tensor(shape, dtype)
        np_sorted = np.sort(x)
        if descending:
            np_sorted = np_sorted[:, :, ::-1].copy()
        torch_res = torch.from_numpy(np_sorted).npu()
    else:
        x = test_common.generate_tensor(shape, dtype).npu()
        torch_res = torch.sort(x, descending=descending)[0]

    x = x.npu()
    triton_res = torch.zeros(shape, dtype=eval('torch.' + dtype)).npu()
    D0 = x.shape[0]
    D1 = x.shape[1]
    D2 = x.shape[2]
    sort_kernel_3d[(1, )](x, triton_res, D0, D1, D2, descending)
    assert (torch_res == triton_res).all(), (torch_res, triton_res)