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
import triton
import triton.language as tl
from triton.runtime.libentry import libentry

from test_common import _all_dtypes_no_bool, generate_tensor, validate_cmp


def torch_func(x, dim, reverse):
    if reverse:
        x = torch.flip(x, [dim])
    res = torch.cumsum(x, dim=dim)
    if reverse:
        res = torch.flip(res, [dim])
    return res


@libentry()
@triton.jit
def triton_kernel(
        out_ptr0,
        in_ptr0,
        dim: tl.constexpr,
        reverse: tl.constexpr,
        numel_x: tl.constexpr,
        numel_r: tl.constexpr,
        XBLOCK: tl.constexpr,
        RBLOCK: tl.constexpr,
):
    tl.static_assert(
        numel_x == XBLOCK, "numel_x must be equal to XBLOCK in this kernel"
    )
    tl.static_assert(
        numel_r == RBLOCK, "numel_r must be equal to RBLOCK in this kernel"
    )
    idx_x = tl.arange(0, XBLOCK)
    idx_r = tl.arange(0, RBLOCK)
    idx = idx_x[:, None] * numel_r + idx_r[None, :]
    x = tl.load(in_ptr0 + idx)
    ret = tl.cumsum(x, axis=dim, reverse=reverse)
    tl.store(out_ptr0 + idx, ret)


def triton_func(x, dim, reverse):
    res = torch.empty_like(x)
    triton_kernel[1, 1, 1](
        res, x, dim, reverse, x.shape[0], x.shape[1], x.shape[0], x.shape[1]
    )
    return res


# dtype=int8, reverse=True not support;
not_support_dtype = {'int8', 'bool'}
support_dtypes = [dtype for dtype in _all_dtypes_no_bool if dtype not in not_support_dtype]


@pytest.mark.parametrize("dtype", support_dtypes)
@pytest.mark.parametrize("shape", [(7, 23)])
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("reverse", [False])
def test_cumsum(dtype, shape, dim, reverse):
    x0 = generate_tensor(shape=shape, dtype=dtype).npu()
    triton_cal = triton_func(x0, dim, reverse)
    torch_dtype = eval('torch.' + dtype)
    if torch_dtype == torch.float16 or torch_dtype == torch.float32:
        x0 = x0.to(torch.float32)
    torch_ref = torch_func(x0, dim, reverse).to(torch_dtype)
    validate_cmp(dtype, torch_ref, triton_cal)
