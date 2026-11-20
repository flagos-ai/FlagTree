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

import torch
import torch_npu
import pytest
import test_common


@triton.jit
def fn_npu_(output_ptr, x_ptr, output_ptr1, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    xidx = tl.arange(0, XB)
    yidx = tl.arange(0, YB)
    zidx = tl.arange(0, ZB)

    idx = xidx[:, None, None] * YB * ZB + yidx[None, :, None] * ZB + zidx[None, None, :]

    X = tl.load(x_ptr + idx)

    xx, yy = tl.split(X)

    oidx = xidx[:, None] * YB + yidx[None, :]

    tl.store(output_ptr + oidx, xx)
    tl.store(output_ptr1 + oidx, yy)


@pytest.mark.parametrize('para_type,data_type,XB,YB,ZB', [
    ['float32', torch.float32, 8, 8, 2],
    ['float16', torch.float16, 8, 8, 2],
    ['int8', torch.int8, 8, 128, 2],
    ['int8', torch.int8, 8, 8, 2],
])
def test_split(para_type, data_type, XB, YB, ZB):

    x = torch.randint(low=-128, high=128, size=(XB, YB, ZB), dtype=data_type).npu()

    a, b = torch.split(x, 1, dim=-1)
    a = a.reshape(XB, YB)
    b = b.reshape(XB, YB)
    print(a)
    print(b)

    output = torch.randint(1, (XB, YB), dtype=data_type).npu()
    output1 = torch.randint(1, (XB, YB), dtype=data_type).npu()
    fn_npu_[1, 1, 1](output, x, output1, XB, YB, ZB, debug=True)

    print(output)
    print(output1)

    test_common.validate_cmp(para_type, a, output)
    test_common.validate_cmp(para_type, b, output1)


@triton.jit
def _transposed_nested_split(x, y, transposed_out, joined_out, Q: tl.constexpr, EXTRA: tl.constexpr):
    flat = tl.load(x + tl.arange(0, 4 * Q))
    transposed = tl.trans(tl.reshape(flat, (4, Q)))
    if EXTRA & 1:
        tl.store(transposed_out + tl.arange(0, 4 * Q), tl.reshape(transposed, (4 * Q, )))
    lo, hi = tl.split(tl.reshape(transposed, (Q, 2, 2)))
    a, b = tl.split(lo)
    c, d = tl.split(hi)
    i = tl.arange(0, Q)
    tl.store(y + i, a)
    tl.store(y + Q + i, b)
    tl.store(y + 2 * Q + i, c)
    tl.store(y + 3 * Q + i, d)
    if EXTRA & 2:
        joined = tl.join(tl.join(a, b), tl.join(c, d))
        tl.store(joined_out + tl.arange(0, 4 * Q), tl.reshape(joined, (4 * Q, )))


@pytest.mark.parametrize("q", [4, 8, 32, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("extra", [0, 1, 2, 3])
def test_transposed_nested_split(q, dtype, extra):
    # Secondary consumers keep the transposed tensor and split values alive.
    # A lone split can be folded into the load and miss strided UB lowering.
    x_cpu = torch.arange(4 * q, dtype=dtype)
    x = x_cpu.npu()
    y = torch.full_like(x, -777)
    transposed = torch.full_like(x, -777)
    joined = torch.full_like(x, -777)
    _transposed_nested_split[(1, )](x, y, transposed, joined, Q=q, EXTRA=extra)
    expected = x_cpu.reshape(4, q)[[0, 2, 1, 3], :].reshape(-1)
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)
    expected_transposed = x_cpu.reshape(4, q).T.contiguous().reshape(-1)
    if extra & 1:
        torch.testing.assert_close(transposed.cpu(), expected_transposed, rtol=0, atol=0)
    if extra & 2:
        torch.testing.assert_close(joined.cpu(), expected_transposed, rtol=0, atol=0)
