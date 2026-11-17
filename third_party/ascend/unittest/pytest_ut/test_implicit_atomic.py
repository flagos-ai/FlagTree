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
import pytest
import triton
import test_common
import triton.language as tl

types_all = [
    (torch.float32, 'float32'),
]


def ceil_div(a, b):
    return (a + b - 1) // b


@triton.jit
def addptr_implicit_perm_atomic_add_2d(
    ptr,
    out,
    ynumel,
    xnumel,
    YBLOCK: tl.constexpr,
    XBLOCK: tl.constexpr,
):
    y = tl.program_id(1) * YBLOCK + tl.arange(0, YBLOCK)[None, :]  # [1, YB]
    x = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]  # [XB, 1]

    val = 1.0 + (x.to(tl.float32) * 0.01) + (y.to(tl.float32) * 0.001)  # [XB, YB]
    xmask = x < xnumel
    ymask = y < ynumel
    old = tl.atomic_add(ptr + (x + 4 * y), val, xmask & ymask)

    tl.store(out + (x + 4 * y), old)


@triton.jit
def addptr_implicit_perm_atomic_cas_2d(
    ptr,
    out,
    cmp_ptr,
    val_ptr,
    ynumel,
    xnumel,
    YBLOCK: tl.constexpr,
    XBLOCK: tl.constexpr,
):
    y = tl.program_id(1) * YBLOCK + tl.arange(0, YBLOCK)[None, :]
    x = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]

    xmask = x < xnumel
    ymask = y < ynumel
    mask = xmask & ymask

    offset = x + 4 * y

    cmp = tl.load(cmp_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    val = tl.load(val_ptr + offset, mask=mask, other=0.0).to(tl.float32)

    old = tl.atomic_cas(ptr + offset, cmp, val)

    tl.store(out + offset, old, mask=mask)


@pytest.mark.parametrize('dtype,sigtype', types_all)
@pytest.mark.parametrize('xnumel, ynumel, XBLOCK, YBLOCK', [(4, 512, 4, 64)])
def test_addptr_implicit_perm_atomic_add_2d(
    dtype,
    sigtype,
    xnumel,
    ynumel,
    XBLOCK,
    YBLOCK,
):
    in_ptr = torch.zeros((ynumel * 4, ), dtype=dtype).npu()
    out_ptr = torch.ones_like(in_ptr)

    grid = (ceil_div(xnumel, XBLOCK), ceil_div(ynumel, YBLOCK), 1)
    addptr_implicit_perm_atomic_add_2d[grid](in_ptr, out_ptr, ynumel, xnumel, YBLOCK=YBLOCK, XBLOCK=XBLOCK)

    y_idx = torch.arange(ynumel).unsqueeze(1).npu()
    x_idx = torch.arange(xnumel).unsqueeze(0).npu()
    idx = (x_idx + 4 * y_idx).reshape(-1)
    torch.testing.assert_close(out_ptr[idx], torch.zeros_like(out_ptr[idx]))

    val_ref = (1.0 + 0.01 * x_idx.to(torch.float32) + 0.001 * y_idx.to(torch.float32)).reshape(-1)
    torch.testing.assert_close(in_ptr[idx], val_ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize('dtype,sigtype', types_all)
@pytest.mark.parametrize('xnumel, ynumel, XBLOCK, YBLOCK', [(4, 512, 4, 64)])
def test_addptr_implicit_perm_atomic_cas_2d(
    dtype,
    sigtype,
    xnumel,
    ynumel,
    XBLOCK,
    YBLOCK,
):
    in_ptr = torch.full((ynumel * 4, ), 2, dtype=dtype).npu()
    out_ptr = torch.full((ynumel * 4, ), 1, dtype=dtype).npu()
    cmp_ptr = torch.full((ynumel * 4, ), 2, dtype=dtype).npu()
    val_ptr = torch.full((ynumel * 4, ), 1, dtype=dtype).npu()

    grid = (ceil_div(xnumel, XBLOCK), ceil_div(ynumel, YBLOCK), 1)
    addptr_implicit_perm_atomic_cas_2d[grid](in_ptr, out_ptr, cmp_ptr, val_ptr, ynumel, xnumel, YBLOCK=YBLOCK,
                                             XBLOCK=XBLOCK)

    y_idx = torch.arange(ynumel).unsqueeze(1).npu()
    x_idx = torch.arange(xnumel).unsqueeze(0).npu()
    idx = (x_idx + 4 * y_idx).reshape(-1)

    torch.testing.assert_close(out_ptr[idx], torch.full_like(out_ptr[idx], 2.0))

    torch.testing.assert_close(in_ptr[idx], torch.ones_like(in_ptr[idx]))


if __name__ == '__main__':
    case_2d = (4, 512, 4, 64)
    test_addptr_implicit_perm_atomic_add_2d(*types_all[0], *case_2d)
    test_addptr_implicit_perm_atomic_cas_2d(*types_all[0], *case_2d)
