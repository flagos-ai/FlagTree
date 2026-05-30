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


def torch_expanddims_load(in0, in1, out0, YBLOCK, XBLOCK):
    for i in range(0, YBLOCK):
        tmp = in0[i] + 10
        res = in1[tmp]
        out0[i, :] = res


@triton.jit
def triton_expanddims_load(in0, in1, out0, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    base_y = tl.arange(0, YBLOCK)
    y_idx = base_y[:, None]
    y_mask = y_idx < YBLOCK
    base_x = tl.arange(0, XBLOCK)
    x_idx = base_x[None, :]
    x_mask = x_idx < XBLOCK

    y = tl.load(in0 + y_idx, mask=y_mask)
    tmp0 = tl.full([YBLOCK, XBLOCK], 10, tl.int32)
    tmp1 = y + tmp0
    res = tl.load(in1 + tmp1, mask=y_mask)
    tl.store(out0 + (y_idx * XBLOCK + x_idx), res, mask=y_mask & x_mask)


def test_case():
    YBLOCK = 4
    XBLOCK = 8
    in0 = torch.arange(0, YBLOCK, device="npu", dtype=torch.int32)
    in1 = torch.arange(0, YBLOCK + 10, device="npu", dtype=torch.int32)
    out0 = torch.empty((YBLOCK, XBLOCK), device="npu", dtype=torch.int32)

    torch_expanddims_load(in0, in1, out0, YBLOCK, XBLOCK)

    in0_triton = in0
    in1_triton = in1
    out0_triton = torch.empty((YBLOCK * XBLOCK, ), device="npu", dtype=torch.int32)

    triton_expanddims_load[(1, 1, 1)](in0_triton, in1_triton, out0_triton, YBLOCK, XBLOCK)
    out0_triton = out0_triton.view(YBLOCK, XBLOCK)

    assert torch.allclose(out0, out0_triton, rtol=1e-03, atol=1e-03, equal_nan=True)
