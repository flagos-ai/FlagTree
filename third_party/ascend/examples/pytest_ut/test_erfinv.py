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
import triton.language.extra.ascend.libdevice as libdevice

@triton.jit
def triton_erfinv(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        xindex = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]
        xmask = xindex < xnumel
        x0 = tl.load(in_ptr0 + xindex, xmask)
        y = libdevice.erfinv(x0)
        tl.store(out_ptr0 + xindex, y, xmask)

@pytest.mark.parametrize('param_list',
                         [
                             ['float32', (2, 4096, 8), 2, 32768, 1024],
                         ]
                         )
def test_erfinv_case(param_list):
    dtype, shape, ncore, xblock, xblock_sub = param_list
    x = test_common.generate_tensor(shape, dtype).npu()
    x[0][0][0] = 1  # erfinv(1) -> ∞
    x[0][0][1] = -1  # erfinv(-1) -> -∞

    # Avoid numerical instability near ±1
    # Move values in (threshold, 1) to threshold and (-1, -threshold) to -threshold 
    threshold = 1 - 1.1e-4
    too_close_pos = (x > threshold) & (x < 1)
    too_close_neg = (x < -threshold) & (x > -1)
    x[too_close_pos] = threshold
    x[too_close_neg] = -threshold
    y_ref = torch.erfinv(x).npu()
    y_cal = torch.zeros(shape, dtype=eval('torch.' + dtype)).npu()
    triton_erfinv[ncore, 1, 1](x, y_cal, x.numel(), xblock, xblock_sub)
    test_common.validate_cmp(dtype, y_cal, y_ref)

@pytest.mark.parametrize('param_list',
                         [
                             ['float32', (2, 4096, 8), 2, 32768, 1024],
                         ]
                         )
def test_all_blocks_parallel(param_list, monkeypatch):
    monkeypatch.setenv("TRITON_ALL_BLOCKS_PARALLEL", "1")
    dtype, shape, ncore, xblock, xblock_sub = param_list
    x = test_common.generate_tensor(shape, dtype).npu()
    x[0][0][0] = 1  # erfinv(1) -> ∞
    x[0][0][1] = -1  # erfinv(-1) -> -∞

    # Avoid numerical instability near ±1
    # Move values in (threshold, 1) to threshold and (-1, -threshold) to -threshold 
    threshold = 1 - 1.1e-4
    too_close_pos = (x > threshold) & (x < 1)
    too_close_neg = (x < -threshold) & (x > -1)
    x[too_close_pos] = threshold
    x[too_close_neg] = -threshold
    y_ref = torch.erfinv(x).npu()
    y_cal = torch.zeros(shape, dtype=eval('torch.' + dtype)).npu()
    triton_erfinv[ncore, 1, 1](x, y_cal, x.numel(), xblock, xblock_sub)
    test_common.validate_cmp(dtype, y_cal, y_ref)
    monkeypatch.delenv("TRITON_ALL_BLOCKS_PARALLEL")