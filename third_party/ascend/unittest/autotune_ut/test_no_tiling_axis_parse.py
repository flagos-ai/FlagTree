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
import os
import shutil
import pytest
import torch
import torch_npu
import triton
import triton.backends.ascend.runtime
import triton.language as tl
from triton.tools.get_ascend_devices import is_compile_on_910_95

import test_common

os.environ['TRITON_ALWAYS_COMPILE'] = '1'
os.environ['TRITON_AUTOTUNE_PARALLEL_COMPILE'] = '0'


def case_torch(x):
    return torch.permute(x, (1, 0))


@triton.autotune(configs=[], key=['xnumel', 'ynumel'], hints={
    "auto_gen_config": True,
})
@triton.jit
def triton_permute_2d(
    output_ptr,
    x_ptr,
    xnumel: tl.constexpr,
    ynumel: tl.constexpr,
    XBLOCK: tl.constexpr,
    YBLOCK: tl.constexpr,
):
    xpid = tl.program_id(0)
    ypid = tl.program_id(1)

    x_off = xpid * XBLOCK + tl.arange(0, XBLOCK)[:, None]
    y_off = ypid * YBLOCK + tl.arange(0, YBLOCK)[None, :]
    mask = (x_off < xnumel) & (y_off < ynumel)
    offs = y_off + x_off * ynumel
    b = tl.load(x_ptr + offs, mask=mask)
    ox_off = ypid * YBLOCK + tl.arange(0, YBLOCK)[:, None]
    oy_off = xpid * XBLOCK + tl.arange(0, XBLOCK)[None, :]
    o_mask = (ox_off < ynumel) & (oy_off < xnumel)
    o_offs = oy_off + ox_off * xnumel
    ret = tl.permute(b, (1, 0))
    tl.store(output_ptr + o_offs, ret, mask=o_mask)


def case_triton(x_cal, is_simt_only=False):
    xnumel = x_cal.shape[0]
    ynumel = x_cal.shape[1]
    output = torch.randint(1, (ynumel, xnumel), dtype=x_cal.dtype, device=x_cal.device)
    if is_simt_only:
        (triton_permute_2d[lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']), triton.cdiv(ynumel, meta['YBLOCK']), 1)](
            output, x_cal, xnumel, ynumel, force_simt_only=True))
    else:
        (triton_permute_2d[lambda meta:
                           (triton.cdiv(xnumel, meta['XBLOCK']), triton.cdiv(ynumel, meta['YBLOCK']), 1)](output, x_cal,
                                                                                                          xnumel,
                                                                                                          ynumel))
    return output


@pytest.mark.parametrize('shape', [(1024, 32), (32, 8)])
@pytest.mark.parametrize('dtype', ['bfloat16'])
def test_permute(shape, dtype):
    x_cal = test_common.generate_tensor(shape, dtype).npu()
    torch_output = case_torch(x_cal)
    triton_output = case_triton(x_cal)
    torch.testing.assert_close(torch_output, triton_output, rtol=1e-03, atol=1e-03, equal_nan=True)


@pytest.mark.skipif(not is_compile_on_910_95, reason="only support A5")
@pytest.mark.parametrize('shape', [(1024, 32)])
@pytest.mark.parametrize('dtype', ['bfloat16'])
def test_permute_simt(shape, dtype):
    x_cal = test_common.generate_tensor(shape, dtype).npu()
    torch_output = case_torch(x_cal)
    triton_output = case_triton(x_cal, True)
    torch.testing.assert_close(torch_output, triton_output, rtol=1e-03, atol=1e-03, equal_nan=True)
