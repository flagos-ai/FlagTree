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

import pytest
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["active_count"])
def scalar_masked_load_kernel(src, dst, active_count, WITH_OTHER: tl.constexpr):
    pid = tl.program_id(0)
    active = pid < active_count
    if WITH_OTHER:
        value = tl.load(src + pid, mask=active, other=0)
        tl.store(dst + pid, value)
    else:
        value = tl.load(src + pid, mask=active)
        # The inactive load result is unspecified; leave the destination alone.
        tl.store(dst + pid, value, mask=active)


@pytest.mark.parametrize("dtype", [torch.int8, torch.int32, torch.float32, torch.bool])
@pytest.mark.parametrize("active_count", [0, 5, 32])
@pytest.mark.parametrize("with_other", [False, True])
def test_scalar_masked_load(dtype, active_count, with_other):
    source = (torch.arange(32) % 7 - 3).to(dtype)
    src = source.npu()
    dst = torch.full((32, ), 1, dtype=dtype, device="npu")
    scalar_masked_load_kernel[(32, )](src, dst, active_count, with_other)
    expected = torch.full((32, ), 0 if with_other else 1, dtype=dtype)
    expected[:active_count] = source[:active_count]
    torch.testing.assert_close(dst.cpu(), expected, rtol=0, atol=0)
