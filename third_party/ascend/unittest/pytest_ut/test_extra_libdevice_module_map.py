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
import torch_npu  # noqa: F401
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _extra_libdevice_log1p_kernel(x_ptr, o_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    y = libdevice.log1p(x)
    tl.store(o_ptr + offs, y)


def test_extra_libdevice_module_map_log1p():
    n = 32
    # Domain of log1p: x > -1. Use values near 0 for numerical stability.
    x = torch.linspace(-0.5, 0.5, n, dtype=torch.float32, device="npu")
    o = torch.empty(n, dtype=torch.float32, device="npu")

    _extra_libdevice_log1p_kernel[(1, )](x, o, BLOCK=n)
    torch.npu.synchronize()

    torch.testing.assert_close(o.cpu(), torch.log1p(x.cpu()), rtol=1e-5, atol=1e-5)
