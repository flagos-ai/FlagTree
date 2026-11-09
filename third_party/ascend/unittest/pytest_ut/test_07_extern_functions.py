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
"""
Libdevice (`tl.extra.libdevice`) function
==============================
"""

import pytest
import torch
import torch_npu

import triton
import triton.language as tl
import triton.language.extra.cann.libdevice as libdevice
from triton.backends.ascend.compiler import get_libdevice

DEV = "npu"


@triton.jit
def asin_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x = libdevice.asin(x)
    tl.store(y_ptr + offsets, x, mask=mask)


extern_libs = {'libdevice': get_libdevice()}


def run_asin_case(size, use_extern_libs):
    torch.manual_seed(0)
    x = torch.rand(size, device=DEV)
    output_triton = torch.empty_like(x)
    output_torch = torch.asin(x)
    assert x.device.type == DEV and output_triton.device.type == DEV

    n_elements = output_torch.numel()

    def grid(meta):
        return (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    launch_kwargs = {"BLOCK_SIZE": 1024}
    if use_extern_libs:
        launch_kwargs["extern_libs"] = extern_libs

    asin_kernel[grid](x, output_triton, n_elements, **launch_kwargs)
    torch.testing.assert_close(output_torch, output_triton, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [98432, 1024])
def test_asin_kernel_matches_torch(size):
    run_asin_case(size=size, use_extern_libs=False)


@pytest.mark.parametrize("size", [98432, 1024])
def test_asin_kernel_matches_torch_with_extern_libs(size):
    run_asin_case(size=size, use_extern_libs=True)
