# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
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
import torch
import torch_npu

import triton
import triton.language as tl
import triton.language.extra.cann.libdevice as libdevice

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


def test_asin_kernel_matches_torch():
    size = 98432
    torch.manual_seed(0)
    x = torch.rand(size, device=DEV)
    output_triton = torch.empty_like(x)
    output_torch = torch.asin(x)
    n_elements = output_torch.numel()

    def grid(meta):
        return (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    asin_kernel[grid](x, output_triton, n_elements, BLOCK_SIZE=1024)
    torch.testing.assert_close(output_torch, output_triton, rtol=1e-4, atol=1e-4)
    # Demo-style print
    torch.manual_seed(0)
    x_demo = torch.rand(98432, device=DEV)
    output_triton_demo = torch.empty_like(x_demo)
    output_torch_demo = torch.asin(x_demo)
    n_elements = output_torch_demo.numel()

    def grid(meta):
        return (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    asin_kernel[grid](x_demo, output_triton_demo, n_elements, BLOCK_SIZE=1024)
    print(output_torch_demo)
    print(output_triton_demo)
    print(f'The maximum difference between torch and triton is '
          f'{torch.max(torch.abs(output_torch_demo - output_triton_demo))}')


if __name__ == "__main__":
    test_asin_kernel_matches_torch()
    print("======Extern Functions Test Passed!======")
