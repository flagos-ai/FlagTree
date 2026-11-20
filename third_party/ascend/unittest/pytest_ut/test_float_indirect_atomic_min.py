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
import torch_npu
import triton
import triton.language as tl


@triton.jit
def float_indirect_atomic_min_kernel(
    value_ptr,
    index_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    values = tl.load(value_ptr + offsets, mask=mask, other=0.0)
    indices = tl.load(index_ptr + offsets, mask=mask, other=0)
    tl.atomic_min(output_ptr + indices, values, mask=mask)


def test_float_indirect_atomic_min():
    n_elements = 257
    block_size = 128

    values_cpu = torch.linspace(-4.0, 4.0, n_elements, dtype=torch.float32, device="cpu")
    indices_cpu = torch.arange(n_elements - 1, -1, -1, dtype=torch.int64, device="cpu")

    values = values_cpu.npu()
    indices = indices_cpu.npu()
    output = torch.full((n_elements, ), float("inf"), dtype=torch.float32, device="npu")

    grid = (triton.cdiv(n_elements, block_size), )
    float_indirect_atomic_min_kernel[grid](
        values,
        indices,
        output,
        n_elements,
        BLOCK_SIZE=block_size,
    )

    expected = torch.full((n_elements, ), float("inf"), dtype=torch.float32, device="cpu")
    expected[indices_cpu] = values_cpu

    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
