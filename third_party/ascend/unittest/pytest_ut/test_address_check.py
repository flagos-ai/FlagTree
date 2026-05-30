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
import pytest


@triton.jit
def simple_kernel(x_ptr, y_ptr, output_ptr, n_elements):
    pid = tl.program_id(axis=0)
    offsets = pid * 1024 + tl.arange(0, 1024)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    ret = x + y
    tl.store(output_ptr + offsets, ret, mask=mask)


def test_npu_tensor_should_success():
    print("Test the NPU tensor. The NPU tensor should be passed and executed properly.")

    size = 1024
    x_npu = torch.rand(size, device='npu')
    y_npu = torch.rand(size, device='npu')
    output = torch.empty(size, device='npu')

    simple_kernel[(1, )](x_npu, y_npu, output, size)

    expected = x_npu + y_npu
    actual = output

    torch.testing.assert_close(expected, actual, rtol=1e-03, atol=1e-03)


def test_cpu_tensor_should_fail():
    print("Test the CPU tensor. An address check exception should be raised.")

    size = 1024
    x_cpu = torch.rand(size, device='cpu')
    y_cpu = torch.rand(size, device='cpu')
    output = torch.empty(size, device='npu')

    with pytest.raises(ValueError) as exc_info:
        simple_kernel[(1, )](x_cpu, y_cpu, output, size)

    error_msg = str(exc_info.value)
    assert "cannot be accessed from Triton (cpu tensor?)" in error_msg, \
        f"Expected error message to contain CPU tensor rejection hint, but got: {error_msg}"
