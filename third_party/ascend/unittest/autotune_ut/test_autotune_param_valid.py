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

import pytest
import torch
import torch_npu
import triton
import triton.backends.ascend.runtime
import triton.language as tl


@triton.autotune(
    configs=[], key={"x": "n_elements"}, hints={
        "split_params": {"x": "BLOCK_SIZE"},
        "tiling_params": {"x": "BLOCK_SIZE_SUB"},
        "low_dim_axes": ["x"],
        "reduction_axes": [],
    })
@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_SUB: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK_SIZE
    loops1 = (BLOCK_SIZE + BLOCK_SIZE_SUB - 1) // BLOCK_SIZE_SUB
    for loop in range(0, loops1):
        x0 = offset + loop * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE_SUB)
        mask = x0 < n_elements
        x = tl.load(x_ptr + x0, mask)
        y = tl.load(y_ptr + x0, mask)
        output = x + y
        tl.store(output_ptr + x0, output)


def add_torch(x, y):
    return x + y


def add_autotune(x, y):
    output = torch.empty_like(x)
    n_elements = output.numel()
    add_kernel[lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )](x, y, output, n_elements)
    return output


@pytest.mark.autotune
@pytest.mark.parametrize('size', [
    2048,
])
def test_add(size: int):
    x = torch.rand(size, device="npu")
    y = torch.rand(size, device="npu")

    output_torch = add_torch(x, y)
    output_triton = add_autotune(x, y)
    assert torch.allclose(output_triton, output_torch)


@pytest.mark.autotune
def test_add_no_reduction_axes():
    try:

        @triton.autotune(
            configs=[], key={"x": "n_elements"}, hints={
                "split_params": {"x": "BLOCK_SIZE"},
                "tiling_params": {"x": "BLOCK_SIZE_SUB"},
                "low_dim_axes": ["x"],
            })
        @triton.jit
        def add_kernel_exception():
            pass
    except ValueError as e:
        assert "reduction_axes must be a list" in str(e)


@pytest.mark.autotune
def test_add_no_low_dim_axes():
    try:

        @triton.autotune(
            configs=[], key={"x": "n_elements"}, hints={
                "split_params": {"x": "BLOCK_SIZE"},
                "tiling_params": {"x": "BLOCK_SIZE_SUB"},
                "reduction_axes": [],
            })
        @triton.jit
        def add_kernel_exception():
            pass
    except ValueError as e:
        assert "low_dim_axes must be a list" in str(e)


@pytest.mark.autotune
def test_add_no_tiling_params():
    try:

        @triton.autotune(configs=[], key={"x": "n_elements"}, hints={
            "split_params": {"x": "BLOCK_SIZE"},
            "low_dim_axes": ["x"],
            "reduction_axes": [],
        })
        @triton.jit
        def add_kernel_exception():
            pass
    except ValueError as e:
        assert "tiling_params must be a dict" in str(e)


@pytest.mark.autotune
def test_add_no_split_params():
    try:

        @triton.autotune(
            configs=[], key={"x": "n_elements"}, hints={
                "tiling_params": {"x": "BLOCK_SIZE_SUB"},
                "low_dim_axes": ["x"],
                "reduction_axes": [],
            })
        @triton.jit
        def add_kernel_exception():
            pass
    except ValueError as e:
        assert "split_params must be a dict" in str(e)


@pytest.mark.autotune
def test_add_no_keyname():
    try:

        @triton.autotune(
            configs=[], key={"x0": "n_elements"}, hints={
                "tiling_params": {"x": "BLOCK_SIZE_SUB"},
                "low_dim_axes": ["x"],
                "reduction_axes": [],
            })
        @triton.jit
        def add_kernel_exception():
            pass
    except ValueError as e:
        assert "All keys in 'key' must be valid axis names" in str(e)
