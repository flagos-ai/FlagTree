# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is herey_size granted, free of charge, to any person obtaining a copy
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
import test_common


@triton.jit
def negative_padding_with_load_kernel(
    input_ptr,
    output_ptr,
    x_offset: tl.constexpr,
    y_offset: tl.constexpr,
    x_size: tl.constexpr,
    y_size: tl.constexpr,
):
    in_ptr = tl.make_block_ptr(
        base=input_ptr,
        shape=(x_size, y_size),
        strides=(y_size, 1),
        offsets=(x_offset, y_offset),
        block_shape=(x_size, y_size),
        order=(1, 0),
    )
    out_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(x_size, y_size),
        strides=(y_size, 1),
        offsets=(0, 0),
        block_shape=(x_size, y_size),
        order=(1, 0),
    )
    in_val = tl.load(in_ptr, boundary_check=(0, 1), padding_option="zero")
    tl.store(out_ptr, in_val)


@triton.jit
def negative_padding_with_store_kernel(
    input_ptr,
    output_ptr,
    x_offset: tl.constexpr,
    y_offset: tl.constexpr,
    x_size: tl.constexpr,
    y_size: tl.constexpr,
):
    in_ptr = tl.make_block_ptr(
        base=input_ptr,
        shape=(x_size, y_size),
        strides=(y_size, 1),
        offsets=(0, 0),
        block_shape=(x_size, y_size),
        order=(1, 0),
    )
    out_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(x_size, y_size),
        strides=(y_size, 1),
        offsets=(x_offset, y_offset),
        block_shape=(x_size, y_size),
        order=(1, 0),
    )
    in_val = tl.load(in_ptr)
    tl.store(out_ptr, in_val, boundary_check=(0, 1))


@pytest.mark.parametrize('param_list', [(8, 8), (16, 16), (32, 32), (64, 64)])
def test_makeblockptr_load_with_negative_padding(param_list):
    shape = param_list
    torch.manual_seed(1)
    x_offset = torch.randint(shape[0], size=()).item()
    # y_offset = torch.randint(shape[1], size=()).item()
    y_offset = 0
    input_tensor = torch.arange(start=1, end=shape[0] * shape[1] + 1, dtype=torch.int32).view(shape).npu()
    output = torch.zeros(shape, dtype=torch.int32).npu()
    negative_padding_with_load_kernel[(1, )](
        input_tensor,
        output,
        -x_offset,
        -y_offset,
        shape[0],
        shape[1],
    )
    output_ref = torch.zeros((shape[0] + x_offset, shape[1] + y_offset), dtype=torch.int32).cpu()
    output_subview = torch.narrow(output_ref, 0, x_offset, shape[0])
    output_subview = torch.narrow(output_subview, 1, y_offset, shape[1])
    output_subview.copy_(input_tensor)
    output_ref = torch.narrow(output_ref, 0, 0, shape[0])
    output_ref = torch.narrow(output_ref, 1, 0, shape[1])
    test_common.validate_cmp("int32", output, output_ref)


@pytest.mark.parametrize('param_list', [(8, 8), (16, 16), (32, 32), (64, 64)])
def test_makeblockptr_store_with_negative_padding(param_list):
    shape = param_list
    torch.manual_seed(1)
    x_offset = torch.randint(shape[0], size=()).item()
    # y_offset = torch.randint(shape[1], size=()).item()
    y_offset = 0
    input_tensor = torch.arange(start=1, end=shape[0] * shape[1] + 1, dtype=torch.int32).view(shape).npu()
    output = torch.zeros(shape, dtype=torch.int32).npu()
    negative_padding_with_store_kernel[(1, )](
        input_tensor,
        output,
        -x_offset,
        -y_offset,
        shape[0],
        shape[1],
    )
    output_ref = torch.zeros(shape, dtype=torch.int32).cpu()
    input_subview = torch.narrow(input_tensor, 0, x_offset, shape[0] - x_offset)
    input_subview = torch.narrow(input_subview, 1, y_offset, shape[1] - y_offset)
    output_subview = torch.narrow(output_ref, 0, 0, shape[0] - x_offset)
    output_subview = torch.narrow(output_subview, 1, 0, shape[1] - y_offset)
    output_subview.copy_(input_subview)
    test_common.validate_cmp("int32", output, output_ref)
