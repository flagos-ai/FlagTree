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

import pytest

import triton
import triton.language as tl
import test_common

import torch
import torch_npu
import numpy as np


@triton.jit
def copy(
    data_ptrs,
    tgt_loc_ptr,
    src_loc_ptr,
    num_locs,
    stride: tl.constexpr,
    num_locs_upper: tl.constexpr,
):
    bid = tl.program_id(0)
    data_ptr = tl.load(data_ptrs + bid)
    data_ptr = tl.cast(data_ptr, tl.pointer_type(tl.float16))
    num_locs_offset = tl.arange(0, num_locs_upper)
    tgt_locs = tl.load(tgt_loc_ptr + num_locs_offset)
    src_locs = tl.load(src_loc_ptr + num_locs_offset)
    copy_offset = tl.arange(0, stride)
    value = tl.load(data_ptr + src_locs[:, None] * stride + copy_offset[None, :])
    value += 1
    tl.store(data_ptr + tgt_locs[:, None] * stride + copy_offset[None, :], value)


@triton.jit
def copy_all_layer_kv_cache(
    data_ptrs,
    strides,
    tgt_loc_ptr,
    src_loc_ptr,
    num_locs,
    num_locs_upper: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 128

    bid = tl.program_id(0)
    stride = tl.load(strides + bid)

    data_ptr = tl.load(data_ptrs + bid)
    data_ptr = tl.cast(data_ptr, tl.pointer_type(tl.uint8))

    num_locs_offset = tl.arange(0, num_locs_upper)
    tgt_locs = tl.load(tgt_loc_ptr + num_locs_offset)
    src_locs = tl.load(src_loc_ptr + num_locs_offset)

    num_loop = tl.cdiv(stride, BLOCK_SIZE)
    for i in range(num_loop):
        copy_offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = (num_locs_offset < num_locs)[:, None] & (copy_offset < stride)[None, :]
        value = tl.load(data_ptr + src_locs[:, None] * stride + copy_offset[None, :], mask=mask)
        value *= 1
        tl.store(
            data_ptr + tgt_locs[:, None] * stride + copy_offset[None, :],
            value,
            mask=mask,
        )


@triton.jit
def copy_all_layer_kv_cache2(
    data_ptrs,
    strides,
    tgt_loc_ptr,
    src_loc_ptr,
    num_locs,
    num_locs_upper: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 128

    bid = tl.program_id(0)
    stride = tl.load(strides + bid)

    data_ptr = tl.load(data_ptrs + bid)
    data_ptr = tl.cast(data_ptr, tl.pointer_type(tl.uint8))

    num_locs_offset = tl.arange(0, num_locs_upper)
    tgt_locs = tl.load(tgt_loc_ptr + num_locs_offset)
    src_locs = tl.load(src_loc_ptr + num_locs_offset)

    num_loop = tl.cdiv(stride, BLOCK_SIZE)
    for i in range(num_loop):
        copy_offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = (num_locs_offset < num_locs)[:, None] & (copy_offset < stride)[None, :]
        value = tl.load(data_ptr + (src_locs[:, None] * stride + copy_offset[None, :]), mask=mask)
        value *= 1
        tl.store(
            data_ptr + (tgt_locs[:, None] * stride + copy_offset[None, :]),
            value,
            mask=mask,
        )


@pytest.mark.parametrize('param_list', [
    ['float16', (10, 10), 'npu'],
])
def test_copy(param_list):
    dtype, shape, device = param_list
    data = torch.zeros(shape, dtype=eval('torch.' + dtype), device=device)
    data_ref = torch.zeros(shape, dtype=eval('torch.' + dtype), device=device)

    src_loc = torch.tensor([0], dtype=torch.int32, device=device)
    tgt_loc = torch.tensor([0], dtype=torch.int32, device=device)

    data_ptr = torch.tensor([data.data_ptr()], dtype=torch.uint64, device=device)
    stride = shape[1]
    copy[(1, )](data_ptr, tgt_loc, src_loc, 1, stride, 1)
    data_ref[0, :] += 1
    test_common.validate_cmp(dtype, data, data_ref)


@pytest.mark.parametrize('param_list', [
    ['float16', 3, 20, 16, 4, 16, 'npu'],
])
def test_hoistbroadcast_compare(param_list):
    dtype, layer_num, page_num, page_size, head_num, head_dim, device = param_list
    kv_buffer = torch.randn((2, layer_num, page_num, page_size, head_num, head_dim), dtype=eval('torch.' + dtype),
                            device=device)
    kv_buffer_ref = kv_buffer.clone()
    k_buffer = kv_buffer[0]
    v_buffer = kv_buffer[1]
    k_buffer_ref = kv_buffer_ref[0]
    v_buffer_ref = kv_buffer_ref[1]

    data_ptrs = torch.tensor([x.data_ptr() for x in [k_buffer]] + [x.data_ptr() for x in [v_buffer]],
                             dtype=torch.uint64, device=device)
    data_ptrs_ref = torch.tensor([x.data_ptr() for x in [k_buffer_ref]] + [x.data_ptr() for x in [v_buffer_ref]],
                                 dtype=torch.uint64, device=device)

    data_strides = torch.cat([
        torch.tensor([np.prod(x.shape[1:]) * x.dtype.itemsize for x in k_buffer], device=device),
        torch.tensor([np.prod(x.shape[1:]) * x.dtype.itemsize for x in v_buffer], device=device)
    ], dim=0)
    data_strides_ref = data_strides.clone()

    src_loc = torch.tensor([0], dtype=torch.int32, device=device)
    tgt_loc = torch.tensor([0], dtype=torch.int32, device=device)

    copy_all_layer_kv_cache[(len(data_ptrs), )](data_ptrs, data_strides, tgt_loc, src_loc, len(tgt_loc), 1)
    copy_all_layer_kv_cache2[(len(data_ptrs_ref), )](data_ptrs_ref, data_strides_ref, tgt_loc, src_loc, len(tgt_loc), 1)
    test_common.validate_cmp(dtype, kv_buffer, kv_buffer_ref)


def torch_pointwise(x0):
    if x0.dtype != torch.uint32:
        return torch.abs(x0)
    else:
        return torch.abs(x0.to(torch.float32))


@triton.jit
def fn_npu_(output_ptr, x_ptr, y_ptr, z_ptr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr, XNUMEL: tl.constexpr,
            YNUMEL: tl.constexpr, ZNUMEL: tl.constexpr):
    xoffs = tl.program_id(0) * XB
    yoffs = tl.program_id(1) * YB
    zoffs = tl.program_id(2) * ZB

    xidx = tl.arange(0, XB) + xoffs
    yidx = tl.arange(0, YB) + yoffs
    zidx = tl.arange(0, ZB) + zoffs

    X = tl.load(x_ptr + xidx[:, None, None] * YNUMEL * ZNUMEL + yidx[None, :, None] * ZNUMEL + zidx[None, None, :])
    ret = tl.abs(X)
    tl.store(output_ptr + xidx[:, None, None] * YNUMEL * ZNUMEL + yidx[None, :, None] * ZNUMEL + zidx[None, None, :],
             ret)


@pytest.mark.parametrize('shape', [(8, 16, 16)])
@pytest.mark.parametrize('dtype', ['float32'])
def test_case2(dtype, shape):
    # 生成数据
    x = test_common.generate_tensor(shape, dtype).npu()
    y = test_common.generate_tensor(shape, dtype).npu()
    z = test_common.generate_tensor(shape, dtype).npu()
    new_shape = shape

    output = torch.randint(1, new_shape, dtype=eval('torch.' + dtype)).npu()
    output1 = output

    ans = torch_pointwise(x)

    XB = shape[0]
    xnumel = shape[0]
    YB = shape[1]
    ynumel = shape[1]
    ZB = shape[2]
    znumel = shape[2]

    grid = (1, 1, 1)

    fn_npu_[grid](output, x, y, z, XB, YB, ZB, xnumel, ynumel, znumel)

    test_common.validate_cmp(dtype, ans, output)
