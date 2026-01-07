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

def torch_save_cache_to_buffer_indirect(
    buffer,
    cache,
    loc,
    buffer_stride,
    cache_stride,
    BLOCK
):
    max_len = min(buffer_stride, loc.shape[0])
    for i in range(buffer.shape[0]):
        index = (loc // BLOCK) * BLOCK + loc % BLOCK
        tmp = cache[i, 0, index]
        buffer[i, 0, :max_len] = tmp[:max_len]


@triton.jit
def save_cache_to_buffer_indirect(
    buffer_ptr,
    cache_ptr,
    loc_ptr,
    buffer_stride: tl.constexpr,
    BLOCK: tl.constexpr
):
    pid_loc = tl.program_id(0)
    index = tl.arange(0, buffer_stride)
    loc = tl.load(loc_ptr + index)
    buffer_offset = pid_loc * buffer_stride
    buffer_index = buffer_offset + index

    cache_offset = pid_loc * buffer_stride * 2
    cache_index = cache_offset + loc
    cache_index_0 = cache_index // BLOCK
    cache_index_1 = cache_index % BLOCK


    tmp = tl.load(cache_ptr + (BLOCK*cache_index_0 + cache_index_1))
    tl.store(buffer_ptr + buffer_index, tmp)


def biggest_divisor(num):
    for i in range(2, num):
        if num % i == 0:
            return num // i
    return num


types = [
    (torch.float32, 'float32'),
]


cache_shapes = [
    (5, 15),
]


@pytest.mark.parametrize('dtype,sigtype', types)
@pytest.mark.parametrize('batch_size,buffer_len', cache_shapes)
def test_linearize_indirect_load(batch_size, buffer_len, dtype, sigtype):
    block = biggest_divisor(buffer_len)
    cache_len = buffer_len * 2
    buffer_ref = torch.zeros(batch_size, 1, buffer_len, dtype=dtype)
    buffer = buffer_ref.npu()
    cache_ref = test_common.generate_tensor(shape=(batch_size, 1, cache_len), dtype=sigtype)
    cache = cache_ref.npu()

    loc_ref = torch.arange(0, buffer_len)
    loc = loc_ref.npu()
    torch_save_cache_to_buffer_indirect(buffer_ref, cache_ref, loc_ref, buffer_len, cache_len, block)
    save_cache_to_buffer_indirect[(batch_size, 1, 1)](buffer, cache, loc, buffer_len, block)
    test_common.validate_cmp(sigtype, buffer, buffer_ref)
