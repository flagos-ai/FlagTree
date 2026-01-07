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

import math
import torch
import triton
import triton.language as tl

import pytest


@triton.jit
def if_tensor_kernel(
        kv_start_idx,
        output_ptr,
):
    pid = tl.program_id(0)
    if kv_start_idx:
        value = tl.load(kv_start_idx + pid)
        tl.store(output_ptr + pid, value)


def test_kernel():
    n = 8
    device = 'npu'

    kv_start_idx = torch.arange(n, dtype=torch.float32, device=device)
    output1 = torch.zeros(n, dtype=torch.float32, device=device)
    if_tensor_kernel[(n,)](
        kv_start_idx, output1,
    )

    expected = torch.arange(n, dtype=torch.float32, device=device)
    assert torch.allclose(output1, expected), f"Output {output1} != Expected {expected}"
    print(f"RESULT: output1 = {output1}")
    print("✅ Test passed!")


@triton.jit
def mul_if_block_kernel(
        value,
        value_stride0,
        value_stride1,
        output,
        output_stride0,
        output_stride1,
        lengths,
        bs,
        dim,
        max_seq_len,
        DIM_SIZE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr
):
    batch_idx = tl.program_id(0)
    if batch_idx >= bs:
        return
    
    need_reverse = tl.program_id(1) == 0

    block_idx = tl.program_id(2)
    block_start = block_idx * BLOCK_SIZE

    reverse_len = tl.load(lengths + batch_idx)
    copy_len = max_seq_len - reverse_len

    value = value + batch_idx * value_stride0
    output = output + batch_idx * output_stride0

    if need_reverse:
        if block_start >= reverse_len:
            return
        value_block_ptr = tl.make_block_ptr(
            base=value,
            shape=(reverse_len, dim),
            strides=(value_stride0, 1),
            offsets=(block_start, 0),
            block_shape=(BLOCK_SIZE, DIM_SIZE),
            order=(0, 1)
        )

        block_values = tl.load(value_block_ptr, boundary_check=(0, 1), padding_option="zero")

        output_block_ptr = tl.make_block_ptr(
            base=output + (reverse_len - 1) * output_stride1,
            shape=(reverse_len, dim),
            strides=(output_stride0, 1),
            offsets=(block_start, 0),
            block_shape=(BLOCK_SIZE, DIM_SIZE),
            order=(0, 1)
        )
        tl.store(output_block_ptr, block_values, boundary_check=(0, 1))
    else:
        if block_start >= copy_len:
            return
        value_block_ptr = tl.make_block_ptr(
            base=value + reverse_len * value_stride1,
            shape=(max_seq_len, dim),
            strides=(value_stride0, 1),
            offsets=(block_start, 0),
            block_shape=(BLOCK_SIZE, DIM_SIZE),
            order=(0, 1)
        )

        block_values = tl.load(value_block_ptr, boundary_check=(0, 1), padding_option="zero")

        output_block_ptr = tl.make_block_ptr(
            base=output + reverse_len * output_stride1,
            shape=(max_seq_len, dim),
            strides=(output_stride0, 1),
            offsets=(block_start, 0),
            block_shape=(BLOCK_SIZE, DIM_SIZE),
            order=(0, 1)
        )
        tl.store(output_block_ptr, block_values, boundary_check=(0, 1))


def ref_reverse(value: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    bs, max_seq_len, dim = value.shape
    out = value.clone()
    for b in range(bs):
        r = int(lengths[b])
        if r > 0:
            out[b, :r] = value[b, :r].flip(0)
    return out


@pytest.mark.parametrize("bs,max_seq_len,dim,BLOCK_SIZE", [
    (2, 32, 16, 8),
])
def test_reverse_sequence_kernel(bs, max_seq_len, dim, BLOCK_SIZE):
    device = "npu"
    value = torch.randn(bs, max_seq_len, dim, device=device, dtype=torch.float32)
    lengths = torch.tensor(
        [0, max_seq_len // 2, max_seq_len] + [max_seq_len // 3] * max(0, bs - 3),
        device=device, dtype=torch.int32
    )[:bs]
    output = torch.empty_like(value)

    value_stride0 = value.stride(0)
    value_stride1 = value.stride(1)
    output_stride0 = output.stride(0)
    output_stride1 = output.stride(1)

    DIM_SIZE = dim
    grid = (bs, 2, math.ceil(max_seq_len / BLOCK_SIZE))
    mul_if_block_kernel[grid](
        value, value_stride0, value_stride1,
        output, output_stride0, output_stride1,
        lengths, bs, dim, max_seq_len,
        DIM_SIZE=DIM_SIZE, BLOCK_SIZE=BLOCK_SIZE
    )


if __name__ == "__main__":
    test_kernel()
