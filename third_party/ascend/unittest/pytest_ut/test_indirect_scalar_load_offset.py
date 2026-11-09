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


@triton.jit
def gather_after_reduce_kernel(
    logits_ptr,
    topk_ids_ptr,
    output_ptr,
    logits_stride,
    vocab_size,
    BLOCK: tl.constexpr,
):
    req_idx = tl.program_id(0)

    max_val = -float('inf')
    for start in range(0, vocab_size, BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < vocab_size
        vals = tl.load(
            logits_ptr + req_idx * logits_stride + offsets,
            mask=mask,
            other=-float('inf'),
        )
        block_max = tl.max(vals)
        max_val = tl.maximum(max_val, block_max)

    topk_id = tl.load(topk_ids_ptr + req_idx + tl.arange(0, 1))
    val = tl.load(logits_ptr + req_idx * logits_stride + topk_id)
    tl.store(output_ptr + req_idx + tl.arange(0, 1), val - max_val)


def torch_reference(logits, topk_ids):
    num_rows = logits.shape[0]
    output = torch.empty(num_rows, dtype=logits.dtype)
    for i in range(num_rows):
        max_val = logits[i].max()
        output[i] = logits[i, topk_ids[i]] - max_val
    return output


shapes = [
    (4, 128),
    (8, 256),
    (16, 1024),
]


@pytest.mark.parametrize('num_rows,vocab_size', shapes)
def test_gather_after_reduce(num_rows, vocab_size):
    BLOCK = 128

    logits_ref = test_common.generate_tensor(shape=(num_rows, vocab_size), dtype='float32')
    logits = logits_ref.npu()
    logits_flat = logits.reshape(-1)

    topk_ids_ref = torch.randint(0, vocab_size, (num_rows, ), dtype=torch.int64)
    topk_ids = topk_ids_ref.npu()

    output = torch.empty(num_rows, dtype=torch.float32).npu()

    gather_after_reduce_kernel[(num_rows, )](
        logits_flat,
        topk_ids,
        output,
        vocab_size,
        vocab_size,
        BLOCK=BLOCK,
    )

    output_ref = torch_reference(logits_ref, topk_ids_ref)
    test_common.validate_cmp('float32', output, output_ref)
