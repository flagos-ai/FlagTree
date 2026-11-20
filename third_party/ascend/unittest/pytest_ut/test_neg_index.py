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

import triton
import triton.language as tl
import time
import torch
import torch_npu
import test_common


@triton.jit
def triton_neg_index_load_kernel(in_ptr, out_ptr, BLOCK_SIZE: tl.constexpr, NEG_INDEX: tl.constexpr):
    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset >= NEG_INDEX
    tmp = tl.load(in_ptr + ((-NEG_INDEX) + offset), mask, other=0.0)
    tl.store(out_ptr + offset, tmp)


def triton_neg_index_load(in_tensor, index):
    out_tensor = torch.zeros(in_tensor.shape, device=in_tensor.device, dtype=in_tensor.dtype)
    triton_neg_index_load_kernel[(1, )](in_tensor, out_tensor, in_tensor.numel(), index)
    return out_tensor


def torch_neg_index_load(in_tensor, index):
    out = torch.zeros(in_tensor.shape, device=in_tensor.device, dtype=in_tensor.dtype)
    out[index:] = in_tensor[:index]
    return out


def test_neg_index_load():
    input_data = torch.arange(12, device="npu", dtype=torch.float32)
    triton_out = triton_neg_index_load(input_data, 6)
    torch_out = torch_neg_index_load(input_data, 6)
