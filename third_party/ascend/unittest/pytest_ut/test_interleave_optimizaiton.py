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

import torch
import torch_npu
import pytest
import test_common


def torch_interleave_load(q, k, head_dim_half, bias):
    d_indices = torch.arange(0, head_dim_half)
    k[d_indices * 2 + bias] = q[d_indices * 2 + bias]
    k[d_indices * 2 + 1 + bias] = -q[d_indices * 2 + 1 + bias]
    return k


def torch_interleave_load_with_mask(q, k, head_dim_half, bias, numel):
    d_indices = torch.arange(0, min(head_dim_half, numel))
    k[d_indices * 2 + bias] = q[d_indices * 2 + bias]
    k[d_indices * 2 + 1 + bias] = -q[d_indices * 2 + 1 + bias]
    return k


def torch_interleave_loadstore_with_mask(q, head_dim_half, bias, numel):
    d_indices = torch.arange(0, min(head_dim_half, numel))
    # it's unneccessary since we store it back without edit: q[d_indices * 2 + bias] = q[d_indices * 2 + bias]
    q[d_indices * 2 + 1 + bias] = -q[d_indices * 2 + 1 + bias]
    return q


@triton.jit
def triton_interleave_load(q_ptr, k_ptr, head_dim_half: tl.constexpr, bias: tl.constexpr):
    d_indices = tl.program_id(0) + tl.arange(0, head_dim_half)
    q_real = tl.load(q_ptr + d_indices * 2 + bias)
    q_imag = tl.load(q_ptr + d_indices * 2 + 1 + bias)
    new_q_real = q_real
    new_q_imag = -q_imag
    tl.store(k_ptr + d_indices * 2 + bias, new_q_real)
    tl.store(k_ptr + d_indices * 2 + 1 + bias, new_q_imag)


@triton.jit
def triton_interleave_load_with_mask(q_ptr, k_ptr, head_dim_half: tl.constexpr, bias: tl.constexpr,
                                     numel: tl.constexpr):
    d_indices = tl.program_id(0) + tl.arange(0, head_dim_half)
    mask = d_indices < numel
    q_real = tl.load(q_ptr + d_indices * 2 + bias, mask)
    q_imag = tl.load(q_ptr + d_indices * 2 + 1 + bias, mask)
    new_q_real = q_real
    new_q_imag = -q_imag
    tl.store(k_ptr + d_indices * 2 + bias, new_q_real, mask)
    tl.store(k_ptr + d_indices * 2 + 1 + bias, new_q_imag, mask)


# when load and store are on the same pointer, sometimes we can only optimize the store with mask
@triton.jit
def triton_interleave_loadstore_with_mask(q_ptr, head_dim_half: tl.constexpr, bias: tl.constexpr, numel: tl.constexpr):
    d_indices = tl.arange(0, head_dim_half)
    mask = d_indices < numel
    q_real = tl.load(q_ptr + d_indices * 2 + bias, mask)
    q_imag = tl.load(q_ptr + d_indices * 2 + 1 + bias, mask)
    new_q_real = q_real
    new_q_imag = -q_imag
    tl.store(q_ptr + d_indices * 2 + bias, new_q_real, mask)
    tl.store(q_ptr + d_indices * 2 + 1 + bias, new_q_imag, mask)


@pytest.mark.parametrize('para_type,data_type,head_dim_half,bias', [
    ['float32', torch.float32, 16, 4],
])
def test_interleave(para_type, data_type, head_dim_half, bias):
    length = bias + head_dim_half * 2
    q = torch.randn((length, ), dtype=data_type).npu()
    k = torch.zeros_like(q, dtype=data_type).npu()
    k_ref = torch.zeros_like(q, dtype=data_type).npu()

    triton_interleave_load[(1, )](q, k, head_dim_half, bias)
    k_ref = torch_interleave_load(q, k_ref, head_dim_half, bias)
    assert torch.allclose(k, k_ref)


@pytest.mark.parametrize('para_type,data_type,head_dim_half,bias,numel', [
    ['float32', torch.float32, 16, 0, 8],
])
def test_interleave_with_mask(para_type, data_type, head_dim_half, bias, numel):
    length = bias + head_dim_half * 2
    q = torch.randn((length, ), dtype=data_type).npu()
    k = torch.zeros_like(q, dtype=data_type).npu()
    k_ref = torch.zeros_like(q, dtype=data_type).npu()

    triton_interleave_load_with_mask[(1, )](q, k, head_dim_half, bias, numel)
    k_ref = torch_interleave_load_with_mask(q, k_ref, head_dim_half, bias, numel)
    assert torch.allclose(k, k_ref)


@pytest.mark.parametrize('para_type,data_type,head_dim_half,bias,numel', [
    ['float32', torch.float32, 16, 0, 8],
])
def test_interleave_loadstore_with_mask(para_type, data_type, head_dim_half, bias, numel):
    length = bias + head_dim_half * 2
    q = torch.randn((length, ), dtype=data_type).npu()
    q_ref = q.clone()

    triton_interleave_loadstore_with_mask[(1, )](q, head_dim_half, bias, numel)
    q_ref = torch_interleave_loadstore_with_mask(q_ref, head_dim_half, bias, numel)
    assert torch.allclose(q, q_ref)
