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


"""
Rotary embedding kernel implemented by Triton.
GPT-J style
"""

import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def rotary_embedding_kernel(
    state,  # [num_tokens, head_num, head_dim]
    cos,  # [num_tokens, 1, head_dim // 2]
    sin,  # [num_tokens, 1, head_dim // 2]
    stride_state_n,
    stride_state_h,
    stride_state_d,
    stride_cos_n,
    stride_cos_d,
    # stride_sin_n,
    # stride_sin_d,
    num_tokens,
    num_heads,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_index = tl.program_id(0)
    token_range = token_index * BLOCK_N + tl.arange(0, BLOCK_N)
    head_index = tl.program_id(1)
    head_range = head_index * BLOCK_H + tl.arange(0, BLOCK_H)

    dim_range = tl.arange(0, BLOCK_D // 2)
    dim_range_x = dim_range * 2
    dim_range_y = dim_range * 2 + 1

    # tl.device_print("dim x", dim_range_x)
    # tl.device_print("dim y", dim_range_y)

    state_x_offset = (
        token_range[:, None, None] * stride_state_n
        + head_range[None, :, None] * stride_state_h
        + dim_range_x[None, None, :] * stride_state_d
    )

    state_y_offset = (
        token_range[:, None, None] * stride_state_n
        + head_range[None, :, None] * stride_state_h
        + dim_range_y[None, None, :] * stride_state_d
    )

    cos_sim_offset = (
        token_range[:, None, None] * stride_cos_n
        + dim_range[None, None, :] * stride_cos_d
    )

    state_x = tl.load(
        state + state_x_offset,
        mask=(token_range[:, None, None] < num_tokens)
        & (head_range[None, :, None] < num_heads),
        other=0.0,
    )
    state_y = tl.load(
        state + state_y_offset,
        mask=(token_range[:, None, None] < num_tokens)
        & (head_range[None, :, None] < num_heads),
        other=0.0,
    )

    cos_loaded = tl.load(
        cos + cos_sim_offset,
        mask=token_range[:, None, None] < num_tokens,
        other=0.0,
    )
    sin_loaded = tl.load(
        sin + cos_sim_offset,
        mask=token_range[:, None, None] < num_tokens,
        other=0.0,
    )

    out_x = state_x * cos_loaded - state_y * sin_loaded
    out_y = state_x * sin_loaded + state_y * cos_loaded

    tl.store(
        state + state_x_offset,
        out_x,
        mask=(token_range[:, None, None] < num_tokens)
        & (head_range[None, :, None] < num_heads),
    )
    tl.store(
        state + state_y_offset,
        out_y,
        mask=(token_range[:, None, None] < num_tokens)
        & (head_range[None, :, None] < num_heads),
    )


@torch.inference_mode()
def rotary_embedding(state, cos, sin):
    num_tokens = state.shape[0]
    num_heads = state.shape[1]
    head_dim = state.shape[2]

    BLOCK_N = 8
    BLOCK_H = 4
    grid = (
        triton.cdiv(num_tokens, BLOCK_N),
        triton.cdiv(num_heads, BLOCK_H),
    )
    if head_dim >= 128:
        num_warps = 8
    else:
        num_warps = 4

    kernel = rotary_embedding_kernel[grid](
        state,
        cos,
        sin,
        state.stride(0),
        state.stride(1),
        state.stride(2),
        cos.stride(0),
        cos.stride(2),
        # sin.stride(0),
        # sin.stride(2),
        num_tokens,
        num_heads,
        BLOCK_N=BLOCK_N,
        BLOCK_H=BLOCK_H,
        BLOCK_D=head_dim,
        num_warps=num_warps,
        num_stages=1,
    )
    # print(kernel.asm['ttir'])
    return


def torch_rotary_embedding(state, cos, sin):
    _, _, dim = state.shape
    state_x = state[:, :, 0 : dim : 2]
    state_y = state[:, :, 1 : dim : 2]
    out_x = state_x * cos - state_y * sin
    out_y = state_x * sin + state_y * cos
    out = torch.empty_like(state).npu()
    out[:, :, 0 : dim : 2] = out_x
    out[:, :, 1 : dim : 2] = out_y
    return out


def _rotary_emb(dtype):
    tokens_num = 256
    num_heads = 96
    head_dim = 128
    max_positions = 1024

    # torch.float16 has floating point problem in Triton 2.0.0
    # But it works fine in Triton 2.1.0
    state = torch.randn((tokens_num, num_heads, head_dim), dtype=dtype, device="npu")
    cos_shape = (tokens_num, 1, head_dim // 2)
    cos = -1.2 + 0.5 * torch.randn(cos_shape, dtype=dtype, device="npu")
    sin = -2.0 + 0.5 * torch.randn(cos_shape, dtype=dtype, device="npu")
    # forward pass
    torch_result = torch_rotary_embedding(state, cos, sin)
    rotary_embedding(state, cos, sin)
    triton_result = state  # state is modified in-place
    # print(torch_result[1][0])
    # print(triton_result[1][0])
    # Note: This test is not accurate enough.
    assert torch.allclose(torch_result, triton_result, atol=1e-2, rtol=1e-7)

def test_rotary_emb():
    _rotary_emb(torch.float16)
    _rotary_emb(torch.float32)
