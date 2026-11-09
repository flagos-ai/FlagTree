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


@triton.jit
def fn_npu_(output_ptr, x_ptr, y_ptr, z_ptr, output_ptr1, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    xidx = tl.arange(0, XB)
    yidx = tl.arange(0, YB)
    zidx = tl.arange(0, ZB)
    idx = xidx[:, None, None] * YB * ZB + yidx[None, :, None] * ZB + zidx[None, None, :]
    block_ptr_in = tl.make_block_ptr(
        base=x_ptr,
        shape=(XB, YB, ZB),
        strides=(YB * ZB, ZB, 1),
        offsets=(0, 0, 0),
        block_shape=(XB, YB, ZB),
        order=(2, 1, 0),
    )
    X = tl.load(block_ptr_in)

    oidx = xidx[:, None, None] * YB * ZB + yidx[None, :, None] * ZB + zidx[None, None, :]

    block_ptr_out = tl.make_block_ptr(
        base=output_ptr,
        shape=(XB, YB, ZB),
        strides=(YB * ZB, ZB, 1),
        offsets=(0, 0, 0),
        block_shape=(XB, YB, ZB),
        order=(2, 1, 0),
    )
    tl.store(block_ptr_out, X)


paras = [
    ('*fp32', eval('torch.float32'), 2, 256, 16),
    ('*fp32', eval('torch.float32'), 8, 8, 4),
    ('*fp16', eval('torch.float16'), 2, 256, 16),
    ('*fp16', eval('torch.float16'), 8, 8, 4),
    ('*i8', eval('torch.int8'), 2, 256, 16),
    ('*i8', eval('torch.int8'), 8, 8, 4),
]


@pytest.mark.parametrize('para_type,data_type,XB,YB,ZB', paras)
def test_npu(para_type, data_type, XB, YB, ZB):

    x = torch.randint(low=-128, high=128, size=(XB, YB, ZB), dtype=data_type).npu()
    y = torch.randint(low=-128, high=128, size=(XB, YB, ZB), dtype=data_type).npu()
    z = torch.randint(low=-128, high=128, size=(XB, YB, ZB), dtype=data_type).npu()

    print(f"shape = {x.shape}")
    print(x.dtype)

    output = torch.randint(1, (XB, YB, ZB), dtype=data_type).npu()
    output1 = output
    print(f"output.dtype={output.dtype}")

    a = x
    print(a)
    fn_npu_[1, 1, 1](output, x, y, z, output1, XB=XB, YB=YB, ZB=ZB, debug=True)
    print(output)
    torch.testing.assert_close(output, a)


@triton.jit
def dma_block_ptr(
    input_ptr,
    output_ptr,
    scale_ptr,
    batch_size,
    cu_seqlens_ptr,
    stride_i_m,
    stride_i_n,
    stride_o_m,
    stride_o_n,
    stride_s_b,
    HEAD_DIM,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    n_progs = tl.num_programs(0)
    pid = tl.program_id(0)

    cu_num_blocks = 0
    for bid in range(batch_size):
        start_loc = tl.load(cu_seqlens_ptr + bid)
        end_loc = tl.load(cu_seqlens_ptr + bid + 1)
        scale = tl.load(scale_ptr + bid * stride_s_b)

        len_loc = end_loc - start_loc
        prev_num_blocks = cu_num_blocks
        new_num_blocks = tl.cdiv(len_loc, BLOCK_SIZE_M).to(tl.int32)
        i_block_ptr_bbase = tl.make_block_ptr(
            input_ptr + start_loc * stride_i_m,
            shape=(len_loc, HEAD_DIM),
            strides=(stride_i_m, stride_i_n),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
            order=(1, 0),
        )
        o_block_ptr_bbase = tl.make_block_ptr(
            output_ptr + start_loc * stride_o_m,
            shape=(len_loc, HEAD_DIM),
            strides=(stride_o_m, stride_o_n),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
            order=(1, 0),
        )
        cu_num_blocks += new_num_blocks
        for m_id in range((prev_num_blocks + pid) % n_progs, new_num_blocks, n_progs):
            i_block_ptr = tl.advance(i_block_ptr_bbase, (m_id * BLOCK_SIZE_M, 0))
            o_block_ptr = tl.advance(o_block_ptr_bbase, (m_id * BLOCK_SIZE_M, 0))
            i_tile = tl.load(i_block_ptr, boundary_check=[0, 1], padding_option="zero")
            o_tile = i_tile.to(tl.float32) * scale
            tl.store(o_block_ptr, o_tile.to(i_tile.dtype), boundary_check=[0, 1])


def ref_func(inputs, scale, cu_lens):
    outputs = torch.zeros_like(inputs)
    bsz = cu_lens.size(0) - 1
    for bid in range(bsz):
        tmp = inputs[cu_lens[bid]:cu_lens[bid + 1]].to(torch.float32) * scale[bid]
        outputs[cu_lens[bid]:cu_lens[bid + 1]] = tmp.to(outputs.dtype)
    return outputs


def tt_func(inputs, scale, cu_lens):
    bsz = cu_lens.size(0) - 1
    outputs = torch.zeros_like(inputs)
    head_dim = inputs.size(-1)
    assert head_dim <= 1024
    BLOCK_SIZE_N = 1024
    BLOCK_SIZE_M = 4
    dma_block_ptr[
        20,
    ](
        inputs,
        outputs,
        scale,
        bsz,
        cu_lens,
        inputs.stride(0),
        inputs.stride(1),
        outputs.stride(0),
        outputs.stride(1),
        scale.stride(0),
        head_dim,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    return outputs


@pytest.mark.parametrize('param_list', [
    [8, 1024, 1024, True],
    [8, 1024, 1024, False],
])
def test_func(param_list):
    bsz, max_len, max_n, test_align = param_list
    lens = torch.randint(max_len // 2, max_len, (bsz, ), dtype=torch.int32, device="npu")
    n = torch.randint(max_n // 2, max_n, (1, ), dtype=torch.int32, device="npu")[0].item()
    if test_align:
        lens = (lens + 1023) // 1024 * 1024
        n = (n + 1023) // 1024 * 1024
    cu_lens = torch.cumsum(lens, dim=0)
    cu_lens = torch.cat([torch.zeros(1, dtype=torch.int32, device="npu"), cu_lens], dim=0)
    inputs = torch.randn(cu_lens[-1], n, dtype=torch.float16, device="npu")
    scale = torch.randn(bsz, dtype=torch.float32, device="npu")
    ref_output = ref_func(inputs, scale, cu_lens)
    tt_output = tt_func(inputs, scale, cu_lens)
    torch.testing.assert_close(ref_output, tt_output)
