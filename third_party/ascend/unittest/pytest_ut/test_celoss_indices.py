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

import torch
import triton
import triton.language as tl


@triton.jit
def celoss_indices_kernel(
    inp_ptr,
    tgt_ptr,
    w_ptr,
    out_ptr,
    w_tgt_ptr,
    ignore_index,
    C,
    D,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_d = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    offset_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D).to(tl.int64)

    tgt_ptrs = tgt_ptr + pid_n * D + offset_d
    tgt_mask = offset_d < D
    tgt = tl.load(tgt_ptrs, mask=tgt_mask, other=0)

    ignore_mask = not (tgt == ignore_index) and tgt_mask

    tmp_max = tl.zeros([BLOCK_C, BLOCK_D], dtype=tl.float32)
    tmp_sum = tl.zeros([BLOCK_C, BLOCK_D], dtype=tl.float32)

    for off in range(0, C, BLOCK_C):
        offset_c = off + tl.arange(0, BLOCK_C)
        inp_ptrs = inp_ptr + pid_n * C * D + offset_c[:, None] * D + offset_d[None, :]
        inp_mask = offset_c[:, None] < C and offset_d[None, :] < D
        inp = tl.load(inp_ptrs, inp_mask, other=-float("inf")).to(tl.float32)
        cur_max = tl.maximum(tmp_max, inp)
        cur_exp = tl.exp(inp - cur_max)
        tmp_sum = tmp_sum * tl.exp(tmp_max - cur_max) + cur_exp
        tmp_max = cur_max

    final_max = tl.max(tmp_max, axis=0)
    tmp_sum = tmp_sum * tl.exp(tmp_max - final_max[None, :])
    final_sum = tl.log(tl.sum(tmp_sum, axis=0))
    inp_tgt_ptrs = inp_ptr + pid_n * C * D + tgt * D + offset_d
    inp_tgt = tl.load(inp_tgt_ptrs, mask=tgt_mask, other=-float("inf")).to(tl.float32)

    out = final_sum + final_max - inp_tgt
    w_tgt_ptrs = w_tgt_ptr + pid_n * D + offset_d

    if w_ptr is None:
        w_tgt = ignore_mask
    else:
        w_tgt = tl.load(w_ptr + tgt, mask=tgt_mask, other=0).to(tl.float32)
        w_tgt = tl.where(ignore_mask, w_tgt, 0)

    tl.store(w_tgt_ptrs, w_tgt, mask=tgt_mask)
    out *= w_tgt
    out_ptrs = out_ptr + pid_n * D + offset_d
    tl.store(out_ptrs, out, mask=tgt_mask)


def test_celoss_indices_kernel(shape=(1, 2)):
    device = "npu"
    dtype = torch.float16
    ignore_index = -100
    BLOCK_C = 256
    BLOCK_D = 1

    N, C = shape
    D = 1

    inp = torch.randn(shape, dtype=dtype, device=device)
    tgt = torch.randint(0, C, (N, ), dtype=torch.int64, device=device)
    wgt = torch.randn(C, dtype=dtype, device=device)

    out_triton = torch.empty((N * D, ), dtype=torch.float32, device=device)
    w_tgt_triton = torch.empty((N * D, ), dtype=torch.float32, device=device)

    grid = (triton.cdiv(D, BLOCK_D), N)
    celoss_indices_kernel[grid](
        inp,
        tgt,
        wgt,
        out_triton,
        w_tgt_triton,
        ignore_index,
        C,
        D,
        BLOCK_C=BLOCK_C,
        BLOCK_D=BLOCK_D,
    )
