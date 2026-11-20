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
import torch
import torch_npu
import triton
import triton.language as tl
import test_common


def torch_gather_gemv(weight, indices, vector):
    """Reference: gather matrices from weight by indices (with negative clamping),
    then do batched matrix-vector multiply.

    Args:
        weight:  (B, S, S) float16
        indices: (N,) int64, values in [-B, B-1]
        vector:  (S,) float16
    Returns:
        result:  (N, S) float16
    """
    B = weight.shape[0]
    indices_clamped = torch.where(indices < 0, indices + B, indices)
    gathered = weight[indices_clamped].float()  # (N, S, S)
    result = torch.matmul(gathered, vector.float().unsqueeze(-1)).squeeze(-1)
    return result.to(torch.float16)


@triton.jit
def triton_gather_gemv_kernel(
    in_ptr0,  # indices: (N,) int64
    in_ptr1,  # weight:  (B, S, S) float16
    in_ptr2,  # vector:  (S,) float16
    out_ptr1,  # output:  (N*S,) float16
    xnumel,
    rnumel,
    XBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr,
):
    xoffset = tl.program_id(0).to(tl.int64) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None].to(tl.int64)
    rbase = tl.arange(0, RBLOCK)[None, :].to(tl.int64)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 // rnumel), None, eviction_policy="evict_last")
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        r1 = rindex
        tmp7 = tl.load(in_ptr2 + (r1), None, eviction_policy="evict_last").to(tl.float32)
        tmp1 = tmp0 + 8
        tmp2 = tmp0 < 0
        tmp3 = tl.where(tmp2, tmp1, tmp0)
        tmp4 = tl.load(
            in_ptr1 + (r1 + (rnumel * (x0 % rnumel)) + (rnumel * rnumel * tmp3)),
            None,
            eviction_policy="evict_first",
        )
        tmp5 = tmp4.to(tl.float32)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp5 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tmp12
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tmp13 = tmp11.to(tl.float32)
    tl.store(out_ptr1 + (x0), tmp13, None)


@pytest.mark.parametrize('S', [2048])
@pytest.mark.parametrize('config', [(1, 2048), (64, 8)], ids=['xblock1_rblock2048', 'xblock64_rblock8'])
def test_gather_gemv(S, config):
    XBLOCK, RBLOCK = config
    B = 8
    N = 2

    indices = torch.randint(0, B, (N, ), dtype=torch.int64).npu()
    weight = test_common.generate_tensor((B, S, S), 'float16').npu()
    vector = test_common.generate_tensor((S, ), 'float16').npu()

    xnumel = N * S
    rnumel = S
    out = torch.empty((xnumel, ), dtype=torch.float16).npu()
    grid = (triton.cdiv(xnumel, XBLOCK), )
    triton_gather_gemv_kernel[grid](
        indices,
        weight,
        vector,
        out,
        xnumel,
        rnumel,
        XBLOCK=XBLOCK,
        RBLOCK=RBLOCK,
        num_stages=1,
        num_warps=8,
    )
    triton_res = out.view(N, S)
    torch_res = torch_gather_gemv(weight, indices, vector)
    test_common.validate_cmp("float16", triton_res, torch_res)
