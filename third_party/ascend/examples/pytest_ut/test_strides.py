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

import logging
import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def mv_kernel(
    A,
    B,
    C,
    N,
    M,
    stride_an,
    stride_am,
    stride_bm,
    stride_cn,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offset_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)[:, None]
    offset_m = tl.arange(0, BLOCK_M)[None, :]
    n_mask = offset_n < N
    A_ptrs = A + offset_n * stride_an + offset_m * stride_am
    B_ptrs = B + offset_m * stride_bm
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for m in range(0, M, BLOCK_M):
        m_mask = m + offset_m < M
        a = tl.load(A_ptrs, mask=n_mask & m_mask, other=0.0).to(tl.float32)
        b = tl.load(B_ptrs, mask=m_mask, other=0.0).to(tl.float32)
        acc += a * b
        A_ptrs += BLOCK_M * stride_am
        B_ptrs += BLOCK_M * stride_bm

    acc = tl.sum(acc, axis=1)
    C_ptrs = C + offset_n * stride_cn
    tl.store(C_ptrs, acc[:, None], mask=n_mask)


def mv(inp, vec):
    assert inp.shape[1] == vec.shape[0], "incompatible dimensions"
    N, M = inp.shape
    out = torch.empty((N,), device=inp.device, dtype=inp.dtype)
    
    def grid(META): 
        return (triton.cdiv(N, META["BLOCK_N"]),)
    mv_kernel[grid](
        inp,
        vec,
        out,
        N,
        M,
        inp.stride(0),
        inp.stride(1),
        vec.stride(0),
        out.stride(0),
        32,
        4,
    )
    return out


def to_reference(inp, upcast=False):
    if inp is None:
        return None
    ref_inp = inp
    ref_inp = ref_inp.to("cpu")
    if upcast:
        if ref_inp.is_complex():
            ref_inp = ref_inp.to(torch.complex128)
        else:
            ref_inp = ref_inp.to(torch.float64)
    return ref_inp


RESOLUTION = {
    torch.bool: 0,
    torch.int16: 0,
    torch.int32: 0,
    torch.int64: 0,
    torch.float16: 1e-3,
    torch.float32: 1.3e-6,
    torch.bfloat16: 0.016,
    torch.float64: 1e-7,
    torch.complex32: 1e-3,
    torch.complex64: 1.3e-6,
}


def assert_close(res, ref, dtype, equal_nan=False, reduce_dim=1):
    assert res.dtype == dtype
    ref = ref.to(dtype)
    atol = 1e-4 * reduce_dim
    rtol = RESOLUTION[dtype]
    torch.testing.assert_close(res, ref, atol=atol, rtol=rtol, equal_nan=equal_nan)


def gems_assert_close(res, ref, dtype, equal_nan=False, reduce_dim=1):
    res = res.to("cpu")
    assert ref.device == torch.device("cpu")
    assert_close(
        res, ref, dtype, equal_nan=equal_nan, reduce_dim=reduce_dim
    )


@pytest.mark.mv
@pytest.mark.parametrize("M, N", [(1, 32), (160, 1024), (5333, 497)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_accuracy_mv(M, N, dtype):
    matrix = torch.randn((N, M), dtype=dtype, device="npu")
    vector = torch.randn((M,), dtype=dtype, device="npu")
    ref_matrix = to_reference(matrix, True)
    ref_vector = to_reference(vector, True)

    ref_out = torch.mv(ref_matrix, ref_vector)

    res_out = mv(matrix, vector)

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=M)

