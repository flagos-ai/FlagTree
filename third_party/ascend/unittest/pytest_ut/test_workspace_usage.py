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


@triton.jit
def matmul_mul_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    Out_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    m_mask = rm[:, None] < M
    n_mask = rn[None, :] < N
    full_mask = m_mask & n_mask

    A = tl.load(A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak, mask=m_mask, other=0.0)

    B = tl.load(B_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn, mask=n_mask, other=0.0)

    C = tl.load(C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn, mask=full_mask, other=0.0)

    AB = tl.dot(A, B)
    Out = AB * C

    tl.store(Out_ptr + rm[:, None] * stride_om + rn[None, :] * stride_on, Out, mask=full_mask)


def _make_inputs(M: int, N: int, K: int, dtype=torch.float16, device="cpu"):
    """Build a packed set of A, B, C, Out on the given device."""
    torch.manual_seed(0)
    A = torch.randn(M, K, dtype=dtype, device=device)
    B = torch.randn(K, N, dtype=dtype, device=device)
    C = torch.randn(M, N, dtype=dtype, device=device)
    Out = torch.empty(M, N, dtype=dtype, device=device)
    return A, B, C, Out


def _ref(A, B, C):
    return torch.matmul(A, B) * C


def _launch_kernel(A, B, C, Out, BLOCK_M=32, BLOCK_N=32, BLOCK_K=32):
    M, K = A.shape
    _, N = B.shape
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_mul_kernel[grid](
        A,
        B,
        C,
        Out,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        Out.stride(0),
        Out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )


def test_workspace():
    M = 256 * 64
    N = 256 * 64
    K = 32

    A, B, C, Out = _make_inputs(M, N, K, device="npu")
    # cv pipeline workspace * 4
    _launch_kernel(A, B, C, Out, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32)
    ref = _ref(A, B, C)
    torch.testing.assert_close(Out, ref, atol=1e-3, rtol=1e-3)
