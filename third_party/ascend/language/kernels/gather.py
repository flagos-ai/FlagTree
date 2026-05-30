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
Gather kernel optimized for Ascend NPU.
"""
__all__ = ["gather_2d_simd"]

import triton
import triton.language as tl
from triton.language.core import constexpr


@triton.jit
def gather_2d_simd(src_ptr, idx_ptr, out_ptr, M: constexpr, N: constexpr, K: constexpr, XBLOCK: constexpr,
                   XBLOCK_SUB: constexpr):
    """
    2D gather kernel for axis=1 (tail axis) with SIMD-style vectorization.

    This kernel is optimized for Ascend NPU architecture with focus on:
    - Vectorized memory access using XBLOCK_SUB
    - Efficient global memory (GM) access patterns
    - Suitable for cases where N and K are not extremely large

    Args:
        src_ptr: [M, N] source tensor in GM (Global Memory)
        idx_ptr: [M, K] indices tensor in GM
        out_ptr: [M, K] output tensor in GM
        M: batch dimension size
        N: source dimension size (gather from this dimension)
        K: output dimension size (number of indices per batch)
        XBLOCK: outer block size for M dimension (for program distribution)
        XBLOCK_SUB: inner block size for M dimension (for SIMD vectorization)

    Example:
        import torch
        import triton
        from third_party.ascend.language.kernels import gather_2d_simd

        M, N, K = 128, 256, 64
        src = torch.randn(M, N, device='npu')
        indices = torch.randint(0, N, (M, K), dtype=torch.int32, device='npu')
        output = torch.empty((M, K), dtype=src.dtype, device='npu')

        grid = (triton.cdiv(M, 32),)
        gather_2d_simd[grid](src, indices, output, M, N, K,
                             XBLOCK=32, XBLOCK_SUB=4)
    """
    pid = tl.program_id(0)
    m_start = pid * XBLOCK
    m_end = min(m_start + XBLOCK, M)
    m_base = tl.arange(0, XBLOCK_SUB)

    # Process multiple rows at once using XBLOCK_SUB for vectorization
    for m_tile_start in range(m_start, m_end, XBLOCK_SUB):
        # M dimension offsets: [XBLOCK_SUB]
        m_offs = m_tile_start + m_base
        m_mask = m_offs < M

        # Load indices: [XBLOCK_SUB, K]
        k_offs = tl.arange(0, K)
        idx_tile = tl.load(idx_ptr + m_offs[:, None] * K + k_offs[None, :])

        # Load source data: [XBLOCK_SUB, N]
        n_offs = tl.arange(0, N)
        src_tile = tl.load(src_ptr + m_offs[:, None] * N + n_offs[None, :])

        # Gather operation along axis=1
        gathered_values = tl.gather(src_tile, idx_tile, axis=1)

        # Store results
        tl.store(out_ptr + m_offs[:, None] * K + k_offs[None, :], gathered_values, mask=m_mask[:, None])
