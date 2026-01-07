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
import pytest
import test_common

# eg: pytest -v test_matmul_exp.py::test_matmul_exp
#############################


@triton.jit
def triton_matmul_exp(A_ptr, B_ptr, C_ptr, TBuff_ptr,
                      M, N, K: tl.constexpr):
    # Each program computes one element C[row, col] using 2D tl.dot
    row = tl.program_id(0)
    col = tl.program_id(1)

    # Build small 2D grids so tl.dot sees [1,K] x [K,1]
    offs_i = tl.arange(0, 1)[:, None]         # [1,1] (row axis)
    offs_j = tl.arange(0, 1)[None, :]         # [1,1] (col axis)
    offs_k = tl.arange(0, K)                  # [K]

    # A row: [1, K]
    a_ptrs = A_ptr + (row + offs_i) * K + offs_k[None, :]
    a_vals = tl.load(a_ptrs)                  # [1, K]

    # B column: [K, 1]
    b_ptrs = B_ptr + offs_k[:, None] * N + (col + offs_j)
    b_vals = tl.load(b_ptrs)                  # [K, 1]

    tbuff_ptrs = TBuff_ptr + (row + offs_i) * N + (col + offs_j)

    # Dot: [1, K] @ [K, 1] -> [1, 1]
    acc_11 = tl.dot(a_vals, b_vals)           # [1, 1]
    tl.store(tbuff_ptrs, acc_11)
    
    tl.sync_block_set("cube", "vector", 5)
    tl.sync_block_wait("cube", "vector", 5)
    
    acc_11_reload = tl.load(tbuff_ptrs)
    # Pointer grid for the single output element: shape [1,1]
    c_ptrs = C_ptr + (row + offs_i) * N + (col + offs_j)

    # Store exp(acc) without scalar indexing
    tl.store(c_ptrs, tl.exp(acc_11_reload))


@pytest.mark.parametrize(
    'dtype, ashape, bshape',
    [
        # dtype, A-shape, B-shape
        ['float32', (4, 4), (4, 4)],
        ['float32', (2, 3), (3, 5)],
    ]
)
def test_matmul_exp(dtype, ashape, bshape):
    M, K = ashape
    K2, N = bshape
    assert K == K2, "Inner dimensions must match"

    # generate input tensors
    A = test_common.generate_tensor(ashape, dtype).npu()
    B = test_common.generate_tensor(bshape, dtype).npu()
    C = test_common.generate_tensor((M, N), dtype).npu()
    TBuff = test_common.generate_tensor((M, N), dtype).npu()

    # run kernel
    grid = (M, N)  # one program per output element
    triton_matmul_exp[grid](A, B, C, TBuff, M, N, K)

    # reference result
    C_ref = (A @ B).exp()

    # compare
    test_common.validate_cmp(dtype, C, C_ref)

if __name__ == "__main__":
    test_matmul_exp('float32', (4, 4), (4, 4))
