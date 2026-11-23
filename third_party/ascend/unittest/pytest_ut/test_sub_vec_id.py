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
import triton.language.extra.cann.extension as al


@triton.jit
def triton_matmul_exp(
    A_ptr,
    B_ptr,
    C_ptr,
    TBuff_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    sub_M: tl.constexpr,
):
    """function:
        1) The matrix matmul
        2) The matrix exp computation
    For example,
        1) [2, 3] @ [3, 5] -> [2, 5]

          [[-1.4310,  0.3144,  0.1952],        [[-0.1099,  0.7062,  0.6576,  1.3056,  0.3783],
           [ 1.6719, -0.2581, -1.0243]]   @     [ 0.9769, -0.6924,  0.4765,  1.1012,  0.3814]
                                                [-1.4598, -0.5444,  0.5582, -2.0959, -0.0568]]
          ->
          [[ 0.1795, -1.3346, -0.6822, -1.9311, -0.4324],
           [ 1.0593,  1.9171,  0.4047,  4.0454,  0.5921]]

        2) exp([2, 5])
          exp([[ 0.1795, -1.3346, -0.6822, -1.9311, -0.4324],   ->    [[ 1.1966,  0.2633,  0.5055,  0.1450,  0.6489],
              [ 1.0593,  1.9171,  0.4047,  4.0454,  0.5921]])          [ 2.8845,  6.8013,  1.4988, 57.1358,  1.8078]]
    """
    # Each program computes one element C[row, col] using 2D tl.dot
    row_matmul = tl.program_id(0)
    col = tl.program_id(1)

    # Build small 2D grids so tl.dot sees [M,K] x [K,N]
    offs_i = tl.arange(0, tl.constexpr(M))[:, None]  # [M,1] (row axis)
    offs_j = tl.arange(0, N)[None, :]  # [1,N] (col axis)
    offs_k = tl.arange(0, K)  # [K]

    # A row: [M, K]
    a_ptrs = A_ptr + (row_matmul + offs_i) * K + offs_k[None, :]
    a_vals = tl.load(a_ptrs)  # [M, K]

    # B column: [K, N]
    b_ptrs = B_ptr + offs_k[:, None] * N + (col + offs_j)
    b_vals = tl.load(b_ptrs)  # [K, N]

    tbuff_ptrs = TBuff_ptr + (row_matmul + offs_i) * N + (col + offs_j)

    # Dot: [M, K] @ [K, N] -> [M, N]
    acc_11 = tl.dot(a_vals, b_vals)  # [M, N]
    tl.store(tbuff_ptrs, acc_11)

    # Load Matrix [M/2, N]
    sub_vec_id = al.sub_vec_id()
    row_exp = row_matmul + sub_M * sub_vec_id
    offs_exp_i = tl.arange(0, tl.constexpr(sub_M))[:, None]  # [M/2, 1] (row axis)
    tbuff_exp_ptrs = TBuff_ptr + (row_exp + offs_exp_i) * N + (col + offs_j)
    acc_11_reload = tl.load(tbuff_exp_ptrs)
    # Pointer grid for the single output element: shape [M/2, N]
    c_ptrs = C_ptr + (row_exp + offs_exp_i) * N + (col + offs_j)

    # Store exp(acc) without scalar indexing
    tl.store(c_ptrs, tl.exp(acc_11_reload))


@pytest.mark.parametrize(
    "dtype, ashape, bshape",
    [
        # dtype, A-shape, B-shape
        ["float32", (2, 3), (3, 5)],
        ["float32", (2, 1), (1, 5)],
    ],
)
def test_sub_vec_id_1to2(dtype, ashape, bshape):
    """function:
        A 1:2 demo using sub_vec_id.
    1. The matrix computation and the vector computation unit each have their own independent Scalar scheduler units,
    deploying separately on cube core and vector core.
    2. Combine cube core and vector core in a certain ratio (1:2)

    For example, [2, 3] @ [3, 5] -> [2, 5] matrix matmul computation and matrix exp([2, 5]) computation
                 using sub_vec_id was used during the matrix exp.
    """
    M, K = ashape
    K2, N = bshape
    assert K == K2, "Inner dimensions must match"
    assert M % 2 == 0, "M dimensions must be divisible by 2"
    sub_M = int(M / 2)

    # Generate input tensors
    A = test_common.generate_tensor(ashape, dtype).npu()
    B = test_common.generate_tensor(bshape, dtype).npu()
    C = test_common.generate_tensor((M, N), dtype).npu()
    TBuff = test_common.generate_tensor((M, N), dtype).npu()

    # Run
    grid_matmul_exp = (1, )  # grid
    triton_matmul_exp[grid_matmul_exp](A, B, C, TBuff, M, N, K, sub_M)

    # Reference result
    C_ref = (A @ B).exp()
    test_common.validate_cmp(dtype, C, C_ref)


if __name__ == "__main__":
    test_sub_vec_id_1to2("float32", (2, 3), (3, 5))
