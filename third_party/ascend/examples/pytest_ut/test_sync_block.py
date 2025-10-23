import triton
import triton.language as tl
import pytest
import test_common

# eg: pytest -v test_matmul_exp.py::test_matmul_exp
#############################


@triton.jit
def triton_matmul_exp(A_ptr, B_ptr, C_ptr,
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

    
    tl.sync_block_set("cube", "vector", 5)
    # Dot: [1, K] @ [K, 1] -> [1, 1]
    acc_11 = tl.dot(a_vals, b_vals)           # [1, 1]
    tl.sync_block_wait("cube", "vector", 5)

    # Pointer grid for the single output element: shape [1,1]
    c_ptrs = C_ptr + (row + offs_i) * N + (col + offs_j)

    # Store exp(acc) without scalar indexing
    tl.store(c_ptrs, tl.exp(acc_11))


@pytest.mark.parametrize(
    'param_list',
    [
        # dtype, A-shape, B-shape
        ['float32', (4, 4), (4, 4)],
        ['float32', (2, 3), (3, 5)],
    ]
)
def test_matmul_exp(param_list):
    dtype, ashape, bshape = param_list
    M, K = ashape
    K2, N = bshape
    assert K == K2, "Inner dimensions must match"

    # generate input tensors
    A = test_common.generate_tensor(ashape, dtype).npu()
    B = test_common.generate_tensor(bshape, dtype).npu()
    C = test_common.generate_tensor((M, N), dtype).npu()

    # run kernel
    grid = (M, N)  # one program per output element
    triton_matmul_exp[grid](A, B, C, M, N, K)

    # reference result
    C_ref = (A @ B).exp()

    # compare
    test_common.validate_cmp(dtype, C, C_ref)

if __name__ == "__main__":
    test_matmul_exp('float32', (4, 4), (4, 4))