import logging
import pytest
import torch
import torch_npu
import triton
import triton.language as tl

import test_common
from test_common import TestUtils, avoid_not_support, get_dtype_size


@triton.jit
def matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        acc_dtype: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    matxa_ptr_in = tl.make_block_ptr(a_ptr,
                                    (M, K),
                                    (K, 1),
                                    (0, 0),
                                    (M, K),
                                    order=(1, 0))
    matxb_ptr_in = tl.make_block_ptr(b_ptr,
                                    (K, N),
                                    (N, 1),
                                    (0, 0),
                                    (K, N),
                                    order=(1, 0))
    matxc_ptr_in = tl.make_block_ptr(c_ptr,
                                    (M, N),
                                    (N, 1),
                                    (0, 0),
                                    (M, N),
                                    order=(1, 0))

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)
    a = tl.load(matxa_ptr_in)
    b = tl.load(matxb_ptr_in)
    accumulator = tl.dot(a, b, accumulator, out_dtype=acc_dtype)
    c = accumulator.to(c_ptr.dtype.element_ty)
    tl.store(matxc_ptr_in, c)


@avoid_not_support('matmul')
@pytest.mark.parametrize('shape', [(16, 32)])
@pytest.mark.parametrize('dtype', ['float32'])
def test_matmul(shape, dtype):
    M, N, K = shape[0], shape[0], shape[1]

    BLOCK_M, BLOCK_N, BLOCK_K = M, N, K
    a = test_common.generate_tensor((M, K), dtype)
    b = test_common.generate_tensor((K, N), dtype)

    triton_res = torch.zeros((M, N), dtype=eval('torch.' + dtype)).npu()
    accumulator_type = tl.float32
    
    matmul_kernel[1, ](a.npu(), b.npu(), triton_res, M, N, K, accumulator_type,
                        BLOCK_M, BLOCK_N, BLOCK_K, enable_nd2nz_on_vector=False)

    print("PASSED")
