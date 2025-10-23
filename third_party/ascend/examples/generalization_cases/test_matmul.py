import logging
import pytest
import torch
import torch_npu
import triton
import triton.language as tl

import acc_util
import test_common
from test_common import TestUtils, avoid_not_support, get_dtype_size


@triton.jit
def matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        acc_dtype: tl.constexpr,
        stride_am: tl.constexpr,
        stride_ak: tl.constexpr,
        stride_bk: tl.constexpr,
        stride_bn: tl.constexpr,
        stride_cm: tl.constexpr,
        stride_cn: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M))
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator, out_dtype=acc_dtype)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c = accumulator.to(c_ptr.dtype.element_ty)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

@avoid_not_support('matmul')
@pytest.mark.parametrize('shape', TestUtils.test_shape2d)
@pytest.mark.parametrize('dtype', TestUtils.dtype_list)
def test_matmul(shape, dtype):
    M, N, K = shape[0], shape[0], shape[1]
    # 32byte/Dtype_bytes
    kalign = 32 // get_dtype_size(dtype)  
    BLOCK_M, BLOCK_N, BLOCK_K = min(max(M, 16), 32), min(max(N, 16), 32), min(max(K, kalign), 32)
    a = test_common.generate_tensor((M, K), dtype)
    b = test_common.generate_tensor((K, N), dtype)

    if dtype == "int8":
        triton_res = torch.zeros((M, N), dtype=torch.int32).npu()
        accumulator_type = tl.int32
    else:
        triton_res = torch.zeros((M, N), dtype=eval('torch.' + dtype)).npu()
        accumulator_type = tl.float32
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    
    matmul_kernel[grid](a.npu(), b.npu(), triton_res, M, N, K, accumulator_type,
                        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                        triton_res.stride(0), triton_res.stride(1),
                        BLOCK_M, BLOCK_N, BLOCK_K)

    a_gold = a.to(torch.float32)
    b_gold = b.to(torch.float32)
    cpu_res = torch.mm(a_gold, b_gold)

    if dtype == "int8":
        # torch_npu do not support int8 matmul
        a_npu = a.npu().to(torch.float32)
        b_npu = b.npu().to(torch.float32)
        torch_res = torch.mm(a_npu, b_npu)
        triton_res = triton_res.to(torch.float32)
    else:
        a_npu = a.npu()
        b_npu = b.npu()
        torch_res = torch.mm(a_npu, b_npu)

    try:
        print("starting compare of cpu vs triton:")
        acc_util.assert_close(cpu_res, triton_res)
    except Exception as e:
        print(e)
        print("starting compare of cpu vs triton vs torch_npu:")
        acc_util.benchmark_compare_close(cpu_res, triton_res, torch_res)
    print("PASSED")


@avoid_not_support('matmul')
@pytest.mark.parametrize('batch', TestUtils.batch)
@pytest.mark.parametrize('shape', TestUtils.test_shape2d)
@pytest.mark.parametrize('dtype', TestUtils.dtype_list)
def test_batch_matmul(shape, dtype, batch):
    M, N, K = shape[0], shape[0], shape[1]
    # 32byte/Dtype_bytes
    kalign = 32 // get_dtype_size(dtype)  
    BLOCK_M, BLOCK_N, BLOCK_K = min(max(M, 16), 32), min(max(N, 16), 32), min(max(K, kalign), 32)

    aa = test_common.generate_tensor((batch, M, K), dtype)
    bb = test_common.generate_tensor((batch, K, N), dtype)

    if dtype == "int8":
        final_triton_res = torch.zeros((batch, M, N), dtype=torch.int32).npu()
        accumulator_type = tl.int32
    else:
        final_triton_res = torch.zeros((batch, M, N), dtype=eval('torch.' + dtype)).npu()
        accumulator_type = tl.float32
    
    for i in range(0, batch):
        if dtype == "int8":
            triton_res = torch.zeros((M, N), dtype=torch.int32).npu()
        else:
            triton_res = torch.zeros((M, N), dtype=eval('torch.' + dtype)).npu()
        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
        a = aa[i]
        b = bb[i]
        matmul_kernel[grid](a.npu(), b.npu(), triton_res, M, N, K, accumulator_type,
                        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                        triton_res.stride(0), triton_res.stride(1),
                        BLOCK_M, BLOCK_N, BLOCK_K)
        final_triton_res[i] = triton_res

    a_gold = aa.to(torch.float32)
    b_gold = bb.to(torch.float32)
    cpu_res = torch.bmm(a_gold, b_gold)

    if dtype == "int8":
        a_npu = aa.npu().to(torch.float32)
        b_npu = bb.npu().to(torch.float32)
        final_triton_res = final_triton_res.to(torch.float32)
    else:
        a_npu = aa.npu()
        b_npu = bb.npu()
    torch_res = torch.bmm(a_npu, b_npu)
    
    try:
        print("starting compare of cpu vs triton:")
        acc_util.assert_close(cpu_res, final_triton_res)
    except Exception as e:
        print(e)
        print("starting compare of cpu vs triton vs torch_npu:")
        acc_util.benchmark_compare_close(cpu_res, final_triton_res, torch_res)
    print("PASSED")


if __name__ == "__main__":
    test_matmul((16, 32), 'float32')
    test_matmul((16, 32), 'int8')
    test_batch_matmul(2, (16, 32), 'float32')
    test_batch_matmul(2, (16, 32), 'int8')
