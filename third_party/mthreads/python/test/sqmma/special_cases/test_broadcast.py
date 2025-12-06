import numpy as np
import pytest
import torch

import triton
import triton.language as tl


@triton.jit
def broadcat_kernel_axis_0(a_desc_ptr, b_desc_ptr, c_desc_ptr, R_ptr,  #
                           M, N, K, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N
    offs_k = 0

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    R_block = tl.load(R_ptr + offs_n, mask=offs_n < N, other=0.0)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl._experimental_descriptor_load(a_desc_ptr, [offs_am, offs_k], [BLOCK_SIZE_M, BLOCK_SIZE_K], tl.float16)
        b = tl._experimental_descriptor_load(b_desc_ptr, [offs_k, offs_bn], [BLOCK_SIZE_K, BLOCK_SIZE_N], tl.float16)
        accumulator = tl.dot(a, b, acc=accumulator)
        offs_k += BLOCK_SIZE_K
    accumulator = accumulator - R_block[None, :]
    accumulator = accumulator.to(tl.float16)
    tl._experimental_descriptor_store(c_desc_ptr, accumulator, [offs_am, offs_bn])


@triton.jit
def broadcat_kernel_axis_1(a_desc_ptr, b_desc_ptr, c_desc_ptr, R_ptr,  #
                           M, N, K, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N
    offs_k = 0

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    R_block = tl.load(R_ptr + offs_m, mask=offs_m < M, other=0.0)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl._experimental_descriptor_load(a_desc_ptr, [offs_am, offs_k], [BLOCK_SIZE_M, BLOCK_SIZE_K], tl.float16)
        b = tl._experimental_descriptor_load(b_desc_ptr, [offs_k, offs_bn], [BLOCK_SIZE_K, BLOCK_SIZE_N], tl.float16)
        accumulator = tl.dot(a, b, acc=accumulator)
        offs_k += BLOCK_SIZE_K
    accumulator = accumulator - R_block[:, None]
    accumulator = accumulator.to(tl.float16)
    tl._experimental_descriptor_store(c_desc_ptr, accumulator, [offs_am, offs_bn])


@pytest.mark.parametrize("num_stages", [1])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K", [
    (32, 32, 32),
])
def test_boradcast_axis_0(num_stages, BLOCK_M, BLOCK_N, BLOCK_K):
    device = "musa"
    M, N, K = 64, 32, 32
    torch.manual_seed(42)
    A = torch.randn((M, K), dtype=torch.float16, device=device)
    B = torch.arange(N, dtype=torch.float16, device=device).repeat(1, K).reshape((K, N))
    C = torch.zeros((M, N), dtype=torch.float16, device=device)
    R = torch.randn(N, device='musa', dtype=torch.float16)
    TMA_SIZE = 64
    desc_a = np.empty(TMA_SIZE, dtype=np.int8)
    desc_b = np.empty(TMA_SIZE, dtype=np.int8)
    desc_c = np.empty(TMA_SIZE, dtype=np.int8)
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(A.data_ptr(), M, K, BLOCK_M, BLOCK_K, A.element_size(),
                                                              desc_a)
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(B.data_ptr(), K, N, BLOCK_K, BLOCK_N, B.element_size(),
                                                              desc_b)
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(C.data_ptr(), M, N, BLOCK_M, BLOCK_N, C.element_size(),
                                                              desc_c)
    desc_a = torch.tensor(desc_a, device=device)
    desc_b = torch.tensor(desc_b, device=device)
    desc_c = torch.tensor(desc_c, device=device)
    kernel = broadcat_kernel_axis_0[(triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1,
                                     1)](desc_a, desc_b, desc_c, R, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_warps=4,
                                         num_stages=num_stages)
    #print(kernel.asm['ttir'])
    # print(kernel.asm['ttgir'])
    # print(kernel.asm['llir'])
    torch.set_printoptions(threshold=float('inf'), linewidth=800, edgeitems=50, sci_mode=False, precision=1)

    # print(R)
    ref_out = torch.matmul(A.to(torch.float32), B.to(torch.float32)).to(torch.float16) - R[None, :]
    torch.testing.assert_close(ref_out, C, rtol=1e-3, atol=1e-3)
    print("Successfully!")
    if BLOCK_M >= 64 and BLOCK_N >= 64:
        assert "tme" in kernel.asm["llir"]


@pytest.mark.parametrize("num_stages", [1])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K", [
    (32, 32, 32),
])
def test_boradcast_axis_1(num_stages, BLOCK_M, BLOCK_N, BLOCK_K):
    device = "musa"
    M, N, K = 64, 32, 32
    torch.manual_seed(42)
    A = torch.ones((M, K), dtype=torch.float16, device=device)
    B = torch.arange(N, dtype=torch.float16, device=device).repeat(1, K).reshape((K, N))
    C = torch.zeros((M, N), dtype=torch.float16, device=device)
    R = torch.randn(M, device='musa', dtype=torch.float16)
    TMA_SIZE = 64
    desc_a = np.empty(TMA_SIZE, dtype=np.int8)
    desc_b = np.empty(TMA_SIZE, dtype=np.int8)
    desc_c = np.empty(TMA_SIZE, dtype=np.int8)
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(A.data_ptr(), M, K, BLOCK_M, BLOCK_K, A.element_size(),
                                                              desc_a)
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(B.data_ptr(), K, N, BLOCK_K, BLOCK_N, B.element_size(),
                                                              desc_b)
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(C.data_ptr(), M, N, BLOCK_M, BLOCK_N, C.element_size(),
                                                              desc_c)
    desc_a = torch.tensor(desc_a, device=device)
    desc_b = torch.tensor(desc_b, device=device)
    desc_c = torch.tensor(desc_c, device=device)
    kernel = broadcat_kernel_axis_1[(triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1,
                                     1)](desc_a, desc_b, desc_c, R, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_warps=4,
                                         num_stages=num_stages)
    # print(kernel.asm['ttir'])
    # print(kernel.asm['ttgir'])
    # print(kernel.asm['llir'])
    torch.set_printoptions(threshold=float('inf'), linewidth=800, edgeitems=50, sci_mode=False, precision=1)

    # print(R)
    ref_out = torch.matmul(A.to(torch.float32), B.to(torch.float32)).to(torch.float16) - R[:, None]
    torch.testing.assert_close(ref_out, C, rtol=1e-3, atol=1e-3)
    print("Successfully!")
    if BLOCK_M >= 64 and BLOCK_N >= 64:
        assert "tme" in kernel.asm["llir"]


# test_boradcast_axis_0(1, 32, 32, 32)
# test_boradcast_axis_0(2, 32, 32, 32) # segment fault
# test_boradcast_axis_1(1, 32, 32, 32)
# test_boradcast_axis_1(2, 32, 32, 32) # segment fault
