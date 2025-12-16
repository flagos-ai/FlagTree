import numpy as np
import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel_tma(a_desc_ptr, b_desc_ptr, c_desc_ptr, M, N, K, BLOCK_SIZE_M: tl.constexpr,
                      BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn0 = pid_n * BLOCK_SIZE_N
    offs_bn1 = offs_bn0 + (BLOCK_SIZE_N // 2)
    offs_k = 0
    c0 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N // 2), dtype=tl.float32)
    c1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N // 2), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl._experimental_descriptor_load(a_desc_ptr, [offs_am, offs_k], [BLOCK_SIZE_M, BLOCK_SIZE_K], tl.float16)
        b0 = tl._experimental_descriptor_load(b_desc_ptr, [offs_k, offs_bn0], [BLOCK_SIZE_K, BLOCK_SIZE_N // 2],
                                              tl.float16)
        c0 = tl.dot(a, b0, acc=c0)
        offs_k += BLOCK_SIZE_K
    tl._experimental_descriptor_store(c_desc_ptr, c0.to(tl.float16), [offs_am, offs_bn0])
    offs_k = 0
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl._experimental_descriptor_load(a_desc_ptr, [offs_am, offs_k], [BLOCK_SIZE_M, BLOCK_SIZE_K], tl.float16)
        b1 = tl._experimental_descriptor_load(b_desc_ptr, [offs_k, offs_bn1], [BLOCK_SIZE_K, BLOCK_SIZE_N // 2],
                                              tl.float16)
        c1 = tl.dot(a, b1, acc=c1)
        offs_k += BLOCK_SIZE_K
    tl._experimental_descriptor_store(c_desc_ptr, c1.to(tl.float16), [offs_am, offs_bn1])


@pytest.mark.parametrize("M, N, K", [(4096, 3072, 2048), (4096, 576, 2048), (4096, 4096, 512), (4096, 2048, 2048),
                                     (4096, 5632, 2048), (4096, 2048, 2816), (4096, 2816, 2048), (2048, 2816, 4096),
                                     (4096, 2048, 5632), (5632, 2048, 4096), (2048, 4096, 2048), (4096, 512, 4096),
                                     (4096, 2048, 576), (4096, 2048, 3072), (3072, 2048, 4096), (2048, 2048, 4096)])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K", [(128, 256, 64)])
def test_experimental_tma_matmul(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K):
    device = "musa"
    A = torch.randn((M, K), dtype=torch.float16, device=device)
    B = torch.randn((K, N), dtype=torch.float16, device=device)
    C = torch.zeros((M, N), dtype=torch.float16, device=device)
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
    kernel = matmul_kernel_tma[(triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1, 1)](desc_a, desc_b, desc_c, M, N,
                                                                                          K, BLOCK_M, BLOCK_N, BLOCK_K,
                                                                                          num_warps=4, num_stages=1)
    ref_out = torch.matmul(A.to(torch.float32), B.to(torch.float32)).to(torch.float16)
    # print(f"ref_out, {ref_out}")
    # print(f"C, {C}")
    torch.testing.assert_close(ref_out, C, rtol=1e-2, atol=1e-2)
