import numpy as np
import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def addmm_kernel(a_desc_ptr, b_desc_ptr, bias_desc_ptr, c_desc_ptr, M, N, K, BLOCK_SIZE_M: tl.constexpr,
                 BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, alhpa: tl.constexpr, beta: tl.constexpr,
                 ab_type: tl.constexpr, c_type: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N
    offs_k = 0
    input_type = ab_type
    output_type = c_type
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl._experimental_descriptor_load(a_desc_ptr, [offs_am, offs_k], [BLOCK_SIZE_M, BLOCK_SIZE_K], input_type)
        b = tl._experimental_descriptor_load(b_desc_ptr, [offs_k, offs_bn], [BLOCK_SIZE_K, BLOCK_SIZE_N], input_type)
        accumulator = tl.dot(a, b, acc=accumulator)
        offs_k += BLOCK_SIZE_K
    bias = tl._experimental_descriptor_load(bias_desc_ptr, [offs_am, offs_bn], [BLOCK_SIZE_M, BLOCK_SIZE_N], input_type)
    result = (alhpa * accumulator.to(output_type) + beta * bias.to(output_type)).to(output_type)
    tl._experimental_descriptor_store(c_desc_ptr, result, [offs_am, offs_bn])


def get_triton_type(elem_type):
    type_map = {torch.float16: tl.float16, torch.bfloat16: tl.bfloat16, torch.float8_e4m3fn: tl.float8e4nv}
    return type_map.get(elem_type, None)


def get_rtol(elem_type):
    rtol_map = {
        torch.float16: 1e-3,
        torch.bfloat16: 7.9e-3,
        torch.float8_e4m3fn: 1.25e-1,
    }
    return rtol_map.get(elem_type, None)


def get_atol(elem_type):
    atol_map = {
        torch.float16: 1e-3,
        torch.bfloat16: 1e-3,
        torch.float8_e4m3fn: 1.25e-1,
    }
    return atol_map.get(elem_type, None)


elem_type_list = [torch.float16, torch.bfloat16, torch.float8_e4m3fn]
alpha_beta_list = [(0.5, 0.5)]
size_list = [(4096, 4096, 4096), (2048, 2048, 2048), (1024, 1024, 1024), (384, 384, 384)]
block_size_list = [(128, 128, 64, 4)]
num_stages_list = [1]
test_params = [(elem_type, alpha, beta, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)
               for elem_type in elem_type_list
               for (alpha, beta) in alpha_beta_list
               for (M, N, K) in size_list
               for (BLOCK_M, BLOCK_N, BLOCK_K, num_warps) in block_size_list
               for num_stages in num_stages_list
               if M % BLOCK_M == 0 and N % BLOCK_N == 0 and K % BLOCK_K == 0]


@pytest.mark.parametrize("elem_type, alpha, beta, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages",
                         test_params)
def test_addmm_kernel_perf(elem_type, alpha, beta, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages):
    device = "musa"
    ab_type = elem_type
    c_type = elem_type if (elem_type != torch.bfloat16) else torch.float16
    A = torch.randn((M, K), dtype=torch.float16, device=device).to(ab_type)
    B = torch.randn((K, N), dtype=torch.float16, device=device).to(ab_type)
    Bias = torch.randn((M, N), dtype=torch.float16, device=device).to(ab_type)
    C = torch.zeros((M, N), dtype=torch.float16, device=device).to(c_type)
    TMA_SIZE = 64
    desc_a = np.empty(TMA_SIZE, dtype=np.int8)
    desc_b = np.empty(TMA_SIZE, dtype=np.int8)
    desc_bias = np.empty(TMA_SIZE, dtype=np.int8)
    desc_c = np.empty(TMA_SIZE, dtype=np.int8)
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(A.data_ptr(), M, K, BLOCK_M, BLOCK_K, A.element_size(),
                                                              desc_a)
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(B.data_ptr(), K, N, BLOCK_K, BLOCK_N, B.element_size(),
                                                              desc_b)
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(Bias.data_ptr(), M, N, BLOCK_M, BLOCK_N,
                                                              Bias.element_size(), desc_bias)
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(C.data_ptr(), M, N, BLOCK_M, BLOCK_N, C.element_size(),
                                                              desc_c)
    desc_a = torch.tensor(desc_a, device=device)
    desc_b = torch.tensor(desc_b, device=device)
    desc_bias = torch.tensor(desc_bias, device=device)
    desc_c = torch.tensor(desc_c, device=device)
    ms_torch = triton.musa_testing.do_bench(lambda: torch.addmm(Bias, A, B, alpha=alpha, beta=beta))
    ms_triton = triton.musa_testing.do_bench(
        lambda: addmm_kernel[(triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1, 1)]
        (desc_a, desc_b, desc_bias, desc_c, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, alpha, beta, get_triton_type(ab_type),
         get_triton_type(c_type), num_warps=num_warps, num_stages=num_stages))
    print("\n")
    print(
        f"elem_type={elem_type}, alpha={alpha}, beta={beta}, M={M}, N={N}, K={K}, BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}, BLOCK_K={BLOCK_K}, num_warps={num_warps}, num_stages={num_stages}"
    )
    print(f"torch addmm latency: {ms_torch:.4f} ms")
    print(f"triton addmm kernel latency: {ms_triton:.4f} ms")
    print(f"triton2torch speed_up: {round(ms_torch / ms_triton, 6)}")
    ref_out = torch.addmm(Bias, A, B, alpha=alpha, beta=beta)
    torch.testing.assert_close(ref_out.to(torch.float16), C.to(torch.float16), rtol=get_rtol(elem_type),
                               atol=get_atol(elem_type))
