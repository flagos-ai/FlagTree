import pytest

import numpy as np
import torch
import triton
import triton.language as tl
import torch_musa


def cpu_matmul(A: torch.Tensor, B: torch.Tensor):
    A_cpu = A.to(torch.float32).cpu()
    B_cpu = B.to(torch.float32).cpu()
    return torch.mm(A_cpu, B_cpu).to(torch.float32)


@triton.jit
def matmul_kernel_tma(a_desc_ptr, b_desc_ptr, c_desc_ptr,  #
                      M, N, K, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                      tl_dtype: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N
    offs_k = 0
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    dtype = tl_dtype
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl._experimental_descriptor_load(a_desc_ptr, [offs_am, offs_k], [BLOCK_SIZE_M, BLOCK_SIZE_K], dtype)
        b = tl._experimental_descriptor_load(b_desc_ptr, [offs_k, offs_bn], [BLOCK_SIZE_K, BLOCK_SIZE_N], dtype)
        accumulator = tl.dot(a, b, acc=accumulator)
        offs_k += BLOCK_SIZE_K
    accumulator = accumulator.to(tl.float32)
    tl._experimental_descriptor_store(c_desc_ptr, accumulator, [offs_am, offs_bn])


def get_triton_dtype(dtype):
    dtype_map = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32,
    }
    return dtype_map.get(dtype, None)


def create_tma_device_descriptor(tensor, block_m, block_n, device):
    assert tensor.dim() == 2, "TMA descriptor only supports 2D tensors"
    TMA_DESCRIPTOR_SIZE = 64
    desc = np.empty(TMA_DESCRIPTOR_SIZE, dtype=np.int8)
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(
        tensor.data_ptr(),
        tensor.shape[0],
        tensor.shape[1],
        block_m,
        block_n,
        tensor.element_size(),
        desc,
    )
    desc = torch.tensor(desc, device=device)
    return desc


def triton_matmul(
    desc_a,
    desc_b,
    desc_c,
    M,
    N,
    K,
    block_size_m,
    block_size_n,
    block_size_k,
    num_warps,
    num_stages,
    dtype,
):
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]),
        triton.cdiv(N, meta["BLOCK_SIZE_N"]),
    )
    tl_dtype = get_triton_dtype(dtype)
    kernel = matmul_kernel_tma[(triton.cdiv(M, block_size_m) * triton.cdiv(N, block_size_n), 1,
                                1)](desc_a, desc_b, desc_c, M, N, K, block_size_m, block_size_n, block_size_k, tl_dtype,
                                    num_warps=num_warps, num_stages=num_stages)
    return kernel


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("M, N, K", [(128, 128, 128), (256, 256, 256), (512, 512, 512), (1024, 1024, 1024),
                                     (2048, 2048, 2048), (4096, 4096, 4096)])
@pytest.mark.parametrize("block_m, block_n, block_k", [(32, 32, 32), (128, 128, 64)])
# @pytest.mark.parametrize("num_warps", [4, 8])
# @pytest.mark.parametrize("num_stages", [1, 2, 3, 4])
@pytest.mark.parametrize("num_warps", [4])
@pytest.mark.parametrize("num_stages", [1])
def test_matmul(dtype, M, N, K, block_m, block_n, block_k, num_warps, num_stages):
    device = "musa"
    capability = torch_musa.get_device_capability()
    assert (capability[0] == 3 and capability[1] == 1) and "Only used for PH1."
    A = torch.randn(M, K, device=device, dtype=dtype)
    B = torch.randn(K, N, device=device, dtype=dtype)
    C = torch.empty((M, N), device=device, dtype=torch.float32)

    desc_a = create_tma_device_descriptor(A, block_m, block_k, device=device)
    desc_b = create_tma_device_descriptor(B, block_k, block_n, device=device)
    desc_c = create_tma_device_descriptor(C, block_m, block_n, device=device)

    kernel = triton_matmul(desc_a, desc_b, desc_c, M, N, K, block_m, block_n, block_k, num_warps, num_stages, dtype)
    #print(kernel.asm['ttir'])
    # print(kernel.asm['ttgir'])
    # print(kernel.asm['llir'])

    C_cpu = cpu_matmul(A, B)
    C_gpu = C.cpu()
    # print(f"C_cpu: {C_cpu}")
    # print(f"C_gpu: {C_gpu}")
    assert torch.allclose(C_cpu, C_gpu, atol=1e-1)
