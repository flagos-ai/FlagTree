import numpy as np
import pytest
import torch

import triton
import triton.language as tl


def get_tolerance(dtype):
    tolerance_map = {
        torch.float16: {"rtol": 1e-3, "atol": 1e-3},
        torch.bfloat16: {"rtol": 7.9e-3, "atol": 2e-3},
        torch.float8_e4m3fn: {"rtol": 1.25e-1, "atol": 4.5e-2},
    }
    return tolerance_map.get(dtype, {"rtol": 1e-3, "atol": 1e-3})


def get_triton_dtype(dtype):
    dtype_map = {
        torch.float16: tl.float16,
        torch.float32: tl.float32,
        torch.bfloat16: tl.bfloat16,
        torch.float8_e4m3fn: tl.float8e4nv,
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


@triton.jit
def matmul_kernel_tmeLoad_sqmma_tmeStore(
    a_desc_ptr,
    b_desc_ptr,
    c_desc_ptr,  #
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    dtype: tl.constexpr,
    save_dtype: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N
    offs_k = 0

    # handle an AttributeError
    dtype = dtype

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl._experimental_descriptor_load(a_desc_ptr, [offs_am, offs_k], [BLOCK_SIZE_M, BLOCK_SIZE_K], dtype)
        b = tl._experimental_descriptor_load(b_desc_ptr, [offs_k, offs_bn], [BLOCK_SIZE_K, BLOCK_SIZE_N], dtype)
        accumulator = tl.dot(a, b, acc=accumulator)
        offs_k += BLOCK_SIZE_K
    # f32->bf16 implemention in triton could ran out of registers when block_size > 128*128*64,
    # thus convert f32 to f16 first in triton, then convert f16 to dtype after triton kernel.
    accumulator = accumulator.to(save_dtype)
    tl._experimental_descriptor_store(c_desc_ptr, accumulator, [offs_am, offs_bn])


@pytest.mark.parametrize("num_stages", [4])
@pytest.mark.parametrize(
    "M, N, K",
    [
        (4096, 3072, 2048),
        (4096, 576, 2048),
        (4096, 4096, 512),
        (4096, 2048, 2048),
        (4096, 5632, 2048),
        (4096, 2048, 2816),
        (4096, 2816, 2048),
        (2048, 2816, 4096),
        (4096, 2048, 5632),
        (5632, 2048, 4096),
        (2048, 4096, 2048),
        (4096, 512, 4096),
        (4096, 2048, 576),
        (4096, 2048, 3072),
        (3072, 2048, 4096),
        (2048, 2048, 4096),
    ],
)
@pytest.mark.parametrize(
    "BLOCK_M, BLOCK_N, BLOCK_K, num_warps",
    [
        (32, 32, 32, 4),
        (128, 64, 64, 4),
        (128, 128, 64, 4),
        (128, 256, 64, 4),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float8_e4m3fn])
def test_tmeLoad_sqmma_tmeStore(num_stages, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_warps, dtype):
    # print(
    #     f"num_stage: {num_stages}\tM: {M}\tN: {N}\tK: {K}\tBLOCK_M: {BLOCK_M}\tBLOCK_N: {BLOCK_N}\tBLOCK_K: {BLOCK_K}\tdtype: {dtype}",
    #     end="\t\t",
    # )

    device = "musa"
    tolerance = get_tolerance(dtype)
    rtol = tolerance["rtol"]
    atol = tolerance["atol"]
    # save_dtype = dtype if dtype != torch.bfloat16 else torch.float16
    save_dtype = torch.float16
    torch.manual_seed(42)
    A = torch.randn((M, K), dtype=torch.float32, device=device).to(dtype)
    B = torch.randn((K, N), dtype=torch.float32, device=device).to(dtype)
    C = torch.zeros((M, N), dtype=torch.float32, device=device).to(save_dtype)

    desc_a = create_tma_device_descriptor(A, BLOCK_M, BLOCK_K, device=device)
    desc_b = create_tma_device_descriptor(B, BLOCK_K, BLOCK_N, device=device)
    desc_c = create_tma_device_descriptor(C, BLOCK_M, BLOCK_N, device=device)

    kernel = matmul_kernel_tmeLoad_sqmma_tmeStore[(triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1, 1)](
        desc_a,
        desc_b,
        desc_c,
        # C,
        M,
        N,
        K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        get_triton_dtype(dtype),
        get_triton_dtype(save_dtype),
        num_warps=num_warps,
        num_stages=num_stages,
    )
    torch.set_printoptions(threshold=float("inf"), linewidth=800, edgeitems=50, sci_mode=False, precision=1)

    C = C.to(dtype)
    ref_out = torch.matmul(A, B).to(dtype)
    torch.testing.assert_close(ref_out.to(torch.float32), C.to(torch.float32), rtol=rtol, atol=atol)
    # print("Successfully!")
