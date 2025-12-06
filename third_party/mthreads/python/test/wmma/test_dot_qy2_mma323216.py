import pytest
import torch
import triton
import triton.language as tl
import torch_musa


def get_resolution(dtype):
    atol_resolution_map = {
        torch.float16: 2.1e-3,
        torch.bfloat16: 1.4e-3,
    }
    rtol_resolution_map = {
        torch.float16: 1e-3,
        torch.bfloat16: 7.9e-3,
    }
    return atol_resolution_map.get(dtype, None), rtol_resolution_map.get(dtype, None)


def torch_matmul(A: torch.Tensor, B: torch.Tensor):
    return torch.mm(A, B)


@triton.jit
def matmul_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    A_block = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=A_ptr.dtype.element_ty)
    B_block = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=B_ptr.dtype.element_ty)
    C_block = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        mask_A = tl.where(offs_m[:, None] < M, k + offs_k[None, :] < K, False)
        mask_B = tl.where(k + offs_k[:, None] < K, offs_n[None, :] < N, False)
        A_block = tl.load(
            A_ptr + offs_m[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak,
            mask=mask_A,
            other=0.0,
        )
        B_block = tl.load(
            B_ptr + (k + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=mask_B,
            other=0.0,
        )
        C_block += tl.dot(A_block, B_block)

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(
        C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        C_block,
        mask=mask,
    )


def triton_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    block_size_m,
    block_size_n,
    block_size_k,
    num_warps,
):
    M, K = A.shape
    K, N = B.shape
    C = torch.empty((M, N), device="musa", dtype=torch.float32)
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]),
        triton.cdiv(N, meta["BLOCK_SIZE_N"]),
    )
    matmul_kernel[grid](
        A,
        B,
        C,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        num_warps=num_warps,
        num_stages=1,
        en_wmma=True,
    )
    return C


MNK = [
    (4096, 3072, 2048),
    (4096, 576, 2048),
    (4096, 4096, 512),
    (4096, 2048, 2048),
    (4096, 5632, 2048),
    (4096, 2048, 2816),
    (4096, 2816, 2048),
    (2048, 2816, 4096),
    (4096, 2048, 5632),
    (4096, 2048, 64),
    (5632, 2048, 4096),
    (2048, 4096, 2048),
    (4096, 512, 4096),
    (4096, 2048, 576),
    (4096, 2048, 3072),
    (3072, 2048, 4096),
    (2048, 2048, 4096),
]
BLOCK_WARP_CONFIG = [
    (32, 32, 16, 1),
    (64, 64, 32, 1),
    (64, 32, 16, 2),
    (32, 64, 16, 2),
    (64, 32, 32, 2),
    (128, 16, 16, 2),  # from real case of M1000
    (64, 64, 16, 4),
    (64, 64, 32, 4),
    (64, 64, 64, 4),
    (64, 64, 128, 4),
    (64, 64, 256, 4),
    (64, 64, 256, 8),
    (128, 64, 64, 8),
]

CONFIG = [(M, N, K, block_m, block_n, block_k, num_warps)
          for M, N, K in MNK
          for block_m, block_n, block_k, num_warps in BLOCK_WARP_CONFIG]
EXTRA_CONFIG = [
    # Basic cases with 1 warp and 1 MMA
    (32, 32, 16, 32, 32, 16, 1),
    (16, 32, 16, 32, 32, 16, 1),
    (32, 16, 16, 32, 32, 16, 1),
    (16, 16, 16, 32, 32, 16, 1),
    (8, 32, 16, 32, 32, 16, 1),
    (32, 8, 16, 32, 32, 16, 1),
    # Cases with 1 warp and multiple MMAs
    (32, 32, 32, 32, 32, 32, 1),
    (64, 32, 16, 64, 32, 16, 1),
    (32, 64, 16, 32, 64, 16, 1),
    # Cases with multiple warps
    (64, 32, 16, 64, 32, 16, 2),
    (32, 64, 16, 32, 64, 16, 2),
    (64, 32, 32, 64, 32, 32, 2),
    (32, 64, 32, 32, 64, 32, 2),
    # Edge cases
    (32, 32, 12, 32, 32, 16, 1),
    (32, 30, 16, 32, 32, 16, 1),
    (30, 32, 16, 32, 32, 16, 1),
    (32, 32, 30, 32, 32, 32, 1),
    (32, 30, 32, 32, 32, 32, 1),
    (30, 32, 32, 32, 32, 32, 1),
    # Special cases
    (16, 64, 16, 32, 32, 16, 2),
    (64, 16, 16, 32, 32, 16, 2),
    (16, 32, 16, 32, 32, 16, 1),
    (16, 64, 16, 32, 32, 16, 1),
    (32, 32, 8, 32, 32, 16, 1),
    (16, 16, 8, 32, 32, 16, 1),
    # Larger cases (will use integer comparison)
    (16, 512, 32, 32, 512, 32, 1),
    (16, 32, 64, 32, 32, 64, 1),
    (16, 32, 512, 32, 32, 512, 1),
    (32, 512, 32, 32, 512, 32, 1),
    (32, 32, 64, 32, 32, 64, 1),
    (32, 32, 512, 32, 32, 512, 1),
]
FINAL_CONFIG = CONFIG + EXTRA_CONFIG


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("M,N,K,block_m,block_n,block_k,num_warps", FINAL_CONFIG)
def test_matmul(dtype, M, N, K, block_m, block_n, block_k, num_warps):
    atol, rtol = get_resolution(dtype)
    capability = torch_musa.get_device_capability()
    assert (capability[0] == 2 and capability[1] == 2) and "Only used for QY2."
    A = torch.randn(M, K, device="musa", dtype=dtype)
    B = torch.randn(K, N, device="musa", dtype=dtype)
    C_triton = triton_matmul(A, B, block_m, block_n, block_k, num_warps)
    C_torch = torch_matmul(A, B).to(torch.float32)
    # print(f"C_torch: {C_torch}")
    # print(f"C_triton: {C_triton}")
    torch.testing.assert_close(C_torch, C_triton, atol=atol, rtol=rtol)
    print("Successfully!")
