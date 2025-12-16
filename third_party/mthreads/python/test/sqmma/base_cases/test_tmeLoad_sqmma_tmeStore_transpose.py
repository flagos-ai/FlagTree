import numpy as np
import pytest
import torch

import triton
import triton.language as tl


def get_resolution(dtype):
    rtol_resolution_map = {
        torch.float16:
        1e-3,
        torch.bfloat16:
        7.9e-3,
        # f32->fp8 implemention in triton could ran out of registers when block_size > 256*256*64,
        # thus convert f32 to f16 first in triton, then convert f16 to dtype after triton kernel.
        # but this cause some precision problem, thus here use 1.25e-1 as resolution of fp8.
        # torch.float8_e4m3fn: 1e-3,
        torch.float8_e4m3fn:
        1.25e-1,
    }
    return rtol_resolution_map.get(dtype, None)


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
    desc_np = np.empty(TMA_DESCRIPTOR_SIZE, dtype=np.int8)
    shapes = [tensor.shape[0], tensor.shape[1]]
    if not tensor.is_contiguous():
        shapes.reverse()
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(
        tensor.data_ptr(),
        shapes[0],
        shapes[1],
        block_m,
        block_n,
        tensor.element_size(),
        desc_np,
    )
    desc = torch.tensor(desc_np, device=device)
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
    is_transpose_a: tl.constexpr = False,
    is_transpose_b: tl.constexpr = False,
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
        a = tl._experimental_descriptor_load(
            a_desc_ptr,
            [offs_am, offs_k],
            [BLOCK_SIZE_M, BLOCK_SIZE_K],
            dtype,
            is_transpose_a,
        )
        b = tl._experimental_descriptor_load(
            b_desc_ptr,
            [offs_k, offs_bn],
            [BLOCK_SIZE_K, BLOCK_SIZE_N],
            dtype,
            is_transpose_b,
        )
        accumulator = tl.dot(a, b, acc=accumulator)
        offs_k += BLOCK_SIZE_K
    # f32->bf16 implemention in triton could ran out of registers when block_size > 128*128*64,
    # thus convert f32 to f16 first in triton, then convert f16 to dtype after triton kernel.
    accumulator = accumulator.to(save_dtype)
    tl._experimental_descriptor_store(c_desc_ptr, accumulator, [offs_am, offs_bn])


@pytest.mark.parametrize("num_stages", [1])
@pytest.mark.parametrize(
    "M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_warps",
    [
        # basic cases
        (32, 32, 32, 32, 32, 32, 4),
        (128, 128, 128, 32, 32, 64, 4),
        (128, 128, 128, 32, 64, 32, 4),
        (128, 128, 128, 64, 32, 32, 4),
        (128, 128, 64, 32, 32, 32, 4),
        (128, 128, 64, 32, 32, 64, 4),
        (128, 128, 64, 32, 64, 32, 4),
        (128, 128, 64, 64, 32, 32, 4),
        (128, 128, 64, 128, 128, 64, 4),

        # multi-replicate
        # TODO: waiting for branch dev_sqmma_main to merge

        # split leading dimension
        # TODO: waiting for branch dev_sqmma_main to merge

        # LSU + SQMMA + transpose
        # TODO

        # real case from Qwen3-8B
        (10, 6144, 4096, 128, 128, 64, 16),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("is_transposeA", [True, False])
@pytest.mark.parametrize("is_transposeB", [True, False])
def test_tmeLoad_sqmma_tmeStore(
    num_stages,
    M,
    N,
    K,
    BLOCK_M,
    BLOCK_N,
    BLOCK_K,
    num_warps,
    dtype,
    is_transposeA,
    is_transposeB,
):
    if M < 32 and dtype == torch.float8_e4m3fn:
        pytest.skip("create_tma_device_descriptor could be core dumped.")

    print(
        f"num_stage: {num_stages}\tM: {M}\tN: {N}\tK: {K}\tBLOCK_M: {BLOCK_M}\tBLOCK_N: {BLOCK_N}\tBLOCK_K: {BLOCK_K}\tdtype: {dtype}",
        end="\t\t",
    )

    device = "musa"
    rtol = get_resolution(dtype)
    # save_dtype = dtype if dtype != torch.bfloat16 else torch.float16
    save_dtype = torch.float16
    torch.manual_seed(42)

    if is_transposeA:
        A = torch.randn((K, M), dtype=torch.float32, device=device).to(dtype)
        # A = torch.arange((K*M), device=device).reshape((K, M)).to(dtype) # used for debug
        # A = torch.ones((K, M), dtype=torch.float32, device=device).to(dtype)

        A = A.transpose(1, 0)
    else:
        A = torch.randn((M, K), dtype=torch.float32, device=device).to(dtype)
        # A = torch.arange((M*K), device=device).reshape((M, K)).to(dtype)
        # A = torch.eye(max(M, K), device=device).to(dtype)[:M, :K].contiguous()
    if is_transposeB:
        B = torch.randn((N, K), dtype=torch.float32, device=device).to(dtype)
        # B = torch.arange((N*K), device=device).reshape((N, K)).to(dtype).contiguous()

        B = B.transpose(1, 0)
    else:
        B = torch.randn((K, N), dtype=torch.float32, device=device).to(dtype)
        # B = torch.eye(max(K, N), device=device).to(dtype)[:K, :N].contiguous()
    print(f"\nA.shape : {A.shape}, A.stride: {A.stride()}, A.is_contiguous: {A.is_contiguous()}")
    print(f"\nB.shape : {B.shape}, B.stride: {B.stride()}, B.is_contiguous: {B.is_contiguous()}")
    C = torch.zeros((M, N), dtype=torch.float32, device=device).to(save_dtype)

    desc_a = create_tma_device_descriptor(A, BLOCK_M, BLOCK_K, device=device)
    desc_b = create_tma_device_descriptor(B, BLOCK_K, BLOCK_N, device=device)
    desc_c = create_tma_device_descriptor(C, BLOCK_M, BLOCK_N, device=device)

    kernel = matmul_kernel_tmeLoad_sqmma_tmeStore[(triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1, 1)](
        desc_a,
        desc_b,
        desc_c,
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
        is_transpose_a=is_transposeA,
        is_transpose_b=is_transposeB,
    )
    # print(kernel.asm['ttir'])
    # print(kernel.asm['ttgir'])
    # print(kernel.asm['llir'])
    # print("C.numel() :", C.numel())
    torch.set_printoptions(threshold=float("inf"), linewidth=800, edgeitems=50, sci_mode=False, precision=1)

    C = C.to(dtype)
    ref_out = torch.matmul(A, B).to(dtype)

    # print tensors
    # step = 16
    # for i in range(N // step):
    #     print(f"C[:, {i*step}:{(i+1)*step}]: \n{C[:, i*step:(i+1)*step]}")
    #     print(f"ref_out[:, {i*step}:{(i+1)*step}]: \n{ref_out[:, i*step:(i+1)*step]}")
    # print(f"A: {A}")
    # print(f"B: {B}")
    # print(f"C: {C}")
    # print(f"ref_out: {ref_out}")

    torch.testing.assert_close(ref_out.to(torch.float32), C.to(torch.float32), rtol=rtol, atol=1e-3)
    print("Successfully!")


# test_tmeLoad_sqmma_tmeStore(1, 32, 32, 32, 32, 32, 32, 4, torch.float16, is_transposeA=True, is_transposeB=True)
