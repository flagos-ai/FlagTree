"""
最简单的矩阵乘法示例，使用 tl.make_block_ptr 进行内存访问。
C = A @ B, 其中 A: [M, K], B: [K, N], C: [M, N]
"""

import torch
import triton
import triton.language as tl
from triton.backends.spacemit.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())


@triton.jit
def mm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # K == BLOCK_K，单次加载完整 K 维度
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )
    a = tl.load(a_block_ptr, boundary_check=(0, 1))
    b = tl.load(b_block_ptr, boundary_check=(0, 1))
    acc = tl.dot(a, b)

    # 存储结果
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(c_block_ptr, acc.to(c_ptr.dtype.element_ty), boundary_check=(0, 1))


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"incompatible dimensions: {K} vs {K2}"
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, triton.next_power_of_2(K)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    mm_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c


if __name__ == "__main__":
    M, N, K = 256, 256, 256
    a = torch.randn((M, K), device="cpu", dtype=torch.float16)
    b = torch.randn((K, N), device="cpu", dtype=torch.float16)

    c_triton = matmul(a, b)
    c_torch = torch.matmul(a, b)

    print("c_triton", c_triton)
    print("c_torch", c_torch)

    torch.testing.assert_close(c_triton, c_torch, atol=1e-2, rtol=1e-2)
    print(f"✅ PASS: mm_block_ptr ({M}x{K}) @ ({K}x{N}) = ({M}x{N})")
