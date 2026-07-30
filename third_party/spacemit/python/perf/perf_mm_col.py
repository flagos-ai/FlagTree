import time
import torch
import triton
import triton.language as tl
import triton.language.extra.smt as smt
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())

ARCH_ID = triton.runtime.driver.active.current_arch_id


@triton.jit
def mm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
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
    MICRO_M: tl.constexpr,
    MICRO_K: tl.constexpr,
    MICRO_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=[M, K],
        strides=[stride_am, stride_ak],
        offsets=[pid_m * BLOCK_SIZE_M, 0],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
        order=[1, 0],
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=[N, K],
        strides=[stride_bk, stride_bn],
        offsets=[pid_n * BLOCK_SIZE_N, 0],
        block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
        order=[1, 0],
    )

    a_desc = smt.descriptor_load(a_block_ptr, (0, 0))
    a = smt.view(a_desc, (0, 0), (BLOCK_SIZE_M, BLOCK_SIZE_K), (MICRO_M, MICRO_K))

    b_desc = smt.descriptor_load(b_block_ptr, (0, 0))
    b = smt.view(b_desc, (0, 0), (BLOCK_SIZE_N, BLOCK_SIZE_K), (MICRO_N, MICRO_K))

    accumulator = smt.dot(a, b)
    accumulator = smt.view(accumulator, (0, 0), (BLOCK_SIZE_M, BLOCK_SIZE_N), (1, 1))
    c = accumulator.to(c_ptr.dtype.element_ty)

    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=[M, N],
        strides=[stride_cm, stride_cn],
        offsets=[pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        order=[1, 0],
    )
    tl.store(c_block_ptr, c, boundary_check=(0, 1))


def triton_mm(a, b, c, block_size_m=128, block_size_n=128):
    M, K = a.shape
    N, Kb = b.shape
    assert K == Kb

    micro_m, micro_n, micro_k = 16, 32, 8
    block_size_k = triton.next_power_of_2(K)
    grid = (triton.cdiv(M, block_size_m), triton.cdiv(N, block_size_n))

    mm_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        MICRO_M=micro_m,
        MICRO_N=micro_n,
        MICRO_K=micro_k,
    )


if __name__ == "__main__":
    M, N, K = 2048, 2048, 512
    dtype = torch.float16
    block_size_m = 128
    block_size_n = 256
    num_warmup = 20
    num_iterations = 200

    print(f"MM fp16 test: M={M}, N={N}, K={K}, BLOCK_M={block_size_m}, BLOCK_N={block_size_n}, ARCH_ID={ARCH_ID}")
    print("=" * 70)

    torch.manual_seed(0)
    A = torch.randn((M, K), dtype=dtype, device="cpu").contiguous()
    B = torch.randn((N, K), dtype=dtype, device="cpu").contiguous()
    C = torch.empty((M, N), dtype=dtype, device="cpu")

    # warmup
    for _ in range(num_warmup):
        triton_mm(A, B, C, block_size_m=block_size_m, block_size_n=block_size_n)

    start = time.time()
    for _ in range(num_iterations):
        triton_mm(A, B, C, block_size_m=block_size_m, block_size_n=block_size_n)

    triton_time = 1000 * (time.time() - start) / num_iterations

    gflops = 2 * M * N * K / (triton_time / 1000) / 1e9
    print(f"Performance: {triton_time:.4f} ms, {gflops:.2f} GFLOPS")
