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
        shape=[K, N],
        strides=[stride_bk, stride_bn],
        offsets=[0, pid_n * BLOCK_SIZE_N],
        block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
        order=[1, 0],
    )

    a_descriptor_load = smt.descriptor_load(a_block_ptr, (0, 0))
    a = smt.view(a_descriptor_load, (0, 0), (BLOCK_SIZE_M, BLOCK_SIZE_K), (MICRO_M, MICRO_K))
    b_descriptor_load = smt.descriptor_load(b_block_ptr, (0, 0))
    b = smt.view(b_descriptor_load, (0, 0), (BLOCK_SIZE_K, BLOCK_SIZE_N), (MICRO_K, MICRO_N))

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


def triton_mm(a, b, block_size_m=128, block_size_n=128):
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()

    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape

    if a.dtype == torch.float16 and ARCH_ID == "0xA064":
        micro_m, micro_n, micro_k = 16, 32, 8
    elif a.dtype == torch.float16 and ARCH_ID == "0xA03C":
        micro_m, micro_n, micro_k = 8, 16, 16
    else:
        raise ValueError(f"Unsupported ARCH_ID={ARCH_ID} or dtype={a.dtype}")

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    def grid(META):
        return (triton.cdiv(M, META["BLOCK_SIZE_M"]), triton.cdiv(N, META["BLOCK_SIZE_N"]))

    BLOCK_SIZE_K = triton.next_power_of_2(K)

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
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        MICRO_M=micro_m,
        MICRO_N=micro_n,
        MICRO_K=micro_k,
    )
    return c


if __name__ == "__main__":
    M, N, K = 512, 512, 512
    dtype = torch.float16
    block_size_m = 128
    block_size_n = 128
    num_warmup = 5
    num_iterations = 100

    print(f"MM fp16 test: M={M}, N={N}, K={K}, BLOCK_M={block_size_m}, BLOCK_N={block_size_n}, ARCH_ID={ARCH_ID}")
    print("=" * 70)

    # --- Correctness ---
    torch.manual_seed(0)
    A = torch.randn((M, K), dtype=dtype, device="cpu")
    B = torch.randn((K, N), dtype=dtype, device="cpu")

    out_triton = triton_mm(A, B, block_size_m=block_size_m, block_size_n=block_size_n)
    out_torch = torch.mm(A, B)

    max_diff = torch.max(torch.abs(out_triton - out_torch)).item()
    passed = torch.allclose(out_triton, out_torch, atol=1e-2, rtol=0)
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"Correctness: {status}  (max_diff={max_diff:.6f})")

    # --- Performance ---
    for _ in range(num_warmup):
        _ = triton_mm(A, B, block_size_m=block_size_m, block_size_n=block_size_n)

    start = time.time()
    for _ in range(num_iterations):
        _ = triton_mm(A, B, block_size_m=block_size_m, block_size_n=block_size_n)
    triton_time = 1000 * (time.time() - start) / num_iterations

    gflops = 2 * M * N * K / (triton_time / 1000) / 1e9
    print(f"Performance: {triton_time:.4f} ms, {gflops:.2f} GFLOPS")
