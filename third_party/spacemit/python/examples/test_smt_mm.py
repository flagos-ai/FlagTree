import torch
import triton
import triton.language as tl
import triton.language.extra.smt as smt
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())


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
    SPLIT_M: tl.constexpr,
    SPLIT_MN: tl.constexpr,
    SPLIT_K: tl.constexpr,
    SUB_BLK_M: tl.constexpr,
    SUB_BLK_N: tl.constexpr,
    SUB_BLK_K: tl.constexpr,
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

    if SPLIT_M:
        b_descriptor_load = smt.descriptor_load(b_block_ptr, (0, 0))
        b = smt.view(b_descriptor_load, (0, 0), (BLOCK_SIZE_K, BLOCK_SIZE_N), (MICRO_K, MICRO_N))
        sub_num = (min(BLOCK_SIZE_M, M - BLOCK_SIZE_M * pid_m) + SUB_BLK_M - 1) // SUB_BLK_M
        for s in smt.parallel(0, sub_num):
            a_descriptor_load = smt.descriptor_load(a_block_ptr, (0, 0))
            a = smt.view(a_descriptor_load, (s * SUB_BLK_M, 0), (SUB_BLK_M, BLOCK_SIZE_K), (MICRO_M, MICRO_K))
            accumulator = smt.dot(a, b)
            accumulator = smt.view(accumulator, (0, 0), (SUB_BLK_M, BLOCK_SIZE_N), (1, 1))
            c = accumulator.to(c_ptr.dtype.element_ty)
            c_block_ptr = tl.make_block_ptr(
                base=c_ptr,
                shape=[M, N],
                strides=[stride_cm, stride_cn],
                offsets=[pid_m * BLOCK_SIZE_M + s * SUB_BLK_M, pid_n * BLOCK_SIZE_N],
                block_shape=[SUB_BLK_M, BLOCK_SIZE_N],
                order=[1, 0],
            )
            tl.store(c_block_ptr, c, boundary_check=(0, 1))

    elif SPLIT_MN:
        sub_num_m = (min(BLOCK_SIZE_M, M - BLOCK_SIZE_M * pid_m) + SUB_BLK_M - 1) // SUB_BLK_M
        sub_num_n = (min(BLOCK_SIZE_N, N - BLOCK_SIZE_N * pid_n) + SUB_BLK_N - 1) // SUB_BLK_N
        total_sub_blocks = sub_num_m * sub_num_n
        b_alloc_ptr = smt.alloc(shape=[BLOCK_SIZE_K, BLOCK_SIZE_N])
        b_alloc_view_ptr = smt.view(b_alloc_ptr, (0, 0), (BLOCK_SIZE_K, BLOCK_SIZE_N), (MICRO_K, MICRO_N))
        bar0 = smt.mbarrier(flag=0, expect_count=1)
        bar1 = smt.mbarrier(flag=0, expect_count=1)
        for s in smt.parallel(0, total_sub_blocks):
            s_m = s // sub_num_n
            s_n = s % sub_num_n
            a_descriptor_load = smt.descriptor_load(a_block_ptr, (0, 0))
            a = smt.view(a_descriptor_load, (s_m * SUB_BLK_M, 0), (SUB_BLK_M, BLOCK_SIZE_K), (MICRO_M, MICRO_K))
            b_alloc_sub_ptr = smt.view(b_alloc_view_ptr, (0, s_n * SUB_BLK_N), (BLOCK_SIZE_K, SUB_BLK_N))
            if s_m == 0:
                b_descriptor_load = smt.descriptor_load(b_block_ptr, (0, 0))
                b = smt.view(b_descriptor_load, (0, s_n * SUB_BLK_N), (BLOCK_SIZE_K, SUB_BLK_N), (MICRO_K, MICRO_N))
                tl.store(b_alloc_sub_ptr, b, boundary_check=(0, 1, 2, 3))
                if s_n == 0:
                    smt.barrier_arrive(bar0)
                elif s_n == 1:
                    smt.barrier_arrive(bar1)
            else:
                if s_n == 0:
                    smt.barrier_wait(bar0, flag=1)
                elif s_n == 1:
                    smt.barrier_wait(bar1, flag=1)

            b_alloc = tl.load(b_alloc_sub_ptr, boundary_check=(0, 1, 2, 3))
            accumulator = smt.dot(a, b_alloc)
            accumulator = smt.view(accumulator, (0, 0), (SUB_BLK_M, SUB_BLK_N), (1, 1))
            c = accumulator.to(c_ptr.dtype.element_ty)
            c_block_ptr = tl.make_block_ptr(
                base=c_ptr,
                shape=[M, N],
                strides=[stride_cm, stride_cn],
                offsets=[pid_m * BLOCK_SIZE_M + s_m * SUB_BLK_M, pid_n * BLOCK_SIZE_N + s_n * SUB_BLK_N],
                block_shape=[SUB_BLK_M, SUB_BLK_N],
                order=[1, 0],
            )
            tl.store(c_block_ptr, c, boundary_check=(0, 1))

    elif SPLIT_K:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=a_ptr.type.element_ty)
        accumulator = smt.view(accumulator, (0, 0), (BLOCK_SIZE_M, BLOCK_SIZE_N), (MICRO_M, MICRO_N))
        sub_num = (K + SUB_BLK_K - 1) // SUB_BLK_K
        for k in tl.range(0, sub_num):
            a_descriptor_load = smt.descriptor_load(a_block_ptr, (0, 0))
            a = smt.view(a_descriptor_load, (0, k * SUB_BLK_K), (BLOCK_SIZE_M, SUB_BLK_K), (MICRO_M, MICRO_K))
            b_descriptor_load = smt.descriptor_load(b_block_ptr, (0, 0))
            b = smt.view(b_descriptor_load, (k * SUB_BLK_K, 0), (SUB_BLK_K, BLOCK_SIZE_N), (MICRO_K, MICRO_N))
            accumulator += smt.dot(a, b)
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


def triton_mm(a, b):
    # handle non-contiguous inputs if necessary
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    # checks constraints
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    # allocates output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    # launch kernel
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
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
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        SPLIT_M=0,
        SPLIT_MN=0,
        SPLIT_K=1,
        SUB_BLK_M=8,
        SUB_BLK_N=64,
        SUB_BLK_K=256,
        MICRO_M=32,
        MICRO_N=32,
        MICRO_K=32,
    )
    return c


def _run_correctness(M: int, N: int, K: int, dtype=torch.float16, device: str = "cpu"):
    A = torch.randn((M, K), dtype=dtype, device=device, requires_grad=False)
    B = torch.randn((K, N), dtype=dtype, device=device, requires_grad=False)

    C = triton_mm(A, B)
    C_ref = torch.mm(A, B)
    torch.testing.assert_close(C, C_ref, atol=1e-1, rtol=0)
    print("PASS")


# -----------------------------
# Performance benchmark (GOPS)
# -----------------------------
def benchmark_gops(
    M: int = 512,
    N: int = 512,
    K: int = 512,
    iters: int = 100,
    warmup: int = 10,
    dtype=torch.float16,
    device: str = "cpu",
):
    """Run matmul benchmark and report GOPS.

    GOPS definition here uses GEMM FLOP count: 2 * M * N * K operations.
    """
    import time

    A = torch.randn((M, K), dtype=dtype, device=device, requires_grad=False)
    B = torch.randn((K, N), dtype=dtype, device=device, requires_grad=False)

    # Warmup
    for _ in range(warmup):
        _ = triton_mm(A, B)

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = triton_mm(A, B)
    t1 = time.perf_counter()

    avg_s = (t1 - t0) / iters
    ops = 2.0 * float(M) * float(N) * float(K)
    gops = ops / avg_s / 1e9

    print(f"[PERF] M={M}, N={N}, K={K}, iters={iters}, "
          f"avg={avg_s * 1e3:.3f} ms, GOPS={gops:.3f}")


# Run perf by env var:
#   RUN_PERF=1 python test_smt_mm.py
if __name__ == "__main__":
    import os

    M, N, K = 512, 512, 512
    dtype = torch.float16
    device = "cpu"

    _run_correctness(M, N, K, dtype=dtype, device=device)

    if os.getenv("RUN_PERF", "0") == "1":
        benchmark_gops(M=M, N=N, K=K, iters=100, dtype=dtype, device=device)
