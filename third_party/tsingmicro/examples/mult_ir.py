import torch

import triton
import triton.language as tl
# import benchmark

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.jit
def matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr, BLOCK_SIZE_N1: tl.constexpr, BLOCK_SIZE_N2: tl.constexpr, #
        BLOCK_SIZE_K1: tl.constexpr, BLOCK_SIZE_K2: tl.constexpr,
        ACTIVATION: tl.constexpr  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetics` section for details
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    mask_m = offs_m < M
    mask_n = offs_n < N
    offs_nn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N1)[:, None] * BLOCK_SIZE_N2 + tl.arange(0, BLOCK_SIZE_N2)[None, :]
    offs_kk = tl.arange(0, BLOCK_SIZE_K1)[:, None] * BLOCK_SIZE_K2 + tl.arange(0, BLOCK_SIZE_K2)[None, :]
    mask_nn = offs_nn < N
    a_ptrs = a_ptr + (offs_m[None, :, None] * stride_am + offs_kk[:, None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[None, :, None] * stride_bk + offs_nn[:, None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    # accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float16)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        mask_k = offs_k < K - k * BLOCK_SIZE_K
        mask_kk = offs_kk < K - k * BLOCK_SIZE_K
        # a = tl.load(a_ptrs, mask=(mask_m[None, :, None] & mask_kk[:, None, :]), other=0.0)
        # b = tl.load(b_ptrs, mask=(mask_k[None, :, None] & mask_nn[:, None, :]), other=0.0)
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        if BLOCK_SIZE_K1 != 1:
            a = tl.trans(a, (1, 0, 2))
        a = tl.reshape(a, (BLOCK_SIZE_M, BLOCK_SIZE_K))
        if BLOCK_SIZE_N1 != 1:
            b = tl.trans(b, (1, 0, 2))
        b = tl.reshape(b, (BLOCK_SIZE_K, BLOCK_SIZE_N))
        # We accumulate along the K dimension.
        acc += tl.dot(a, b, out_dtype=tl.float16)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if BLOCK_SIZE_N1 == 1:
        acc = tl.reshape(acc, (BLOCK_SIZE_N1, BLOCK_SIZE_M, BLOCK_SIZE_N2))
    else:
        acc = tl.reshape(acc, (BLOCK_SIZE_M, BLOCK_SIZE_N1, BLOCK_SIZE_N2))
        acc = tl.trans(acc, (1, 0, 2))

    if ACTIVATION == "leaky_relu":
        acc = leaky_relu(acc)
    c = acc

    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    # if ACTIVATION == "leaky_relu":
    #     accumulator = leaky_relu(accumulator)
    # c = accumulator.to(tl.float32)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    c_ptrs = c_ptr + stride_cm * offs_m[None, :, None] + stride_cn * offs_nn[:, None, :]
    tl.store(c_ptrs, c)
    # tl.store(c_ptrs, c, mask=(mask_m[None, :, None] & mask_nn[:, None, :]))

# We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `_matmul`.
@triton.jit
def leaky_relu(x):
    x = x + 1
    return tl.where(x >= 0, x, 0.01 * x)


def matmul(a, b, activation=""):
    BLOCK_M = 1024
    BLOCK_N = 1024
    BLOCK_K = 128

    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    M, K = a.shape
    K, N = b.shape

    assert M % BLOCK_M == 0 and N % BLOCK_N == 0 and K % BLOCK_K == 0

    ALIGN = 64
    BLOCK_K1 = max(1, BLOCK_K // ALIGN)
    BLOCK_K2 = min(ALIGN, BLOCK_K)
    BLOCK_N1 = max(1, BLOCK_N // ALIGN)
    BLOCK_N2 = min(ALIGN, BLOCK_N)
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        ACTIVATION=activation,  #
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=BLOCK_N,
        BLOCK_SIZE_K=BLOCK_K,
        GROUP_SIZE_M=8,
        BLOCK_SIZE_N1=BLOCK_N1,
        BLOCK_SIZE_N2=BLOCK_N2,
        BLOCK_SIZE_K1=BLOCK_K1,
        BLOCK_SIZE_K2=BLOCK_K2,
    )
    return c

# @benchmark.measure(repeats=20)
def bench_matmul(a, b):
    x = a.to(DEVICE)
    y = b.to(DEVICE)
    z = matmul(x, y)
    z = z.to("cpu")
    return z

if __name__ == "__main__":
    # benchmark.select_cpu_backend()
    M = 1024
    K = 1024
    N = 1024
    a = torch.randn((M, K), device='cpu', dtype=torch.float16)
    b = torch.randn((K, N), device='cpu', dtype=torch.float16)
    out = bench_matmul(a, b)
    print(out)
    ref = torch.matmul(a, b)
    print(ref)
    print(ref - out)
