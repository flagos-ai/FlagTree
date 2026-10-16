"""HCU matmul through tle_raw.call_smem.

Same shape as the CUDA shared-memory example: allocate the tiles with
tle.gpu.alloc, copy them in, and let the HIP function update the accumulator
in shared memory. The call passes output_indices because the accumulator is
updated in place.
"""

from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.experimental.tle.raw import dialect
import triton.experimental.tle.language.gpu as tle_gpu
import triton.experimental.tle.language.raw as tle_raw

DEVICE = triton.runtime.driver.active.get_active_torch_device()
BLOCK_M = 32
BLOCK_N = 32
BLOCK_K = 32


# deferred=True is the fast path: clang bitcode is linked with its target attributes.
@dialect(name="hcu", file=Path(__file__).parent / "04-matmul-smem.hip", extern_func_name="matmul_smem", deferred=True)
def edsl(*args, **kwargs):
    ...


@triton.jit
def matmul_kernel(
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
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_smem = tle_gpu.alloc(shape=[BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32, layout=None, scope=tle_gpu.smem,
                             nv_mma_shared_layout=False)
    rows = tl.broadcast_to(tl.arange(0, BLOCK_SIZE_M)[:, None], (BLOCK_SIZE_M, BLOCK_SIZE_N))
    cols = tl.broadcast_to(tl.arange(0, BLOCK_SIZE_N)[None, :], (BLOCK_SIZE_M, BLOCK_SIZE_N))
    acc_ptrs = tle_gpu.local_ptr(acc_smem, (rows, cols))
    tl.store(acc_ptrs, acc)
    for _ in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a_smem = tle_gpu.alloc(shape=[BLOCK_SIZE_M, BLOCK_SIZE_K], dtype=tl.float16, layout=None, scope=tle_gpu.smem,
                               nv_mma_shared_layout=False)
        tle_gpu.copy(a_ptrs, a_smem, shape=[BLOCK_SIZE_M, BLOCK_SIZE_K])
        b_smem = tle_gpu.alloc(shape=[BLOCK_SIZE_K, BLOCK_SIZE_N], dtype=tl.float16, layout=None, scope=tle_gpu.smem,
                               nv_mma_shared_layout=False)
        tle_gpu.copy(b_ptrs, b_smem, shape=[BLOCK_SIZE_K, BLOCK_SIZE_N])
        acc_smem = tle_raw.call_smem(edsl, [acc_smem, a_smem, b_smem], output_indices=[0])
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    acc = tl.load(acc_ptrs)
    c = acc.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a, b):
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"incompatible dimensions: {tuple(a.shape)} x {tuple(b.shape)}")
    m, k = a.shape
    _, n = b.shape
    c = torch.empty((m, n), device=a.device, dtype=torch.float16)
    grid = (triton.cdiv(m, BLOCK_M) * triton.cdiv(n, BLOCK_N), )
    matmul_kernel[grid](
        a,
        b,
        c,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=BLOCK_N,
        BLOCK_SIZE_K=BLOCK_K,
        num_warps=4,
        num_stages=1,
    )
    return c


if __name__ == "__main__":
    torch.manual_seed(0)
    a = torch.rand((64, 64), device=DEVICE, dtype=torch.float16) - 0.5
    b = torch.rand((64, 64), device=DEVICE, dtype=torch.float16) - 0.5
    got = matmul(a, b)
    torch.cuda.synchronize()
    ref = torch.matmul(a, b)
    ok = torch.allclose(got, ref, atol=1e-2, rtol=1e-2)
    err = (got - ref).abs().max().item()
    print("allclose", bool(ok))
    print("max_abs_err", err)
    raise SystemExit(0 if ok else 1)
