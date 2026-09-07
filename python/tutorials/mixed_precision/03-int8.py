"""
INT8
====

In this tutorial, you will use the INT8 support of FlagTree.

In doing so, you will learn about:

* INT8 x INT8 -> INT32 `tl.dot`: the one matrix combination that is native on
  every product, and bit-exact.

* Carrying an explicit int32 accumulator across the K loop.

* Quantization built from ordinary ops: symmetric quantize, integer matmul,
  and requantization of the int32 accumulator.

"""

# %%
# INT8 Matmul Is Bit-Exact
# ------------------------
# INT8 is a first-class dtype, and int8 x int8 -> int32 is the one matrix
# combination with a native instruction on every product -- no capability
# branching needed. Integer arithmetic is exact, so results compare with
# `torch.equal` rather than tolerances. The dot returns int32 regardless of
# `out_dtype`, and an explicit accumulator is int32 as well. Other integer
# operands (int16/int32/uint8) are rejected at compile time: int32 exists in
# `tl.dot` only as the accumulator and output dtype.

import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def int8_matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.int32)  # int8 dot accumulates in int32
    for k0 in range(0, K, BK):
        offs_k = k0 + tl.arange(0, BK)
        a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
        b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])
        acc = tl.dot(a, b, acc=acc)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)


def int8_matmul(a, b):
    M, K = a.shape
    _, N = b.shape
    bm, bn, bk = 64, 64, 64
    c = torch.empty((M, N), dtype=torch.int32, device=DEVICE)
    int8_matmul_kernel[(M // bm, N // bn)](a, b, c, M, N, K, BM=bm, BN=bn, BK=bk)
    return c


def demo_int8_matmul():
    M = N = K = 256
    a = torch.randint(-128, 128, (M, K), dtype=torch.int8, device=DEVICE)
    b = torch.randint(-128, 128, (K, N), dtype=torch.int8, device=DEVICE)
    c = int8_matmul(a, b)
    ref = a.cpu().to(torch.int64) @ b.cpu().to(torch.int64)  # exact reference
    assert torch.equal(c.cpu().to(torch.int64), ref)
    print("int8 matmul: bit-exact against the int64 reference")


demo_int8_matmul()

# %%
# A Quantized Matmul End to End
# -----------------------------
# INT8 quantization needs no dedicated ops: it is clamp + cast (`float -> int`
# casts truncate). The integer matmul in the middle is exact, so the only
# rounding in the whole pipeline happens at the two casts. Requantization
# rescales the int32 accumulator back to int8 with the combined scale.


@triton.jit
def int8_quant_kernel(x_ptr, q_ptr, scale, N, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    q = tl.clamp(x / scale, -128.0, 127.0).to(tl.int8)  # float -> int truncates
    tl.store(q_ptr + offs, q, mask=mask)


@triton.jit
def requant_kernel(acc_ptr, q_ptr, scale_ab, scale_out, N, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    acc = tl.load(acc_ptr + offs, mask=mask)
    y = acc.to(tl.float32) * scale_ab  # dequantize the exact int32 result
    q = tl.clamp(y / scale_out, -128.0, 127.0).to(tl.int8)
    tl.store(q_ptr + offs, q, mask=mask)


def quantize(x, scale):
    q = torch.empty(x.shape, dtype=torch.int8, device=DEVICE)
    n = x.numel()
    int8_quant_kernel[(triton.cdiv(n, 1024), )](x, q, scale, n, BLOCK=1024)
    return q


def demo_quantized_matmul():
    M = N = K = 256
    x = torch.randn(M, K, device=DEVICE)
    w = torch.randn(K, N, device=DEVICE)
    scale_x = x.abs().max().item() / 127
    scale_w = w.abs().max().item() / 127
    q_x, q_w = quantize(x, scale_x), quantize(w, scale_w)

    acc = int8_matmul(q_x, q_w)
    ref = q_x.cpu().to(torch.int64) @ q_w.cpu().to(torch.int64)
    assert torch.equal(acc.cpu().to(torch.int64), ref)

    y = torch.empty((M, N), dtype=torch.int8, device=DEVICE)
    scale_y = (acc.to(torch.float32) * scale_x * scale_w).abs().max().item() / 127
    requant_kernel[(triton.cdiv(M * N, 1024), )](acc, y, scale_x * scale_w, scale_y, M * N, BLOCK=1024)

    err = (y.cpu().to(torch.float32) * scale_y - x.cpu() @ w.cpu()).abs().max().item()
    print(f"quantized matmul: integer core bit-exact, end-to-end max err {err:.3f} "
          f"(output quantization step {scale_y:.3f})")


demo_quantized_matmul()
