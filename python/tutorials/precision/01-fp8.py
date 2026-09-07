"""
FP8 (E4M3FN)
============

In this tutorial, you will use the FP8 support of FlagTree.

In doing so, you will learn about:

* Querying per-product capability declarations and branching on them.

* Casting to and from FP8 with satfinite saturation and strict RTNE/RTZ rounding,
  which works even on products without native FP8 cvt instructions (software cast).

* FP8 matmul: native where declared, otherwise a preserved emulation path that
  warns at compile time and is declared native=false.

"""

# %%
# Capability Declarations
# -----------------------
# FP8 support differs per product, so backends declare what they actually
# provide: `supported_fp8_dtypes` for storage (dual-whitelist backends refine
# it with `supported_fp8_storage_dtypes`), `supported_fp8_cast_dtypes`
# for numerical casts (`custom_cast_fp8_dtypes` marks the ones lowered in
# software), and `async_copy_dtypes` / `descriptor_dtypes` for data movement.
# The declarations ship with every compiled kernel (`h.metadata`), so programs
# can branch on them instead of guessing.

import warnings

import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()
TARGET = triton.runtime.driver.active.get_current_target()


def backend_options():
    from triton.compiler import make_backend
    return make_backend(TARGET).parse_options({})


opts = backend_options()
FP8_STORAGE = getattr(opts, "supported_fp8_storage_dtypes", getattr(opts, "supported_fp8_dtypes", ()))
FP8_CAST = getattr(opts, "supported_fp8_cast_dtypes", FP8_STORAGE)
FP8_SW_CAST = getattr(opts, "custom_cast_fp8_dtypes", ())

print(f"target:            {TARGET.backend} / {TARGET.arch}")
print(f"fp8 storage:       {FP8_STORAGE}")
print(f"fp8 cast:          {FP8_CAST}")
print(f"fp8 software cast: {FP8_SW_CAST}")

HAS_E4M3FN = "fp8e4nv" in FP8_CAST

# %%
# FP8 Quantization Is Just a Cast
# -------------------------------
# E4M3FN downcasts saturate to +-448 (satfinite) and round with RTNE by default
# (RTZ is also available via `fp_downcast_rounding`). NaN maps to the NaN code.
# FP8 values travel as plain bytes: bitcast to uint8 for storage, bitcast back
# to reinterpret. There is no implicit FP8 <-> integer numerical conversion.


@triton.jit
def fp8_quant_kernel(x_ptr, q_ptr, scale, N, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    q = (x / scale).to(tl.float8e4nv)  # satfinite + RTNE built in
    tl.store(q_ptr + offs, q.to(tl.uint8, bitcast=True), mask=mask)


@triton.jit
def fp8_dequant_kernel(q_ptr, y_ptr, scale, N, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    bits = tl.load(q_ptr + offs, mask=mask)
    y = bits.to(tl.float8e4nv, bitcast=True).to(tl.float32) * scale
    tl.store(y_ptr + offs, y, mask=mask)


def demo_fp8_quantization():
    n, block, scale = 4096, 1024, 0.05
    x = torch.randn(n, device=DEVICE) * 10
    q = torch.empty(n, dtype=torch.uint8, device=DEVICE)
    y = torch.empty(n, dtype=torch.float32, device=DEVICE)
    grid = (triton.cdiv(n, block), )
    fp8_quant_kernel[grid](x, q, scale, n, BLOCK=block)
    fp8_dequant_kernel[grid](q, y, scale, n, BLOCK=block)
    # torch's CPU float8 conversion is the reference for in-range values; on
    # overflow the two differ by design: torch yields NaN, Triton saturates
    xs = x.cpu() / scale
    ref_bits = xs.to(torch.float8_e4m3fn).view(torch.uint8)
    in_range = xs.abs() <= 448.0
    assert torch.equal(q.cpu()[in_range], ref_bits[in_range])
    err = (y - x)[in_range.to(DEVICE)].abs().max().item()
    print(f"fp8 quant/dequant: bit-exact vs torch reference in range, max roundtrip err {err:.4f}")

    # satfinite: overflow clamps to +-448 instead of mapping to the NaN code
    big = torch.tensor([1e6, -1e6], device=DEVICE)
    qb = torch.empty(2, dtype=torch.uint8, device=DEVICE)
    yb = torch.empty(2, dtype=torch.float32, device=DEVICE)
    fp8_quant_kernel[(1, )](big, qb, 1.0, 2, BLOCK=2)
    fp8_dequant_kernel[(1, )](qb, yb, 1.0, 2, BLOCK=2)
    print(f"satfinite: 1e6 -> {yb[0].item()}, -1e6 -> {yb[1].item()}")


if HAS_E4M3FN:
    demo_fp8_quantization()
else:
    print("E4M3FN cast is not declared on this product -- skipping (as declared)")

# %%
# FP8 Matmul and the Emulation Warning
# ------------------------------------
# `tl.dot` legality is resolved by the backend's capability rules. Products with
# native FP8 matrix cores run it natively; products that keep the legacy
# FP16-promotion path still execute it, but emit a compile-time warning and
# declare native=false, so silent degradation cannot happen. The warning shows
# up when the kernel is actually compiled (a warm cache skips compilation).


@triton.jit
def fp8_matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        offs_k = k0 + tl.arange(0, BK)
        a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])  # fp8 typed pointer
        b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])
        acc = tl.dot(a, b, acc=acc)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)


def demo_fp8_matmul():
    M = N = K = 256
    bm, bn, bk = 64, 64, 64
    a8 = (torch.randn(M, K) * 0.5).to(torch.float8_e4m3fn).view(torch.uint8).to(DEVICE)
    b8 = (torch.randn(K, N) * 0.5).to(torch.float8_e4m3fn).view(torch.uint8).to(DEVICE)
    c = torch.empty((M, N), dtype=torch.float32, device=DEVICE)
    grid = (M // bm, N // bn)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fp8_matmul_kernel[grid](triton.reinterpret(a8, tl.float8e4nv), triton.reinterpret(b8, tl.float8e4nv), c, M, N,
                                K, BM=bm, BN=bn, BK=bk)
    for msg in dict.fromkeys(str(w.message) for w in caught):
        print(f"compile-time warning: {msg}")
    a_ref = a8.cpu().view(torch.float8_e4m3fn).to(torch.float32)
    b_ref = b8.cpu().view(torch.float8_e4m3fn).to(torch.float32)
    torch.testing.assert_close(c.cpu(), a_ref @ b_ref, rtol=1e-4, atol=1e-2)
    print("fp8 matmul: numerics match the upcast reference")


if HAS_E4M3FN:
    demo_fp8_matmul()
else:
    print("fp8 matmul is not available on this product -- skipping (as declared)")
