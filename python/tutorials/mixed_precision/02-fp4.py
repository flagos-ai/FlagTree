"""
FP4 (E2M1)
==========

In this tutorial, you will use the FP4 support of FlagTree.

In doing so, you will learn about:

* The packed-uint8 container convention: FP4 has no dtype of its own.

* The in-kernel decode recipe: split the bit fields, rebase the exponent,
  bitcast to fp16 -- all 16 patterns are exact.

* `tl.dot_scaled` with the "e2m1" format tag as the only matrix entry point.

"""

# %%
# Packed Containers
# -----------------
# FP4 has no dtype of its own: a 4-bit itemsize would break pointer arithmetic
# and layout analysis. Instead, two E2M1 values live in each uint8 (low nibble
# first) and the container moves through ordinary byte paths without unpacking.
# Decoding is a three-step bit recipe: split the fields, rebase the exponent
# (bias 1 -> 15), and bitcast to fp16. All 16 patterns are exact.

import warnings

import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def decode_e2m1(nibbles):
    sign = torch.where(nibbles >> 3 != 0, -1.0, 1.0)
    return sign * E2M1_VALUES[(nibbles & 0x7).long()]


@triton.jit
def fp4_decode_kernel(packed_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    packed = tl.load(packed_ptr + offs)
    nib = tl.interleave(packed & 0xF, packed >> 4).to(tl.uint16)  # low nibble first
    s, e, m = nib >> 3, (nib >> 1) & 0x3, nib & 0x1
    bits = (s << 15) | tl.where(e == 0, m * 0x3800, ((e + 14) << 10) | (m << 9))
    tl.store(out_ptr + tl.arange(0, 2 * BLOCK), bits.to(tl.float16, bitcast=True))


def demo_fp4_decode():
    packed = torch.arange(256, dtype=torch.uint8).to(DEVICE)
    out = torch.empty(512, dtype=torch.float16, device=DEVICE)
    fp4_decode_kernel[(1, )](packed, out, BLOCK=256)
    nibbles = torch.stack([packed.cpu() & 0xF, packed.cpu() >> 4], dim=-1).reshape(-1)
    assert torch.equal(out.cpu().float(), decode_e2m1(nibbles))
    print("fp4 decode: all 256 byte patterns (512 nibbles) match the E2M1 table")


demo_fp4_decode()

# %%
# FP4 Matmul via `tl.dot_scaled`
# ------------------------------
# A packed container fed to plain `tl.dot` is just a uint8 tensor and is
# rejected by the integer rules. The only matrix entry point is
# `tl.dot_scaled`, where the "e2m1" format tag gives the container its
# semantics. Scales use the e8m0 format (127 encodes 1.0). Products without a
# native scaled-MMA path decompose to a promoted dot and warn at compile time,
# declared native=false, so silent degradation cannot happen.


@triton.jit
def fp4_matmul_kernel(a_ptr, as_ptr, b_ptr, c_ptr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    offs_m = tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a = tl.load(a_ptr + offs_m[:, None] * (BK // 2) + tl.arange(0, BK // 2)[None, :])  # packed uint8
    b = tl.load(b_ptr + offs_k[:, None] * BN + offs_n[None, :])  # bf16
    a_scale = tl.load(as_ptr + offs_m[:, None] * (BK // 32) + tl.arange(0, BK // 32)[None, :])
    c = tl.dot_scaled(a, a_scale, "e2m1", b, None, "bf16")
    tl.store(c_ptr + offs_m[:, None] * BN + offs_n[None, :], c)


def demo_fp4_matmul():
    bm = bn = bk = 64
    a_packed = torch.randint(0, 256, (bm, bk // 2), dtype=torch.uint8, device=DEVICE)
    b = (torch.rand(bk, bn, device=DEVICE) * 4 - 2).to(torch.bfloat16)
    ones = torch.full((bm, bk // 32), 127, dtype=torch.uint8, device=DEVICE)  # e8m0 for 1.0
    c = torch.empty((bm, bn), dtype=torch.float32, device=DEVICE)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fp4_matmul_kernel[(1, )](a_packed, ones, b, c, BM=bm, BN=bn, BK=bk)
    for msg in dict.fromkeys(str(w.message) for w in caught):
        print(f"compile-time warning: {msg}")
    nibbles = torch.stack([a_packed.cpu() & 0xF, a_packed.cpu() >> 4], dim=-1).reshape(bm, bk)
    a_ref = decode_e2m1(nibbles)
    ref = a_ref @ b.cpu().to(torch.float32)
    torch.testing.assert_close(c.cpu(), ref, rtol=2e-2, atol=1e-1)
    print("fp4 dot_scaled: numerics match the decode-table reference")


demo_fp4_matmul()
