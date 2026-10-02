"""
INT4
====

In this tutorial, you will use the INT4 support of FlagTree.

In doing so, you will learn about:

* The packed-uint8 container convention: INT4 has no dtype of its own.

* The unpack recipe: shift-based sign extension into the int8 value domain
  [-8, 7], and packing back -- both bit-exact.

* INT4 matmul: unpack the weights in-kernel and take the native
  INT8 x INT8 -> INT32 path.

"""

# %%
# Unpack and Repack
# -----------------
# Like FP4, INT4 has no dtype: a 4-bit itemsize would break pointer arithmetic
# and layout analysis. Two values live in each uint8, low nibble first, and the
# container moves through ordinary byte paths without unpacking. Unpacking is
# shift arithmetic: once the byte is reinterpreted as int8, `(v << 4) >> 4`
# sign-extends the low nibble and `v >> 4` extracts the high one. The unpacked
# values land in int8's [-8, 7] range and reuse the first-class integer ops.
# With an odd number of values, the last high nibble is padded with zero and
# ignored by the algorithm layer.

import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def int4_roundtrip_kernel(packed_ptr, unpacked_ptr, repacked_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    packed = tl.load(packed_ptr + offs)
    lo = (packed << 4).to(tl.int8, bitcast=True) >> 4  # sign-extend low nibble
    hi = packed.to(tl.int8, bitcast=True) >> 4
    vals = tl.interleave(lo, hi)  # int8 values in [-8, 7], low nibble first
    tl.store(unpacked_ptr + tl.arange(0, 2 * BLOCK), vals)
    even, odd = tl.split(tl.reshape(vals, (BLOCK, 2)))
    repacked = (((odd & 0xF) << 4) | (even & 0xF)).to(tl.uint8)
    tl.store(repacked_ptr + offs, repacked)


def unpack_int4(packed):
    lo = (packed.to(torch.int8) << 4) >> 4
    hi = packed.to(torch.int8) >> 4
    return torch.stack([lo, hi], dim=-1).reshape(*packed.shape[:-1], -1)


def demo_int4_roundtrip():
    packed = torch.arange(256, dtype=torch.uint8).to(DEVICE)  # every byte pattern
    unpacked = torch.empty(512, dtype=torch.int8, device=DEVICE)
    repacked = torch.empty_like(packed)
    int4_roundtrip_kernel[(1, )](packed, unpacked, repacked, BLOCK=256)
    assert torch.equal(unpacked.cpu(), unpack_int4(packed.cpu()))
    assert torch.equal(repacked, packed)
    lo, hi = unpacked.min().item(), unpacked.max().item()
    print(f"int4 roundtrip: all 256 byte patterns bit-exact, values in [{lo}, {hi}]")


demo_int4_roundtrip()

# %%
# INT4 Matmul via the Native INT8 Path
# ------------------------------------
# A packed container fed to `tl.dot` is just uint8 and is rejected by the
# integer rules. Instead, unpack to int8 in-kernel and use the one native
# integer matrix path -- the result stays bit-exact. This is the standard
# weight-only layout: int8 activations, weights packed two per byte along N.


@triton.jit
def int4_weight_matmul_kernel(a_ptr, wp_ptr, c_ptr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    offs_m = tl.arange(0, BM)
    offs_k = tl.arange(0, BK)
    a = tl.load(a_ptr + offs_m[:, None] * BK + offs_k[None, :])  # int8 (BM, BK)
    packed = tl.load(wp_ptr + offs_k[:, None] * (BN // 2) + tl.arange(0, BN // 2)[None, :])
    lo = (packed << 4).to(tl.int8, bitcast=True) >> 4
    hi = packed.to(tl.int8, bitcast=True) >> 4
    w = tl.interleave(lo, hi)  # (BK, BN) int8 in [-8, 7]
    c = tl.dot(a, w)  # native INT8 x INT8 -> INT32
    tl.store(c_ptr + offs_m[:, None] * BN + tl.arange(0, BN)[None, :], c)


def demo_int4_weight_matmul():
    bm = bn = bk = 64
    a = torch.randint(-128, 128, (bm, bk), dtype=torch.int8, device=DEVICE)
    w = torch.randint(-8, 8, (bk, bn), dtype=torch.int8)  # int4 value domain
    w_packed = (((w[:, 1::2] & 0xF) << 4) | (w[:, 0::2] & 0xF)).to(torch.uint8).to(DEVICE)
    c = torch.empty((bm, bn), dtype=torch.int32, device=DEVICE)
    int4_weight_matmul_kernel[(1, )](a, w_packed, c, BM=bm, BN=bn, BK=bk)
    ref = a.cpu().to(torch.int64) @ w.to(torch.int64)
    assert torch.equal(c.cpu().to(torch.int64), ref)
    print("int4 weight matmul: unpack + native int8 dot, bit-exact")


demo_int4_weight_matmul()
