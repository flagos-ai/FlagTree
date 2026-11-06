# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Fused multiply-add family correctness and signature checks.

Covers the four vector FMA instructions HIVM does not expose as ops
(`vmla` / `vmadd` / `vmaddrelu` / `vaxpy`). All four destinations are
read-modify-write, so every kernel here loads a non-constant initial value
into `out` to keep that operand observable.

Rounding: these instructions fuse the instruction and the datapath, not the
arithmetic. They are *not* IEEE-754 `fma` — the product is rounded to the
compute dtype before the add, so the result carries two roundings. Measured
bit-exact against that model on 910B4-1 over 4096 random values per dtype
(an IEEE single-rounding model disagrees on ~1190/4096 for both fp16 and
fp32). For the fp16-source / fp32-destination form the two models coincide,
because an fp16 product is exactly representable in fp32.

Consequence for these tests: a numerical result cannot distinguish the
fused instruction from an explicit multiply plus add, since the two agree
bit for bit. What the exact references below do catch is operand order,
stride, mask and dtype-selection regressions. Confirming that the fused
instruction is actually emitted is an instruction-level check, done by
inspecting the compiled bitcode for `llvm.hivm.V{MLA,MADD,MADDRELU,AXPY}`.
"""
import numpy as np
import torch
import torch_npu  # noqa: F401
import triton
import triton.experimental.tle.language.raw as tle_raw
import triton.language as tl
from triton.experimental.tle.language.dsa.ascend.custom_ops import (axpy, fused_mul_add, mul_add_dst, mul_add_relu)

_TORCH_DTYPE = {tl.float16: torch.float16, tl.float32: torch.float32}
_NP_DTYPE = {tl.float16: np.float16, tl.float32: np.float32}


def _mul_add(prod, addend, src_dt, out_dt):
    """Reference for all four ops: round the product, then round the sum.

    `prod` is the exact product (compute it in float64 so the only rounding
    is the one modelled here). The product lands in the source dtype when
    source and destination match, and in the destination dtype for the
    mixed form, where it is exact either way. See the module docstring for
    why this is not an IEEE `fma`.
    """
    prod_dt = src_dt if src_dt == out_dt else out_dt
    rounded = prod.astype(_NP_DTYPE[prod_dt]).astype(_NP_DTYPE[out_dt])
    return (rounded + addend).astype(_NP_DTYPE[out_dt])


def _f64(array):
    return array.astype(np.float64)


@triton.jit
def mul_add_dst_kernel(A, B, C, Out, N: tl.constexpr, OUT_DTYPE: tl.constexpr):
    offs = tl.arange(0, N)
    a = tl.load(A + offs)
    b = tl.load(B + offs)
    c = tl.load(C + offs).to(OUT_DTYPE)
    r = tle_raw.call("ascend_mul_add_dst", a, b, N, out=c)
    tl.store(Out + offs, r)


@triton.jit
def fused_mul_add_kernel(A, B, C, Out, N: tl.constexpr):
    offs = tl.arange(0, N)
    a = tl.load(A + offs)
    b = tl.load(B + offs)
    c = tl.load(C + offs)
    r = tle_raw.call("ascend_fused_mul_add", a, b, N, out=c)
    tl.store(Out + offs, r)


@triton.jit
def mul_add_relu_kernel(A, B, C, Out, N: tl.constexpr):
    offs = tl.arange(0, N)
    a = tl.load(A + offs)
    b = tl.load(B + offs)
    c = tl.load(C + offs)
    r = tle_raw.call("ascend_mul_add_relu", a, b, N, out=c)
    tl.store(Out + offs, r)


@triton.jit
def axpy_kernel(X, C, Out, scalar, N: tl.constexpr, OUT_DTYPE: tl.constexpr):
    offs = tl.arange(0, N)
    x = tl.load(X + offs)
    c = tl.load(C + offs).to(OUT_DTYPE)
    r = tle_raw.call("ascend_axpy", x, scalar, N, out=c)
    tl.store(Out + offs, r)


def _rand(rng, n, dtype):
    # Keep magnitudes small so fp16 rounding stays the only error source.
    return (rng.standard_normal(n) * 0.5).astype(_NP_DTYPE[dtype])


def _npu(array):
    return torch.from_numpy(array).npu()


def test_mul_add_dst():
    """dst = src0 * src1 + dst, including the fp16-src / fp32-dst form."""
    rng = np.random.default_rng(5)
    cases = 0
    for src_dt, out_dt in ((tl.float16, tl.float16), (tl.float32, tl.float32), (tl.float16, tl.float32)):
        for n in (256, 1024, 4096):
            a = _rand(rng, n, src_dt)
            b = _rand(rng, n, src_dt)
            c = _rand(rng, n, out_dt)
            out = torch.zeros(n, dtype=_TORCH_DTYPE[out_dt], device="npu")
            mul_add_dst_kernel[(1, )](_npu(a), _npu(b), _npu(c), out, N=n, OUT_DTYPE=out_dt)
            want = _mul_add(_f64(a) * _f64(b), c, src_dt, out_dt)
            np.testing.assert_array_equal(out.cpu().numpy(), want, err_msg=f"src={src_dt}, dst={out_dt}, n={n}")
            cases += 1
    print(f"[PASS] mul_add_dst: {cases} dtype/size cases")


def test_fused_mul_add():
    """dst = src0 * dst + src1 — out is a multiplicand, src1 the addend."""
    rng = np.random.default_rng(6)
    cases = 0
    for dtype in (tl.float16, tl.float32):
        for n in (256, 1024, 4096):
            a = _rand(rng, n, dtype)
            b = _rand(rng, n, dtype)
            c = _rand(rng, n, dtype)
            out = torch.zeros(n, dtype=_TORCH_DTYPE[dtype], device="npu")
            fused_mul_add_kernel[(1, )](_npu(a), _npu(b), _npu(c), out, N=n)
            # out holds the multiplicand here, src1 (b) the addend.
            want = _mul_add(_f64(a) * _f64(c), b, dtype, dtype)
            np.testing.assert_array_equal(out.cpu().numpy(), want, err_msg=f"dtype={dtype}, n={n}")
            cases += 1
    print(f"[PASS] fused_mul_add: {cases} dtype/size cases")


def test_mul_add_relu():
    """dst = relu(src0 * dst + src1); inputs straddle zero on purpose."""
    rng = np.random.default_rng(7)
    cases = 0
    for dtype in (tl.float16, tl.float32):
        for n in (256, 1024, 4096):
            a = _rand(rng, n, dtype)
            b = _rand(rng, n, dtype)
            c = _rand(rng, n, dtype)
            out = torch.zeros(n, dtype=_TORCH_DTYPE[dtype], device="npu")
            mul_add_relu_kernel[(1, )](_npu(a), _npu(b), _npu(c), out, N=n)
            want = np.maximum(_mul_add(_f64(a) * _f64(c), b, dtype, dtype), _NP_DTYPE[dtype](0))
            got = out.cpu().numpy()
            np.testing.assert_array_equal(got, want, err_msg=f"dtype={dtype}, n={n}")
            assert (got >= 0).all(), "mul_add_relu produced a negative value"
            cases += 1
    print(f"[PASS] mul_add_relu: {cases} dtype/size cases (relu clamps "
          f"{int((want == 0).sum())}/{want.size} in the last case)")


def test_axpy():
    """dst = src * scalar + dst, including the fp16-src / fp32-dst form."""
    rng = np.random.default_rng(8)
    cases = 0
    for src_dt, out_dt in ((tl.float16, tl.float16), (tl.float32, tl.float32), (tl.float16, tl.float32)):
        for n in (256, 1024):
            for scalar in (0.0, 1.0, -2.5, 0.125):
                x = _rand(rng, n, src_dt)
                c = _rand(rng, n, out_dt)
                out = torch.zeros(n, dtype=_TORCH_DTYPE[out_dt], device="npu")
                axpy_kernel[(1, )](_npu(x), _npu(c), out, float(scalar), N=n, OUT_DTYPE=out_dt)
                # The scalar is narrowed to the source dtype before the multiply.
                scalar_src = np.float64(_NP_DTYPE[src_dt](scalar))
                want = _mul_add(_f64(x) * scalar_src, c, src_dt, out_dt)
                np.testing.assert_array_equal(out.cpu().numpy(), want,
                                              err_msg=f"src={src_dt}, dst={out_dt}, n={n}, scalar={scalar}")
                cases += 1
    print(f"[PASS] axpy: {cases} dtype/size/scalar cases")


def test_dst_init_is_read():
    """The destination must contribute; a zeroed dst must change the result.

    Every op here is read-modify-write on dst, unlike the write-only `out`
    of the other registered custom ops. Running the same inputs against a
    zero and a non-zero dst pins that down.
    """
    rng = np.random.default_rng(9)
    n = 256
    a = _rand(rng, n, tl.float32)
    b = _rand(rng, n, tl.float32)
    c = (np.abs(_rand(rng, n, tl.float32)) + 1.0).astype(np.float32)
    zero = np.zeros(n, dtype=np.float32)

    out_c = torch.zeros(n, dtype=torch.float32, device="npu")
    out_z = torch.zeros(n, dtype=torch.float32, device="npu")
    mul_add_dst_kernel[(1, )](_npu(a), _npu(b), _npu(c), out_c, N=n, OUT_DTYPE=tl.float32)
    mul_add_dst_kernel[(1, )](_npu(a), _npu(b), _npu(zero), out_z, N=n, OUT_DTYPE=tl.float32)

    got_c = out_c.cpu().numpy()
    got_z = out_z.cpu().numpy()
    np.testing.assert_allclose(got_z, a * b, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(got_c, a * b + c, rtol=1e-6, atol=1e-6)
    assert np.abs(got_c - got_z).max() > 0.9, "dst init was dropped"
    print("[PASS] dst init is read (a*b+c vs a*b differ as expected)")


def _tensor(dtype, shape):
    # Shape must be a list: inside triton.language.core the name `tuple` is
    # rebound to tl.tuple, so block_type's isinstance check rejects a plain
    # Python tuple.
    return tl.tensor(None, tl.block_type(dtype, list(shape)))


def _init(cls, *args, **kwargs):
    # Match the custom-op dispatcher initialization without generating IR.
    op = cls.__new__(cls)
    op.arg_type = {}
    cls.__init__(op, *args, **kwargs)
    return op


def test_signature_checks():
    """Rejected argument combinations and the resulting symbol selection."""
    f32_256 = _tensor(tl.float32, (256, ))
    f16_256 = _tensor(tl.float16, (256, ))

    # Symbol selection, including the mixed-precision forms.
    assert _init(mul_add_dst, f32_256, f32_256, 256, out=f32_256).symbol == "custom_mul_add_dst_float"
    assert _init(mul_add_dst, f16_256, f16_256, 256, out=f16_256).symbol == "custom_mul_add_dst_half"
    assert _init(mul_add_dst, f16_256, f16_256, 256, out=f32_256).symbol == "custom_mul_add_dst_fmix"
    assert _init(axpy, f16_256, 1.0, 256, out=f32_256).symbol == "custom_axpy_fmix"
    assert _init(fused_mul_add, f32_256, f32_256, 256, out=f32_256).symbol == "custom_fused_mul_add_float"
    assert _init(mul_add_relu, f16_256, f16_256, 256, out=f16_256).symbol == "custom_mul_add_relu_half"

    def rejects(cls, *args, **kwargs):
        try:
            _init(cls, *args, **kwargs)
        except AssertionError:
            return True
        return False

    # out is mandatory: it carries the addend.
    assert rejects(mul_add_dst, f32_256, f32_256, 256)
    # fp32 source into an fp16 destination is not a hardware combination.
    assert rejects(mul_add_dst, f32_256, f32_256, 256, out=f16_256)
    # fused_mul_add has no mixed form.
    assert rejects(fused_mul_add, f16_256, f16_256, 256, out=f32_256)
    # Mismatched source lengths and count/size disagreement.
    assert rejects(mul_add_dst, f32_256, _tensor(tl.float32, (128, )), 256, out=f32_256)
    assert rejects(mul_add_dst, f32_256, f32_256, 128, out=f32_256)
    # count must be a compile-time int.
    assert rejects(mul_add_dst, f32_256, f32_256, f32_256, out=f32_256)
    # 2D operands and unsupported dtypes.
    assert rejects(mul_add_dst, _tensor(tl.float32, (16, 16)), _tensor(tl.float32, (16, 16)), 256,
                   out=_tensor(tl.float32, (16, 16)))
    assert rejects(axpy, _tensor(tl.int32, (256, )), 1.0, 256, out=_tensor(tl.int32, (256, )))
    # Over the documented per-call element limit.
    big = _tensor(tl.float32, (16384, ))
    assert rejects(mul_add_dst, big, big, 16384, out=big)
    print("[PASS] signature checks")


if __name__ == "__main__":
    test_mul_add_dst()
    test_fused_mul_add()
    test_mul_add_relu()
    test_axpy()
    test_dst_init_is_read()
    test_signature_checks()
