# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Packed scalar comparison correctness and signature checks."""
import numpy as np
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl
from triton.experimental import tle
from triton.experimental.tle.language.dsa.ascend.custom_ops import compare_scalar


@triton.jit
def compare_kernel(X, Mask, scalar, N: tl.constexpr):
    values = tl.load(X + tl.arange(0, N))
    mask = tl.full((N // 16, ), 0, tl.uint16)
    mask = tle.dsa.ascend.raw("compare_scalar", values, scalar, out=mask)
    tl.store(Mask + tl.arange(0, N // 16), mask)


def _pack(bits):
    words = bits.reshape(-1, 16).astype(np.uint32)
    return np.sum(words << np.arange(16, dtype=np.uint32), axis=1).astype(np.uint16)


def test_compare_scalar():
    rng = np.random.default_rng(17)
    cases = 0
    for n in (256, 512, 1024, 2048, 4096):
        values = rng.integers(-4, 5, size=n).astype(np.float32) * 0.5
        values[:6] = [0.0, -0.0, np.inf, -np.inf, np.nan, 1.5]
        x = torch.from_numpy(values).to("npu")
        mask = torch.empty(n // 16, dtype=torch.uint16, device="npu")
        for scalar in (0.0, -0.0, 1.5, np.inf, -np.inf, np.nan, 32.0):
            compare_kernel[(1, )](x, mask, float(scalar), N=n)
            np.testing.assert_array_equal(mask.cpu().numpy(), _pack(values == scalar))
            cases += 1
    print(f"[PASS] compare_scalar: {cases} size/scalar cases")


def _tensor(dtype, shape):
    return tl.tensor(None, tl.block_type(dtype, shape))


def _init(cls, *args, **kwargs):
    # Match the custom-op dispatcher initialization without generating IR.
    op = cls.__new__(cls)
    op.arg_type = {}
    cls.__init__(op, *args, **kwargs)
    return op


@triton.jit
def compare_mode(X, Y, scalar, N: tl.constexpr, MODE: tl.constexpr):
    x = tl.load(X + tl.arange(0, N))
    y = tle.dsa.ascend.raw("compare_scalar", x, scalar, MODE, out=tl.full((N // 16, ), 0, tl.uint16))
    tl.store(Y + tl.arange(0, N // 16), y)


def test_modes():
    rng = np.random.default_rng(11)
    for dtype, sizes in ((np.float16, (256, 4096, 8192, 32768)), (np.float32, (256, 4096))):
        for n in sizes:
            values = rng.integers(-10, 11, n).astype(dtype)
            values[:7] = [0., -0., np.inf, -np.inf, np.nan, 1., 1.001]
            x = torch.from_numpy(values).npu()
            out = torch.empty(n // 16, dtype=torch.uint16, device="npu")
            for scalar in (0., 1.0003, np.inf, -np.inf, np.nan):
                for mode, op in enumerate((np.equal, np.greater, np.greater_equal)):
                    compare_mode[(1, )](x, out, float(scalar), n, mode)
                    np.testing.assert_array_equal(out.cpu().numpy(), _pack(op(values, dtype(scalar))))
    print("[PASS] FP16/FP32 EQ/GT/GE, including the 252-repeat boundary")


def test_validation():
    src = _tensor(tl.float32, [256])
    mask = _tensor(tl.uint16, [16])
    for args, out in [((src, 0., 3), mask), ((_tensor(tl.int32, [256]), 0.), mask),
                      ((_tensor(tl.float32, [128]), 0.), mask), ((src, 0.), None), ((src, 0.), _tensor(tl.uint32,
                                                                                                       [16]))]:
        try:
            _init(compare_scalar, *args, out=out)
        except AssertionError:
            continue
        raise AssertionError("Accepted invalid comparison signature")


@triton.jit
def compare_mask32(X, Y, scalar, N: tl.constexpr, MODE: tl.constexpr):
    x = tl.load(X + tl.arange(0, N))
    y = tle.dsa.ascend.raw("compare_scalar", x, scalar, MODE, out=tl.full((N // 32, ), 0, tl.uint32))
    tl.store(Y + tl.arange(0, N // 32), y)


def test_mask32():
    rng = np.random.default_rng(32)
    for n in (256, 512, 1024, 2048, 4096):
        values = rng.integers(-3, 4, n).astype(np.float32)
        values[:5] = [0., -0., np.inf, -np.inf, np.nan]
        x = torch.from_numpy(values).npu()
        y = torch.empty(n // 32, dtype=torch.uint32, device="npu")
        for mode, op in enumerate((np.equal, np.greater, np.greater_equal)):
            for scalar in (0., 1., np.inf, np.nan):
                compare_mask32[(1, )](x, y, scalar, n, mode)
                np.testing.assert_array_equal(y.cpu().numpy(), _pack(op(values, scalar)).view(np.uint32))
    print("[PASS] uint32 masks for FP32 GatherMask consumers")


def main():
    test_validation()
    test_compare_scalar()
    test_modes()
    test_mask32()
    print("All comparison tests passed.")


if __name__ == "__main__":
    main()
