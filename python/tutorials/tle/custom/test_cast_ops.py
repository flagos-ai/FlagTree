# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Standalone correctness checks for the signed INT4-to-FP16 custom op."""

import numpy as np
import torch
import torch_npu  # noqa: F401
import triton
import triton.experimental.tle as tle
import triton.language as tl
from triton.experimental.tle.language.dsa.ascend.custom_ops import cast_int4_to_fp16

CAST_NONE = tl.constexpr(0)


@triton.jit
def cast_kernel(X, Out, N: tl.constexpr):
    packed = tl.load(X + tl.arange(0, N))
    values = tl.full((2 * N, ), 0, tl.float16)
    values = tle.dsa.ascend.raw("cast_int4_to_fp16", packed, CAST_NONE, 2 * N, out=values)
    tl.store(Out + tl.arange(0, 2 * N), values)


def _reference(packed):
    low = (packed.astype(np.int16) & 15)
    high = (packed.astype(np.int16) >> 4)
    values = np.stack((low, high), axis=-1)
    return np.where(values < 8, values, values - 16).reshape(-1).astype(np.float16)


def test_cast_int4_to_fp16():
    cases = 0
    for n in (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192):
        # Cover all 256 byte encodings even for the smallest source buffers.
        for start in range(0, 256, min(n, 256)):
            packed = ((np.arange(n, dtype=np.uint32) + start) % 256).astype(np.uint8)
            x = torch.from_numpy(packed).to("npu")
            storage = torch.full((2 * n + 32, ), 123, dtype=torch.float16, device="npu")
            out = storage[16:-16]
            cast_kernel[(1, )](x, out, N=n)
            np.testing.assert_array_equal(out.cpu().numpy(), _reference(packed))
            np.testing.assert_array_equal(storage[:16].cpu().numpy(), np.full(16, 123))
            np.testing.assert_array_equal(storage[-16:].cpu().numpy(), np.full(16, 123))
            cases += 1
    print(f"[PASS] cast_int4_to_fp16: {cases} size/encoding cases")


@triton.jit
def cast_bench_kernel(X, Y, BYTES: tl.constexpr, TILES: tl.constexpr, CUSTOM: tl.constexpr):
    for tile in range(tl.program_id(0), TILES, tl.num_programs(0)):
        packed = tl.load(X + tile * BYTES + tl.arange(0, BYTES))
        if CUSTOM:
            values = tle.dsa.ascend.raw("cast_int4_to_fp16", packed, 0, 2 * BYTES, out=tl.full((2 * BYTES, ), 0,
                                                                                               tl.float16))
        else:
            codes = packed.to(tl.int16)
            low = codes & 0xF
            high = (codes >> 4) & 0xF
            low = ((low ^ 8) - 8).to(tl.float16)
            high = ((high ^ 8) - 8).to(tl.float16)
            values = tl.interleave(low, high)
        tl.store(Y + tile * 2 * BYTES + tl.arange(0, 2 * BYTES), values)


def bench_cast_int4_to_fp16():
    # Largest packed tile used by the W4A16 caller. The Triton side sign-extends
    # each nibble to FP16 and interleaves them, low nibble first.
    nbytes, tiles = 4096, 5120
    packed = (np.arange(nbytes * tiles, dtype=np.uint32) % 256).astype(np.uint8)
    expected = _reference(packed)
    x = torch.from_numpy(packed).to("npu")
    y = torch.empty(packed.size * 2, dtype=torch.float16, device="npu")

    def launch(custom):
        return cast_bench_kernel[(40, )](x, y, nbytes, tiles, custom, num_warps=1, multibuffer=False,
                                         enable_fp_fusion=False)

    for custom in (True, False):
        launch(custom)
        torch.npu.synchronize()
        np.testing.assert_array_equal(y.cpu().numpy(), expected)
    # do_bench times each launch with device events. Two orders, then the mean.
    measured = {"custom": [], "triton": []}
    for custom_first in (False, True):
        order = (("custom", True), ("triton", False)) if custom_first else (("triton", False), ("custom", True))
        for name, custom in order:
            measured[name].append(triton.testing.do_bench(lambda custom=custom: launch(custom), return_mode="median"))
    custom_us = sum(measured["custom"]) / len(measured["custom"]) * 1000
    triton_us = sum(measured["triton"]) / len(measured["triton"]) * 1000
    print(f"[BENCH] cast_int4_to_fp16 {nbytes} bytes x {tiles} tiles: "
          f"custom {custom_us:.3f} us, triton {triton_us:.3f} us, "
          f"triton/custom {triton_us / custom_us:.2f}x "
          f"rounds_ms={measured}")


def _tensor(dtype, shape):
    return tl.tensor(None, tl.block_type(dtype, shape))


def _init(*args, **kwargs):
    op = cast_int4_to_fp16.__new__(cast_int4_to_fp16)
    op.arg_type = {}
    op.__init__(*args, **kwargs)
    return op


def test_validation():
    for n in (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192):
        op = _init(_tensor(tl.uint8, [n]), 0, 2 * n, out=_tensor(tl.float16, [2 * n]))
        assert op.symbol == "custom_cast_int4_to_fp16"
        assert op.bitcode.endswith("custom_ops.bc")
    src = _tensor(tl.uint8, [64])
    dst = _tensor(tl.float16, [128])
    invalid = [
        (src, 0, 128, None),
        (_tensor(tl.int8, [64]), 0, 128, dst),
        (_tensor(tl.float16, [64]), 0, 128, dst),
        (_tensor(tl.uint8, [8, 8]), 0, 128, dst),
        (_tensor(tl.uint8, [16]), 0, 32, _tensor(tl.float16, [32])),
        (_tensor(tl.uint8, [16384]), 0, 32768, _tensor(tl.float16, [32768])),
        (src, 0, 128, _tensor(tl.bfloat16, [128])),
        (src, 0, 128, _tensor(tl.float16, [64])),
        (src, 0, 128, _tensor(tl.float16, [256])),
        (src, 0, 128, _tensor(tl.float16, [8, 16])),
        # Unsupported round modes: only CAST_NONE (0) is valid for s4 -> f16.
        (src, 1, 128, dst),
        (src, 6, 128, dst),
        # count must equal the output element count 2 * N.
        (src, 0, 64, dst),
        (src, 0, 256, dst),
    ]
    for args in invalid:
        try:
            _init(*args[:-1], out=args[-1])
        except AssertionError:
            continue
        raise AssertionError("cast_int4_to_fp16 accepted an invalid signature")
    print(f"[PASS] cast registration: nine valid and {len(invalid)} rejected signatures")


def main():
    test_validation()
    test_cast_int4_to_fp16()
    bench_cast_int4_to_fp16()
    print("All cast custom op tests passed.")


if __name__ == "__main__":
    main()
