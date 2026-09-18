# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Standalone correctness checks for the signed INT8-to-FP16 custom op."""

import numpy as np
import torch
import torch_npu  # noqa: F401
import triton
import triton.experimental.tle as tle
import triton.language as tl
from triton.experimental.tle.language.dsa.ascend.custom_ops import cast_int8_to_fp16

CAST_NONE = tl.constexpr(0)


@triton.jit
def cast_kernel_int8(X, Out, N: tl.constexpr):
    packed = tl.load(X + tl.arange(0, N))
    values = tl.full((N, ), 0, tl.float16)
    values = tle.dsa.ascend.raw("cast_int8_to_fp16", packed, CAST_NONE, N, out=values)
    tl.store(Out + tl.arange(0, N), values)


def test_cast_int8_to_fp16():
    cases = 0
    for n in (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192):
        for start in (0, 1, 64, 200):
            if start >= 256:
                continue
            packed = ((np.arange(n, dtype=np.int32) + start) % 256 - 128).astype(np.int8)
            x = torch.from_numpy(packed).to("npu")
            storage = torch.full((n + 32, ), 123, dtype=torch.float16, device="npu")
            out = storage[16:-16]
            cast_kernel_int8[(1, )](x, out, N=n)
            np.testing.assert_array_equal(out.cpu().numpy(), packed.astype(np.float16))
            np.testing.assert_array_equal(storage[:16].cpu().numpy(), np.full(16, 123))
            np.testing.assert_array_equal(storage[-16:].cpu().numpy(), np.full(16, 123))
            cases += 1
    print(f"[PASS] cast_int8_to_fp16: {cases} size/encoding cases")


def test_graph_replay_int8():
    n = 4096
    x = torch.zeros(n, dtype=torch.int8, device="npu")
    out = torch.empty(n, dtype=torch.float16, device="npu")
    for _ in range(3):
        cast_kernel_int8[(1, )](x, out, N=n)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        cast_kernel_int8[(1, )](x, out, N=n)
    for value in (-128, -1, 0, 127):
        x.fill_(value)
        graph.replay()
        np.testing.assert_array_equal(out.cpu().numpy(), np.full(n, value, dtype=np.int8).astype(np.float16))
    print("[PASS] cast_int8 graph replay with four changed inputs")


def _tensor(dtype, shape):
    return tl.tensor(None, tl.block_type(dtype, shape))


def _init(*args, **kwargs):
    op = cast_int8_to_fp16.__new__(cast_int8_to_fp16)
    op.arg_type = {}
    op.__init__(*args, **kwargs)
    return op


def test_validation_int8():
    for n in (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192):
        op = _init(_tensor(tl.int8, [n]), 0, n, out=_tensor(tl.float16, [n]))
        assert op.symbol == "custom_cast_int8_to_fp16"
        assert op.bitcode.endswith("custom_ops.bc")
    src = _tensor(tl.int8, [64])
    dst = _tensor(tl.float16, [64])
    invalid = [
        (src, 0, 64, None),
        (_tensor(tl.uint8, [64]), 0, 64, dst),
        (_tensor(tl.float16, [64]), 0, 64, dst),
        (_tensor(tl.int8, [8, 8]), 0, 64, dst),
        (_tensor(tl.int8, [16]), 0, 16, _tensor(tl.float16, [16])),
        (_tensor(tl.int8, [16384]), 0, 16384, _tensor(tl.float16, [16384])),
        (src, 0, 64, _tensor(tl.bfloat16, [64])),
        (src, 0, 64, _tensor(tl.float16, [32])),
        (src, 0, 64, _tensor(tl.float16, [128])),
        (src, 0, 64, _tensor(tl.float16, [8, 8])),
        # Unsupported round modes: only CAST_NONE (0) is valid for s8 -> f16.
        (src, 1, 64, dst),
        (src, 6, 64, dst),
        # count must equal the element count.
        (src, 0, 32, dst),
        (src, 0, 128, dst),
    ]
    for args in invalid:
        try:
            _init(*args[:-1], out=args[-1])
        except AssertionError:
            continue
        raise AssertionError("cast_int8_to_fp16 accepted an invalid signature")
    print(f"[PASS] cast_int8 registration: nine valid and {len(invalid)} rejected signatures")


def main():
    test_validation_int8()
    test_cast_int8_to_fp16()
    test_graph_replay_int8()
    print("All cast_int8 custom op correctness tests passed.")


if __name__ == "__main__":
    main()
