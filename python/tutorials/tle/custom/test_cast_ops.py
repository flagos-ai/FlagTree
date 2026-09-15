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


@triton.jit
def cast_kernel(X, Out, N: tl.constexpr):
    packed = tl.load(X + tl.arange(0, N))
    values = tl.full((2 * N, ), 0, tl.float16)
    values = tle.dsa.ascend.raw("cast_int4_to_fp16", packed, out=values)
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


def test_graph_replay():
    n = 4096
    x = torch.zeros(n, dtype=torch.uint8, device="npu")
    out = torch.empty(2 * n, dtype=torch.float16, device="npu")
    for _ in range(3):
        cast_kernel[(1, )](x, out, N=n)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        cast_kernel[(1, )](x, out, N=n)
    for value in (0x78, 0xF0, 0x00):
        x.fill_(value)
        graph.replay()
        np.testing.assert_array_equal(out.cpu().numpy(), _reference(np.full(n, value, dtype=np.uint8)))
    print("[PASS] cast graph replay with three changed inputs")


def _tensor(dtype, shape):
    return tl.tensor(None, tl.block_type(dtype, shape))


def _init(src, out):
    op = cast_int4_to_fp16.__new__(cast_int4_to_fp16)
    op.arg_type = {}
    cast_int4_to_fp16.__init__(op, src, out=out)
    return op


def test_validation():
    for n in (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192):
        op = _init(_tensor(tl.uint8, [n]), _tensor(tl.float16, [2 * n]))
        assert op.symbol == "custom_cast_int4_to_fp16"
        assert op.bitcode.endswith("custom_ops.bc")
    src = _tensor(tl.uint8, [64])
    dst = _tensor(tl.float16, [128])
    invalid = [
        (src, None),
        (_tensor(tl.int8, [64]), dst),
        (_tensor(tl.float16, [64]), dst),
        (_tensor(tl.uint8, [8, 8]), dst),
        (_tensor(tl.uint8, [16]), _tensor(tl.float16, [32])),
        (_tensor(tl.uint8, [16384]), _tensor(tl.float16, [32768])),
        (src, _tensor(tl.bfloat16, [128])),
        (src, _tensor(tl.float16, [64])),
        (src, _tensor(tl.float16, [256])),
        (src, _tensor(tl.float16, [8, 16])),
    ]
    for source, output in invalid:
        try:
            _init(source, output)
        except AssertionError:
            continue
        raise AssertionError("cast_int4_to_fp16 accepted an invalid signature")
    print(f"[PASS] cast registration: nine valid and {len(invalid)} rejected signatures")


def main():
    test_validation()
    test_cast_int4_to_fp16()
    test_graph_replay()
    print("All cast custom op correctness tests passed.")


if __name__ == "__main__":
    main()
