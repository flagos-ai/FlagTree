# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Native INT8 ND2NZ correctness and an optional equivalent-TLE benchmark."""

import argparse

import numpy as np
import torch
import torch_npu  # noqa: F401
import triton
import triton.experimental.tle as tle
import triton.language as tl
from triton.experimental.tle.language.dsa.ascend.custom_ops import data_copy_gm_to_l1_nd2nz_int8


@triton.jit
def copy_dot_kernel(X, Identity, Out, ROW_STRIDE: tl.constexpr, SOURCE_OFFSET: tl.constexpr, M: tl.constexpr,
                    K: tl.constexpr, CUSTOM: tl.constexpr):
    rows = tl.arange(0, M)
    cols = tl.arange(0, K)
    src = tl.make_block_ptr(X + SOURCE_OFFSET, (M, K), (ROW_STRIDE, 1), (0, 0), (M, K), (1, 0))
    if CUSTOM:
        tile = tle.dsa.ascend.raw("data_copy_gm_to_l1_nd2nz_int8", src, 1, M, K, 0, ROW_STRIDE, M, 1, 1, out=tl.full(
            (M, K), 0, tl.int8))
    else:
        buf = tle.dsa.alloc([M, K], tl.int8, tle.dsa.ascend.L1)
        tle.dsa.copy(src, buf, [M, K])
        tile = tle.dsa.to_tensor(buf)
    identity = tl.load(Identity + cols[:, None] * K + cols[None, :])
    result = tl.dot(tile, identity, out_dtype=tl.int32)
    tl.store(Out + rows[:, None] * K + cols[None, :], result)


def _inputs(m, k, row_stride, offset):
    rng = np.random.default_rng(42)
    values = rng.integers(-128, 128, size=offset + m * row_stride + 32, dtype=np.int8)
    # Cover all encodings without making distant rows repeat modulo 256.
    values[:256] = np.arange(-128, 128, dtype=np.int16).astype(np.int8)
    expected = values[offset + np.arange(m)[:, None] * row_stride + np.arange(k)[None, :]].astype(np.int32)
    x = torch.from_numpy(values).to("npu")
    identity = torch.eye(k, dtype=torch.int32).to(torch.int8).to("npu")
    storage = torch.full((m * k + 64, ), 777777, dtype=torch.int32, device="npu")
    return x, identity, storage, expected


def test_data_copy_gm_to_l1_nd2nz_int8():
    cases = ((16, 32, 32, 0), (16, 128, 160, 32), (32, 128, 256, 64), (128, 128, 128, 32), (256, 128, 128, 64),
             (16, 128, 65504, 32))
    for m, k, row_stride, offset in cases:
        x, identity, storage, expected = _inputs(m, k, row_stride, offset)
        copy_dot_kernel[(1, )](x, identity, storage[32:-32], row_stride, offset, m, k, True,
                               enable_legacy_insert_load_store_for_mix_cv=True)
        torch.npu.synchronize()
        actual = storage.cpu().numpy()
        np.testing.assert_array_equal(actual[32:-32].reshape(m, k), expected)
        np.testing.assert_array_equal(actual[:32], np.full(32, 777777))
        np.testing.assert_array_equal(actual[-32:], np.full(32, 777777))
    print(f"[PASS] data_copy_gm_to_l1_nd2nz_int8: {len(cases)} layout/offset cases")


def bench_data_copy_gm_to_l1_nd2nz_int8():
    m, k, row_stride, offset = 128, 128, 160, 32
    x, identity, storage, expected = _inputs(m, k, row_stride, offset)

    def launch(custom):
        # Keep the compiler options and the identity-dot consumer identical.
        return copy_dot_kernel[(1, )](x, identity, storage[32:-32], row_stride, offset, m, k, custom,
                                      enable_legacy_insert_load_store_for_mix_cv=True)

    for custom in (False, True):
        launch(custom)
        torch.npu.synchronize()
        np.testing.assert_array_equal(storage[32:-32].cpu().numpy().reshape(m, k), expected)
    timings = {"tle": [], "custom": []}
    for order in ((False, True), (True, False)):
        for custom in order:
            name = "custom" if custom else "tle"
            timings[name].append(triton.testing.do_bench(lambda custom=custom: launch(custom), return_mode="median"))
    tle_ms = sum(timings["tle"]) / len(timings["tle"])
    custom_ms = sum(timings["custom"]) / len(timings["custom"])
    print(f"[BENCH] INT8 ND2NZ + identity dot: TLE {tle_ms * 1000:.3f} us, "
          f"custom {custom_ms * 1000:.3f} us, speedup {tle_ms / custom_ms:.3f}x; rounds_ms={timings}")


def _tensor(dtype, shape):
    return tl.tensor(None, tl.block_type(dtype, shape))


def _block_pointer(dtype, shape):
    return tl.tensor(None, tl.pointer_type(tl.block_type(dtype, shape)))


def test_validation():
    src = _block_pointer(tl.int8, [128, 128])
    dst = _tensor(tl.int8, [128, 128])
    args = [src, 1, 128, 128, 0, 160, 128, 1, 1]
    op = data_copy_gm_to_l1_nd2nz_int8(*args, out=dst)
    assert op.symbol == "custom_data_copy_gm_to_l1_nd2nz_int8"
    assert op.bitcode.endswith("custom_ops.bc")
    # Dynamic scalar parameters retain the native API and its runtime bounds.
    dynamic = [src] + [tl.tensor(None, tl.uint16) for _ in range(8)]
    data_copy_gm_to_l1_nd2nz_int8(*dynamic, out=dst)
    # Zero-size transfers are valid no-ops in the official API.
    noop = [src, 0, 0, 0, 65535, 65535, 16384, 16384, 65535]
    data_copy_gm_to_l1_nd2nz_int8(*noop, out=dst)
    invalid = []
    for position, value in ((1, -1), (1, 4096), (2, 16385), (3, 65536), (4, 65536), (5, 0), (5, 114688), (6, 0),
                            (6, 16385), (7, 0), (7, 16385), (8, 65536), (6, 256), (7, 2), (2, 1.5)):
        changed = args.copy()
        changed[position] = value
        invalid.append((changed, dst))
    # A dynamic source stride must not bypass an entirely static output bound.
    for position in (4, 5):
        changed = args.copy()
        changed[position] = tl.tensor(None, tl.uint16)
        changed[6] = 256
        invalid.append((changed, dst))
    # The custom-op argument converter lowers Python bool to i1 before applying
    # its declared type, which cannot match this primitive's uint16 ABI.
    for position in range(1, 9):
        changed = args.copy()
        changed[position] = True
        invalid.append((changed, dst))
    for bad_src in (_block_pointer(tl.float16, [128, 128]), _block_pointer(tl.int8, [16384]),
                    tl.tensor(None, tl.pointer_type(tl.int8)), _tensor(tl.int8, [128, 128])):
        invalid.append(([bad_src] + args[1:], dst))
    for bad_dst in (None, _tensor(tl.float16, [128, 128]), _tensor(tl.int8, [16384]), _tensor(tl.int8, [16, 128])):
        invalid.append((args, bad_dst))
    for bad_args, bad_dst in invalid:
        try:
            data_copy_gm_to_l1_nd2nz_int8(*bad_args, out=bad_dst)
        except AssertionError:
            continue
        raise AssertionError("data_copy_gm_to_l1_nd2nz_int8 accepted an invalid signature")
    print(f"[PASS] ND2NZ registration: three valid and {len(invalid)} rejected signatures")


def main():
    test_validation()
    test_data_copy_gm_to_l1_nd2nz_int8()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    main()
    if args.benchmark:
        bench_data_copy_gm_to_l1_nd2nz_int8()
