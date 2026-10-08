# Copyright 2026- Xcoresigma Technology Co., Ltd
"""
@dialect(name="cann") custom op end-to-end tests
===================================================

Exercises the kernel-level custom op channel (tle_raw.call with a @dialect
function object + output_indices) against the same bitcode symbols used by
the registry-based tests in test_custom_ops.py:

  - gather_gm_to_l1   via format="bitcode": references the prebuilt
                      custom_ops.bc directly, no compilation involved. The
                      gathered L1/CBUF tile feeds a following tl.dot on the
                      CUBE core (CUBE / PIPE_MTE2).
  - gather_gm_to_ub   via source JIT (format omitted): compiles
                      mem_ops/gather_gm_to_ub.cpp with ccec on first use and
                      caches the produced bitcode under ~/.triton/cache. The
                      gathered UB tile is stored back to GM for verification
                      (VECTOR / PIPE_MTE2).

Correctness only, no benchmarking.
"""

from pathlib import Path

import torch
import torch_npu
import triton
import triton.experimental.tle.language.raw as tle_raw
import triton.language as tl
from triton.experimental.tle.raw import dialect

DEVICE = "npu"

CUSTOM_OPS_DIR = Path(__file__).resolve().parents[3] / "triton" / "experimental" / "tle" / \
    "language" / "dsa" / "ascend" / "custom_ops"

# ══════════════════════════════════════════════════════════════════════════
# Dialect-bound custom ops
# ══════════════════════════════════════════════════════════════════════════


# format="bitcode": invoke the prebuilt custom_ops.bc directly. CUBE core,
# so the extern symbol is the dav-c220-cube variant.
@dialect(
    name="cann",
    format="bitcode",
    file=CUSTOM_OPS_DIR / "custom_ops.bc",
    extern_func_name="custom_gather_gm_to_l1_half",
    pipeline={"core": "cube", "pipe": "PIPE_MTE2"},
)
def gather_gm_to_l1(*args, **kwargs):
    ...


# format omitted: the input defaults to source; gather_gm_to_ub.cpp is
# JIT-compiled to bitcode with ccec on first use and cached under
# ~/.triton/cache (vector core -> dav-c220-vec).
@dialect(
    name="cann",
    file=CUSTOM_OPS_DIR / "mem_ops" / "gather_gm_to_ub.cpp",
    extern_func_name="custom_gather_gm_to_ub_half",
    pipeline={"core": "vector", "pipe": "PIPE_MTE2"},
)
def gather_gm_to_ub(*args, **kwargs):
    ...


# ══════════════════════════════════════════════════════════════════════════
# Kernels
# ══════════════════════════════════════════════════════════════════════════


@triton.jit
def gather_gm_to_l1_dot_kernel(
    src,
    src_index,
    query,
    output,
    NUM_ROWS: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    D: tl.constexpr,
):
    src_2d = tl.make_block_ptr(
        base=src,
        shape=(NUM_ROWS, D),
        strides=(D, 1),
        offsets=(0, 0),
        block_shape=(TILE_SIZE, D),
        order=(1, 0),
    )
    src_index_2d = tl.make_block_ptr(
        base=src_index,
        shape=(TILE_SIZE, 1),
        strides=(1, 1),
        offsets=(0, 0),
        block_shape=(TILE_SIZE, 1),
        order=(1, 0),
    )
    tile_k = tl.full((TILE_SIZE, D), 0, tl.float16)
    # args[4] (`tile_k`) is the output/aliased operand; the exported extern
    # function takes (src, index, tile_size, D, dst).
    tile_k = tle_raw.call(gather_gm_to_l1, [src_2d, src_index_2d, TILE_SIZE, D, tile_k], output_indices=[4])

    query_2d = tl.make_block_ptr(
        base=query,
        shape=(TILE_SIZE, D),
        strides=(D, 1),
        offsets=(0, 0),
        block_shape=(TILE_SIZE, D),
        order=(1, 0),
    )
    tile_q = tl.load(query_2d)
    result = tl.dot(tile_q, tl.trans(tile_k))

    output_2d = tl.make_block_ptr(
        base=output,
        shape=(TILE_SIZE, TILE_SIZE),
        strides=(TILE_SIZE, 1),
        offsets=(0, 0),
        block_shape=(TILE_SIZE, TILE_SIZE),
        order=(1, 0),
    )
    tl.store(output_2d, result)


@triton.jit
def gather_gm_to_ub_store_kernel(
    src,
    src_index,
    output,
    NUM_ROWS: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    D: tl.constexpr,
):
    src_2d = tl.make_block_ptr(
        base=src,
        shape=(NUM_ROWS, D),
        strides=(D, 1),
        offsets=(0, 0),
        block_shape=(TILE_SIZE, D),
        order=(1, 0),
    )
    src_index_2d = tl.make_block_ptr(
        base=src_index,
        shape=(TILE_SIZE, 1),
        strides=(1, 1),
        offsets=(0, 0),
        block_shape=(TILE_SIZE, 1),
        order=(1, 0),
    )
    tile_v = tl.full((TILE_SIZE, D), 0, tl.float16)
    # args[4] (`tile_v`) is the output/aliased operand; the exported extern
    # function takes (src, index, tile_size, D, dst).
    tile_v = tle_raw.call(gather_gm_to_ub, [src_2d, src_index_2d, TILE_SIZE, D, tile_v], output_indices=[4])

    output_2d = tl.make_block_ptr(
        base=output,
        shape=(TILE_SIZE, D),
        strides=(D, 1),
        offsets=(0, 0),
        block_shape=(TILE_SIZE, D),
        order=(1, 0),
    )
    tl.store(output_2d, tile_v)


# ══════════════════════════════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════════════════════════════

NUM_ROWS = 32
TILE_SIZE = 16
D = 16
# Both lists contain adjacent index pairs (e.g. 3,4 / 9,10 / 0,1 / 12,13) so
# the two-row coalesced copy path is exercised as well.
L1_INDEX = [3, 4, 9, 10, 2, 7, 0, 1, 12, 13, 5, 8, 11, 6, 15, 14]
UB_INDEX = [20, 21, 19, 30, 31, 25, 24, 18, 27, 28, 22, 29, 26, 23, 17, 16]


def test_gather_gm_to_l1_dialect():
    torch.manual_seed(0)
    src = torch.randn((NUM_ROWS, D), dtype=torch.float16, device=DEVICE)
    query = torch.randn((TILE_SIZE, D), dtype=torch.float16, device=DEVICE)
    output = torch.empty((TILE_SIZE, TILE_SIZE), dtype=torch.float32, device=DEVICE)
    src_index = torch.tensor(L1_INDEX, dtype=torch.int32, device=DEVICE)

    gather_gm_to_l1_dot_kernel[(1, )](
        src,
        src_index,
        query,
        output,
        NUM_ROWS=NUM_ROWS,
        TILE_SIZE=TILE_SIZE,
        D=D,
        enable_legacy_insert_load_store_for_mix_cv=True,
    )
    torch_npu.npu.synchronize()

    gathered_src = src[src_index.long(), :]
    expected = torch.matmul(query.float(), gathered_src.float().transpose(0, 1))
    torch.testing.assert_close(output.cpu(), expected.cpu(), rtol=1e-3, atol=1e-3)
    print("[PASS] dialect gather_gm_to_l1 (bitcode) dot correctness (fp16)")


def test_gather_gm_to_ub_dialect():
    torch.manual_seed(0)
    src = torch.randn((NUM_ROWS, D), dtype=torch.float16, device=DEVICE)
    output = torch.empty((TILE_SIZE, D), dtype=torch.float16, device=DEVICE)
    src_index = torch.tensor(UB_INDEX, dtype=torch.int32, device=DEVICE)

    gather_gm_to_ub_store_kernel[(1, )](
        src,
        src_index,
        output,
        NUM_ROWS=NUM_ROWS,
        TILE_SIZE=TILE_SIZE,
        D=D,
        enable_legacy_insert_load_store_for_mix_cv=True,
    )
    torch_npu.npu.synchronize()

    expected = src[src_index.long(), :]
    torch.testing.assert_close(output.cpu(), expected.cpu(), rtol=0, atol=0)
    print("[PASS] dialect gather_gm_to_ub (source JIT) store correctness (fp16)")


def main():
    test_gather_gm_to_l1_dialect()
    test_gather_gm_to_ub_dialect()
    print("\nAll @dialect(name='cann') custom op tests passed.")


if __name__ == "__main__":
    main()
