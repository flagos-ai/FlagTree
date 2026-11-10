# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Regression for #1586: tiled zero fills must retain async pipe transport."""

import pytest
import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl
from triton._internal_testing import is_hopper


@triton.jit
def _produce(src, valid_rows, writer, N: tl.constexpr):
    rows = tl.arange(0, 128)
    cols = tl.arange(0, 256)
    pointers = src + rows[:, None] * 256 + cols[None, :]

    first = writer.acquire(0)
    # Logical tiling wraps the default zero fill in extract_tile/convert_layout.
    tle.gpu.copy(pointers, first.tile, [N, 256], mask=rows[:, None] < valid_rows)
    writer.commit(0)

    second = writer.acquire(1)
    tle.gpu.copy(pointers, second.tile, [N, 256])
    writer.commit(1)


@triton.jit
def _consume(reader, dst, N: tl.constexpr):
    a = tl.full((64, 256), 1, tl.bfloat16)
    acc = tl.zeros((64, 128), tl.float32)
    for i in tl.static_range(2):
        ready = reader.wait(i)
        acc = tle.gpu.wgmma(a, ready.slot.tile, acc, trans_b=True)
        acc = tle.gpu.wgmma_wait(0, acc)
        reader.release(i)
    rows = tl.arange(0, 64)
    cols = tl.arange(0, 128)
    tl.store(dst + rows[:, None] * N + cols[None, :], acc, cols[None, :] < N)


@triton.jit
def _copy_pipe(src, dst, valid_rows, N: tl.constexpr):
    buffer = tle.gpu.alloc([2, N, 256], dtype=tl.bfloat16, scope=tle.gpu.smem)
    pipe = tle.pipe(capacity=2, name="tiled_zero_fill", tile=buffer)
    tle.gpu.warp_specialize(
        [(_consume, (pipe.reader(), dst, N)), (_produce, (src, valid_rows, pipe.writer(), N))],
        [4],
        [64],
    )


@pytest.mark.skipif(not is_hopper(), reason="requires NVIDIA Hopper WGMMA")
@pytest.mark.parametrize("n", [80, 128])
def test_logical_zero_fill_preserves_pipe_transport(n, fresh_triton_cache):
    # Both logical copies read within this physical carrier, including row 127.
    generator = torch.Generator(device="cuda").manual_seed(0)
    source = torch.randn((128, 256), generator=generator, device="cuda", dtype=torch.bfloat16)
    output = torch.empty((64, n), device="cuda", dtype=torch.float32)
    _copy_pipe[(1, )](source, output, 73, n, num_warps=4)

    expected = source[:n].float().sum(dim=1)
    expected *= 1 + (torch.arange(n, device="cuda") < 73).float()
    torch.testing.assert_close(output, expected[None, :].expand(64, n), atol=1e-3, rtol=1e-3)
