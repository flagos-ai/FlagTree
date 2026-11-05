# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Physical arenas and masked copies remain distinct from logical matrices."""

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle


@triton.jit
def _arena_masked_copy(source, output, ASYNC: tl.constexpr, FILL: tl.constexpr):
    arena = tle.gpu.alloc([12288], tl.uint8, nv_mma_shared_layout=False, alignment_bytes=1024)
    stage = tle.gpu.alloc([4, 32], tl.float32, alias=arena, alias_offset_bytes=4096, nv_mma_shared_layout=False)
    row = tl.arange(0, 4)[:, None]
    column = tl.arange(0, 32)[None, :]
    tle.gpu.copy(source + row * 32 + column, stage, [4, 32], mask=row < 2, other=FILL, is_async=ASYNC)
    if ASYNC:
        tle.gpu.async_commit_group()
        tle.gpu.async_wait_group(0)
    values = tl.load(tle.gpu.local_ptr(stage, (row, column), [4, 32]))
    tl.store(output + row * 32 + column, values)


@pytest.mark.parametrize("asynchronous,fill", [(False, 0.), (False, float("-inf")), (True, 0.)])
def test_exact_linear_arena_and_masked_copy(asynchronous, fill):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("requires NVIDIA Hopper")
    source = torch.arange(128, dtype=torch.float32, device="cuda").reshape(4, 32)
    output = torch.full_like(source, float("nan"))
    kernel = _arena_masked_copy[(1, )](source, output, asynchronous, fill, num_warps=4)
    expected = torch.full_like(source, fill)
    expected[:2] = source[:2]
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    assert "tle.logical_shape" not in kernel.asm["ttir"]
    assert "memdesc<12288xi8" in kernel.asm["ttgir"]
