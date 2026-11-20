# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""TMA policies must survive dense and logical-tile lowering."""

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton._internal_testing import is_hopper

pytestmark = pytest.mark.skipif(not is_hopper(), reason="requires NVIDIA Hopper")


@triton.jit
def _policy_tma(source, result, ROWS: tl.constexpr, POLICY: tl.constexpr):
    desc = tl.make_tensor_descriptor(source, [ROWS, 256], [256, 1], [ROWS, 256])
    stage = tle.gpu.alloc([ROWS, 256], tl.float16)
    barrier = tle.gpu.alloc_barrier(expect_bytes=ROWS * 256 * 2)
    tle.gpu.copy(desc, stage, [ROWS, 256], [0, 0], barrier=barrier, eviction_policy=POLICY)
    tle.gpu.barrier_wait(barrier, phaseIdx=0)
    lhs = tl.full((64, 256), 1, tl.float16)
    values = tle.gpu.wgmma_wait(0, tle.gpu.wgmma(lhs, stage, trans_b=True))
    row = tl.arange(0, 64)[:, None]
    col = tl.arange(0, triton.next_power_of_2(ROWS))[None, :]
    tl.store(result + row * ROWS + col, values, col < ROWS)


@pytest.mark.parametrize("rows", [64, 80, 96])
@pytest.mark.parametrize("policy", ["", "evict_first", "evict_last"])
def test_tma_policy_reaches_ptx_and_executes(rows, policy):
    triton.set_allocator(lambda size, alignment, stream: torch.empty(size, device="cuda", dtype=torch.uint8))
    source = torch.ones((rows, 256), device="cuda", dtype=torch.float16)
    result = torch.empty((64, rows), device="cuda", dtype=torch.float32)
    kernel = _policy_tma[(1, )](source, result, rows, policy, num_warps=4)
    torch.testing.assert_close(result, torch.full_like(result, 256), rtol=0, atol=0)
    ptx = kernel.asm["ptx"]
    if policy:
        assert "createpolicy.fractional.L2::" + policy in ptx
        copies = [line for line in ptx.splitlines() if "cp.async.bulk.tensor" in line]
        assert copies and all("L2::cache_hint" in line for line in copies)
    else:
        assert "createpolicy.fractional" not in ptx
