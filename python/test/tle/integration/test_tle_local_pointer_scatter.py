# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A full-shaped local pointer view must preserve its explicit indices."""

import pytest
import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl


@triton.jit
def _scatter(values, indices, output, MASKED: tl.constexpr):
    slots = tl.arange(0, 128)
    buffer = tle.gpu.alloc((128,), tl.int32, nv_mma_shared_layout=False)
    tl.store(tle.gpu.local_ptr(buffer, (slots,)), -1)
    tl.debug_barrier()
    destination = tl.load(indices + slots)
    value = tl.load(values + slots)
    mask = slots % 3 != 0 if MASKED else tl.full((128,), True, tl.int1)
    tl.store(tle.gpu.local_ptr(buffer, (destination,)), value, mask)
    tl.debug_barrier()
    tl.store(output + slots, tl.load(tle.gpu.local_ptr(buffer, (slots,))))


@pytest.mark.parametrize("masked", [False, True])
def test_dynamic_full_shape_scatter_preserves_destinations(masked):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    values = torch.arange(128, device="cuda", dtype=torch.int32) + 100
    indices = torch.arange(127, -1, -1, device="cuda", dtype=torch.int32).roll(13)
    output = torch.empty_like(values)
    _scatter[(1,)](values, indices, output, masked, num_warps=4)
    expected = torch.full_like(values, -1)
    mask = torch.arange(128, device="cuda") % 3 != 0 if masked else torch.ones(128, device="cuda", dtype=torch.bool)
    expected[indices[mask].long()] = values[mask]
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
