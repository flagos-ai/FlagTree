"""Target the scalar atomic DMA predicates in native-style control flow.

The surrounding block-end/prompt-length branch is part of the native bincount
algorithm. General kernels without that branch remain separate diagnostics;
these tests do not replace their original failures or the frozen native gates.
"""

import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def _native_scalar_atomic_mask_contract(
    ids_ptr,
    bounds_ptr,
    counts_ptr,
    old_values_ptr,
    unrelated_ptr,
    BLOCK: tl.constexpr,
    SENTINEL: tl.constexpr,
):
    block_idx = tl.program_id(1)
    position = block_idx * BLOCK + tl.arange(0, BLOCK)
    lower = tl.load(bounds_ptr)
    upper = tl.load(bounds_ptr + 1)
    # Same dynamic block-end/prompt-length condition as native bincount.
    if (block_idx + 1) * BLOCK >= lower:
        ids = tl.load(ids_ptr + position)
        mask = (position < upper) & (position >= lower)
        old = tl.atomic_add(counts_ptr + ids, 1, mask=mask, sem="acq_rel")
        tl.store(old_values_ptr + position, tl.where(mask, old, SENTINEL))
        # A repair must not move this independent side effect under the RMW mask.
        tl.store(unrelated_ptr + position, position + 23)


@pytest.mark.parametrize(
    "lower,upper",
    [(0, 0), (0, 1024), (72, 175), (135, 185), (40, 41), (1024, 1024)],
)
def test_native_control_flow_scalar_mask(lower, upper):
    block, bins, initial, sentinel = 1024, 17, 9, -777
    position = torch.arange(block, dtype=torch.int32)
    ids = (position * 13 + 7) % bins
    bounds = torch.tensor([lower, upper], dtype=torch.int32)
    ids_device = ids.to("cuda")
    bounds_device = bounds.to("cuda")
    counts = torch.full((bins, ), initial, dtype=torch.int32, device="cuda")
    old = torch.full((block, ), sentinel, dtype=torch.int32, device="cuda")
    unrelated = torch.full((block, ), -1, dtype=torch.int32, device="cuda")
    _native_scalar_atomic_mask_contract[(1, 1)](
        ids_device,
        bounds_device,
        counts,
        old,
        unrelated,
        BLOCK=block,
        SENTINEL=sentinel,
    )
    torch.cuda.synchronize()
    active = (position < upper) & (position >= lower)
    expected = torch.bincount(ids[active].long(), minlength=bins).to(torch.int32)
    torch.testing.assert_close(counts.cpu(), expected + initial, rtol=0, atol=0)
    torch.testing.assert_close(unrelated.cpu(), position + 23, rtol=0, atol=0)
    old = old.cpu()
    torch.testing.assert_close(old[~active], torch.full_like(old[~active], sentinel), rtol=0, atol=0)
    # Atomic execution order is unspecified. For repeated IDs, the multiset of
    # returned old values must still be the complete consecutive count range.
    for token in range(bins):
        observed = old[active & (ids == token)].sort().values
        wanted = torch.arange(initial, initial + int(expected[token]), dtype=torch.int32)
        torch.testing.assert_close(observed, wanted, rtol=0, atol=0)
