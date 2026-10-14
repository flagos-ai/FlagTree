"""Expansion of explicitly sliced TLE tensor layouts."""
import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle


@triton.jit
def _slice_expand(source, output, AXIS: tl.constexpr, WARPS: tl.constexpr):
    parent: tl.constexpr = tle.gpu.BlockEncoding([1, 1], [1, 32], [WARPS, 1], [1, 0])
    vector_layout: tl.constexpr = tle.gpu.SlicedEncoding(1, parent)
    offsets = tl.arange(0, 8)
    value = tle.gpu.set_layout(tl.load(source + offsets), vector_layout)
    if AXIS == 1:
        expanded = tl.broadcast_to(value[:, None], (8, 32))
        destination = tl.arange(0, 8)[:, None] * 32 + tl.arange(0, 32)[None, :]
    else:
        expanded = tl.broadcast_to(value[None, :], (32, 8))
        destination = tl.arange(0, 32)[:, None] * 8 + tl.arange(0, 8)[None, :]
    tl.store(output + destination, expanded)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.require_tle('gpu.set_layout')
@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('warps', [4, 8])
def test_slice_layout_expands_at_matching_and_new_axis(axis, warps):
    source = torch.arange(8, device='cuda', dtype=torch.float32)
    output = torch.empty(256, device='cuda')
    _slice_expand[(1, )](source, output, axis, warps, num_warps=warps)
    expected = source[:, None].expand(8, 32) if axis == 1 else source[None, :].expand(32, 8)
    torch.testing.assert_close(output, expected.flatten(), rtol=0, atol=0)
