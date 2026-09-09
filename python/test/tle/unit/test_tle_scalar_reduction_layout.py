# Copyright 2026 FlagOS Contributors

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle


_MATRIX = tl.constexpr(tle.gpu.BlockEncoding([1, 8], [2, 16], [4, 1], [1, 0]))
_SLICE = tl.constexpr(tle.gpu.SlicedEncoding(0, _MATRIX.value))
_FLAT = tl.constexpr(tle.gpu.BlockEncoding([1], [32], [4], [0]))


@triton.jit
def _reduce_with_layout(source, output, N: tl.constexpr, SLICE: tl.constexpr):
    offsets = tl.arange(0, 128)
    values = tl.load(source + offsets, offsets < N, other=0)
    if SLICE:
        values = tle.gpu.set_layout(values, _SLICE)
    else:
        values = tle.gpu.set_layout(values, _FLAT)
    total = tl.sum(values, 0)
    # The scalar can feed a different tensor domain; it has no layout itself.
    result = tle.gpu.set_layout(tl.full((128,), total, tl.float32), _FLAT)
    tl.store(output + offsets, result)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.require_tle("gpu.set_layout")
@pytest.mark.parametrize("use_slice", [False, True])
@pytest.mark.parametrize("n", [0, 73, 128])
def test_scalar_reduction_terminates_tensor_layout_propagation(use_slice, n):
    source = torch.arange(128, dtype=torch.float32, device="cuda")
    output = torch.empty_like(source)
    _reduce_with_layout[(1,)](source, output, n, use_slice, num_warps=4)
    torch.testing.assert_close(output, torch.full_like(source, n * (n - 1) / 2), rtol=0, atol=0)
