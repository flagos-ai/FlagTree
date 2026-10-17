# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A tensor reduction may feed a different explicit consumer layout."""

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle


_MATRIX = tl.constexpr(tle.gpu.BlockEncoding([1, 1], [4, 8], [1, 4], [1, 0]))
_ROW = tl.constexpr(tle.gpu.SlicedEncoding(1, _MATRIX.value))


@triton.jit
def _reduce_to_matrix(source, output, CHUNKS: tl.constexpr):
    rows = tl.arange(0, 16)[:, None]
    k = tl.arange(0, 128)[None, :]
    total = tl.zeros((4, 4), output.dtype.element_ty)
    for chunk in tl.range(0, CHUNKS):
        values = tl.load(source + rows * (128 * CHUNKS) + chunk * 128 + k)
        values = tl.reshape(values, (4, 4, 128))
        total += tl.sum(values, 2)
    total = tle.gpu.set_layout(total, _MATRIX)
    offsets = tl.arange(0, 4)[:, None] * 4 + tl.arange(0, 4)[None, :]
    tl.store(output + offsets, total)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.require_tle("gpu.set_layout")
@pytest.mark.parametrize("chunks", (1, 3))
@pytest.mark.parametrize("dtype", (torch.float32, torch.int32))
def test_tensor_reduction_to_explicit_matrix_layout(chunks, dtype):
    values = torch.arange(16 * 128 * chunks, dtype=dtype, device="cuda").reshape(16, -1)
    output = torch.empty((4, 4), dtype=dtype, device="cuda")
    for _ in range(2):
        _reduce_to_matrix[(1,)](values, output, chunks, num_warps=4)
        torch.testing.assert_close(output, values.sum(1).reshape(4, 4).to(dtype), rtol=0, atol=0)
        values.add_(1)


@triton.jit
def _shared_mask(output, row_output, limit):
    index = tl.arange(0, 4)
    active = index < limit
    pair = tle.gpu.set_layout(active[:, None] & active[None, :], _MATRIX)
    rows = tle.gpu.set_layout(active[:, None], _MATRIX)
    offsets = index[:, None] * 4 + index[None, :]
    tl.store(output + offsets, pair)
    tl.store(row_output + index[:, None], rows)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.require_tle("gpu.set_layout")
@pytest.mark.parametrize("limit", (-1, 0, 3, 4, 5))
def test_shared_predicate_expands_in_two_layout_domains(limit):
    pair = torch.empty((4, 4), dtype=torch.bool, device="cuda")
    rows = torch.empty((4, 1), dtype=torch.bool, device="cuda")
    _shared_mask[(1,)](pair, rows, limit, num_warps=4)
    active = torch.arange(4, device="cuda") < limit
    torch.testing.assert_close(pair, active[:, None] & active[None, :])
    torch.testing.assert_close(rows, active[:, None])


@triton.jit
def _expand_other_axis(output):
    index = tle.gpu.set_layout(tl.arange(0, 4), _ROW)
    expanded = tle.gpu.set_layout(index[None, :], _MATRIX)
    tl.store(output + expanded, expanded)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.require_tle("gpu.set_layout")
def test_expand_explicit_source_along_another_axis():
    output = torch.empty((1, 4), dtype=torch.int32, device="cuda")
    _expand_other_axis[(1,)](output, num_warps=4)
    torch.testing.assert_close(output, torch.arange(4, device="cuda", dtype=torch.int32)[None, :])
