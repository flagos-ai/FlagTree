# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
import pytest
import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl


@triton.jit
def _maximum(left, right):
    return tl.maximum(left, right)


@triton.jit
def _full_prefix(Input, Prefix, REVERSE: tl.constexpr):
    rows = tl.arange(0, 64)
    columns = tl.arange(0, 16)
    values = tl.load(Input + rows[:, None] * 16 + columns[None, :])
    prefix = tl.associative_scan(values, 0, _maximum, reverse=REVERSE)
    tl.store(Prefix + rows[:, None] * 16 + columns[None, :], prefix)


@triton.jit
def _scan_gather(Input, Indices, Output, WARPS: tl.constexpr, REVERSE: tl.constexpr):
    rows = tl.arange(0, 64)
    columns = tl.arange(0, 16)
    values = tl.load(Input + rows[:, None] * 16 + columns[None, :])
    prefix = tl.associative_scan(values, 0, _maximum, reverse=REVERSE)
    index = tl.load(Indices + tl.program_id(0))
    row_layout: tl.constexpr = tle.gpu.SlicedEncoding(1, tle.gpu.MmaEncoding([2, 0], [1, WARPS], [16, 8]))
    selected_rows = tl.full((1, 16), index, tl.int32)
    selected = tle.gpu.set_layout(tl.gather(prefix, selected_rows, 0).reshape((16, )), row_layout)
    tl.store(Output + tl.program_id(0) * 16 + columns, selected)


@pytest.mark.parametrize('dtype', [torch.float32, torch.int32])
@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('warps', [4, 8])
def test_scan_gather_keeps_explicit_result_encoding(dtype, reverse, warps):
    if not torch.cuda.is_available():
        pytest.skip('CUDA is required')
    torch.manual_seed(1293)
    if dtype == torch.float32:
        values = torch.randn((64, 16), device='cuda', dtype=dtype)
        values[:, 0] = -1
        values[16, 0] = -0.0
        values[32, 0] = 0.0
        values[8, 1] = float('nan')
        values[48, 2] = float('inf')
        values[:, 3] = -float('inf')
    else:
        values = torch.randint(-100, 100, (64, 16), device='cuda', dtype=dtype)
        values[8, 1] = torch.iinfo(dtype).max
        values[:, 3] = torch.iinfo(dtype).min
    indices = torch.tensor([0, 1, 17, 31, 32, 63], device='cuda', dtype=torch.int32)
    reference = torch.empty_like(values)
    output = torch.empty((6, 16), device='cuda', dtype=dtype)
    options = {'num_warps': warps, 'enable_fp_fusion': False}
    _full_prefix[(1, )](values, reference, reverse, **options)
    compiled = _scan_gather[(6, )](values, indices, output, warps, reverse, **options)
    expected = reference[indices.to(torch.int64)]
    assert torch.equal(output.view(torch.uint8), expected.view(torch.uint8))
    assert compiled.n_spills == 0
