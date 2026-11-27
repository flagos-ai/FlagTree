# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

"""GPU regression for the fused projection/residual/norm-stat reduction."""

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

import triton._C.libtriton as libtriton


_WEIGHT_LAYOUT = tl.constexpr(
    tle.gpu.BlockEncoding([1, 2], [2, 16], [8, 1], [1, 0])
)
_OUTPUT_LAYOUT = tl.constexpr(tle.gpu.SlicedEncoding(1, _WEIGHT_LAYOUT.value))
_STATS_LAYOUT = tl.constexpr(tle.gpu.SlicedEncoding(0, _WEIGHT_LAYOUT.value))
_HAS_TLE_EXPLICIT_LAYOUT = hasattr(
    libtriton.ir.builder, "ensure_ttg_layout_attrs"
)


@triton.jit
def _rank_one_sliced_all_dimensions_sum(source, partial_stats):
    offsets = tle.gpu.set_layout(tl.arange(0, 16), _OUTPUT_LAYOUT)
    values = tle.gpu.set_layout(tl.load(source + offsets), _OUTPUT_LAYOUT).to(tl.float32)
    parent_values = tle.gpu.set_layout(values[:, None], _WEIGHT_LAYOUT)
    square_sum = tl.sum(parent_values * parent_values, axis=0)
    stats_offset = tle.gpu.set_layout(tl.arange(0, 1), _STATS_LAYOUT)
    tl.store(partial_stats + stats_offset, square_sum)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(not _HAS_TLE_EXPLICIT_LAYOUT, reason="requires __TLE__ build")
def test_rank_one_sliced_norm_stats_reduce_through_parent_layout():
    """A rank-one sliced value must not create a rank-zero SliceEncoding."""

    source = torch.arange(1, 17, dtype=torch.float32, device="cuda")
    partial_stats = torch.zeros(1, dtype=torch.float32, device="cuda")

    compiled = _rank_one_sliced_all_dimensions_sum.warmup(
        source,
        partial_stats,
        grid=(1,),
        num_warps=8,
    )
    assert compiled is not None
    _rank_one_sliced_all_dimensions_sum[(1,)](
        source,
        partial_stats,
        num_warps=8,
    )

    torch.testing.assert_close(
        partial_stats.cpu(),
        torch.tensor([(source.cpu() ** 2).sum().item()], dtype=torch.float32),
    )
