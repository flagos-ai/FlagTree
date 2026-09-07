# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.ir.distributed_type import BlockCyclicSplit, ContiguousSplit
from triton.flagmega.passes.auto_distributed.split_candidates import (
    DistributedSplitCandidateContext,
)
from triton.flagmega.targets.pyntt_split import PyNttDistributedSplitCandidateProvider


def test_pyntt_split_stages_follow_physical_levels():
    tensor = fm.tensor_type("bfloat16", (4096,))
    placement = fm.Placement((2, 4, 8), "cyx", "cbb")
    context = DistributedSplitCandidateContext(
        tensor,
        0,
        placement,
        (0, 1, 2),
        64,
        4096,
    )

    candidates = PyNttDistributedSplitCandidateProvider(128).get_candidates(context)

    assert len(candidates) == 2
    staged, contiguous = candidates
    assert staged.stages[0].hierarchy_axes == (0,)
    assert isinstance(staged.stages[0].distribution, ContiguousSplit)
    assert staged.stages[1].hierarchy_axes == (1, 2)
    assert staged.stages[1].distribution == BlockCyclicSplit(64)
    assert contiguous == fm.SBP.split_contiguous((0, 1, 2), 64)


def test_pyntt_split_uses_physical_element_bytes_not_vendor_or_model_names():
    placement = fm.Placement((16,), "x", "b")
    output = fm.tensor_type(fm.VectorType(fm.DType.BFLOAT16, (8,)), (256,))
    context = DistributedSplitCandidateContext(output, 0, placement, (0,), 16, 256)

    candidates = PyNttDistributedSplitCandidateProvider(128).get_candidates(context)

    assert candidates[0] == fm.SBP.split_block_cyclic((0,), 8)


def test_pyntt_keeps_block_cyclic_reduction_candidates_for_late_kernel_selection():
    placement = fm.Placement((8, 16), "yx", "bb")
    tensor = fm.tensor_type("bfloat16", (6144,))
    base = PyNttDistributedSplitCandidateProvider(128)
    output_context = DistributedSplitCandidateContext(
        tensor, 0, placement, (0, 1), 48, 6144, "output"
    )
    reduction_context = DistributedSplitCandidateContext(
        tensor, 0, placement, (0, 1), 48, 6144, "reduction"
    )

    output_candidates = base.get_candidates(output_context)
    reduction_candidates = base.get_candidates(reduction_context)

    assert any(not candidate.is_contiguous for candidate in output_candidates)
    assert any(not candidate.is_contiguous for candidate in reduction_candidates)
    assert fm.SBP.split_contiguous((0, 1), 48) in reduction_candidates


@pytest.mark.parametrize("block_bytes", (0, 3, -8))
def test_pyntt_split_rejects_invalid_block_bytes(block_bytes):
    with pytest.raises(ValueError, match="positive power of two"):
        PyNttDistributedSplitCandidateProvider(block_bytes)
