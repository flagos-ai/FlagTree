# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.passes.auto_distributed import DistributedCandidateContext
from triton.flagmega.passes.auto_distributed.providers import MatMulCandidateProvider
from triton.flagmega.targets.pyntt_split import PyNttDistributedSplitCandidateProvider

from .helpers import packed_matmul_module


def _candidates(size: int):
    module = packed_matmul_module(size=size)
    node = module.node_map["output"]
    placement = fm.Placement((8, 16), "yx", "bb")
    context = DistributedCandidateContext(
        module,
        node,
        placement,
        tuple((module.node_map[value].type,) for value in node.inputs),
        PyNttDistributedSplitCandidateProvider(block_bytes=128),
    )
    return MatMulCandidateProvider().get_candidates(context)


def test_block_fp8_reduction_candidates_are_contiguous_scale_group_aligned():
    reductions = tuple(
        candidate
        for candidate in _candidates(1024)
        if isinstance(candidate.return_type, fm.DistributedType)
        and candidate.return_type.partial is not None
    )

    assert reductions
    for candidate in reductions:
        lhs = candidate.input_types[0]
        split = lhs.axis_policies[-1]
        assert isinstance(split, fm.SBPSplit)
        assert split.is_contiguous
        assert fm.local_shape(lhs)[-1].fixed_value % 128 == 0


def test_block_fp8_does_not_offer_partial_sum_when_local_k_is_subgroup():
    assert not any(
        isinstance(candidate.return_type, fm.DistributedType)
        and candidate.return_type.partial is not None
        for candidate in _candidates(128)
    )
