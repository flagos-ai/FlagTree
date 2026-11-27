# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.ir.distributed_type import scale_split_units
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


def test_packed_fp8_output_candidates_scale_n_and_preserve_scalar_work():
    from python.test.flagmega.passes.functions.post_boundary_pack.test_block_fp8_producer_packing import Fp8Consumers

    costs = []
    for lanes in ((), (2, 4)):
        module = Fp8Consumers(n=1024, k=1024, output_lanes=lanes, lanes=(1,)).build()
        node = module.node_map["projection"]
        placement = fm.Placement((8, 16), "yx", "bb")
        context = DistributedCandidateContext(module, node, placement,
                                              tuple((module.node_map[name].type,) for name in node.inputs),
                                              PyNttDistributedSplitCandidateProvider(block_bytes=128))
        candidates = MatMulCandidateProvider().get_candidates(context)
        assert candidates
        costs.append(next(candidate.operation_cost for candidate in candidates if candidate.reason == "broadcast-replicated"))
        for candidate in candidates:
            inputs = tuple(fm.Node(str(index), "builtin.var", (), value, attrs={"name": str(index)})
                           for index, value in enumerate(candidate.input_types))
            assert fm.get_definition(node.op).infer_call_type(inputs, node.attrs) == candidate.return_type
            policy = candidate.return_type.axis_policies[-1]
            if isinstance(policy, fm.SBPSplit):
                lane_count = 8 if lanes else 1
                assert scale_split_units(candidate.input_types[1].axis_policies[0], 1, lane_count) == policy
    assert costs[0] == costs[1]
