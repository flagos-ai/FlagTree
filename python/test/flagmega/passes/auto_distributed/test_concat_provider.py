# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.passes.auto_distributed import DistributedCandidateContext, TypeInferenceCandidateProvider
from triton.flagmega.targets.pyntt_split import PyNttDistributedSplitCandidateProvider
from python.test.flagmega.ir.ops.tensors.test_concat_local_contract import concat_module


def test_inference_provider_expands_every_variadic_operand_and_keeps_row_sharding():
    module = concat_module(((8, 3), (8, 5), (8, 7)))
    node = module.node_map["output"]
    placement = fm.Placement((2, 2), "xy", "bb")
    inputs = tuple(module.node_map[value].type for value in node.inputs)
    context = DistributedCandidateContext(module, node, placement, tuple((value, ) for value in inputs),
                                          PyNttDistributedSplitCandidateProvider(128))
    candidates = TypeInferenceCandidateProvider(frozenset({"tensors.concat"})).get_candidates(context)
    assert candidates
    for candidate in candidates:
        assert len(candidate.input_types) == 3
        assert candidate.return_type.tensor == node.type
    assert any(isinstance(candidate.return_type.axis_policies[0], fm.SBPSplit) for candidate in candidates)
    assert all(candidate.return_type.axis_policies[1] == fm.SBP.broadcast() for candidate in candidates)
