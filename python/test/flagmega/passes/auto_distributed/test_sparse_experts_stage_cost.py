# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.passes.auto_distributed import DistributedCandidateProviderRegistry, build_search_graph
from triton.flagmega.passes.target_independent import decompose_complex_ops
from triton.flagmega.targets import NvidiaSm90Target
from python.test.flagmega.sparse_experts.helpers import build_module, operand_types


def search(*, exported=False):
    module = decompose_complex_ops(build_module(types=operand_types(tokens=1, hidden=64, intermediate=32, routes=3)))
    if exported:
        fn = module.functions[0]
        module = replace(module, functions=(replace(fn, outputs=(*fn.outputs, "experts.dispatch", "experts.down")),))
    target = NvidiaSm90Target()
    registry = DistributedCandidateProviderRegistry()
    target.register_auto_distributed_candidate_providers(registry)
    return build_search_graph(module, fm.Placement((2, 2), "yx", "bb"), registry,
                              target.distributed_reshard_realization_policy(), target.distributed_reshard_cost_model(),
                              target.distributed_operation_cost_model())


def test_all_four_stages_have_independent_candidates_and_route_owners():
    graph = search()
    for name in ("experts.dispatch", "experts.gate_up", "experts.down"):
        assert any(isinstance(c.return_type, fm.DistributedType) and isinstance(c.return_type.axis_policies[1], fm.SBPSplit)
                   for c in graph.bucket_map[name].candidates)
    combines = graph.bucket_map["experts"].candidates
    assert any(isinstance(c.input_types[0], fm.DistributedType) and c.input_types[0].partial is not None for c in combines)
    assert {f.operation for f in graph.fusions} == {"ntt.dispatched_experts_gate_up", "ntt.sparse_experts_down_combine"}
    assert any(f.operation_cost < f.standalone_cost for f in graph.fusions)


def test_exported_stages_cannot_claim_local_fusion_savings():
    assert not search(exported=True).fusions
