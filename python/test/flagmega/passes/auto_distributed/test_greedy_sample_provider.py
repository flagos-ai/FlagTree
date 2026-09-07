# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.passes.auto_distributed.candidates import (
    DistributedCandidateContext, DistributedCandidateProviderRegistry,
)
from triton.flagmega.passes.auto_distributed.policy import NttDistributionPolicy


def make_context(placement):
    builder = fm.IRBuilder(dialect="nn", stage="packed")
    value = builder.var("logits", fm.tensor_type("float32", [1, 128]), id="logits")
    result = builder.call("nn.greedy_sample", (value,), fm.tensor_type("int32", [1]), id="sample")
    builder.function("main", (value,), (result,))
    module = builder.build(entry="main")
    return DistributedCandidateContext(module, module.node_map["sample"], placement, ((),))


@pytest.mark.parametrize("hierarchy", [(2, 4), (2, 2, 2)])
def test_policy_exposes_nonredundant_sampling_and_local_work_objective(hierarchy):
    placement = fm.Placement(hierarchy, "abc"[:len(hierarchy)], "b" * len(hierarchy))
    context = make_context(placement)
    registry = DistributedCandidateProviderRegistry()
    NttDistributionPolicy((placement,)).register_candidate_providers(registry)
    provider = registry.try_get("nn.greedy_sample")
    candidates = provider.get_candidates(context)
    broadcast = next(c for c in candidates if isinstance(c.input_types[0].axis_policies[-1], fm.SBPBroadCast))
    sharded = [c for c in candidates if isinstance(c.input_types[0].axis_policies[-1], fm.SBPSplit)]
    assert sharded
    assert all(c.return_type == broadcast.return_type for c in sharded)
    assert min(c.operation_cost for c in sharded) < broadcast.operation_cost


def test_provider_preserves_producer_cyclic_split_and_target_split_policy():
    from dataclasses import replace

    placement = fm.Placement((2, 4), "yx", "bb")
    context = make_context(placement)
    cyclic = fm.DistributedType(context.module.node_map["logits"].type,
                                (fm.SBP.broadcast(), fm.SBP.split_block_cyclic((0, 1), 3)),
                                placement)

    class CyclicSplits:
        identity = "test-cyclic/v1"

        def get_candidates(self, request):
            return (fm.SBP.split_block_cyclic(request.hierarchy_axes, 5),)

    context = replace(context, available_input_types=((cyclic,),),
                      split_candidate_provider=CyclicSplits())
    registry = DistributedCandidateProviderRegistry()
    NttDistributionPolicy((placement,)).register_candidate_providers(registry)
    candidates = registry.try_get("nn.greedy_sample").get_candidates(context)
    assert any(c.input_types == (cyclic,) for c in candidates)
    assert any(isinstance(c.input_types[0].axis_policies[-1], fm.SBPSplit)
               and c.input_types[0].axis_policies[-1].stages[0].distribution == fm.BlockCyclicSplit(5)
               for c in candidates)
