# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The joint KDA kernel owns complete chip projections, then publishes heads."""

from pathlib import Path

from triton.flagmega import ir as fm
from triton.flagmega.passes.auto_distributed import DistributedCandidateContext, DistributedCandidateProviderRegistry
from triton.flagmega.passes.auto_distributed.policy import NttDistributionPolicy
from triton.flagmega.targets.pyntt_split import PyNttDistributedSplitCandidateProvider

from python.test.flagmega.ir.ops.nn.test_kda_input_projection import graph


def test_joint_projection_has_one_legal_tp4_owner_with_replicated_low_ranks(monkeypatch):
    tutorial = Path(__file__).parents[3] / "tutorials/flagmega/04-glm5.3-flash-fp8/nvidia-h800"
    monkeypatch.syspath_prepend(str(tutorial))
    from local_optimizations import kda_distribution

    module = graph(k=4096, heads=64, width=128, rank_f=128, rank_g=128,
                   lanes=((), (), (), (), (8,), (8,)))
    node = module.node_map["projection"]
    placement = fm.Placement((4, 8, 16), "cyx", "cbb")
    split = PyNttDistributedSplitCandidateProvider(128)
    context = DistributedCandidateContext(module, node, placement,
        tuple((module.node_map[name].type,) for name in node.inputs), split)
    registry = DistributedCandidateProviderRegistry()
    NttDistributionPolicy((placement,), split).register_candidate_providers(registry)
    candidates = registry.try_get(node.op).get_candidates(context)
    selected, = (candidate for candidate in candidates if kda_distribution._owns_input_projection(candidate.return_type))
    def extents(value):
        return tuple(d.fixed_value for d in fm.local_shape(value))

    for weight in selected.input_types[1:4]:
        assert extents(weight) == (16, 128, 4096)
    assert extents(selected.input_types[4]) == (16, 4096)
    assert all(extents(weight) == (128, 4096) for weight in selected.input_types[5:])
    assert all(extents(field) == (1, 16, 128) for field in selected.return_type.fields[:3])
    assert all(extents(field) == (1, 16) for field in selected.return_type.fields[3:])
