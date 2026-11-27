# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Ragged packed N geometry must not silently discard card-local owners."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.passes.auto_distributed import DistributedCandidateContext, DistributedCandidateProviderRegistry
from triton.flagmega.passes.auto_distributed.policy import NttDistributionPolicy
from triton.flagmega.targets.pyntt_split import PyNttDistributedSplitCandidateProvider


@pytest.mark.parametrize("groups,axes", [(192, (1, 2)), (256, (1, 2)), (512, (0, 1, 2)),
                                         (64, (2,)), (16, (2,)), (36, (2,)), (19360, (0, 1, 2))])
def test_native_projection_uses_ragged_block_owners_before_chip_sharding(monkeypatch, groups, axes):
    tutorial = Path(__file__).parents[3] / "tutorials/flagmega/04-glm5.3-flash-fp8/nvidia-h800"
    monkeypatch.syspath_prepend(str(tutorial))
    spec = importlib.util.spec_from_file_location("glm_native_projection_layout", tutorial / "local_optimizations/native_tp4.py")
    profile = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profile)

    class Graph(fm.Module):
        def forward(self):
            x = self.input("x", fm.tensor_type("bfloat16", (1, 4096)), id="x")
            weight = self.input("weight", fm.tensor_type(fm.vector_type("bfloat16", (8, 2, 8)), (256, groups)),
                                id="weight")
            none = fm.F.builtin.none()
            result = fm.F.ntt.packed_matmul(x, weight, none, none, output_data_type="float32", name="projection")
            self.function("main", (x, weight), (result,))

    module = Graph(dialect="ntt", stage="packed", entry="main").build()
    node = module.node_map["projection"]
    placement = fm.Placement((4, 8, 16), "cyx", "cbb")
    provider = PyNttDistributedSplitCandidateProvider(128)
    context = DistributedCandidateContext(module, node, placement,
        tuple((module.node_map[i].type,) for i in node.inputs), provider)
    registry = DistributedCandidateProviderRegistry()
    NttDistributionPolicy((placement,), provider).register_candidate_providers(registry)
    candidates = registry.try_get(node.op).get_candidates(context)
    bucket = SimpleNamespace(node_id=node.id, candidates=candidates)
    selected = profile.select_complete_projection(bucket, groups, placement)
    policy = selected.return_type.axis_policies[-1]
    assert tuple(policy.hierarchy_axes) == axes
    if groups == 192:
        assert policy == fm.SBP.split_block_cyclic((1, 2), 1)
    assert selected.return_type.partial is None
    assert selected.input_types[0].axis_policies == (fm.SBP.broadcast(), fm.SBP.broadcast())
