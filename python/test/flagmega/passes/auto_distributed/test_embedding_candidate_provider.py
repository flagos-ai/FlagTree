# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.passes.auto_distributed import DistributedCandidateContext
from triton.flagmega.passes.auto_distributed.providers import EmbeddingCandidateProvider


def _context(feature_extent: int = 2048) -> DistributedCandidateContext:
    indices_type = fm.tensor_type("int32", (1,))
    weight_type = fm.tensor_type("bfloat16", (151936, feature_extent))
    output_type = fm.tensor_type("bfloat16", (1, feature_extent))
    indices = fm.Node("indices", "builtin.var", (), indices_type, attrs={"name": "indices"})
    weight = fm.Node("weight", "builtin.weight", (), weight_type, attrs={"name": "weight"})
    output = fm.Node(
        "output",
        "nn.embedding",
        (indices.id, weight.id),
        output_type,
        attrs={"padding_idx": None},
    )
    module = fm.IRModule(
        "high_level",
        "packed",
        (indices, weight, output),
        (fm.Function("main", (indices.id,), (output.id,)),),
        "main",
    )
    return DistributedCandidateContext(
        module,
        output,
        fm.Placement((8, 16), "yx", "bb"),
        ((indices_type,), (weight_type,)),
    )


def test_embedding_provider_projects_output_feature_split_to_weight_column():
    candidates = EmbeddingCandidateProvider().get_candidates(_context())
    split = {
        candidate.return_type.axis_policies[-1].hierarchy_axes: candidate
        for candidate in candidates
        if candidate.reason == "embedding-exact-output-sbp"
    }

    assert set(split) == {(0,), (1,), (0, 1)}
    for axes, candidate in split.items():
        assert isinstance(candidate.return_type, fm.DistributedType)
        indices_type, weight_type = candidate.input_types
        assert isinstance(indices_type, fm.DistributedType)
        assert all(
            isinstance(policy, fm.SBPBroadCast)
            for policy in indices_type.axis_policies
        )
        assert isinstance(weight_type, fm.DistributedType)
        assert isinstance(weight_type.axis_policies[0], fm.SBPBroadCast)
        assert weight_type.axis_policies[1] == candidate.return_type.axis_policies[1]
        assert candidate.return_type.partial is None


def test_embedding_provider_keeps_a_replicated_candidate_for_independent_compilation():
    candidates = EmbeddingCandidateProvider().get_candidates(_context())

    replicated = [
        candidate
        for candidate in candidates
        if candidate.reason == "embedding-replicated"
    ]
    assert len(replicated) == 1
    assert all(
        isinstance(policy, fm.SBPBroadCast)
        for policy in replicated[0].return_type.axis_policies
    )


def test_embedding_provider_rejects_only_nondividing_mesh_combinations():
    candidates = EmbeddingCandidateProvider().get_candidates(_context(160))
    axes = {
        candidate.return_type.axis_policies[-1].hierarchy_axes
        for candidate in candidates
        if candidate.reason == "embedding-exact-output-sbp"
    }

    assert axes == {(0,), (1,)}
