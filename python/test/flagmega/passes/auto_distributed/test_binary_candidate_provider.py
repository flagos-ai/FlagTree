# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.passes.auto_distributed import DistributedCandidateContext
from triton.flagmega.passes.auto_distributed.providers import BinaryCandidateProvider


def _context(last_extent: int) -> DistributedCandidateContext:
    value_type = fm.tensor_type("bfloat16", (1, last_extent))
    lhs = fm.Node("lhs", "builtin.var", (), value_type, attrs={"name": "lhs"})
    rhs = fm.Node("rhs", "builtin.var", (), value_type, attrs={"name": "rhs"})
    call = fm.Node("output", "math.add", (lhs.id, rhs.id), value_type)
    module = fm.IRModule(
        "high_level",
        "packed",
        (lhs, rhs, call),
        (fm.Function("main", (lhs.id, rhs.id), (call.id,)),),
        "main",
    )
    return DistributedCandidateContext(
        module,
        call,
        fm.Placement((8, 16), "yx", "bb"),
        ((value_type,), (value_type,)),
    )


def test_binary_provider_enumerates_each_legal_2d_mesh_axis_combination():
    candidates = BinaryCandidateProvider().get_candidates(_context(128))
    split = {
        candidate.return_type.axis_policies[-1].hierarchy_axes: candidate
        for candidate in candidates
        if candidate.reason == "binary-exact-output-sbp"
    }

    assert set(split) == {(0,), (1,), (0, 1)}
    for axes, candidate in split.items():
        assert candidate.input_types == (
            candidate.return_type,
            candidate.return_type,
        )
        assert candidate.return_type.placement.hierarchy == (8, 16)
        shard_count = 1
        for axis in axes:
            shard_count *= (8, 16)[axis]
        assert candidate.operation_cost == max(128 // shard_count, 1)


def test_binary_provider_rejects_only_the_nondividing_mesh_combinations():
    candidates = BinaryCandidateProvider().get_candidates(_context(160))
    axes = {
        candidate.return_type.axis_policies[-1].hierarchy_axes
        for candidate in candidates
        if candidate.reason == "binary-exact-output-sbp"
    }

    assert axes == {(0,), (1,)}
