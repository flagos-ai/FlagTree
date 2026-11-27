# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.passes.auto_distributed import DistributedCandidateContext
from triton.flagmega.passes.auto_distributed.providers import MatMulGluCandidateProvider


def _context(output_n: int) -> DistributedCandidateContext:
    value_type = fm.tensor_type("bfloat16", (1, 128))
    weight_type = fm.tensor_type("bfloat16", (output_n, 128))
    output_type = fm.tensor_type("bfloat16", (1, output_n))
    value = fm.Node("value", "builtin.var", (), value_type, attrs={"name": "value"})
    gate = fm.Node("gate", "builtin.var", (), weight_type, attrs={"name": "gate"})
    up = fm.Node("up", "builtin.var", (), weight_type, attrs={"name": "up"})
    call = fm.Node(
        "output", "nn.matmul_glu", (value.id, gate.id, up.id), output_type
    )
    module = fm.IRModule(
        "high_level",
        "packed",
        (value, gate, up, call),
        (fm.Function("main", (value.id, gate.id, up.id), (call.id,)),),
        "main",
    )
    return DistributedCandidateContext(
        module,
        call,
        fm.Placement((8, 16), "yx", "bb"),
        ((value_type,), (weight_type,), (weight_type,)),
    )


def test_matmul_glu_provider_shards_both_weights_and_output_on_same_mesh_axes():
    candidates = MatMulGluCandidateProvider().get_candidates(_context(256))
    split = {
        candidate.return_type.axis_policies[-1].hierarchy_axes: candidate
        for candidate in candidates
        if candidate.reason == "matmul-glu-output-sbp"
    }

    assert set(split) == {(0,), (1,), (0, 1)}
    for axes, candidate in split.items():
        assert all(
            isinstance(policy, fm.SBPBroadCast)
            for policy in candidate.input_types[0].axis_policies
        )
        assert candidate.input_types[1].axis_policies[0].hierarchy_axes == axes
        assert candidate.input_types[2].axis_policies[0].hierarchy_axes == axes
        assert candidate.return_type.axis_policies[-1].hierarchy_axes == axes


def test_matmul_glu_provider_does_not_offer_nondividing_full_mesh_split():
    candidates = MatMulGluCandidateProvider().get_candidates(_context(160))
    axes = {
        candidate.return_type.axis_policies[-1].hierarchy_axes
        for candidate in candidates
        if candidate.reason == "matmul-glu-output-sbp"
    }

    assert axes == {(0,), (1,)}
