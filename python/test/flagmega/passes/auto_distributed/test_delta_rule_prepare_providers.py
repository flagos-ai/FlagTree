# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.ir.ops.nn.delta_rule_gates import DeltaRuleGates
from python.test.flagmega.ir.ops.nn.delta_rule.test_gates_types import gate_types
from python.test.flagmega.passes.auto_distributed.test_router_primitive_providers import candidates_for


@pytest.mark.parametrize("mesh", ((2,), (2, 2), (2, 2, 2)))
def test_gate_join_preserves_head_and_token_splitting_with_bounded_inference(monkeypatch, mesh):
    original = DeltaRuleGates.infer_type
    calls = 0
    def infer(cls, inputs, attrs):
        nonlocal calls
        calls += 1
        if calls > 1024:
            pytest.fail("Gate relation enumeration regressed to an unconstrained Cartesian product.")
        return original(inputs, attrs)
    monkeypatch.setattr(DeltaRuleGates, "infer_type", classmethod(infer))
    candidates, _ = candidates_for("nn.delta_rule_gates", gate_types(tokens=8), mesh=mesh)
    assert any(isinstance(candidate.return_type.fields[0].axis_policies[1], fm.SBPSplit) for candidate in candidates)
    assert any(isinstance(candidate.return_type.fields[0].axis_policies[0], fm.SBPSplit) for candidate in candidates)


@pytest.mark.parametrize("mesh", ((2,), (2, 2), (2, 2, 2)))
def test_l2_provider_only_splits_outer_axes(mesh):
    candidates, _ = candidates_for("nn.l2_normalization", (fm.tensor_type("bfloat16", (8, 8, 128)),), mesh=mesh)
    assert all(candidate.return_type.axis_policies[-1] == fm.SBP.broadcast() for candidate in candidates)
    assert any(isinstance(candidate.return_type.axis_policies[1], fm.SBPSplit) for candidate in candidates)
