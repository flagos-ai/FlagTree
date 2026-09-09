# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.ir.ops.nn.delta_rule_block_update import DeltaRuleBlockUpdate
from python.test.flagmega.passes.auto_distributed.test_router_primitive_providers import candidates_for


@pytest.mark.parametrize("mesh", [(2, ), (2, 2), (2, 2, 2)])
def test_block_update_proposals_preserve_grouped_heads_and_aliased_reference(mesh, monkeypatch):
    original = DeltaRuleBlockUpdate.infer_type
    calls = 0

    def counted(cls, inputs, attrs):
        nonlocal calls
        calls += 1
        if calls > 256:
            pytest.fail("Head-coupled operands must not enumerate a five-input Cartesian product.")
        return original(inputs, attrs)

    monkeypatch.setattr(DeltaRuleBlockUpdate, "infer_type", classmethod(counted))
    q = fm.tensor_type("bfloat16", (65, 8, 16))
    v = fm.tensor_type("bfloat16", (65, 16, 8))
    c = fm.tensor_type("bfloat16", (2, 16, 64, 64))
    p = fm.tensor_type("float32", (2, 16, 64))
    state = fm.RefType("state", (("matrix", fm.tensor_type("float32", (16, 8, 16))), ))
    candidates, _ = candidates_for("nn.delta_rule_block_update", (q, q, v, c, p, state), mesh=mesh)
    b = fm.SBP.broadcast()
    assert any(isinstance(candidate.return_type.fields[0].axis_policies[1], fm.SBPSplit) for candidate in candidates)
    assert any(candidate.return_type.fields[0].axis_policies == (b, b, b) for candidate in candidates)
    for candidate in candidates:
        query, key, value, coefficients, prefix, reference = candidate.input_types
        assert reference == state == candidate.return_type.fields[1]
        assert candidate.return_type.fields[0] == value
        key_head = key.axis_policies[1]
        head = fm.scale_split_units(key_head, 2, 1) if isinstance(key_head, fm.SBPSplit) else key_head
        assert query.axis_policies == key.axis_policies == (b, key_head, b)
        assert value.axis_policies == prefix.axis_policies == (b, head, b)
        assert coefficients.axis_policies == (b, head, b, b)
        assert all(tensor.partial is None for tensor in (query, key, value, coefficients, prefix))
