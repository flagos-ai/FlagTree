# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from python.test.flagmega.passes.auto_distributed.test_router_primitive_providers import candidates_for


@pytest.mark.parametrize("mesh", [(2, ), (2, 2), (2, 2, 2)])
def test_coefficients_proposals_follow_grouped_head_type_contract(mesh):
    candidates, _ = candidates_for("nn.delta_rule_coefficients",
                                   (fm.tensor_type("bfloat16", (65, 8, 16)), fm.tensor_type("float32",
                                                                                            (65, 16))), mesh=mesh)
    b = fm.SBP.broadcast()
    assert any(isinstance(candidate.return_type.axis_policies[1], fm.SBPSplit) for candidate in candidates)
    assert any(candidate.return_type.axis_policies == (b, b, b, b) for candidate in candidates)
    for candidate in candidates:
        head = candidate.return_type.axis_policies[1]
        assert candidate.return_type.axis_policies == (b, head, b, b)
        key_head = candidate.input_types[0].axis_policies[1]
        assert candidate.input_types[0].axis_policies == (b, key_head, b)
        assert (fm.scale_split_units(key_head, 2, 1) if isinstance(key_head, fm.SBPSplit) else key_head) == head
        assert candidate.input_types[1].axis_policies == (b, head)
        assert all(value.partial is None for value in (*candidate.input_types, candidate.return_type))
