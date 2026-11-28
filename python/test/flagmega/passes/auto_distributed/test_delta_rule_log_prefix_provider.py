# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from python.test.flagmega.passes.auto_distributed.test_router_primitive_providers import candidates_for


@pytest.mark.parametrize("mesh", [(2, ), (2, 2), (2, 2, 2)])
def test_log_prefix_proposals_preserve_group_local_scan(mesh):
    candidates, _ = candidates_for("nn.delta_rule_log_prefix", (fm.tensor_type("float32", (65, 8)), ), mesh=mesh)
    b = fm.SBP.broadcast()
    assert any(isinstance(candidate.return_type.axis_policies[1], fm.SBPSplit) for candidate in candidates)
    assert any(candidate.return_type.axis_policies == (b, b, b) for candidate in candidates)
    for candidate in candidates:
        head = candidate.return_type.axis_policies[1]
        assert candidate.return_type.axis_policies == (b, head, b)
        assert candidate.input_types[0].axis_policies == (b, head)
        assert candidate.input_types[0].partial is None
