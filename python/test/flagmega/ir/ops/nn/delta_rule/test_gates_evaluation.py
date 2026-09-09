# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch

from triton.flagmega.ir.ops.nn.delta_rule_gates import DeltaRuleGates
from python.test.flagmega.ir.ops.primitive_helpers import evaluate


@pytest.mark.parametrize("threshold", (2., 20.))
def test_gate_stable_softplus_tails_and_inputs_unchanged(threshold):
    a = torch.tensor([[-80., -2., 0., 2., 80.]]).bfloat16()
    b = -a
    a_log = torch.tensor([-1., 0., 1., 2., -2.])
    bias = torch.tensor([.25, -.5, 0., -.25, .5]).bfloat16()
    inputs = (a, b, a_log, bias)
    snapshots = tuple(value.clone() for value in inputs)
    alpha, beta = evaluate(DeltaRuleGates, inputs, softplus_threshold=threshold)
    expected = torch.exp(-a_log.exp() * torch.nn.functional.softplus(a.float() + bias.float(), threshold=threshold))
    torch.testing.assert_close(alpha, expected, rtol=2e-7, atol=1e-8)
    torch.testing.assert_close(beta, torch.sigmoid(b.float()), rtol=0, atol=0)
    assert alpha.dtype == beta.dtype == torch.float32
    assert all(torch.equal(value, snapshot) for value, snapshot in zip(inputs, snapshots))
