# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch
import json
from pathlib import Path

from triton.flagmega.ir.ops.nn.l2_normalization import L2Normalization
from python.test.flagmega.ir.ops.primitive_helpers import evaluate


@pytest.mark.parametrize("axes", ((-1,), (0,), (1, 2)))
@pytest.mark.parametrize("mode", ("add", "clamp"))
@pytest.mark.parametrize("division", ("divide", "reciprocal_multiply"))
def test_l2_explicit_reduction_denominator_and_division(axes, mode, division):
    value = (torch.arange(60).reshape(3, 4, 5) / 30 - 1).bfloat16()
    original = value.clone()
    square_sum = value.float().square().sum(axes, keepdim=True)
    denominator = (square_sum + .1 if mode == "add" else square_sum.clamp_min(.1)).sqrt()
    expected = value.float() / denominator if division == "divide" else value.float() * (1. / denominator)
    actual = evaluate(L2Normalization, (value,), axes=axes, epsilon=.1, epsilon_mode=mode, division_mode=division)
    torch.testing.assert_close(actual, expected.bfloat16(), rtol=0, atol=0)
    assert torch.equal(value, original)


def test_l2_epsilon_keeps_zero_and_tiny_inputs_finite():
    value = torch.tensor([[0., 0.], [1e-8, -1e-8]])
    result = evaluate(L2Normalization, (value,), epsilon=1e-6, epsilon_mode="add",
                      division_mode="reciprocal_multiply")
    torch.testing.assert_close(result, value * (1. / (value.square().sum(-1, keepdim=True) + 1e-6).sqrt()),
                               rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA instruction reference required")
def test_l2_native_contiguous_group_eager_reference():
    fixture = json.loads((Path(__file__).resolve().parents[4] / "data" /
                          "native_l2_contiguous_groups.json").read_text())
    value = torch.tensor([fixture["input"]], dtype=torch.bfloat16, device="cuda")
    expected = torch.tensor([fixture["output"]], dtype=torch.bfloat16, device="cuda")
    actual = evaluate(L2Normalization, (value,), epsilon=1e-6, epsilon_mode="add",
                      division_mode="reciprocal_multiply", reduction_group_size=8)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
