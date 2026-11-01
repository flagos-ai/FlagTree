# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Widening an explicitly rounded MMA operand changes the coefficient."""

import json
from pathlib import Path

import pytest
import torch

from triton.flagmega.ir.ops.nn.delta_rule_coefficients import delta_rule_coefficients


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA MMA reference required")
def test_independent_eager_preserves_bf16_and_fp16_mma_operands():
    fixture = json.loads((Path(__file__).resolve().parents[4] / "data" /
                          "delta_rule_coefficients_mma_rounding.json").read_text())
    key = torch.tensor(fixture["key"], dtype=torch.bfloat16, device="cuda")
    beta = torch.tensor(fixture["beta"], dtype=torch.float32, device="cuda")
    expected = torch.tensor(fixture["output"], dtype=torch.bfloat16, device="cuda")
    actual = delta_rule_coefficients(key, beta, 64, torch=torch)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
