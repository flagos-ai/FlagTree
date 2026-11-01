# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The native vector load order is part of the normalization contract."""

import json
from pathlib import Path

import pytest
import torch

from python.test.flagmega.codegen.triton.kernels.l2_normalization.helpers import execute_l2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_contiguous_group_accumulation(tmp_path):
    fixture = json.loads((Path(__file__).resolve().parents[4] / "data" /
                          "native_l2_contiguous_groups.json").read_text())
    value = torch.tensor([fixture["input"]], dtype=torch.bfloat16)
    expected = torch.tensor([fixture["output"]], dtype=torch.bfloat16)
    actual = execute_l2(tmp_path, value, epsilon=1e-6, epsilon_mode="add",
                        division_mode="reciprocal_multiply", reduction_group_size=8)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
