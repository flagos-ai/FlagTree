# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The evaluator and existing kernels agree on the declared rounding boundary."""

import pytest
import torch
from triton.flagmega.ir.ops.nn.sparse_experts_down import SparseExpertsDown
from triton.flagmega.ir.ops.nn.sparse_experts_gate_up import SparseExpertsGateUp

from python.test.flagmega.codegen.triton.kernels.sparse_experts.helpers import execute_and_reference
from python.test.flagmega.ir.ops.nn.test_sparse_experts_fp8_rounding import projection_case


@pytest.mark.parametrize("definition", (SparseExpertsGateUp, SparseExpertsDown))
@pytest.mark.parametrize("rounded", (False, True))
@pytest.mark.parametrize("input_scale", (1., 1.1))
def test_existing_fp8_kernel_agrees_with_the_stage_rounding_contract(tmp_path, definition, rounded, input_scale):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")
    module, values = projection_case(definition, rounded=rounded, input_scale=input_scale)
    actual, expected, runtime = execute_and_reference(module, tmp_path, torch, overrides=values)
    try:
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()
