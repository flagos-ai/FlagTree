# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega.ir.ops.nn.sparse_experts_gate_up import SparseExpertsGateUp
from python.test.flagmega.codegen.triton.kernels.sparse_experts.helpers import stage_module, execute_and_reference


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("round_projections,round_activation", [(False, False), (True, False), (False, True),
                                                                (True, True)])
def test_gate_up_device_explicit_rounding_boundaries(tmp_path, dtype, round_projections, round_activation):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    module = stage_module(SparseExpertsGateUp, dtype=dtype, round_projections=round_projections,
                          round_activation=round_activation)
    output, expected, _ = execute_and_reference(module, tmp_path, torch)
    torch.testing.assert_close(output, expected, rtol=2e-6 if dtype == "float32" else 0,
                               atol=2e-6 if dtype == "float32" else 0)


@pytest.mark.parametrize("packed,distribution", [(False, "token_output"), (True, "token_output"), (True, None)])
def test_gate_up_device_preserves_vector_and_distributed_feature_coordinates(tmp_path, packed, distribution):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    module = stage_module(SparseExpertsGateUp, packed=packed, distribution=distribution, intermediate=48,
                          round_projections=True)
    output, expected, _ = execute_and_reference(module, tmp_path, torch)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.parametrize("tokens,hidden,intermediate", [(2, 513, 35), (0, 72, 40)])
def test_gate_up_device_masks_non_multiple_tiles_and_empty_tokens(tmp_path, tokens, hidden, intermediate):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    module = stage_module(SparseExpertsGateUp, dtype="float32", tokens=tokens, hidden=hidden, intermediate=intermediate)
    output, expected, _ = execute_and_reference(module, tmp_path, torch)
    torch.testing.assert_close(output, expected, rtol=3e-6, atol=3e-6)
