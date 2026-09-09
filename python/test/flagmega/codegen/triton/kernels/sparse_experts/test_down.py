# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega.ir.ops.nn.sparse_experts_down import SparseExpertsDown
from python.test.flagmega.codegen.triton.kernels.sparse_experts.helpers import stage_module, execute_and_reference


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("round_projection,round_weighted_output", [(False, False), (True, False), (False, True),
                                                                    (True, True)])
def test_down_device_respects_route_order_and_rounding(tmp_path, dtype, round_projection, round_weighted_output):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    module = stage_module(SparseExpertsDown, dtype=dtype, round_projection=round_projection,
                          round_weighted_output=round_weighted_output)
    output, expected, _ = execute_and_reference(module, tmp_path, torch)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.parametrize("packed,distribution", [(False, "token_output"), (True, "token_output"), (True, None)])
def test_down_device_preserves_vector_and_output_sharding(tmp_path, packed, distribution):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    module = stage_module(SparseExpertsDown, packed=packed, distribution=distribution, hidden=80, round_projection=True,
                          round_weighted_output=True)
    output, expected, _ = execute_and_reference(module, tmp_path, torch)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.parametrize("packed", [False, True])
def test_down_device_materializes_split_k_through_explicit_boxing(tmp_path, packed):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    module = stage_module(SparseExpertsDown, dtype="float32", packed=packed, distribution="split_k", intermediate=48)
    output, expected, _ = execute_and_reference(module, tmp_path, torch)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.parametrize("tokens,hidden,intermediate", [(2, 35, 513), (0, 72, 40)])
def test_down_device_masks_tiles_and_empty_tokens(tmp_path, tokens, hidden, intermediate):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    module = stage_module(SparseExpertsDown, dtype="float32", tokens=tokens, hidden=hidden, intermediate=intermediate)
    output, expected, _ = execute_and_reference(module, tmp_path, torch)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
