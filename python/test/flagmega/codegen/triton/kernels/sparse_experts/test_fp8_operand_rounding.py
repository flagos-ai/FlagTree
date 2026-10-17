# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Block scales define FP32 matmul operands, including their rounding."""

import pytest
import torch
from triton.flagmega import ir as fm

from python.test.flagmega.codegen.triton.kernels.sparse_experts.helpers import execute_and_reference


@pytest.mark.parametrize("packed", (False, True))
@pytest.mark.parametrize("kind", ("simt", "partitioned", "tma"))
def test_block_scaled_down_rounds_weight_operand_before_product(tmp_path, monkeypatch, packed, kind):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")
    activation_dtype = fm.vector_type("bfloat16", (8,)) if packed else "bfloat16"
    partitioned = kind == "partitioned"
    weight_dtype = "bfloat16" if kind == "tma" else "float8_e4m3fn"
    if kind == "tma":
        from triton.flagmega.compiler import Compiler

        from python.test.flagmega.codegen.triton.kernels.sparse_experts import helpers
        from python.test.flagmega.codegen.triton.kernels.sparse_experts.test_pipeline import pipeline_target
        compiler = Compiler()
        compiler.target = pipeline_target(block_k=128)
        monkeypatch.setattr(helpers, "Compiler", lambda: compiler)
    reduction = 16 if packed else 128
    activation_shape = (1, 1, 1, reduction) if partitioned else (1, 1, reduction)
    weight_shape = (1, 1, 8, 128) if partitioned else (1, 8, 128)
    scale_shape = (1, 1, 1, 1) if partitioned else (1, 1, 1)

    class Projection(fm.Module):
        def forward(self):
            inputs = {
                "activations": self.input("activations", fm.tensor_type(activation_dtype, activation_shape), id="activations"),
                "router_expert_ids": self.input("router_expert_ids", fm.tensor_type("int32", (1, 1)), id="router_expert_ids"),
                "down_input_scale": self.input("down_input_scale", fm.tensor_type("float32", (1, 1)), id="down_input_scale"),
                "down_weight": self.input("down_weight", fm.tensor_type(weight_dtype, weight_shape), id="down_weight"),
                "down_proj_scale": self.input("down_proj_scale", fm.tensor_type("float32", scale_shape), id="down_proj_scale"),
            }
            op = fm.F.nn.partitioned_sparse_experts_down if partitioned else fm.F.nn.sparse_experts_down
            result = op(**inputs, weight_block_n=128, weight_block_k=128)
            self.function("main", tuple(inputs.values()), (result,))

    # Only one dot-product term is nonzero: no reduction-order ambiguity.
    # (x * w) * scale and x * round_fp32(w * scale) differ by one FP32 ULP.
    activation = torch.zeros((*activation_shape, *((8,) if packed else ())), dtype=torch.bfloat16)
    activation.flatten()[0] = 1.5078125
    weight = torch.zeros(weight_shape)
    weight[..., 0] = 7.
    module = Projection(dialect="nn", stage="frozen_constants", entry="main").build()
    actual, expected, runtime = execute_and_reference(module, tmp_path, torch, overrides={
        "activations": activation, "router_expert_ids": torch.zeros((1, 1), dtype=torch.int32),
        "down_input_scale": torch.ones((1, 1)), "down_weight": weight.to(getattr(torch, weight_dtype)),
        "down_proj_scale": torch.full(scale_shape, .013),
    })
    try:
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert runtime.resource_report["spill_bytes"] == 0
        if kind == "tma":
            assert "sparse_experts_down/simt_tma_pipeline" in (tmp_path / "artifact/generated_kernels.py").read_text()
    finally:
        runtime.close()


@pytest.mark.parametrize("packed", (False, True))
@pytest.mark.parametrize("pipeline", (False, True))
def test_block_scaled_gate_up_rounds_weight_before_swiglu(tmp_path, monkeypatch, packed, pipeline):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")
    from triton.flagmega.compiler import Compiler
    from triton.flagmega.ir.ops.nn.sparse_experts_gate_up import SparseExpertsGateUp

    from python.test.flagmega.codegen.triton.kernels.sparse_experts import helpers
    from python.test.flagmega.codegen.triton.kernels.sparse_experts.test_pipeline import pipeline_target
    weight_dtype = "bfloat16" if pipeline else "float8_e4m3fn"
    if pipeline:
        compiler = Compiler()
        compiler.target = pipeline_target(block_k=128)
        monkeypatch.setattr(helpers, "Compiler", lambda: compiler)
    module = helpers.stage_module(SparseExpertsGateUp, packed=packed, tokens=1, hidden=128,
        intermediate=8, weight_dtype=weight_dtype, block_scaled=True, weight_block=128, gate_clamp_max=1.)
    activation = torch.zeros((1, 3, 32, 2, 2) if packed else (1, 3, 128), dtype=torch.bfloat16)
    activation.reshape(1, 3, 128)[..., 0] = 1.5078125
    weight = torch.zeros((5, 8, 128))
    weight[..., 0] = 7.
    # Gate is clamped exactly to 1. The up operand's rounding determines
    # which side of a BF16 midpoint SwiGLU lands on, even with one dot term.
    actual, expected, runtime = helpers.execute_and_reference(module, tmp_path, torch, overrides={
        "dispatched": activation, "gate_input_scale": torch.ones((5, 1)),
        "up_input_scale": torch.ones((5, 1)), "gate_proj_scale": torch.ones((5, 1, 1)),
        "up_proj_scale": torch.full((5, 1, 1), .01300421915948391),
        "gate_weight": weight.to(getattr(torch, weight_dtype)), "up_weight": weight.to(getattr(torch, weight_dtype)),
    })
    try:
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert runtime.resource_report["spill_bytes"] == 0
        if pipeline:
            assert "sparse_experts_gate_up/simt_tma_pipeline" in (tmp_path / "artifact/generated_kernels.py").read_text()
    finally:
        runtime.close()
