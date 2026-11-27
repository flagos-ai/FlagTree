# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The prefill GDN norm remains FP32 until the SiLU product is stored."""

import pytest

from triton.flagmega.importer import Qwen35MoeImporter, apply_numerical_profile
from triton.flagmega.importer.numerics import VLLM_AE10_INDUCTOR_LEVEL3
from triton.flagmega import ir as fm
from python.test.flagmega.importer.qwen3_5_moe.helpers import checkpoint


def test_prefill_normalization_has_no_intermediate_bf16_store():
    torch = pytest.importorskip("torch")
    from triton.flagmega.evaluator import CheckpointWeightResolver, TorchEvaluator
    from triton.flagmega.ir.ops.nn._gdn_state import create_gdn_state
    from triton.flagmega.ir.ops.nn._paged_attention_state import create_paged_attention_state

    source = checkpoint(with_values=True)
    importer = Qwen35MoeImporter(source, execution_phase="prefill", num_tokens=3)
    module = apply_numerical_profile(importer.import_module(), VLLM_AE10_INDUCTOR_LEVEL3)
    _, trace = TorchEvaluator(CheckpointWeightResolver(source)).run_with_trace(module, {
        "input_ids": torch.tensor([7, 11, 5], dtype=torch.int32),
        "gated_delta_net_state": create_gdn_state(importer.gdn_config),
        "paged_attention_state": create_paged_attention_state(importer.paged_config),
    })
    node = module.node_map["prefill_linear_recurrent.norm_apply"]
    core, stats, weight, bias = (trace[identity] for identity in node.inputs)
    expected = core.float() * torch.rsqrt(core.float().square().mean(-1, keepdim=True) + node.attrs["epsilon"])
    expected = expected * weight.float() + bias.float()
    assert torch.any(expected != expected.bfloat16().float()), "Exercise the removed rounding boundary."
    torch.testing.assert_close(trace[node.id], expected, rtol=0, atol=0)
    gate = trace["prefill_linear_recurrent.z_gate"]
    torch.testing.assert_close(trace["prefill_linear_recurrent.gated"], (expected * gate).bfloat16(), rtol=0, atol=0)


@pytest.mark.parametrize("phase,tokens", [("decode", 1), ("prefill", 3)])
def test_native_projection_stores_survive_private_widening(phase, tokens):
    from triton.flagmega.rules import DataflowRewriter
    from triton.flagmega.rules.neutral.fold_matmul_cast import fold_matmul_cast_rule

    importer = Qwen35MoeImporter(checkpoint(), execution_phase=phase, num_tokens=tokens)
    module = apply_numerical_profile(importer.import_module(), VLLM_AE10_INDUCTOR_LEVEL3)
    projections = {node.id for node in module.nodes
                   if node.op == "math.matmul" and node.type.dtype == fm.DType.BFLOAT16}
    result = DataflowRewriter((fold_matmul_cast_rule(),)).rewrite(module)
    assert projections <= result.node_map.keys()
    assert all(result.node_map[identity].type.dtype == fm.DType.BFLOAT16 for identity in projections)
