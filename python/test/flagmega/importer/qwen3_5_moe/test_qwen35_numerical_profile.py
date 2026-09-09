# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Pinned hybrid-decoder frontend numerical contracts."""

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import ImporterError
from triton.flagmega.importer import apply_numerical_profile, import_model
from triton.flagmega.importer.numerics import VLLM_AE10_INDUCTOR_LEVEL3
from triton.flagmega.ir.ops.nn.gdn_recurrent_core import GatedDeltaNetRecurrentCore
from python.test.flagmega.importer.qwen3_5_moe.helpers import checkpoint


def test_profile_preserves_reuse_and_materializes_only_declared_rounding_boundaries():
    original = import_model(checkpoint())
    module = apply_numerical_profile(original, VLLM_AE10_INDUCTOR_LEVEL3)
    assert len(module.functions) == len(original.functions)
    assert [node.attrs["callee"] for node in module.nodes if node.op == "builtin.call"
            ] == [node.attrs["callee"] for node in original.nodes if node.op == "builtin.call"]
    for name in ("decode_linear", "decode_attention"):
        nodes = module.node_map
        assert nodes[name + "_hidden"].type.dtype == fm.DType.FLOAT32
        assert nodes[name + "_input_norm"].type.dtype == fm.DType.BFLOAT16
        assert nodes[name + "_post_norm"].type.dtype == fm.DType.BFLOAT16
        for suffix in ("_attention_residual", "_moe_output", "_output"):
            assert nodes[name + suffix].type.dtype == fm.DType.FLOAT32
        residual = nodes[name + "_attention_residual.residual_bf16"]
        assert residual.op == "tensors.cast" and residual.type.dtype == fm.DType.BFLOAT16
        experts = nodes[name + "_routed_experts"]
        assert experts.attrs["round_projections"] and experts.attrs["round_activation"]
        assert not experts.attrs["round_down_projection"] and experts.attrs["round_weighted_output"]
    conv = module.node_map["decode_linear_convolution"]
    assert conv.attrs["round_products"] and not conv.attrs["round_before_activation"]
    assert conv.attrs["accumulation_order"] == "chronological"
    recurrent = module.node_map["decode_linear_recurrent"]
    assert recurrent.attrs["qk_norm_mode"] == "add" and recurrent.attrs["qk_norm_epsilon"] == 1e-6
    # Native fused_recurrent_gated_delta_rule_packed_decode_kernel keeps
    # normalized Q/K in FP32 and explicitly rounds sigmoid beta to BF16.
    # The separate prefill preparation path has different store boundaries.
    assert not recurrent.attrs["round_normalized_qk"]
    assert recurrent.attrs["round_beta"] and recurrent.attrs["round_core"]
    weight = module.node_map[GatedDeltaNetRecurrentCore.norm_weight.read(recurrent.inputs)]
    assert weight.op == "tensors.cast" and weight.type.dtype == fm.DType.BFLOAT16
    for node in module.nodes:
        if node.op == "nn.rope":
            assert all(module.node_map[value].type.dtype == fm.DType.BFLOAT16 for value in node.inputs)
            assert node.type.dtype == fm.DType.BFLOAT16
            assert all(module.node_map[value].op != "tensors.cast" for value in node.inputs)
    rotary, = (node for node in module.nodes if node.op == "nn.rotary_embedding")
    assert rotary.attrs["output_dtype"] == "bfloat16"


def test_profile_is_idempotent_and_python_round_trip_keeps_contract(tmp_path):
    module = import_model(checkpoint(), numerical_profile=VLLM_AE10_INDUCTOR_LEVEL3)
    assert apply_numerical_profile(module, VLLM_AE10_INDUCTOR_LEVEL3).semantic_hash == module.semantic_hash
    path = tmp_path / "decode.py"
    fm.emit_module(module, path)
    loaded = fm.load_module(path)
    assert loaded.semantic_hash == module.semantic_hash
    assert loaded.metadata["numerical_profile_reference"]["phase"] == "decode"
    with pytest.raises(ImporterError, match="Cannot change"):
        apply_numerical_profile(module, "nncase")
    with pytest.raises(ImporterError, match="require imported"):
        apply_numerical_profile(replace(module, stage="target_independent"), VLLM_AE10_INDUCTOR_LEVEL3)


def test_shared_expert_gate_preserves_eager_sigmoid_store_inside_moe_custom_op():
    module = import_model(checkpoint(), numerical_profile=VLLM_AE10_INDUCTOR_LEVEL3)
    for name in ("decode_linear", "decode_attention"):
        gate = module.node_map[name + "_shared_gate"]
        assert gate.op == "math.sigmoid" and gate.type.dtype == fm.DType.BFLOAT16
        scaled = module.node_map[name + "_shared_scaled"]
        assert scaled.op == "math.mul" and scaled.type.dtype == fm.DType.BFLOAT16
    # This separate gate is in the surrounding compiled graph, where the
    # sigmoid and multiply are fused without a BF16 intermediate store.
    assert module.node_map["decode_attention_attention_gate"].type.dtype == fm.DType.FLOAT32


def test_profile_recognizes_gate_dependencies_in_older_anonymous_python_ir():
    module = import_model(checkpoint())
    renamed = {
        node.id: f"anonymous_sigmoid_{index}"
        for index, node in enumerate(module.nodes)
        if node.op == "math.sigmoid"
    }
    legacy = fm.verify_module(
        replace(
            module, nodes=tuple(
                replace(node, id=renamed.get(node.id, node.id), inputs=tuple(
                    renamed.get(value, value) for value in node.inputs)) for node in module.nodes)))
    namespace = {}
    exec(fm.module_source(legacy), namespace)
    result = apply_numerical_profile(namespace["MODULE"], VLLM_AE10_INDUCTOR_LEVEL3)
    for name in ("decode_linear", "decode_attention"):
        assert result.node_map[renamed[name + "_shared_gate"]].type.dtype == fm.DType.BFLOAT16
    assert result.node_map[renamed["decode_attention_attention_gate"]].type.dtype == fm.DType.FLOAT32


@pytest.mark.parametrize("tokens", (1, 3, 65, 512))
def test_native_prefill_uses_explicit_block_stages_instead_of_decode_recurrence(tokens):
    module = import_model(checkpoint(), mode="prefill", num_tokens=tokens,
                          numerical_profile=VLLM_AE10_INDUCTOR_LEVEL3)
    assert module.metadata["numerical_profile_reference"]["phase"] == "prefill"
    assert not any(node.op == "nn.gdn_recurrent_core" for node in module.nodes)
    assert {node.op for node in module.nodes} >= {"nn.l2_normalization", "nn.delta_rule_gates",
        "nn.delta_rule_coefficients", "nn.delta_rule_log_prefix", "nn.delta_rule_block_update"}
    assert module.node_map["prefill_linear_recurrent"].op == "builtin.tuple"
    assert module.node_map["prefill_linear_recurrent.core"].type.dtype == fm.DType.BFLOAT16
    assert module.node_map["prefill_linear_recurrent.norm_apply"].type.dtype == fm.DType.FLOAT32
    assert set(module.function_map) == {"main", "prefill_linear", "prefill_attention"}
    namespace = {}
    exec(fm.module_source(module), namespace)
    assert namespace["MODULE"].semantic_hash == module.semantic_hash


def test_cp_prefill_contract_does_not_claim_unimplemented_cross_chunk_transfer():
    with pytest.raises(ImporterError, match="chunks of at most 512"):
        import_model(checkpoint(), mode="prefill", num_tokens=513, numerical_profile=VLLM_AE10_INDUCTOR_LEVEL3)


def test_wide_residual_and_moe_additions_follow_independent_arithmetic():
    torch = pytest.importorskip("torch")
    from triton.flagmega.evaluator import CheckpointWeightResolver, TorchEvaluator
    from triton.flagmega.importer import Qwen35MoeImporter
    from triton.flagmega.ir.ops.nn._gdn_state import create_gdn_state
    from triton.flagmega.ir.ops.nn._paged_attention_state import create_paged_attention_state
    source = checkpoint(with_values=True)
    importer = Qwen35MoeImporter(source)
    module = apply_numerical_profile(importer.import_module(), VLLM_AE10_INDUCTOR_LEVEL3)
    _, trace = TorchEvaluator(CheckpointWeightResolver(source)).run_with_trace(
        module, {
            "input_ids": torch.tensor([7], dtype=torch.int32),
            "gated_delta_net_state": create_gdn_state(importer.gdn_config),
            "paged_attention_state": create_paged_attention_state(importer.paged_config),
        })
    for name in ("decode_linear", "decode_attention"):
        residual = trace[name + "_hidden"].bfloat16().float() + trace[name + "_attention_output"].float()
        torch.testing.assert_close(trace[name + "_attention_residual"], residual, rtol=0, atol=0)
        # The shared and routed outputs are individual BF16 stores before
        # their FP32 sum and the next residual/input-normalization segment.
        moe = module.node_map[name + "_moe_output"]
        expected_moe = trace[moe.inputs[0]].float() + trace[moe.inputs[1]].float()
        torch.testing.assert_close(trace[name + "_output"], residual + expected_moe, rtol=0, atol=0)
