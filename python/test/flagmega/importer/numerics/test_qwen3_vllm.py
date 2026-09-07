# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
from dataclasses import replace

import pytest
import torch

from triton.flagmega.importer.numerics.qwen3 import apply_qwen3_vllm_profile
from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.importer import (MemoryCheckpoint, TensorInfo, import_model,
                                     apply_numerical_profile, VLLM_INDUCTOR_LEVEL3)
from triton.flagmega.errors import IRSchemaError, ImporterError


@pytest.fixture
def checkpoint():
    config = dict(architectures=["Qwen3ForCausalLM"], model_type="qwen3", vocab_size=32,
                  num_hidden_layers=2, hidden_size=16, intermediate_size=32, num_attention_heads=2,
                  num_key_value_heads=1, head_dim=8, hidden_act="silu", attention_bias=False,
                  mlp_bias=False, rms_norm_eps=1e-6, rope_theta=10000., max_position_embeddings=128,
                  tie_word_embeddings=True)
    specs = {"model.embed_tokens.weight": (32, 16), "model.norm.weight": (16,)}
    layer_shapes = {"input_layernorm.weight": (16,), "post_attention_layernorm.weight": (16,),
                    "self_attn.q_proj.weight": (16, 16), "self_attn.k_proj.weight": (8, 16),
                    "self_attn.v_proj.weight": (8, 16), "self_attn.o_proj.weight": (16, 16),
                    "self_attn.q_norm.weight": (8,), "self_attn.k_norm.weight": (8,),
                    "mlp.gate_proj.weight": (32, 16), "mlp.up_proj.weight": (32, 16),
                    "mlp.down_proj.weight": (16, 32)}
    for layer in range(2):
        specs.update({f"model.layers.{layer}.{name}": shape for name, shape in layer_shapes.items()})
    checkpoint = MemoryCheckpoint(config, {key: TensorInfo(key, fm.DType.BFLOAT16, shape, "test.safetensors")
                                           for key, shape in specs.items()})
    return checkpoint


@pytest.fixture
def model(checkpoint):
    return import_model(checkpoint)


def fragment(module, output, parameters):
    selected = {}

    def visit(name):
        if name in selected:
            return
        node = module.node_map[name]
        if name in parameters:
            selected[name] = fm.Node(name, "builtin.var", (), node.type, attrs={"name": name})
            return
        for child in node.inputs:
            visit(child)
        selected[name] = node

    visit(output)
    return fm.IRModule("high_level", "imported", tuple(selected.values()),
                       (fm.Function("main", tuple(parameters), (output,)),), "main")


def evaluate(module, output, inputs):
    return TorchEvaluator(DictWeightResolver({})).run(fragment(module, output, inputs), inputs)[0]


def test_preserves_function_reuse_and_round_trips(model, tmp_path):
    result = apply_qwen3_vllm_profile(model)
    decode = result.function_map["decode_layer"]
    assert decode.attrs == model.function_map["decode_layer"].attrs
    assert result.node_map[decode.parameters[0]].type.dtype == fm.DType.FLOAT32
    assert result.node_map[decode.outputs[0]].type.dtype == fm.DType.FLOAT32
    assert len([node for node in result.nodes if node.op == "builtin.call"]) == 2
    assert fm.load_module(fm.emit_module(result, tmp_path / "numeric.py")).semantic_hash == result.semantic_hash
    assert model.node_map[decode.parameters[0]].type.dtype == fm.DType.BFLOAT16


def test_input_norm_consumes_unrounded_residual_sum(model):
    result = apply_qwen3_vllm_profile(model)
    norm = model.node_map["decode_layer_input_norm"]
    generator = torch.Generator().manual_seed(87)
    value = torch.randn(1, 16, generator=generator).bfloat16().float() + .003
    weight = torch.randn(16, generator=generator).bfloat16()
    inputs = {norm.inputs[0]: value, norm.inputs[1]: weight}
    actual = evaluate(result, norm.id, inputs)
    expected = (value * (value.square().mean(-1, keepdim=True) + 1e-6).rsqrt() * weight.float()).bfloat16()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    rounded = evaluate(model, norm.id, {norm.inputs[0]: value.bfloat16(), norm.inputs[1]: weight})
    assert not torch.equal(actual, rounded)


def test_silu_has_only_projection_and_final_bf16_boundaries(model):
    result = apply_qwen3_vllm_profile(model)
    node = model.node_map["decode_layer_mlp_gate_up"]
    generator = torch.Generator().manual_seed(71)
    values = (torch.randn(1, 16, generator=generator).bfloat16(),
              torch.randn(32, 16, generator=generator).bfloat16(),
              torch.randn(32, 16, generator=generator).bfloat16())
    actual = evaluate(result, node.id, dict(zip(node.inputs, values, strict=True)))
    gate = torch.nn.functional.linear(values[0], values[1]).float()
    up = torch.nn.functional.linear(values[0], values[2]).float()
    expected = (torch.nn.functional.silu(gate) * up).bfloat16()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_profile_is_idempotent_but_rejects_later_ir(model):
    specialized = apply_qwen3_vllm_profile(model)
    assert apply_numerical_profile(specialized, VLLM_INDUCTOR_LEVEL3) == specialized
    with pytest.raises(IRSchemaError, match="requires imported"):
        apply_qwen3_vllm_profile(replace(model, stage="distributed"))
    with pytest.raises(ImporterError, match="require imported"):
        apply_numerical_profile(replace(model, stage="distributed"), VLLM_INDUCTOR_LEVEL3)
    with pytest.raises(ImporterError, match="Cannot change"):
        apply_numerical_profile(specialized, "nncase")


def test_qnorm_rope_share_wide_intermediate_but_tables_are_bf16(model):
    result = apply_qwen3_vllm_profile(model)
    norm = model.node_map["decode_layer_query_norm"]
    rope = model.node_map["decode_layer_query_rope"]
    generator = torch.Generator().manual_seed(13)
    value = torch.randn(1, 2, 8, generator=generator).bfloat16()
    weight = torch.randn(8, generator=generator).bfloat16()
    cos = torch.randn(1, 1, 8, generator=generator)
    sin = torch.randn(1, 1, 8, generator=generator)
    actual = evaluate(result, rope.id, {norm.inputs[0]: value, norm.inputs[1]: weight,
                                       rope.inputs[1]: cos, rope.inputs[2]: sin})
    unit = value.float() * (value.float().square().mean(-1, keepdim=True) + 1e-6).rsqrt()
    normalized = unit * weight.float()
    rotated = torch.cat((-normalized[..., 4:], normalized[..., :4]), dim=-1)
    expected = (normalized * cos.bfloat16().float() + rotated * sin.bfloat16().float()).bfloat16()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_residual_rounds_at_attention_boundary_not_at_layer_return(model):
    result = apply_qwen3_vllm_profile(model)
    hidden = model.function_map["decode_layer"].parameters[0]
    generator = torch.Generator().manual_seed(73)
    x = torch.randn(1, 16, generator=generator).bfloat16().float() + .003
    attention = torch.randn(1, 16, generator=generator).bfloat16()
    down = torch.randn(1, 16, generator=generator).bfloat16()
    output = evaluate(result, "decode_layer_output", {hidden: x, "decode_layer_attention_output": attention,
                                                      "decode_layer_mlp_down": down})
    expected = (x.bfloat16().float() + attention.float()) + down.float()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert output.dtype == torch.float32
    assert not torch.equal(output, output.bfloat16().float())


def test_import_profile_and_normal_pass_preserve_wide_activation(model, tmp_path):
    from triton.flagmega.passes.target_independent import decompose_complex_ops

    specialized = apply_numerical_profile(model, VLLM_INDUCTOR_LEVEL3)
    fused = decompose_complex_ops(specialized)
    root = fused.node_map["decode_layer_mlp_gate_up"]
    assert root.op == "nn.dense_matmul_glu" and root.attrs["round_activation"] is False
    node = model.node_map[root.id]
    generator = torch.Generator().manual_seed(71)
    values = (torch.randn(1, 16, generator=generator).bfloat16(),
              torch.randn(32, 16, generator=generator).bfloat16(),
              torch.randn(32, 16, generator=generator).bfloat16())
    inputs = dict(zip(node.inputs, values, strict=True))
    actual = evaluate(fused, root.id, inputs)
    torch.testing.assert_close(actual, evaluate(specialized, root.id, inputs), rtol=0, atol=0)
    assert not torch.equal(actual, evaluate(model, root.id, inputs))
    assert fm.load_module(fm.emit_module(fused, tmp_path / "fused.py")).semantic_hash == fused.semantic_hash


def test_public_import_selects_explicit_contract_and_default_stays_unchanged(checkpoint, model):
    selected = import_model(checkpoint, numerical_profile=VLLM_INDUCTOR_LEVEL3)
    assert selected == apply_numerical_profile(model, VLLM_INDUCTOR_LEVEL3)
    assert import_model(checkpoint, numerical_profile="nncase") == model
    with pytest.raises(ImporterError, match="does not support numerical profile"):
        import_model(checkpoint, numerical_profile="vllm-unvalidated-version")


def test_profile_ir_loads_and_runs_normal_passes_without_tutorial_imports(model, tmp_path):
    import subprocess
    import sys

    selected = apply_numerical_profile(model, VLLM_INDUCTOR_LEVEL3)
    path = fm.emit_module(selected, tmp_path / "imported.py")
    subprocess.run([sys.executable, "-c", (
        "import sys; from triton.flagmega.ir import load_module; "
        "from triton.flagmega.passes.target_independent import decompose_complex_ops; "
        "from triton.flagmega.passes.form_qkv_rope_with_cache import form_qkv_rope_with_cache; "
        f"m=load_module({str(path)!r}); assert m.semantic_hash == {selected.semantic_hash!r}; "
        "m=form_qkv_rope_with_cache(decompose_complex_ops(m)); "
        "assert sum(n.op == 'nn.qkv_rope_with_cache' for n in m.nodes) == 1; "
        "assert sum(n.op == 'nn.dense_matmul_glu' for n in m.nodes) == 1; "
        "assert not any(n.startswith('local_optimizations') for n in sys.modules)"
    )], check=True)
