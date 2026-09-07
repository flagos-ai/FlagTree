# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega.importer import (
    Qwen3LayerImporter,
    import_model_layer,
    source_location_of,
)
from triton.flagmega.ir import emit_module, load_module

from .helpers import checkpoint


def test_importer_builds_the_nncase_no_lm_head_boundary():
    module = Qwen3LayerImporter(checkpoint(), block_size=4, num_blocks=2).import_module()
    function = module.function_map["main"]

    assert module.metadata["architecture"] == "Qwen3ForCausalLM"
    assert module.metadata["output_boundary"] == "final_norm_hidden_fp32"
    assert function.parameters == ("input_ids", "paged_attention_kv_cache")
    assert function.outputs == ("output", "updated_state")
    assert module.node_map["self_attention_qkv_projection"].op == (
        "nn.qkv_parallel_linear"
    )
    assert module.node_map["self_attention_query_rope"].op == "nn.rope"
    assert module.node_map["self_attention_key_cache_update"].op == (
        "nn.update_paged_attention_kv_cache"
    )
    assert module.node_map["self_attention_paged_attention"].op == (
        "nn.paged_attention"
    )
    assert module.node_map["self_attention_query_cache_pack"].op == "tensors.pack"
    assert module.node_map["self_attention_key_cache_pack"].op == "tensors.pack"
    assert module.node_map["self_attention_value_cache_pack"].op == "tensors.pack"
    assert module.node_map["self_attention_paged_attention_unpacked"].op == (
        "tensors.unpack"
    )
    assert module.node_map["self_attention_key_cache_update"].inputs[0] == (
        "self_attention_key_cache_pack"
    )
    assert module.node_map["self_attention_paged_attention"].inputs[0] == (
        "self_attention_query_cache_pack"
    )
    assert module.node_map["attention_output"].op == "math.matmul"
    assert module.node_map["mlp_gate_up"].op == "nn.dense_matmul_glu"
    assert module.node_map["mlp_down"].attrs["transpose_b"] is True
    assert module.node_map["output"].op == "tensors.cast"
    assert not any(node.op == "nn.gated_delta_net" for node in module.nodes)
    assert not any("lm_head" in str(node.attrs.get("key", "")) for node in module.nodes)
    attention_source = source_location_of(
        module.node_map["self_attention_paged_attention"])
    weight_source = source_location_of(module.node_map["w_self_attn_q_proj_weight"])
    assert attention_source.uri == "huggingface://Qwen3ForCausalLM"
    assert attention_source.symbol == "self_attention_paged_attention"
    assert weight_source.uri == "safetensors://qwen3-unit.safetensors"
    assert weight_source.symbol.endswith("self_attn.q_proj.weight")


def test_python_ir_dump_round_trips_qwen3_stateful_graph(tmp_path):
    module = Qwen3LayerImporter(checkpoint(), block_size=4, num_blocks=2).import_module()
    path = emit_module(module, tmp_path / "main.py")
    source = path.read_text(encoding="utf-8")

    assert "F.nn.qkv_parallel_linear(" in source
    assert "F.nn.rotary_embedding(" in source
    assert "F.nn.rope(" in source
    assert "F.nn.update_paged_attention_kv_cache(" in source
    assert "F.nn.paged_attention(" in source
    assert "F.nn.qwen3_paged_attention(" not in source
    assert "F.nn.dense_matmul_glu(" in source
    assert "F.tensors.cast(" in source
    loaded = load_module(path)
    assert loaded.semantic_hash == module.semantic_hash
    assert source_location_of(
        loaded.node_map["self_attention_paged_attention"]
    ) == source_location_of(
        module.node_map["self_attention_paged_attention"]
    )


def test_architecture_dispatch_selects_qwen3_importer():
    module = import_model_layer(checkpoint())

    assert module.metadata["architecture"] == "Qwen3ForCausalLM"
    assert module.node_map["self_attention_qkv_projection"].op == (
        "nn.qkv_parallel_linear"
    )
    assert module.node_map["self_attention_paged_attention"].op == (
        "nn.paged_attention"
    )
