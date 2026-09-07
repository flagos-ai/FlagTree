# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega.compiler import Compiler
from triton.flagmega.importer import MemoryCheckpoint, TensorInfo, import_qwen3_model
from triton.flagmega.ir import DType
from triton.flagmega import ir as fm
from triton.flagmega.passes.tir import find_projection_residual_norm_matches


def _checkpoint(num_layers: int) -> MemoryCheckpoint:
    hidden, intermediate, vocab = 2048, 6144, 4096
    config = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "vocab_size": vocab,
        "num_hidden_layers": num_layers,
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "hidden_act": "silu",
        "attention_bias": False,
        "mlp_bias": False,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_000_000.0,
        "max_position_embeddings": 4096,
        "tie_word_embeddings": True,
        "pad_token_id": None,
    }
    shapes = {
        "model.embed_tokens.weight": (vocab, hidden),
        "model.norm.weight": (hidden,),
    }
    for layer in range(num_layers):
        prefix = f"model.layers.{layer}."
        shapes.update({
            prefix + "input_layernorm.weight": (hidden,),
            prefix + "self_attn.q_proj.weight": (hidden, hidden),
            prefix + "self_attn.k_proj.weight": (1024, hidden),
            prefix + "self_attn.v_proj.weight": (1024, hidden),
            prefix + "self_attn.q_norm.weight": (128,),
            prefix + "self_attn.k_norm.weight": (128,),
            prefix + "self_attn.o_proj.weight": (hidden, hidden),
            prefix + "post_attention_layernorm.weight": (hidden,),
            prefix + "mlp.gate_proj.weight": (intermediate, hidden),
            prefix + "mlp.up_proj.weight": (intermediate, hidden),
            prefix + "mlp.down_proj.weight": (hidden, intermediate),
        })
    infos = {
        name: TensorInfo(name, DType.BFLOAT16, shape, "metadata.safetensors")
        for name, shape in shapes.items()
    }
    return MemoryCheckpoint(config, infos)


def _tir_candidates(num_layers: int = 2):
    return Compiler().compile(
        import_qwen3_model(_checkpoint(num_layers)),
        stop_after="propose-tir",
    ).module


def test_packed_dense_projection_is_a_direct_residual_norm_producer():
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="ntt", stage="distributed", entry="main")

        def forward(self):
            lhs = self.input("lhs", fm.tensor_type("bfloat16", (1, 512)), id="lhs")
            weight = self.input(
                "weight",
                fm.tensor_type("bfloat16", (32, 64, 8, 16)),
                id="weight",
            )
            residual = self.input(
                "residual", fm.tensor_type("bfloat16", (1, 512)), id="residual")
            norm_weight = self.input(
                "norm_weight", fm.tensor_type("bfloat16", (512,)), id="norm_weight")
            projection = fm.F.math.packed_dense_matmul(
                lhs, weight, name="projection")
            value = fm.F.math.add(projection, residual, name="value")
            norm = fm.F.nn.rms_norm(
                value, norm_weight, epsilon=1e-6, name="norm")
            self.function(
                "main", (lhs, weight, residual, norm_weight), (value, norm))

    match = find_projection_residual_norm_matches(Graph().build())["projection"]

    assert match.projection_value == "projection"
    assert match.residual_add == "value"
    assert match.residual_input == "residual"
    assert match.norm_consumer == "norm"


def test_decomposed_attention_keeps_semantic_tir_roles_independent():
    module = _tir_candidates()
    points = {point.id: point for point in module.selection_points}

    paged_attention_id = "tir.decode_layer_paged_attention"
    assert module.selection_map[paged_attention_id].candidate_id == (
        "semantic.ntt.paged_attention_combine"
    )
    paged_attention = points[paged_attention_id]
    assert tuple(value.id for value in paged_attention.candidates) == (
        "semantic.ntt.paged_attention_combine",
    )
    assert not paged_attention.candidates[0].parameters
    assert not paged_attention.candidates[0].facts

    # The imported attention graph remains decomposed through semantic TIR:
    # output projection and residual/statistics combination are independently
    # selectable, instead of being hidden in an SM90-shaped attention rule.
    attention_output_id = "tir.decode_layer_attention_output.vectorized.compute"
    attention_combine_id = (
        "tir.decode_layer_after_attention.vectorized.compute.norm_stats_combine"
    )
    assert attention_output_id in points
    assert module.selection_map[attention_combine_id].candidate_id == (
        "tir.gather_reduce_add_norm_apply.sum"
    )
    attention_combine = next(
        candidate for candidate in points[attention_combine_id].candidates
        if candidate.id == module.selection_map[attention_combine_id].candidate_id
    )
    attention_owner = module.node_map[points[attention_combine_id].owner]
    assert attention_owner.metadata["residual_add"] == (
        "decode_layer_after_attention.vectorized.compute"
    )
    assert attention_owner.metadata["norm_consumer"] == (
        "decode_layer_post_attention_norm.vectorized.compute"
    )
    assert attention_combine.parameters["partial_axes"] == (1,)
    assert attention_combine.parameters["partial_owner_count"] == 16
    assert attention_combine.parameters["owner_count"] == 128
    assert attention_combine.facts["private_norm_stats_workspace"] is True

    down_point_id = "tir.decode_layer_output.vectorized.compute.norm_stats_combine"
    assert module.selection_map[down_point_id].candidate_id == (
        "tir.gather_reduce_add_norm_stats.sum_rms"
    )
    assert module.selection_map[
        "tir.decode_layer_mlp_down.vectorized.compute"
    ].candidate_id == (
        "tir.dense_matmul.split_k_packed_k_major_gemv"
    )
    assert module.selection_map["tir.lm_head.vectorized.compute"].candidate_id == (
        "tir.dense_matmul.packed_tensor_descriptor_smem_pipeline_gemv"
    )
    assert "tir.next_token" in points

    down = points[down_point_id]
    down_selected = next(
        candidate for candidate in down.candidates
        if candidate.id == module.selection_map[down.id].candidate_id
    )
    assert down_selected.parameters["family"] == (
        "gather_reduce_add_norm_stats"
    )
    assert down_selected.parameters["variant"] == "sum_rms"
    assert down_selected.parameters["partial_axes"] == (0, 1)
    assert down_selected.parameters["partial_owner_count"] == 128
    assert down_selected.parameters["owner_count"] == 128
    assert down_selected.parameters["residual_add"] == (
        "decode_layer_output.vectorized.compute"
    )
    assert down_selected.parameters["norm_consumer"] == (
        "decode_layer_input_norm.vectorized.compute"
    )
    assert down_selected.facts["explicit_norm_stats_result"] is True
    assert down_selected.facts["collective_semantics"] == (
        "gather-reduce-add-norm-stats"
    )


def test_final_lm_head_has_no_residual_norm_fusion_candidate():
    module = _tir_candidates(num_layers=1)
    point = next(
        point for point in module.selection_points
        if point.id == "tir.lm_head.vectorized.compute"
    )

    assert all(
        candidate.parameters.get("epilogue") != "residual_norm_stats"
        for candidate in point.candidates
    )
    assert module.selection_map["tir.lm_head.vectorized.compute"].candidate_id == (
        "tir.dense_matmul.packed_tensor_descriptor_smem_pipeline_gemv"
    )
    lm_head_type = module.node_map["lm_head.vectorized.compute"].type
    assert isinstance(lm_head_type, fm.DistributedType)
    assert isinstance(lm_head_type.axis_policies[-1], fm.SBPSplit)
    assert lm_head_type.partial is None
    assert "tir.next_token" in {
        value.id for value in module.selection_points
    }
