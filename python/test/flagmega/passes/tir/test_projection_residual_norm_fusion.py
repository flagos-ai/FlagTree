# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from functools import lru_cache

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


@lru_cache(maxsize=2)
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


def test_decomposed_attention_keeps_projection_epilogue_and_statistics_explicit():
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

    # MatMul + residual + local statistics may fuse after distribution, but
    # neither attention nor the statistics collective is hidden in that op.
    projections = [point for point in points.values()
                   if point.id.startswith("tir.") and module.node_map[point.owner].op == "ntt.matmul_norm_stats"]
    expected = {"decode_layer_attention_output.vectorized.compute": "decode_layer_post_attention_norm.vectorized.compute",
                "decode_layer_mlp_down.vectorized.compute": "decode_layer_input_norm.vectorized.compute"}
    assert {module.node_map[point.owner].metadata["matmul_producer"] for point in projections} == set(expected)
    for point in projections:
        owner = module.node_map[point.owner]
        selected = next(candidate for candidate in point.candidates
                        if candidate.id == module.selection_map[point.id].candidate_id)
        assert owner.metadata["norm_consumer"] == expected[owner.metadata["matmul_producer"]]
        assert selected.parameters["family"] == "dense_matmul"
        assert selected.parameters["epilogue"] == "residual_norm_stats"
        assert selected.parameters["explicit_results"] == ("value", "norm_stats")
        assert selected.parameters["statistics_kind"] == "owner_partial"
        assert selected.facts["explicit_norm_stats_result"] is True
        assert selected.facts["internal_grid_barriers"] == 0
        value, stats = owner.type.fields
        assert value.partial is None
        assert stats.partial.reduce_op is fm.ReduceOp.SUM
        output_axes = {axis for policy in value.axis_policies if isinstance(policy, fm.SBPSplit)
                       for axis in policy.hierarchy_axes}
        assert set(stats.partial.axes) == output_axes
    assert any(node.op == "ntt.gather_reduce_norm_apply" for node in module.nodes)
    assert "tir.next_token" in points


def test_final_lm_head_has_no_residual_norm_fusion_candidate():
    module = _tir_candidates(num_layers=1)
    point, = (point for point in module.selection_points
              if point.id.startswith("tir.logits.")
              and any(candidate.parameters.get("family") == "dense_matmul" for candidate in point.candidates))

    assert all(
        candidate.parameters.get("epilogue") != "residual_norm_stats"
        for candidate in point.candidates
    )
    selected = next(candidate for candidate in point.candidates
                    if candidate.id == module.selection_map[point.id].candidate_id)
    assert selected.parameters["family"] == "dense_matmul"
    lm_head_type = module.node_map[point.owner].type
    assert isinstance(lm_head_type, fm.DistributedType)
    assert isinstance(lm_head_type.axis_policies[-1], fm.SBPSplit)
    assert lm_head_type.partial is None
    assert "tir.next_token" in {
        value.id for value in module.selection_points
    }
