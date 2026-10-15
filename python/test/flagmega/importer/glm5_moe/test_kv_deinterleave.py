# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Interleaved projection stays whole until the paired cache consumer loads it."""

import pytest
import torch

from triton.flagmega.evaluator import CheckpointWeightResolver, TorchEvaluator
from triton.flagmega.importer.glm5_moe import Glm5MoeModelImporter
from triton.flagmega.ir.ops.nn._paged_attention_state import create_paged_attention_state

from python.test.flagmega.importer.glm5_moe.helpers import REDUCED, checkpoint, configuration


def _interleaved_scale(**overrides):
    """A checkpoint scale whose MLA layers carry one interleaved kv_b_proj.

    Only native FP8 checkpoints keep the projection fused; BF16 ones are
    regrouped in the constant island and never reach this path.
    """
    return dict(REDUCED, quantized=True, fp8_block_size=(4, 4), **overrides)


def _deinterleave_nodes(module):
    return [node for node in module.nodes
            if any(mark in node.id for mark in
                   ("_rows_", "_regrouped", "_kv_heads", "_key_heads", "_value_heads"))]


def test_interleaved_projection_updates_both_cache_fields_without_materialized_slices():
    source = checkpoint(_interleaved_scale(), with_values=False, full_model=True,
                        expert_layout="separate")
    module = Glm5MoeModelImporter(source, block_size=8, num_blocks=3).import_module()
    assert not [node for node in _deinterleave_nodes(module) if node.op == "tensors.slice"]
    pairs = [node for node in module.nodes if node.op == "nn.update_paged_attention_kv_cache_pair"]
    assert len(pairs) == 1


@pytest.mark.parametrize("heads", [2, 8, 64])
def test_regrouping_cost_is_independent_of_the_head_count(heads):
    source = checkpoint(_interleaved_scale(num_attention_heads=heads), with_values=False,
                        full_model=True, expert_layout="separate")
    module = Glm5MoeModelImporter(source, block_size=8, num_blocks=3).import_module()
    nodes = _deinterleave_nodes(module)
    assert [node.op for node in nodes] == ["tensors.reshape"]


def test_paired_consumer_receives_the_complete_head_major_view():
    heads = 8
    scale = _interleaved_scale(num_attention_heads=heads)
    source = checkpoint(scale, with_values=False, full_model=True, expert_layout="separate")
    module = Glm5MoeModelImporter(source, block_size=8, num_blocks=3).import_module()
    text = configuration(scale)["text_config"]
    view, = _deinterleave_nodes(module)
    assert tuple(dim.fixed_value for dim in view.type.shape) == (
        1, heads, text["qk_nope_head_dim"] + text["v_head_dim"])
    update, = [node for node in module.nodes if node.op == "nn.update_paged_attention_kv_cache_pair"]
    packed = module.node_map[update.inputs[0]]
    assert packed.op == "tensors.pack" and packed.inputs == (view.id,)
    assert not [node for node in module.nodes if node.op == "nn.update_paged_attention_kv_cache"]


def test_regrouping_preserves_the_cached_key_and_value_numerics():
    scale = _interleaved_scale(num_attention_heads=4)
    source = checkpoint(scale, full_model=True, expert_layout="separate")
    importer = Glm5MoeModelImporter(source, block_size=8, num_blocks=3)
    module = importer.import_module()
    text = configuration(scale)["text_config"]
    inputs = {"input_ids": torch.tensor([11], dtype=torch.int32),
              "paged_attention_state": create_paged_attention_state(importer.paged_config)}
    for layer in importer.kda_layers:
        qkv = text["kda_num_heads"] * text["kda_head_dim"]
        inputs[f"kda_conv_state_{layer}"] = torch.zeros(
            (3, text["kda_num_heads"], text["kda_head_dim"], text["kda_conv_kernel_size"] - 1),
            dtype=torch.bfloat16)
        inputs[f"kda_recurrent_state_{layer}"] = torch.zeros(
            (1, text["kda_num_heads"], text["kda_head_dim"], text["kda_head_dim"]),
            dtype=torch.float32)
    _, trace = TorchEvaluator(CheckpointWeightResolver(source)).run_with_trace(module, inputs)

    heads = text["num_attention_heads"]
    nope, v_head = text["qk_nope_head_dim"], text["v_head_dim"]
    interleaved = trace["decode_mla_attention_sparse_self_attn_kv_b_proj_weight_apply"]
    reference = interleaved.reshape(-1, heads, nope + v_head)
    state = inputs["paged_attention_state"]
    key, value = state.gather(layer_id=len(importer.mla_layers) - 1)
    torch.testing.assert_close(key, reference[..., :nope], rtol=0, atol=0)
    torch.testing.assert_close(value, reference[..., nope:], rtol=0, atol=0)
    assert state.sequence_length == 1
    assert state.query_start_loc.tolist() == [0, 1]
