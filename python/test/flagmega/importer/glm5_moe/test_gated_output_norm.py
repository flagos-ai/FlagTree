# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""KDA's output gate is ``FusedRMSNormGated``, whose interior is entirely FP32.

Building it from ``norm_apply`` + ``sigmoid`` + ``mul`` + ``cast`` introduces two
rounding boundaries the reference does not have: ``NormApply`` rounds the
normalized value to the *input* dtype before honouring ``output_dtype``, and a
BF16-typed ``Math.Sigmoid`` rounds the activation to the gate dtype.  Those are
not removable by a graph rewrite, because every materialized node publishes a
dtype, so the region emits one ``nn.gated_norm_apply`` instead.
"""

import pytest
import torch

from triton.flagmega.evaluator import CheckpointWeightResolver, TorchEvaluator
from triton.flagmega.importer.glm5_moe import Glm5MoeModelImporter
from triton.flagmega.ir.ops.nn._paged_attention_state import create_paged_attention_state

from python.test.flagmega.importer.glm5_moe.helpers import REDUCED, checkpoint, configuration

EPS = 1e-5


def _module(scale, *, with_values=False):
    source = checkpoint(scale, with_values=with_values, full_model=True,
                        expert_layout="separate")
    importer = Glm5MoeModelImporter(source, block_size=8, num_blocks=3)
    return source, importer, importer.import_module()


def _kda_output_nodes(module):
    marks = ("_o_norm", "_out_gate", "_out_gated", "_out_wide", "_out_flat")
    return [node for node in module.nodes
            if node.id.startswith("decode_linear_attention_")
            and any(mark in node.id for mark in marks)]


def test_the_output_gate_is_one_fused_node_per_region():
    _, _, module = _module(dict(REDUCED))
    fused = [node for node in module.nodes if node.op == "nn.gated_norm_apply"]

    # One per KDA region (dense-MLP and sparse-MLP), not one per layer.
    assert len(fused) == 2
    for node in fused:
        assert node.attrs["activation"] == "sigmoid"
        assert node.attrs["use_mean"] is False
        assert node.attrs["round_before_scale"] is False
        assert node.attrs["output_dtype"] == "bfloat16"


def test_the_expanded_gate_chain_leaves_no_residue():
    """The FP32 round-trip nodes are gone, not merely renamed."""
    _, _, module = _module(dict(REDUCED))
    residue = [node.id for node in _kda_output_nodes(module)
               if node.op in {"math.sigmoid", "math.mul", "tensors.cast", "nn.norm_apply"}]

    assert residue == []
    reshapes = [node for node in _kda_output_nodes(module) if node.op == "tensors.reshape"]
    # Only the head view in and the flat view out survive, per region.
    assert len(reshapes) == 4


def test_the_fused_gate_matches_fused_rms_norm_gated_bit_for_bit():
    scale = dict(REDUCED)
    source, importer, module = _module(scale, with_values=True)
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

    prefix = "decode_linear_attention_dense"
    core = trace[f"{prefix}_recurrent_0"]
    gate = trace[f"{prefix}_out_gate_heads"]
    weight = trace[f"{prefix}_self_attn_o_norm_weight"]
    observed = trace[f"{prefix}_o_norm_gated"]

    # vLLM FusedRMSNormGated.forward_native, sigmoid branch, no residual.
    wide = core.float()
    variance = wide.pow(2).mean(dim=-1, keepdim=True)
    normalized = wide * torch.rsqrt(variance + EPS) * weight.float()
    expected = (normalized * torch.sigmoid(gate.float())).to(torch.bfloat16)
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)

    # The rejected expansion: prove the fixture separates the two policies, so
    # the bit-exact assertion above cannot pass on a doubly-rounded graph.
    doubly_rounded = (
        normalized.to(torch.bfloat16).float()
        * torch.sigmoid(gate.float()).to(torch.bfloat16).float()
    ).to(torch.bfloat16)
    assert torch.count_nonzero(expected != doubly_rounded), (
        "This fixture no longer distinguishes the FP32 interior from the "
        "norm_apply/sigmoid/mul expansion, so the assertion above is vacuous."
    )


@pytest.mark.parametrize("heads", [2, 8])
def test_the_gate_view_keeps_the_head_axis(heads):
    _, _, module = _module(dict(REDUCED, num_attention_heads=heads))
    text = configuration(dict(REDUCED, num_attention_heads=heads))["text_config"]
    view = next(node for node in module.nodes
                if node.id == "decode_linear_attention_dense_out_gate_heads")

    assert tuple(dim.fixed_value for dim in view.type.shape) == (
        1, text["kda_num_heads"], text["kda_head_dim"])
