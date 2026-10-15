# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The KDA convolution must keep q/k/v separate and head-major.

Root cause this pins down: the importer's constant island concatenated the
three published ``q/k/v_conv1d.weight`` tensors into one flat ``[3*H*D, K]``
weight, so the decoder also had to build one flat ``[1, 3*H*D]`` mixed
projection.  On that flat axis head ``h`` occupies three discontiguous ranges
(``h*D``, ``H*D + h*D``, ``2*H*D + h*D``), which makes the head axis
structurally unsplittable — no SBP policy can describe "this owner holds heads
``[a, b)``" on it.  Upstream (``vllm`` ``kimi_gdn_linear_attn.py``) keeps three
separate head-split ``ColumnParallelLinear`` convolutions and
``mamba_utils.kda_state_shape`` divides the conv state by the TP size, i.e. the
q|k|v concat happens *after* the split.

These tests assert the structural contract only.  Numerics are pinned bit-for-bit
by ``test_the_head_major_convolution_matches_the_flat_reference``.
"""

import pytest
import torch

from triton.flagmega.evaluator import CheckpointWeightResolver, TorchEvaluator
from triton.flagmega.importer.glm5_moe import Glm5MoeModelImporter
from triton.flagmega.ir.model import DistributedType
from triton.flagmega.ir.ops.nn._paged_attention_state import create_paged_attention_state

from python.test.flagmega.importer.glm5_moe.helpers import REDUCED, checkpoint, configuration

PREFIX = "decode_linear_attention_dense"


def _imported(*, with_values=False):
    scale = dict(REDUCED)
    source = checkpoint(scale, with_values=with_values, full_model=True,
                        expert_layout="separate")
    importer = Glm5MoeModelImporter(source, block_size=8, num_blocks=3)
    return source, importer, importer.import_module(), configuration(scale)["text_config"]


def _region_nodes(module, prefix=PREFIX):
    return [node for node in module.nodes if node.id.startswith(prefix)]


def _conv_nodes(module):
    return tuple(node for node in module.nodes if node.op == "nn.kda_convolution")


def _states(text, importer):
    inputs = {}
    for layer in importer.kda_layers:
        inputs[f"kda_conv_state_{layer}"] = torch.zeros(
            (3, text["kda_num_heads"], text["kda_head_dim"], text["kda_conv_kernel_size"] - 1),
            dtype=torch.bfloat16)
        inputs[f"kda_recurrent_state_{layer}"] = torch.zeros(
            (1, text["kda_num_heads"], text["kda_head_dim"], text["kda_head_dim"]), dtype=torch.float32)
    return inputs


def test_the_convolution_takes_three_head_major_projections():
    _, _, module, text = _imported(with_values=False)
    heads, dim = text["kda_num_heads"], text["kda_head_dim"]
    kernel = text["kda_conv_kernel_size"]
    nodes = _conv_nodes(module)
    assert nodes, "the KDA region must still lower through nn.kda_convolution"
    for node in nodes:
        types = [module.node_map[value].type for value in node.inputs]
        projections, state, weight = types[:3], types[3], types[4]
        assert all(tuple(d.fixed_value for d in value.shape) == (1, heads, dim)
                   for value in projections)
        assert tuple(d.fixed_value for d in state.shape) == (3, heads, dim, kernel - 1)
        assert tuple(d.fixed_value for d in weight.shape) == (3, heads, dim, kernel)


def test_the_convolution_publishes_q_k_v_and_the_state_separately():
    _, _, module, text = _imported(with_values=False)
    heads, dim = text["kda_num_heads"], text["kda_head_dim"]
    for node in _conv_nodes(module):
        fields = node.type.fields
        assert len(fields) == 4
        assert all(tuple(d.fixed_value for d in field.shape) == (1, heads, dim)
                   for field in fields[:3])
        assert tuple(d.fixed_value for d in fields[3].shape) == (
            3, heads, dim, text["kda_conv_kernel_size"] - 1)


def test_no_flat_channel_reshuffle_survives_between_the_projections_and_the_recurrence():
    """The concat/slice pair is what destroyed the head axis; it must be gone."""

    _, _, module, text = _imported(with_values=False)
    heads, dim = text["kda_num_heads"], text["kda_head_dim"]
    flat = 3 * heads * dim
    for node in _region_nodes(module):
        if node.op not in {"tensors.concat", "tensors.slice"}:
            continue
        shape = tuple(d.fixed_value for d in node.type.shape if d.is_fixed)
        assert flat not in shape, f"{node.op} {node.id} still carries the flat q|k|v channel axis"


def test_the_conv_weight_island_keeps_the_published_tensors_apart():
    """Each part is reshaped in place; only axis 0 (the part axis) is joined."""

    _, _, module, text = _imported(with_values=False)
    heads, dim = text["kda_num_heads"], text["kda_head_dim"]
    kernel = text["kda_conv_kernel_size"]
    joins = [node for node in module.nodes
             if node.op == "tensors.concat" and node.id.endswith("qkv_conv_weight")]
    assert joins, "the constant island must still assemble one conv weight per layer"
    for node in joins:
        assert int(node.attrs["axis"]) == 0
        assert tuple(d.fixed_value for d in node.type.shape) == (3, heads, dim, kernel)
        for value in node.inputs:
            assert tuple(d.fixed_value for d in module.node_map[value].type.shape) == (
                1, heads, dim, kernel)


def test_the_head_major_convolution_matches_the_flat_reference():
    """Bit-for-bit against the flat ``[3*H*D, K]`` arithmetic this replaces."""

    source, importer, module, text = _imported(with_values=True)
    heads, dim = text["kda_num_heads"], text["kda_head_dim"]
    kernel = text["kda_conv_kernel_size"]
    generator = torch.Generator().manual_seed(21)
    inputs = {"input_ids": torch.tensor([7], dtype=torch.int32),
              "paged_attention_state": create_paged_attention_state(importer.paged_config)}
    inputs.update(_states(text, importer))
    layer = importer.kda_layers[0]
    state = torch.randn((3, heads, dim, kernel - 1), generator=generator).bfloat16()
    inputs[f"kda_conv_state_{layer}"] = state.clone()

    # main publishes [logits, token, (conv, recurrent) per KDA layer..., paged].
    outputs = TorchEvaluator(CheckpointWeightResolver(source)).run(module, inputs)
    actual = outputs[2 + 2 * importer.kda_layers.index(layer)]
    assert actual.shape == (3, heads, dim, kernel - 1)

    # The published shift is per (part, head, channel) regardless of layout, so
    # the flat reference is the same tensor viewed as [3*H*D, K-1].
    flat_state = state.reshape(3 * heads * dim, kernel - 1)
    assert torch.equal(actual.reshape(3 * heads * dim, kernel - 1)[:, : kernel - 2],
                       flat_state[:, 1:])


@pytest.mark.parametrize("field", (0, 1, 2))
def test_every_published_projection_keeps_a_plain_broadcast_contract(field):
    """3a is a layout change only: the distribution contract is untouched."""

    _, _, module, _ = _imported(with_values=False)
    for node in _conv_nodes(module):
        assert not isinstance(node.type.fields[field], DistributedType)
