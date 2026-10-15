# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Each mHC stage is imported as one fused operation, not as graph algebra.

Hyper-connections add two stages per decoder block.  Expressed as graph algebra
the Sinkhorn normalization costs two reductions and two divisions per iteration,
so the published twenty iterations become hundreds of nodes whose numerical
contract is spread across all of them, and the post stage costs ``hc_mult ** 2``
hidden-sized multiplies plus a concat.  ``nn.mhc_pre``/``nn.mhc_post`` keep both
contracts in one node each, at any iteration count.
"""

import pytest
import torch

from triton.flagmega.evaluator import CheckpointWeightResolver, TorchEvaluator
from triton.flagmega.importer.glm5_moe import Glm5MoeModelImporter
from triton.flagmega.ir.ops.nn._paged_attention_state import create_paged_attention_state

from python.test.flagmega.importer.glm5_moe.helpers import REDUCED, checkpoint, configuration

# The fixture instantiates one function per decoder kind (dense linear
# attention, sparse linear attention, sparse MLA), each with an attention stage
# and an MLP stage.
DECODER_KINDS, STAGES_PER_BLOCK = 3, 2
STAGES = DECODER_KINDS * STAGES_PER_BLOCK


def _mhc_scale(**overrides):
    return {**REDUCED, "quantized": True, "fp8_block_size": (4, 4), "mhc": True,
            "hc_mult": 4, "hc_sinkhorn_iters": 20, **overrides}


def _module(scale):
    source = checkpoint(scale, with_values=False, full_model=True, expert_layout="separate")
    return Glm5MoeModelImporter(source, block_size=8, num_blocks=3).import_module()


@pytest.mark.parametrize("iterations", [2, 20])
def test_stage_cost_is_independent_of_the_sinkhorn_iteration_count(iterations):
    module = _module(_mhc_scale(hc_sinkhorn_iters=iterations))
    counts = {op: sum(1 for node in module.nodes if node.op == op)
              for op in ("nn.mhc_pre", "nn.mhc_post")}
    assert counts == {"nn.mhc_pre": STAGES, "nn.mhc_post": STAGES}
    # No per-iteration Sinkhorn residue and no per-stream-pair mixing residue.
    assert not [node.id for node in module.nodes
                if "comb_" in node.id or "_mix_" in node.id or "_sinkhorn" in node.id]


def test_stages_carry_the_reference_hyperparameters_as_attributes():
    scale = _mhc_scale()
    module = _module(scale)
    text = configuration(scale)["text_config"]
    stages = [node for node in module.nodes if node.op == "nn.mhc_pre"]
    assert len(stages) == STAGES
    for node in stages:
        assert node.attrs["sinkhorn_iterations"] == text["hc_sinkhorn_iters"]
        assert node.attrs["sinkhorn_epsilon"] == pytest.approx(text["hc_eps"])
        assert node.attrs["epsilon"] == pytest.approx(text["rms_norm_eps"])
        # vLLM's glm5next passes hc_post_mult_value straight to the kernel.
        assert node.attrs["post_multiplier"] == pytest.approx(2.0)


def test_post_consumes_the_coefficients_the_matching_pre_produced():
    module = _module(_mhc_scale())
    pre_stages = {node.id: node for node in module.nodes if node.op == "nn.mhc_pre"}
    post_stages = [node for node in module.nodes if node.op == "nn.mhc_post"]
    assert len(pre_stages) == len(post_stages) == STAGES
    for post in post_stages:
        # inputs are (value, residual, post, comb); the last two unpack one pre.
        sources = {module.node_map[module.node_map[post.inputs[index]].inputs[0]].id
                   for index in (2, 3)}
        assert len(sources) == 1 and sources.pop() in pre_stages


def _coefficients(iterations):
    """Evaluate the fixture and return every traced mHC mixing matrix."""
    scale = _mhc_scale(hc_sinkhorn_iters=iterations)
    source = checkpoint(scale, full_model=True, expert_layout="separate")
    text = configuration(scale)["text_config"]
    importer = Glm5MoeModelImporter(source, block_size=8, num_blocks=3)
    module = importer.import_module()
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
    combs = {node.id: trace[module.node_map[node.inputs[3]].id]
             for node in module.nodes if node.op == "nn.mhc_post"}
    assert combs, "no mHC post stage was traced"
    return combs


def test_the_sinkhorn_iteration_count_reaches_the_evaluated_coefficients():
    """The attribute must drive the numerics, not merely be carried along.

    Asserted on ``comb`` rather than on the logits: at the reduced fixture scale
    the bfloat16 rounding downstream absorbs the difference entirely, so equal
    logits would say nothing about whether the iteration count was honoured.
    """
    few, many = _coefficients(2), _coefficients(20)
    assert set(few) == set(many)
    for name, matrix in few.items():
        assert not torch.equal(matrix, many[name]), f"{name} ignored the iteration count"
    # More iterations tighten the row direction; the column direction is
    # normalized last and is already tight after the first pass.
    def row_error(combs):
        return max((matrix.sum(dim=-1) - 1.0).abs().max().item() for matrix in combs.values())

    assert row_error(many) < row_error(few)
