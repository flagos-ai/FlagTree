# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Preserve the pinned profile while removing RoPE representation casts."""

import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import CheckpointWeightResolver, TorchEvaluator, create_paged_attention_state
from triton.flagmega.importer import Qwen35MoeImporter, apply_numerical_profile
from triton.flagmega.importer.numerics import VLLM_AE10_INDUCTOR_LEVEL3
from triton.flagmega.ir.ops.nn._gdn_state import create_gdn_state
from triton.flagmega.passes import DataflowPass
from triton.flagmega.rules import RewriteResult, RewriteRule
from triton.flagmega.rules.neutral._utility import make_node
from python.test.flagmega.importer.qwen3_5_moe.helpers import checkpoint


def _explicit_fp32_rope(module):

    def rewrite(node, graph):
        operands = tuple(graph.node_map[name] for name in node.inputs)
        wide = tuple(
            make_node("tensors.cast", f"{node.id}.reference.arg{i}", (value, ), {"dtype": "float32"}, {})
            for i, value in enumerate(operands))
        rope = make_node(node.op, f"{node.id}.reference.rope", wide, node.attrs, {})
        result = make_node("tensors.cast", node.id, (rope, ), {"dtype": "bfloat16"}, node.metadata)
        return RewriteResult(result, (*wide, rope))

    return DataflowPass(
        "ExplicitRoPEReference",
        (RewriteRule("ExpandRoPECasts", lambda n, _: n.op == "nn.rope" and n.type.dtype == fm.DType.BFLOAT16,
                     rewrite), )).run(module)


def test_native_rope_matches_explicit_profile_casts_across_decode_steps():
    source = checkpoint(with_values=True)
    importer = Qwen35MoeImporter(source, block_size=2, num_blocks=4)
    module = apply_numerical_profile(importer.import_module(), VLLM_AE10_INDUCTOR_LEVEL3)
    reference = _explicit_fp32_rope(module)
    fm.verify_module(reference)
    evaluator = TorchEvaluator(CheckpointWeightResolver(source))
    states = [(create_gdn_state(importer.gdn_config), create_paged_attention_state(importer.paged_config))
              for _ in range(2)]
    token = 7
    for _ in range(6):
        results = [
            evaluator.run(
                graph, {
                    "input_ids": torch.tensor([token], dtype=torch.int32), "gated_delta_net_state": gdn,
                    "paged_attention_state": paged
                }) for graph, (gdn, paged) in zip((module, reference), states)
        ]
        torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
        assert results[0][1].item() == results[1][1].item()
        assert torch.equal(states[0][1].kv_caches, states[1][1].kv_caches)
        token = results[0][1].item()
