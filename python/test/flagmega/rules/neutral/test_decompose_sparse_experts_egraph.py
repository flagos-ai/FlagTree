# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes.rewriter import EGraphRulesPass
from triton.flagmega.rules.neutral import decompose_sparse_experts_rule
from python.test.flagmega.sparse_experts.helpers import build_module, values_for


def test_egraph_can_extract_sparse_expert_stages_with_shared_outputs():
    module = build_module(attrs={"round_projections": True, "round_weighted_output": True}, duplicate_use=True)
    rewritten = EGraphRulesPass(
        "DecomposeSparseExperts",
        (decompose_sparse_experts_rule(), ),
        cost=lambda node, _module: 100.0 if node.op == "nn.sparse_experts" else 1.0,
        cost_model="unit-test/force-stage-extraction",
    ).run(module)
    fm.verify_module(rewritten)
    assert not any(node.op == "nn.sparse_experts" for node in rewritten.nodes)
    assert sum(node.op == "nn.sparse_experts_gate_up" for node in rewritten.nodes) == 1
    assert sum(node.op == "nn.sparse_experts_down" for node in rewritten.nodes) == 1
    evaluator = TorchEvaluator(DictWeightResolver({}))
    inputs = values_for(module)
    for actual, expected in zip(evaluator.run(rewritten, inputs), evaluator.run(module, inputs)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
