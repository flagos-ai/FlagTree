# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import itertools

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes.rewriter import DataflowPass
from triton.flagmega.passes.target_independent import decompose_complex_ops
from triton.flagmega.rules.neutral import decompose_sparse_experts_rule
from python.test.flagmega.sparse_experts.helpers import build_module, values_for


@pytest.mark.parametrize("roundings", list(itertools.product((False, True), repeat=4)))
def test_sparse_expert_decomposition_preserves_every_rounding_contract(roundings):
    attrs = dict(
        zip(("round_projections", "round_activation", "round_down_projection", "round_weighted_output"), roundings))
    module = build_module(attrs=attrs, duplicate_use=True)
    pass_ = DataflowPass("DecomposeSparseExperts", (decompose_sparse_experts_rule(), ))
    rewritten = pass_.run(module)
    assert not any(node.op == "nn.sparse_experts" for node in rewritten.nodes)
    gate, down = rewritten.node_map["experts.gate_up"], rewritten.node_map["experts"]
    assert gate.op == "nn.sparse_experts_gate_up"
    assert down.op == "nn.sparse_experts_down"
    assert down.inputs[0] == gate.id
    assert gate.attrs["round_projections"] == roundings[0]
    assert gate.attrs["round_activation"] == roundings[1]
    assert down.attrs["round_projection"] == roundings[2]
    assert down.attrs["round_weighted_output"] == roundings[3]
    assert gate.metadata["test_tag"] == down.metadata["test_tag"] == "kept"
    assert rewritten.node_map["output"].inputs == (down.id, down.id)
    assert pass_.run(rewritten) == rewritten
    evaluator = TorchEvaluator(DictWeightResolver({}))
    inputs = values_for(module)
    for actual, expected in zip(evaluator.run(rewritten, inputs), evaluator.run(module, inputs)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_sparse_experts_is_decomposed_by_the_real_target_independent_pipeline():
    module = decompose_complex_ops(build_module())
    assert module.node_map["experts"].op == "nn.sparse_experts_down"
    assert module.node_map["experts.gate_up"].op == "nn.sparse_experts_gate_up"


@pytest.mark.parametrize("decomposed", [False, True])
def test_sparse_experts_edit_resume_preserves_vector_dtypes_and_rounding(tmp_path, decomposed):
    module = build_module(attrs={
        "intermediate_dtype": fm.vector_type("bfloat16", (2,
                                                          2)), "round_projections": True, "round_weighted_output": True
    })
    if decomposed:
        module = decompose_complex_ops(module)
    path = fm.emit_module(module, tmp_path / "experts.py")
    assert "F.nn.sparse_experts" in path.read_text()
    assert "fm.vector_type(" in path.read_text()
    assert "'kind': 'vector'" not in path.read_text()
    restored = fm.load_module(path)
    assert restored == module
    values = values_for(module)
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(evaluator.run(restored, values)[0], evaluator.run(module, values)[0], rtol=0, atol=0)
