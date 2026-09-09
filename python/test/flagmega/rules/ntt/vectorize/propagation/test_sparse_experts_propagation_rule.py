# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.ir.ops.nn.sparse_experts_down import SparseExpertsDown
from triton.flagmega.passes.rewriter import DataflowPass
from triton.flagmega.passes.auto_distributed.policy import lower_vectorization_contracts
from triton.flagmega.rules.ntt.vectorize.propagation import propagation_rules, sparse_experts_propagation_rules
from python.test.flagmega.sparse_experts.helpers import operand_types, values_for


def make_module(lanes, axes, *, shared=False, already_vector=False, generated=False):

    class Module(fm.Module):

        def forward(self):
            types = operand_types()
            inputs = tuple(
                self.input(parameter.name, types[parameter.name]) for parameter in SparseExpertsDown.input_parameters)
            down = fm.F.nn.sparse_experts_down(*inputs, round_weighted_output=True,
                                               output_dtype=fm.vector_type("bfloat16", 2) if already_vector else None,
                                               name="down")
            pack = fm.F.tensors.pack(
                down, lanes=lanes, axes=axes, name="pack",
                metadata={"vectorization_internal": True, "vectorization_root": "pack"} if generated else None)
            self.function("main", inputs, (pack, down) if shared else (pack, ))

    return Module(dialect="nn", stage="typed", entry="main").build()


@pytest.mark.parametrize("lanes,axes", [((2, ), (1, )), ((2, 2), (1, 1)), ((2, 4), (-1, -1))])
@pytest.mark.parametrize("shared", [False, True])
def test_down_output_pack_absorption_preserves_rounding_and_other_users(lanes, axes, shared):
    module = make_module(lanes, axes, shared=shared)
    rewritten = DataflowPass("VectorizeSparseExperts", sparse_experts_propagation_rules()).run(module)
    result = rewritten.node_map["pack"]
    assert result.op == "nn.sparse_experts_down"
    assert result.type == module.node_map["pack"].type
    assert result.attrs["round_weighted_output"] is True
    assert result.inputs == module.node_map["down"].inputs
    if shared:
        assert rewritten.node_map["down"].type == module.node_map["down"].type
    evaluator = TorchEvaluator(DictWeightResolver({}))
    inputs = values_for(module)
    for actual, expected in zip(evaluator.run(rewritten, inputs), evaluator.run(module, inputs)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_down_output_pack_does_not_absorb_token_axis():
    module = make_module((2, ), (0, ))
    assert sparse_experts_propagation_rules()[0].apply(module.node_map["pack"], module) is None


def test_down_output_pack_does_not_revectorize_an_existing_vector_result():
    module = make_module((2, ), (1, ), already_vector=True)
    assert sparse_experts_propagation_rules()[0].apply(module.node_map["pack"], module) is None


def test_sparse_expert_rule_is_in_the_auto_vectorize_propagation_set():
    assert "VectorizeSparseExpertsPropagation" in {rule.name for rule in propagation_rules()}


@pytest.mark.parametrize("generated", [False, True])
def test_lower_vectorization_contracts_keeps_sparse_expert_physical_output(generated):
    module = make_module((2, 2), (1, 1), generated=generated)
    packed = DataflowPass("VectorizeSparseExperts", sparse_experts_propagation_rules()).run(module)
    lowered = lower_vectorization_contracts(packed)
    fm.verify_module(lowered)
    assert lowered.node_map["pack"].op == "nn.sparse_experts_down"
    assert lowered.node_map["pack"].type == packed.node_map["pack"].type
