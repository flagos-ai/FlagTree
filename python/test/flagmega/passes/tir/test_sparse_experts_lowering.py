# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace
from itertools import product

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.ir.ops.nn.sparse_experts_combine import SparseExpertsCombine, SparseExpertsWeightedSum
from triton.flagmega.passes.target_independent import decompose_complex_ops
from triton.flagmega.passes.tir.lower_sparse_experts import lower_sparse_experts
from python.test.flagmega.sparse_experts.helpers import build_module, operand_types, operands, values_for


@pytest.mark.parametrize("roundings", tuple(product((False, True), repeat=4)))
def test_private_stage_fusion_preserves_all_numerical_boundaries(roundings, tmp_path):
    attrs = dict(zip(("round_projections", "round_activation", "round_down_projection", "round_weighted_output"), roundings))
    module = build_module(attrs=attrs)
    lowered = lower_sparse_experts(decompose_complex_ops(module))
    assert lowered.node_map["experts.gate_up"].op == "ntt.dispatched_experts_gate_up"
    assert lowered.node_map["experts"].op == "ntt.sparse_experts_down_combine"
    assert not any(n.op in {"nn.sparse_experts_dispatch", "nn.sparse_experts_down", "tensors.cast"} for n in lowered.nodes)
    evaluator = TorchEvaluator(DictWeightResolver({}))
    values = values_for(module)
    torch.testing.assert_close(evaluator.run(lowered, values)[0], evaluator.run(module, values)[0], rtol=0, atol=0)
    assert fm.load_module(fm.emit_module(lowered, tmp_path / "lowered.py")) == lowered
    assert lower_sparse_experts(lowered) == lowered


def combine_types(*, partial=True):
    b = fm.SBP.broadcast()
    route = fm.SBP.split_block_cyclic((0,), 2)
    hidden = fm.SBP.split_contiguous((2,))
    mesh = fm.Placement((2, 2, 2), "xyz", "bbb")
    return {
        "projections": fm.DistributedType(fm.tensor_type("float32", (2, 5, 32)), (b, route, hidden), mesh,
                                          fm.SBP.partial((1,)) if partial else None),
        "router_expert_weights": fm.DistributedType(fm.tensor_type("float32", (2, 5)), (b, route), mesh),
    }


@pytest.mark.parametrize("packed", [False, True])
def test_route_and_k_reductions_are_explicit_and_precede_final_cast(packed):
    dtype = fm.vector_type("bfloat16", (2, 4)) if packed else fm.DType.BFLOAT16
    module = build_module(SparseExpertsCombine, types=combine_types(), attrs={"output_dtype": dtype})
    lowered = lower_sparse_experts(module)
    local = lowered.node_map["experts.weighted_sum"]
    assert local.type.partial == fm.SBP.partial((0, 1))
    assert local.type.tensor.dtype == (fm.vector_type("float32", (2, 4)) if packed else fm.DType.FLOAT32)
    boxing = lowered.node_map["experts.owner_sum"]
    assert boxing.op == "distributed.boxing" and boxing.inputs == (local.id,)
    assert boxing.type.partial is None
    assert lowered.node_map["experts"].inputs == (boxing.id,)
    assert lowered.node_map["experts"].op == ("ntt.vectorized_cast" if packed else "tensors.cast")
    inputs = operands(SparseExpertsCombine, combine_types())
    call = SparseExpertsCombine.prepare(inputs, {"output_dtype": dtype})
    assert SparseExpertsCombine.cost_factors(inputs, call.attrs, call.result_type).grid_synchronizations == 1
    assert SparseExpertsWeightedSum.cost_factors(inputs, call.attrs, local.type).grid_synchronizations == 0


def test_weighted_rounding_rejects_partial_k_but_allows_route_sharding():
    with pytest.raises(IRSchemaError, match="materialized"):
        SparseExpertsCombine.prepare(operands(SparseExpertsCombine, combine_types()), {"round_weighted_output": True})
    inputs = operands(SparseExpertsCombine, combine_types(partial=False))
    local = SparseExpertsWeightedSum.prepare(inputs, {"output_dtype": "bfloat16", "round_weighted_output": True})
    assert local.result_type.partial == fm.SBP.partial((0,))


@pytest.mark.parametrize("shared", ["experts.dispatch", "experts.down"])
def test_exported_intermediates_are_not_absorbed(shared):
    module = decompose_complex_ops(build_module())
    fn = module.functions[0]
    module = replace(module, functions=(replace(fn, outputs=(*fn.outputs, shared)),))
    lowered = lower_sparse_experts(module)
    assert lowered.node_map[shared] == module.node_map[shared]
    fm.verify_module(lowered)
