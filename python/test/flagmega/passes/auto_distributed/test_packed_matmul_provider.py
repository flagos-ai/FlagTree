# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Packed RHS layout propagation, independently of a model/compiler pipeline."""

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.passes.auto_distributed import DistributedCandidateContext, DistributedCandidateProviderRegistry
from triton.flagmega.passes.auto_distributed.policy import NttDistributionPolicy
from triton.flagmega.targets.pyntt_split import PyNttDistributedSplitCandidateProvider


def _context(*, fused_reduce=False, addend=False):
    class Module(fm.Module):
        def forward(self):
            lhs = self.input("lhs", fm.tensor_type("bfloat16", (16, 6144)), id="lhs")
            rhs = self.input("rhs", fm.tensor_type(fm.vector_type("bfloat16", (8, 2, 8)), (384, 256)), id="rhs")
            none = fm.F.builtin.none(name="none")
            residual = self.input("residual", fm.tensor_type(fm.vector_type("bfloat16", (8,)), (16, 256)), id="residual")
            result = fm.F.ntt.packed_matmul(lhs, rhs, none, residual if addend else none,
                                          output_data_type=fm.DType.BFLOAT16, fused_reduce=fused_reduce, name="result")
            self.function("main", (lhs, rhs, residual), (result,))
    module = Module(dialect="ntt", stage="packed", entry="main").build()
    node = module.node_map["result"]
    return DistributedCandidateContext(module, node, fm.Placement((8, 16), "yx", "bb"),
        tuple((module.node_map[i].type,) for i in node.inputs), PyNttDistributedSplitCandidateProvider(128))


def _candidates(context):
    registry = DistributedCandidateProviderRegistry()
    NttDistributionPolicy((context.placement,), context.split_candidate_provider).register_candidate_providers(registry)
    return registry.try_get("ntt.packed_matmul").get_candidates(context)


def test_packed_rhs_n_policy_uses_rhs_vector_bytes_not_result_vector_bytes():
    context = _context()
    policy = fm.SBP.split_block_cyclic((0, 1), 1)
    assert any(c.input_types[1].axis_policies == (fm.SBP.broadcast(), policy)
               and c.return_type.axis_policies == (fm.SBP.broadcast(), policy)
               and c.return_type.partial is None for c in _candidates(context))


def test_available_rhs_n_policy_is_preserved_while_only_k_is_aligned():
    context = _context()
    lhs = fm.DistributedType(context.module.node_map["lhs"].type,
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((0,), 768)), context.placement)
    n_policy = fm.SBP.split_block_cyclic((1,), 3)
    rhs = fm.DistributedType(context.module.node_map["rhs"].type,
        (fm.SBP.broadcast(), n_policy), context.placement)
    context = replace(context, available_input_types=((lhs,), (rhs,), (fm.NoneType(),), (fm.NoneType(),)))
    candidates = _candidates(context)
    assert any(c.input_types[0] == lhs and c.input_types[1].axis_policies == (
        fm.SBP.split_contiguous((0,), 48), n_policy)
        and c.return_type.partial == fm.SBP.partial((0,)) for c in candidates)


@pytest.mark.parametrize("fused_reduce,addend", [(False, False), (False, True), (True, False), (True, True)])
def test_every_relation_agrees_with_op_inference_including_fused_reduce_and_addend(fused_reduce, addend):
    context = _context(fused_reduce=fused_reduce, addend=addend)
    candidates = _candidates(context)
    assert candidates
    for candidate in candidates:
        inputs = tuple(fm.Node(f"arg{i}", "builtin.var", (), t) for i, t in enumerate(candidate.input_types))
        assert fm.get_definition("ntt.packed_matmul").infer_type(inputs, context.source_call.attrs) == candidate.return_type
        if fused_reduce or addend:
            assert candidate.return_type.partial is None


def test_available_lhs_row_policy_survives_rhs_reduction_alignment():
    context = _context()
    lhs = fm.DistributedType(context.module.node_map["lhs"].type,
        (fm.SBP.split_contiguous((0,), 2), fm.SBP.broadcast()), context.placement)
    context = replace(context, available_input_types=((lhs,), *context.available_input_types[1:]))
    assert any(c.input_types[0] == lhs and c.return_type.axis_policies[0] == lhs.axis_policies[0]
               for c in _candidates(context))
