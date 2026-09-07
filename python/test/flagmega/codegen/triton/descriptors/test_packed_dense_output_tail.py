# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Output tails do not invalidate a proved affine TMA source rectangle."""

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.targets import NvidiaSm90Target


IMPLEMENTATION = "tir.dense_matmul.packed_tensor_descriptor_table_smem_pipeline_gemv_tn64_bk512"


def _projection(n, *, k=2048, cyclic=False):
    placement = fm.Placement((2, 4), "yx", "bb")
    broadcast = fm.SBP.broadcast()
    split = fm.SBP.split_block_cyclic((0, 1), 1) if cyclic else fm.SBP.split_contiguous((0, 1), (n // 8 + 7) // 8)
    lhs = fm.Node("lhs", "builtin.var", (), fm.DistributedType(
        fm.tensor_type("bfloat16", (1, k)), (broadcast, broadcast), placement))
    rhs = fm.Node("rhs", "builtin.var", (), fm.DistributedType(
        fm.tensor_type(fm.vector_type("bfloat16", (8, 2, 8)), (k // 16, n // 8)),
        (broadcast, split), placement))
    result = fm.Node("result", "ntt.packed_matmul", (lhs.id, rhs.id), fm.DistributedType(
        fm.tensor_type(fm.vector_type("bfloat16", (8,)), (1, n // 8)),
        (broadcast, split), placement), metadata={
            "selected_vectorization": "vectorization.matmul.n",
            "selected_vector_axes": (1,), "selected_vector_lanes": (8,),
        })
    return fm.IRModule(dialect="distributed", stage="distributed", nodes=(lhs, rhs, result), functions=(), entry="main")


@pytest.mark.parametrize("n,cyclic", [(576, False), (2048, False), (576, True), (1096, True), (2048, True)])
def test_affine_output_tail_has_a_typed_pipeline_candidate(n, cyclic):
    proposed = NvidiaSm90Target().propose_tir(_projection(n, cyclic=cyclic))
    point = next(point for point in proposed.selection_points if point.owner == "result")
    assert IMPLEMENTATION in {candidate.id for candidate in point.candidates}
    if n != 2048:
        assert point.default_candidate == IMPLEMENTATION


def test_partial_k_tile_is_not_silently_accepted_by_output_tail_contract():
    proposed = NvidiaSm90Target().propose_tir(_projection(1096, k=528, cyclic=True))
    point = next(point for point in proposed.selection_points if point.owner == "result")
    assert IMPLEMENTATION not in {candidate.id for candidate in point.candidates}


def test_owner_table_candidate_requires_a_distributed_rhs_abi():
    module = _projection(576)
    module = replace(module, nodes=tuple(replace(node, type=node.type.tensor) for node in module.nodes))
    proposed = NvidiaSm90Target().propose_tir(module)
    point = next(point for point in proposed.selection_points if point.owner == "result")
    assert IMPLEMENTATION not in {candidate.id for candidate in point.candidates}
