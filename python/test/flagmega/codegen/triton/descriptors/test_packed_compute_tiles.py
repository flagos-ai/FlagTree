# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.targets import NvidiaSm90Target


@pytest.mark.parametrize("n,k", [(2048, 2048), (80, 320), (2056, 528)])
def test_compute_tiles_are_independent_of_packing_atoms_and_support_tail_masks(n, k):
    placement = fm.Placement((2, 4), "yx", "bb")
    broadcast = fm.SBP.broadcast()
    split = fm.SBP.split_block_cyclic((0, 1), 2)
    lhs = fm.Node("lhs", "builtin.var", (), fm.DistributedType(
            fm.tensor_type("bfloat16", (1, k)), (broadcast, broadcast), placement,
        ))
    rhs = fm.Node("rhs", "builtin.var", (), fm.DistributedType(
            fm.tensor_type(fm.vector_type("bfloat16", (8, 2, 8)), (k // 16, n // 8)),
            (broadcast, split), placement,
        ))
    result = fm.Node("result", "ntt.packed_matmul", ("lhs", "rhs"), fm.DistributedType(
            fm.tensor_type(fm.vector_type("bfloat16", (8,)), (1, n // 8)),
            (broadcast, split), placement,
        ), metadata={"selected_vectorization": "vectorization.matmul.n",
                     "selected_vector_axes": (1,), "selected_vector_lanes": (8,)})
    module = fm.IRModule(dialect="distributed", stage="distributed",
                         nodes=(lhs, rhs, result), functions=(), entry="main")
    proposed = NvidiaSm90Target().propose_tir(module)
    point, = proposed.selection_points
    candidates = {candidate.id: candidate for candidate in point.candidates}
    for tile_n, block_k in ((16, 128), (32, 128), (64, 256)):
        candidate = candidates[f"tir.dense_matmul.packed_k_major_gemv_tn{tile_n}_bk{block_k}"]
        assert candidate.parameters["tile_n"] == tile_n
        assert candidate.parameters["block_k"] == block_k
        assert candidate.parameters["packed_layout"] == "k_major_n8_k16"
        assert candidate.facts["portable_triton"] is True
    assert point.default_candidate == "tir.dense_matmul.packed_k_major_gemv_tn64_bk256"
