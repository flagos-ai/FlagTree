# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Dual TMA output tails require an explicit mask capability, not full tiles."""

from dataclasses import replace
import pytest
from triton.flagmega import ir as fm
from triton.flagmega.targets import NvidiaSm90Target
from triton.flagmega.targets.nvidia import sm90_triton_implementation_model


SINGLE = "tir.dense_matmul_glu.packed_tensor_descriptor_paired_smem_pipeline_inline_gemv"
TABLE = "tir.dense_matmul_glu.packed_tensor_descriptor_table_paired_smem_pipeline_inline_gemv"


def _projection(local_n, k=2048, *, cyclic=False, plain=False):
    placement = fm.Placement((2, 4), "yx", "bb")
    broadcast = fm.SBP.broadcast()
    split = fm.SBP.split_block_cyclic((0, 1), 1) if cyclic else fm.SBP.split_contiguous((0, 1), local_n // 8)

    def distributed(value, policies):
        return value if plain else fm.DistributedType(value, policies, placement)

    lhs = fm.Node("lhs", "builtin.var", (), distributed(fm.tensor_type("bfloat16", (1, k)), (broadcast, broadcast)))
    weight = distributed(fm.tensor_type(fm.vector_type("bfloat16", (8, 2, 8)), (k // 16, local_n)), (broadcast, split))
    gate, up = (fm.Node(name, "builtin.var", (), weight) for name in ("gate", "up"))
    result = fm.Node("result", "nn.packed_dense_matmul_glu", ("lhs", "gate", "up"),
        distributed(fm.tensor_type(fm.vector_type("bfloat16", (8,)), (1, local_n)), (broadcast, split)),
        attrs={"activation": "silu", "packed_layout": "k_major_n8_k16"}, metadata={
            "selected_vectorization": "vectorization.dense_matmul_glu.n",
            "selected_vector_axes": (1,), "selected_vector_lanes": (8,),
        })
    return fm.IRModule(dialect="distributed", stage="distributed", nodes=(lhs, gate, up, result), functions=(), entry="main")


def _candidates(module, model=None):
    proposed = NvidiaSm90Target(triton_implementation_model=model).propose_tir(module)
    return {value.id for point in proposed.selection_points if point.owner == "result" for value in point.candidates}


@pytest.mark.parametrize("local_n", (8, 24, 40, 64))
def test_tma_tail_candidates_preserve_local_owner_extent(local_n):
    assert {SINGLE, TABLE} <= _candidates(_projection(local_n))


def test_mask_capability_is_required_for_partial_output_tile():
    model = sm90_triton_implementation_model()
    model = replace(model, implementations=tuple(replace(value, contract={**value.contract, "supports_masked_output_tiles": False})
        if value.id in {SINGLE, TABLE} else value for value in model.implementations))
    assert not {SINGLE, TABLE} & _candidates(_projection(24), model)


@pytest.mark.parametrize("k,cyclic", ((1536, False), (2048, True)))
def test_output_mask_does_not_relax_k_or_contiguous_owner_contract(k, cyclic):
    assert not {SINGLE, TABLE} & _candidates(_projection(24, k, cyclic=cyclic))


def test_owner_table_requires_distributed_weight_abi():
    candidates = _candidates(_projection(64, plain=True))
    assert SINGLE in candidates and TABLE not in candidates
