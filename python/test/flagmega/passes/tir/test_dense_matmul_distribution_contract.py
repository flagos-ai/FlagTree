# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.distribution import (
    dense_matmul_distribution_contract,
)
from triton.flagmega.errors import CodegenError
from triton.flagmega.targets import NvidiaSm90Target


_PLACEMENT = fm.Placement((8, 16), "yx", "bb")
_BROADCAST = fm.SBP.broadcast()


def _distributed_matmul(
    *,
    partial: fm.SBPPartial | None = fm.SBP.partial((0, 1)),
    lhs_policies=None,
    rhs_policies=None,
    output_policies=None,
    op: str = "math.matmul",
) -> tuple[fm.IRModule, fm.Node]:
    lhs_tensor = fm.tensor_type("bfloat16", (128, 6144))
    rhs_tensor = fm.tensor_type("bfloat16", (6144, 2048))
    output_tensor = fm.tensor_type("bfloat16", (128, 2048))
    lhs_type = fm.DistributedType(
        lhs_tensor,
        tuple(lhs_policies or (_BROADCAST, fm.SBP.split_contiguous((0, 1)))),
        _PLACEMENT,
    )
    rhs_type = fm.DistributedType(
        rhs_tensor,
        tuple(rhs_policies or (fm.SBP.split_contiguous((0, 1)), _BROADCAST)),
        _PLACEMENT,
    )
    output_type = fm.DistributedType(
        output_tensor,
        tuple(output_policies or (_BROADCAST, _BROADCAST)),
        _PLACEMENT,
        partial,
    )
    lhs = fm.Node("lhs", "builtin.var", (), lhs_type)
    rhs = fm.Node("rhs", "builtin.var", (), rhs_type)
    result = fm.Node(
        "result",
        op,
        (lhs.id, rhs.id),
        output_type,
        metadata={
            "selected_vectorization": "vectorization.matmul.n",
            "selected_vector_axes": (1,),
            "selected_vector_lanes": (8,),
        },
    )
    return fm.IRModule(
        dialect="distributed",
        stage="distributed",
        nodes=(lhs, rhs, result),
        functions=(),
        entry="main",
    ), result


def test_dense_matmul_contract_recognizes_exact_nncase_split_k_form():
    module, result = _distributed_matmul()

    assert dense_matmul_distribution_contract(result, module) == {
        "kind": "reduction_split",
        "partial_axes": (0, 1),
        "owner_count": 128,
        "lhs_k_axis": 1,
        "rhs_k_axis": 0,
        "output_axes": (),
    }


def test_dense_matmul_contract_accepts_block_cyclic_local_k_shards():
    """The microkernel consumes dense local K; SBP mapping is an ABI concern."""

    reduction = fm.SBP.split_block_cyclic((0, 1), 16)
    module, result = _distributed_matmul(
        lhs_policies=(_BROADCAST, reduction),
        rhs_policies=(reduction, _BROADCAST),
    )

    assert dense_matmul_distribution_contract(result, module) == {
        "kind": "reduction_split",
        "partial_axes": (0, 1),
        "owner_count": 128,
        "lhs_k_axis": 1,
        "rhs_k_axis": 0,
        "output_axes": (),
    }


def test_dense_matmul_contract_accepts_disjoint_local_m_and_k_shards():
    m_split = fm.SBP.split_contiguous((0,))
    k_split = fm.SBP.split_block_cyclic((1,), 16)
    module, result = _distributed_matmul(
        partial=fm.SBP.partial((1,)),
        lhs_policies=(m_split, k_split),
        rhs_policies=(k_split, _BROADCAST),
        output_policies=(m_split, _BROADCAST),
    )

    assert dense_matmul_distribution_contract(result, module) == {
        "kind": "output_reduction_split",
        "partial_axes": (1,),
        "owner_count": 16,
        "lhs_k_axis": 1,
        "rhs_k_axis": 0,
        "output_axes": (0,),
    }


@pytest.mark.parametrize(
    ("updates", "message"),
    (
        ({"partial": fm.SBP.partial((0, 1), fm.ReduceOp.MAX)}, "Sum partial"),
        ({"op": "math.packed_dense_matmul"}, "unsupported"),
        (
            {
                "output_policies": (
                    _BROADCAST,
                    fm.SBP.split_contiguous((0, 1)),
                )
            },
            "same placement axes",
        ),
        (
            {
                "rhs_policies": (
                    fm.SBP.split_contiguous((1, 0)),
                    _BROADCAST,
                )
            },
            "reduction axis must be",
        ),
        (
            {
                "partial": fm.SBP.partial((1,)),
                "lhs_policies": (
                    fm.SBP.split_contiguous((0,)),
                    fm.SBP.split_contiguous((1,)),
                ),
                "rhs_policies": (
                    fm.SBP.split_contiguous((1,)),
                    _BROADCAST,
                ),
            },
            "output-M and lhs-M splits",
        ),
    ),
)
def test_dense_matmul_contract_rejects_unimplemented_partial_forms(updates, message):
    module, result = _distributed_matmul(**updates)

    with pytest.raises(CodegenError, match=message):
        dense_matmul_distribution_contract(result, module)


def test_tir_provider_only_exposes_executable_split_k_implementation_for_partial():
    module, _ = _distributed_matmul()

    proposed = NvidiaSm90Target().propose_tir(module)
    point = next(value for value in proposed.selection_points if value.id == "tir.result")

    assert point.default_candidate == "tir.dense_matmul.split_k_gemv"
    assert tuple(value.id for value in point.candidates) == (
        "tir.dense_matmul.split_k_gemv",
    )
    schedule = point.candidates[0].parameters["distribution_schedule"]
    assert schedule["kind"] == "reduction_split"
    assert schedule["partial_axes"] == (0, 1)
    assert schedule["owner_count"] == 128


def test_packed_dense_provider_exposes_disjoint_output_and_reduction_split():
    """A target implementation, not a graph rule, makes hybrid K/N executable."""

    lhs_tensor = fm.tensor_type("bfloat16", (128, 6144))
    packed_weight = fm.tensor_type("bfloat16", (384, 256, 2, 64))
    output_tensor = fm.tensor_type("bfloat16", (128, 2048))
    lhs_type = fm.DistributedType(
        lhs_tensor,
        (_BROADCAST, fm.SBP.split_contiguous((0,), 768)),
        _PLACEMENT,
    )
    weight_type = fm.DistributedType(
        packed_weight,
        (
            fm.SBP.split_contiguous((0,), 48),
            fm.SBP.split_contiguous((1,), 16),
            _BROADCAST,
            _BROADCAST,
        ),
        _PLACEMENT,
    )
    output_type = fm.DistributedType(
        output_tensor,
        (_BROADCAST, fm.SBP.split_contiguous((1,), 128)),
        _PLACEMENT,
        fm.SBP.partial((0,)),
    )
    lhs = fm.Node("lhs", "builtin.var", (), lhs_type)
    weight = fm.Node("weight", "builtin.var", (), weight_type)
    result = fm.Node(
        "result",
        "math.packed_dense_matmul",
        (lhs.id, weight.id),
        output_type,
        attrs={"packed_layout": "k_major_n8_k16", "logical_n": None},
        metadata={
            "selected_vectorization": "vectorization.matmul.n",
            "selected_vector_axes": (1,),
            "selected_vector_lanes": (8,),
        },
    )
    module = fm.IRModule(
        dialect="distributed",
        stage="distributed",
        nodes=(lhs, weight, result),
        functions=(),
        entry="main",
    )

    proposed = NvidiaSm90Target().propose_tir(module)
    point = next(value for value in proposed.selection_points if value.id == "tir.result")

    assert point.default_candidate == (
        "tir.dense_matmul.split_k_n_packed_k_major_gemv"
    )
    assert tuple(value.id for value in point.candidates) == (
        "tir.dense_matmul.split_k_n_packed_k_major_gemv",
    )
    schedule = point.candidates[0].parameters["distribution_schedule"]
    assert schedule == {
        "kind": "output_reduction_split",
        "partial_axes": (0,),
        "output_axes": (1,),
        "owner_count": 8,
        "lhs_k_axis": 1,
        "rhs_k_axis": 0,
    }


def test_non_distributed_matmul_excludes_split_k_implementation():
    tensor_lhs = fm.tensor_type("bfloat16", (1, 6144))
    tensor_rhs = fm.tensor_type("bfloat16", (6144, 2048))
    tensor_output = fm.tensor_type("bfloat16", (1, 2048))
    lhs = fm.Node("lhs", "builtin.var", (), tensor_lhs)
    rhs = fm.Node("rhs", "builtin.var", (), tensor_rhs)
    result = fm.Node(
        "result",
        "math.matmul",
        (lhs.id, rhs.id),
        tensor_output,
        metadata={
            "selected_vectorization": "vectorization.matmul.n",
            "selected_vector_axes": (1,),
            "selected_vector_lanes": (8,),
        },
    )
    module = fm.IRModule(
        dialect="distributed",
        stage="distributed",
        nodes=(lhs, rhs, result),
        functions=(),
        entry="main",
    )

    proposed = NvidiaSm90Target().propose_tir(module)
    point = next(value for value in proposed.selection_points if value.id == "tir.result")

    assert "tir.dense_matmul.split_k_gemv" not in {
        value.id for value in point.candidates
    }
    assert {value.parameters["distribution_schedule"]["kind"] for value in point.candidates} == {
        "canonical"
    }
