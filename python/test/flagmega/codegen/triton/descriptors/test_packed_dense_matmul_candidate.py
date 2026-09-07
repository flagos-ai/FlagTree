# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.targets import NvidiaSm90Target


def _packed_projection() -> fm.IRModule:
    lhs = fm.Node(
        "lhs", "builtin.var", (), fm.tensor_type("bfloat16", (1, 2048))
    )
    rhs = fm.Node(
        "rhs",
        "builtin.var",
        (),
        fm.tensor_type(fm.vector_type("bfloat16", (8, 2, 8)), (128, 256)),
    )
    result = fm.Node(
        "result",
        "ntt.packed_matmul",
        (lhs.id, rhs.id),
        fm.tensor_type(fm.vector_type("bfloat16", (8,)), (1, 256)),
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
    )


def _hybrid_split_packed_projection() -> fm.IRModule:
    placement = fm.Placement((8, 16), "yx", "bb")
    lhs = fm.Node(
        "lhs",
        "builtin.var",
        (),
        fm.DistributedType(
            fm.tensor_type("bfloat16", (1, 6144)),
            (fm.SBP.broadcast(), fm.SBP.split_contiguous((1,), 384)),
            placement,
        ),
    )
    rhs = fm.Node(
        "rhs",
        "builtin.var",
        (),
        fm.DistributedType(
            fm.tensor_type(fm.vector_type("bfloat16", (8, 2, 8)), (384, 256)),
            (
                fm.SBP.split_contiguous((1,), 24),
                fm.SBP.split_contiguous((0,), 32),
            ),
            placement,
        ),
    )
    result = fm.Node(
        "result",
        "ntt.packed_matmul",
        (lhs.id, rhs.id),
        fm.DistributedType(
            fm.tensor_type(fm.vector_type("bfloat16", (8,)), (1, 256)),
            (fm.SBP.broadcast(), fm.SBP.split_contiguous((0,), 32)),
            placement,
            fm.SBP.partial((1,)),
        ),
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
    )


def test_packed_descriptor_pipeline_is_default_for_its_proven_local_contract():
    proposed = NvidiaSm90Target().propose_tir(_packed_projection())
    point = next(value for value in proposed.selection_points if value.id == "tir.result")

    assert point.default_candidate == (
        "tir.dense_matmul.packed_tensor_descriptor_smem_pipeline_gemv"
    )
    candidate = next(
        value for value in point.candidates
        if value.id
        == "tir.dense_matmul.packed_tensor_descriptor_smem_pipeline_gemv"
    )
    assert candidate.parameters["packed_layout"] == "k_major_n8_k16"
    assert candidate.parameters["block_k"] == 1024
    assert candidate.parameters["tile_n"] == 16
    assert candidate.parameters["reduction_group"] == 32
    assert candidate.parameters["consumer_warps"] == 8
    assert candidate.parameters["worker_width"] == 32
    assert candidate.facts["requires"] == ("tma", "warp_specialize")
    assert candidate.facts["host_tensor_descriptor"] is True
    assert candidate.facts["transfer_pipeline"] is True


def test_packed_descriptor_pipeline_supports_disjoint_output_and_k_splits():
    proposed = NvidiaSm90Target().propose_tir(_hybrid_split_packed_projection())
    point = next(value for value in proposed.selection_points if value.id == "tir.result")

    assert point.default_candidate == (
        "tir.dense_matmul.split_k_n_packed_tensor_descriptor_smem_pipeline_gemv"
    )
    candidate = next(
        value
        for value in point.candidates
        if value.id
        == "tir.dense_matmul.split_k_n_packed_tensor_descriptor_smem_pipeline_gemv"
    )
    assert candidate.parameters["block_k"] == 128
    assert candidate.parameters["tile_n"] == 64
    assert candidate.parameters["num_stages"] == 4
    assert "inline_consumer_stage" not in candidate.parameters
    assert candidate.parameters["distribution_schedule"] == {
        "kind": "output_reduction_split",
        "partial_axes": (1,),
        "output_axes": (0,),
        "owner_count": 16,
        "lhs_k_axis": 1,
        "rhs_k_axis": 0,
    }
    assert candidate.facts["requires"] == ("tma", "warp_specialize")
    assert candidate.facts["host_tensor_descriptor"] is True
    assert candidate.facts["transfer_pipeline"] is True
    assert candidate.facts["chip_visible_partial_owners"] is True


def test_packed_descriptor_table_pipeline_is_an_explicit_hybrid_candidate():
    proposed = NvidiaSm90Target().propose_tir(_hybrid_split_packed_projection())
    point = next(value for value in proposed.selection_points if value.id == "tir.result")
    implementation = (
        "tir.dense_matmul."
        "split_k_n_packed_tensor_descriptor_table_smem_pipeline_gemv"
    )

    candidate = next(
        value for value in point.candidates if value.id == implementation
    )
    assert candidate.parameters["descriptor_kind"] == "table"
    assert candidate.parameters["distribution_schedule"] == {
        "kind": "output_reduction_split",
        "partial_axes": (1,),
        "output_axes": (0,),
        "owner_count": 16,
        "lhs_k_axis": 1,
        "rhs_k_axis": 0,
    }
    assert candidate.facts["host_tensor_descriptor"] is True
    assert candidate.facts["host_tensor_descriptor_table"] is True
    assert point.default_candidate != implementation


def test_packed_descriptor_pipeline_requires_complete_n_and_k_tiles():
    module = _packed_projection()
    rhs = module.node_map["rhs"]
    result = module.node_map["result"]
    incompatible = fm.IRModule(
        dialect=module.dialect,
        stage=module.stage,
        nodes=(
            module.node_map["lhs"],
            fm.Node(
                rhs.id,
                rhs.op,
                rhs.inputs,
                fm.tensor_type(
                    fm.vector_type("bfloat16", (8, 2, 8)), (128, 255)
                ),
            ),
            fm.Node(
                result.id,
                result.op,
                result.inputs,
                fm.tensor_type(fm.vector_type("bfloat16", (8,)), (1, 255)),
                metadata=result.metadata,
            ),
        ),
        functions=(),
        entry="main",
    )

    proposed = NvidiaSm90Target().propose_tir(incompatible)
    point = next(value for value in proposed.selection_points if value.id == "tir.result")

    assert "tir.dense_matmul.packed_tensor_descriptor_smem_pipeline_gemv" not in {
        value.id for value in point.candidates
    }


def test_packed_descriptor_pipeline_rejects_block_cyclic_local_tiles():
    module = _packed_projection()
    placement = fm.Placement((2,), "x", "b")
    lhs = module.node_map["lhs"]
    rhs = module.node_map["rhs"]
    result = module.node_map["result"]
    distributed = fm.IRModule(
        dialect=module.dialect,
        stage=module.stage,
        nodes=(
            fm.Node(
                lhs.id,
                lhs.op,
                lhs.inputs,
                fm.DistributedType(
                    lhs.type,
                    (fm.SBP.broadcast(), fm.SBP.broadcast()),
                    placement,
                ),
            ),
            fm.Node(
                rhs.id,
                rhs.op,
                rhs.inputs,
                fm.DistributedType(
                    rhs.type,
                    (
                        fm.SBP.broadcast(),
                        fm.SBP.split_block_cyclic((0,), 2),
                    ),
                    placement,
                ),
            ),
            fm.Node(
                result.id,
                result.op,
                result.inputs,
                fm.DistributedType(
                    result.type,
                    (
                        fm.SBP.broadcast(),
                        fm.SBP.split_block_cyclic((0,), 2),
                    ),
                    placement,
                ),
                metadata=result.metadata,
            ),
        ),
        functions=(),
        entry="main",
    )

    proposed = NvidiaSm90Target().propose_tir(distributed)
    point = next(value for value in proposed.selection_points if value.id == "tir.result")

    assert point.default_candidate == "tir.dense_matmul.packed_k_major_gemv_tn64_bk256"
    assert "tir.dense_matmul.packed_tensor_descriptor_smem_pipeline_gemv" not in {
        value.id for value in point.candidates
    }
