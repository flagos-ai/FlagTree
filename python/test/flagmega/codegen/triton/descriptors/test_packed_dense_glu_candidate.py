# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Selection tests for the generic packed dual-weight TMA contract."""

from dataclasses import replace

from triton.flagmega import ir as fm
from triton.flagmega.targets import NvidiaSm90Target
from triton.flagmega.targets.nvidia import sm90_triton_implementation_model


_IMPLEMENTATION = (
    "tir.dense_matmul_glu.packed_tensor_descriptor_smem_pipeline_full_lhs_gemv"
)
_DEFAULT_IMPLEMENTATION = (
    "tir.dense_matmul_glu.packed_tensor_descriptor_smem_pipeline_gemv"
)
_INLINE_IMPLEMENTATION = (
    "tir.dense_matmul_glu."
    "packed_tensor_descriptor_smem_pipeline_inline_gemv"
)
_PAIRED_INLINE_IMPLEMENTATION = (
    "tir.dense_matmul_glu."
    "packed_tensor_descriptor_paired_smem_pipeline_inline_gemv"
)
_PAIRED_TABLE_INLINE_IMPLEMENTATION = (
    "tir.dense_matmul_glu."
    "packed_tensor_descriptor_table_paired_smem_pipeline_inline_gemv"
)
_PAIRED_FULL_LHS_IMPLEMENTATION = (
    "tir.dense_matmul_glu."
    "packed_tensor_descriptor_paired_smem_pipeline_full_lhs_gemv"
)
_FULL_LHS_INLINE_IMPLEMENTATION = (
    "tir.dense_matmul_glu."
    "packed_tensor_descriptor_smem_pipeline_full_lhs_inline_gemv"
)


def _packed_glu_projection(*, n=256) -> fm.IRModule:
    lhs = fm.Node(
        "lhs", "builtin.var", (), fm.tensor_type("bfloat16", (1, 2048))
    )
    weight_type = fm.tensor_type("bfloat16", (128, n // 8, 2, 64))
    gate = fm.Node("gate", "builtin.var", (), weight_type)
    up = fm.Node("up", "builtin.var", (), weight_type)
    result = fm.Node(
        "result",
        "nn.packed_dense_matmul_glu",
        (lhs.id, gate.id, up.id),
        fm.tensor_type(fm.vector_type("bfloat16", (8,)), (1, n // 8)),
        attrs={"activation": "silu", "packed_layout": "k_major_n8_k16"},
        metadata={
            "selected_vectorization": "vectorization.dense_matmul_glu.n",
            "selected_vector_axes": (1,),
            "selected_vector_lanes": (8,),
        },
    )
    return fm.IRModule(
        dialect="distributed",
        stage="distributed",
        nodes=tuple(replace(value, type=fm.DistributedType(value.type,
                    (fm.SBP.broadcast(),) * value.type.rank, fm.Placement((2, 4), "yx", "bb")))
                    for value in (lhs, gate, up, result)),
        functions=(),
        entry="main",
    )


def _with_node(module: fm.IRModule, replacement: fm.Node) -> fm.IRModule:
    return fm.IRModule(
        dialect=module.dialect,
        stage=module.stage,
        nodes=tuple(
            replacement if node.id == replacement.id else node
            for node in module.nodes
        ),
        functions=module.functions,
        entry=module.entry,
    )


def test_packed_glu_complete_lhs_pipeline_is_available_but_paired_table_is_default():
    proposed = NvidiaSm90Target().propose_tir(_packed_glu_projection())
    point = next(value for value in proposed.selection_points if value.id == "tir.result")

    # On H800 the complete-LHS variant reduces registers but its extra copy
    # and CTA barrier cost more than cached direct reads for K=2048. Pairing
    # gate/up transfers under one pipeline sequence preserves the direct-LHS
    # path while reducing generated instructions.  An owner-rebased descriptor
    # table further removes global-shard coordinate work from every TMA issue.
    assert point.default_candidate == _PAIRED_TABLE_INLINE_IMPLEMENTATION
    candidate = next(value for value in point.candidates if value.id == _IMPLEMENTATION)
    assert candidate.parameters["block_k"] == 1024
    assert candidate.parameters["tile_n"] == 16
    assert candidate.parameters["reduction_group"] == 32
    assert candidate.parameters["packed_layout"] == "k_major_n8_k16"
    assert candidate.facts["requires"] == ("tma", "warp_specialize")
    assert candidate.facts["host_tensor_descriptor"] is True
    assert candidate.facts["transfer_pipeline"] is True
    assert candidate.facts["complete_consumer_lhs_stage"] is True
    assert candidate.parameters["lhs_stage_extent"] == 2048
    implementation = sm90_triton_implementation_model().implementation(
        _IMPLEMENTATION
    )
    assert implementation is not None
    assert implementation.shared_workspaces[1].name == "lhs_stage"
    assert implementation.shared_workspaces[1].type.shape[1].fixed_value == 2048
    assert implementation.transfer_pipeline is not None
    assert implementation.transfer_pipeline.consumer_shared_workspace_indices == (1,)


def test_packed_glu_inline_consumer_remains_an_explicit_candidate():
    proposed = NvidiaSm90Target().propose_tir(_packed_glu_projection())
    point = next(value for value in proposed.selection_points if value.id == "tir.result")

    assert point.default_candidate == _PAIRED_TABLE_INLINE_IMPLEMENTATION
    candidate = next(
        value for value in point.candidates if value.id == _INLINE_IMPLEMENTATION
    )
    assert "inline_consumer_stage" not in candidate.parameters
    assert candidate.parameters["block_k"] == 1024
    assert candidate.parameters["tile_n"] == 16
    implementation = sm90_triton_implementation_model().implementation(
        _INLINE_IMPLEMENTATION
    )
    assert implementation is not None
    assert "inline_consumer_stage" not in implementation.parameters


def test_packed_glu_paired_weight_pipeline_is_a_typed_candidate():
    proposed = NvidiaSm90Target().propose_tir(_packed_glu_projection())
    point = next(value for value in proposed.selection_points if value.id == "tir.result")

    assert point.default_candidate == _PAIRED_TABLE_INLINE_IMPLEMENTATION

    candidate = next(
        value
        for value in point.candidates
        if value.id == _PAIRED_INLINE_IMPLEMENTATION
    )
    assert candidate.parameters["paired_weight_fields"] is True
    assert "inline_consumer_stage" not in candidate.parameters
    assert candidate.facts["paired_weight_transfer"] is True
    implementation = sm90_triton_implementation_model().implementation(
        _PAIRED_INLINE_IMPLEMENTATION
    )
    assert implementation is not None
    assert tuple(
        value.name for value in implementation.shared_workspaces
    ) == ("gate_stage", "up_stage")
    assert implementation.transfer_pipeline is not None
    assert implementation.transfer_pipeline.capacity == 2
    assert implementation.transfer_pipeline.channels[0].shared_workspace_indices == (
        0,
        1,
    )


def test_packed_glu_owner_descriptor_table_is_an_explicit_candidate():
    proposed = NvidiaSm90Target().propose_tir(_packed_glu_projection())
    point = next(value for value in proposed.selection_points if value.id == "tir.result")
    candidate = next(
        value
        for value in point.candidates
        if value.id == _PAIRED_TABLE_INLINE_IMPLEMENTATION
    )

    assert point.default_candidate == _PAIRED_TABLE_INLINE_IMPLEMENTATION
    assert candidate.parameters["descriptor_kind"] == "table"
    assert candidate.parameters["paired_weight_fields"] is True
    assert candidate.facts["host_tensor_descriptor_table"] is True
    implementation = sm90_triton_implementation_model().implementation(
        _PAIRED_TABLE_INLINE_IMPLEMENTATION
    )
    assert implementation is not None
    assert implementation.transfer_pipeline is not None
    assert implementation.transfer_pipeline.capacity == 2


def test_packed_glu_paired_full_lhs_matches_the_typed_nncase_geometry():
    proposed = NvidiaSm90Target().propose_tir(_packed_glu_projection())
    point = next(value for value in proposed.selection_points if value.id == "tir.result")

    candidate = next(
        value
        for value in point.candidates
        if value.id == _PAIRED_FULL_LHS_IMPLEMENTATION
    )
    assert candidate.parameters["paired_weight_fields"] is True
    assert candidate.parameters["lhs_stage_extent"] == 2048
    assert candidate.parameters["num_stages"] == 4
    assert "inline_consumer_stage" not in candidate.parameters
    implementation = sm90_triton_implementation_model().implementation(
        _PAIRED_FULL_LHS_IMPLEMENTATION
    )
    assert implementation is not None
    assert tuple(
        value.name for value in implementation.shared_workspaces
    ) == ("gate_stage", "up_stage", "lhs_stage")
    assert implementation.transfer_pipeline is not None
    assert implementation.transfer_pipeline.capacity == 2
    assert implementation.transfer_pipeline.consumer_shared_workspace_indices == (2,)


def test_packed_glu_full_lhs_inline_pipeline_is_an_explicit_candidate():
    proposed = NvidiaSm90Target().propose_tir(_packed_glu_projection())
    point = next(value for value in proposed.selection_points if value.id == "tir.result")

    candidate = next(
        value
        for value in point.candidates
        if value.id == _FULL_LHS_INLINE_IMPLEMENTATION
    )
    assert "inline_consumer_stage" not in candidate.parameters
    assert candidate.parameters["lhs_stage_extent"] == 2048
    assert candidate.facts["complete_consumer_lhs_stage"] is True
    assert "inline_consumer_stage" not in candidate.facts
    implementation = sm90_triton_implementation_model().implementation(
        _FULL_LHS_INLINE_IMPLEMENTATION
    )
    assert implementation is not None
    assert "inline_consumer_stage" not in implementation.parameters
    assert implementation.transfer_pipeline is not None
    assert implementation.transfer_pipeline.consumer_shared_workspace_indices == (1,)


def test_packed_glu_descriptor_pipeline_accepts_a_valid_output_tail():
    # Both RHS tensors and the result describe the same logical tail. The old
    # fixture changed only the result, which made its logical shape invalid.
    proposed = NvidiaSm90Target().propose_tir(_packed_glu_projection(n=248))
    point = next(value for value in proposed.selection_points if value.id == "tir.result")

    assert _IMPLEMENTATION in {value.id for value in point.candidates}


def test_packed_glu_descriptor_pipeline_rejects_one_block_cyclic_weight():
    module = _packed_glu_projection()
    placement = fm.Placement((2,), "x", "b")
    up = module.node_map["up"]
    incompatible = _with_node(
        module,
        fm.Node(
            up.id,
            up.op,
            up.inputs,
            fm.DistributedType(
                fm.logical_type(up.type),
                (
                    fm.SBP.broadcast(),
                    fm.SBP.split_block_cyclic((0,), 2),
                    fm.SBP.broadcast(),
                    fm.SBP.broadcast(),
                ),
                placement,
            ),
        )
    )

    proposed = NvidiaSm90Target().propose_tir(incompatible)
    point = next(value for value in proposed.selection_points if value.id == "tir.result")

    assert _IMPLEMENTATION not in {value.id for value in point.candidates}
