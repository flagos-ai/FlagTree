# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.selection import override_plan


_AUX_PIPELINE = (
    "tir.dense_matmul.tensor_descriptor_smem_pipeline_aux_gemv"
)


def _proposed(output_extent=128):
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    value_type = fm.tensor_type("bfloat16", (1, 128))
    output_type = fm.tensor_type("bfloat16", (1, output_extent))
    weight_type = fm.tensor_type("bfloat16", (output_extent, 128))
    value = builder.var("value", value_type, id="value")
    weight = builder.var("weight", weight_type, id="weight")
    projection = builder.call(
        "math.matmul",
        (value, weight),
        output_type,
        id="projection",
        attrs={"transpose_a": False, "transpose_b": True},
    )
    builder.function("main", (value, weight), (projection,))
    compiler = Compiler()
    proposed_vector = compiler.compile(
        builder.build(entry="main"), stop_after="propose-vectorization"
    ).module
    vectorized = compiler.run_stage(
        proposed_vector,
        "apply-vectorization",
        plan=override_plan(
            proposed_vector,
            (("vectorization.projection", "vectorization.matmul.n"),),
        ),
    ).module
    return Compiler().compile(vectorized, stop_after="propose-tir").module


def test_auxiliary_candidate_is_non_default_and_not_model_extent_specific():
    proposed = _proposed()
    point = next(
        value for value in proposed.selection_points
        if value.id == "tir.projection"
    )
    candidate = next(value for value in point.candidates if value.id == _AUX_PIPELINE)

    assert point.default_candidate == "tir.dense_matmul.gemv"
    assert candidate.parameters["auxiliary_consumer_warps"] == 4
    assert candidate.parameters["auxiliary_consumer_registers"] == 160
    assert candidate.facts["auxiliary_consumer"] is True
    assert _AUX_PIPELINE in {
        value.id
        for point in _proposed(256).selection_points
        if point.id == "tir.projection"
        for value in point.candidates
    }


def test_auxiliary_selection_materializes_channel_owned_workspace():
    proposed = _proposed()
    selected = Compiler().run_stage(
        proposed,
        "lower-tir",
        plan=override_plan(proposed, (("tir.projection", _AUX_PIPELINE),)),
    ).module
    dispatch = fm.kernel_dispatch_of(selected.kernel_definitions[-1])

    assert dispatch is not None
    selection = dispatch.microkernel
    assert selection is not None
    assert tuple(value.name for value in selection.shared_workspaces) == (
        "weight_stage",
    )
    assert selection.transfer_pipeline is not None
    assert selection.transfer_pipeline.shared_workspace_indices == (0,)
    assert not selection.transfer_pipeline.consumer_shared_workspace_indices
    assert selection.transfer_pipeline.auxiliary_consumer is not None
    assert selection.transfer_pipeline.auxiliary_consumer.channel_indices == (0,)
    assert not (
        selection.transfer_pipeline.auxiliary_consumer
        .consumer_shared_workspace_indices
    )
