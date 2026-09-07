# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.selection import override_plan


_PIPELINE = "tir.dense_matmul.tensor_descriptor_smem_pipeline_gemv"


def _proposed(*, transpose_b: bool):
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    value_type = fm.tensor_type("bfloat16", (1, 128))
    weight_type = fm.tensor_type("bfloat16", (128, 128))
    value = builder.var("value", value_type, id="value")
    weight = builder.var("weight", weight_type, id="weight")
    projection = builder.call(
        "math.matmul",
        (value, weight),
        value_type,
        id="projection",
        attrs={"transpose_a": False, "transpose_b": transpose_b},
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


def test_pipeline_candidate_is_typed_non_default_and_layout_guarded():
    proposed = _proposed(transpose_b=True)
    point = next(value for value in proposed.selection_points if value.id == "tir.projection")
    candidate = next(value for value in point.candidates if value.id == _PIPELINE)

    assert point.default_candidate == "tir.dense_matmul.gemv"
    assert candidate.parameters["num_stages"] == 2
    assert candidate.facts["requires"] == ("tma", "warp_specialize")
    assert candidate.facts["transfer_pipeline"] is True
    assert _PIPELINE not in {
        value.id
        for value in next(
            point for point in _proposed(transpose_b=False).selection_points
            if point.id == "tir.projection"
        ).candidates
    }


def test_direct_selection_materializes_catalog_resources_before_bufferize():
    proposed = _proposed(transpose_b=True)
    selected = Compiler().run_stage(
        proposed,
        "lower-tir",
        plan=override_plan(proposed, (("tir.projection", _PIPELINE),)),
    ).module
    dispatch = fm.kernel_dispatch_of(selected.kernel_definitions[-1])

    assert dispatch is not None
    assert dispatch.microkernel is not None
    assert dispatch.microkernel.transfer_pipeline is not None
    assert dispatch.microkernel.transfer_pipeline.capacity == 2
    assert tuple(value.name for value in dispatch.microkernel.shared_workspaces) == (
        "weight_stage",
    )
    assert dispatch.shared_workspace_buffers[0].dimensions == tuple(
        fm.dim(value) for value in (2, 16, 128)
    )
