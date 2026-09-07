# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton import render_triton_package
from triton.flagmega.compiler import Compiler
from triton.flagmega.selection import override_plan


def _module():
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    value_type = fm.tensor_type("float32", (257,))
    lhs = builder.var("lhs", value_type, id="lhs")
    rhs = builder.var("rhs", value_type, id="rhs")
    output = builder.call("math.add", (lhs, rhs), value_type, id="output")
    builder.function("main", (lhs, rhs), (output,))
    return builder.build(entry="main")


def _compile_scalar_selection():
    proposed = Compiler().compile(
        _module(), stop_after="propose-vectorization"
    ).module
    plan = override_plan(
        proposed,
        (("vectorization.output", "vectorization.scalar"),),
    )
    scalar = Compiler().run_stage(
        proposed, "apply-vectorization", plan=plan
    ).module
    return Compiler().compile(scalar).module


def test_agent_vectorization_choice_lowers_to_distinct_target_candidate_and_source(tmp_path):
    vectorized = Compiler().compile(_module()).module
    scalar = _compile_scalar_selection()

    vector_dispatch = fm.kernel_dispatch_for_call(
        vectorized, vectorized.node_map["output"]
    )
    scalar_dispatch = fm.kernel_dispatch_for_call(
        scalar, scalar.node_map["output"]
    )
    assert vector_dispatch is not None and scalar_dispatch is not None
    assert vector_dispatch.candidate == "tir.elementwise.add"
    assert scalar_dispatch.candidate == "tir.elementwise.add.scalar"
    assert vector_dispatch.parameters["vector_schedule"]["physical"] == {
        "elements_per_program": 128,
    }
    assert scalar_dispatch.parameters["vector_schedule"]["physical"] == {
        "elements_per_program": 1,
    }

    vector_dir = tmp_path / "vector"
    scalar_dir = tmp_path / "scalar"
    vector_package = render_triton_package(vectorized, vector_dir)
    scalar_package = render_triton_package(scalar, scalar_dir)
    vector_source = (vector_dir / "generated_kernels.py").read_text("utf-8")
    scalar_source = (scalar_dir / "generated_kernels.py").read_text("utf-8")

    assert vector_package["kind"] == "tir_call_graph/v1"
    assert scalar_package["kind"] == "tir_call_graph/v1"
    assert "tl.arange(0, 128)" in vector_source
    assert "tl.arange(0, 1)" in scalar_source
    assert vector_source != scalar_source
