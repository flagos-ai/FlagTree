# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.codegen.triton import render_triton_package
from triton.flagmega.codegen.triton import describe_tir_package
from triton.flagmega.compiler import Compiler
from triton.flagmega.runtime import load
from triton.flagmega.selection import override_plan


def _module() -> fm.IRModule:
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    value_type = fm.tensor_type("bfloat16", (3, 17))
    lhs = builder.var("lhs", value_type, id="lhs")
    rhs = builder.var("rhs", value_type, id="rhs")
    output = builder.call("math.add", (lhs, rhs), value_type, id="output")
    builder.function("main", (lhs, rhs), (output,))
    return builder.build(entry="main")


def _compile_axis_zero():
    compiler = Compiler()
    proposed = compiler.compile(
        _module(), stop_after="propose-vectorization"
    ).module
    plan = override_plan(
        proposed, (("vectorization.output", "vectorization.axes_0"),)
    )
    applied = compiler.run_stage(
        proposed, "apply-vectorization", plan=plan
    ).module
    return compiler.compile(applied).module


def test_noncontiguous_axis_selection_becomes_a_padded_physical_traversal(tmp_path):
    module = _compile_axis_zero()
    dispatch = fm.kernel_dispatch_for_call(module, module.node_map["output"])
    assert dispatch is not None
    schedule = dispatch.parameters["vector_schedule"]
    assert schedule["contract"]["kind"] == "axes"
    assert schedule["contract"]["axes"] == (0,)
    assert schedule["lowering"] == "packed_axes"

    package = render_triton_package(module, tmp_path)
    call = next(
        value
        for value in describe_tir_package(module)["render_calls"]
        if value["semantic_op"] == "math.add"
    )
    source = (tmp_path / "generated_kernels.py").read_text("utf-8")
    assert package["kind"] == "tir_call_graph/v1"
    assert call["logical_shape"] == (3, 17)
    assert call["padded_shape"] == (8, 17)
    assert call["local_capacity"] == 136
    assert call["logical_capacity"] == 51
    assert "tl.range(0, 136, 256)" in source
    assert "((local_offsets) // 17)" in source
    assert "((local_offsets) % 17)" in source
    assert "< (3)" in source


def test_noncontiguous_axis_selection_executes_on_h800(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")
    artifact = write_artifact(
        _compile_axis_zero(),
        tmp_path / "axis-zero",
        target="nvidia-sm90",
        emit_executable=True,
    )
    runtime = load(artifact, device="cuda:0")
    lhs = torch.randn((3, 17), dtype=torch.bfloat16, device="cuda:0")
    rhs = torch.randn_like(lhs)
    runtime.prepare(lhs, rhs)
    actual = runtime.run(lhs, rhs)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, lhs + rhs)
