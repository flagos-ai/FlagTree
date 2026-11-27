# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.runtime import (
    GeneratedTirSingleTensorModule,
    load as load_runtime,
)


def _elementwise_module(architecture: str) -> fm.IRModule:
    builder = fm.IRBuilder(
        dialect="high_level",
        stage="imported",
        metadata={"architecture": architecture},
    )
    value_type = fm.tensor_type("float32", [17])
    lhs = builder.var("lhs", value_type, id="lhs")
    rhs = builder.var("rhs", value_type, id="rhs")
    output = builder.call("math.add", [lhs, rhs], value_type, id="output")
    builder.function("main", [lhs, rhs], [output])
    return builder.build(entry="main")


def test_codegen_dispatch_is_independent_of_model_architecture(tmp_path):
    compiler = Compiler()
    first = compiler.compile(_elementwise_module("ModelA")).module
    second = compiler.compile(_elementwise_module("ModelB")).module

    first_plan = dict(first.metadata["codegen_package_plan"])
    second_plan = dict(second.metadata["codegen_package_plan"])
    assert first_plan == second_plan
    assert first_plan["kind"] == "tir_call_graph"
    assert any(call["family"] == "elementwise" for call in first_plan["calls"])

    first_package = write_artifact(
        first,
        tmp_path / "first",
        target="nvidia-sm90",
        emit_executable=True,
    )
    second_package = write_artifact(
        second,
        tmp_path / "second",
        target="nvidia-sm90",
        emit_executable=True,
    )
    assert ((first_package / "generated_kernels.py").read_text() == (second_package /
                                                                     "generated_kernels.py").read_text())


def test_builtin_renderer_ignores_importer_metadata(tmp_path):
    compiled = Compiler().compile(_elementwise_module("ModelA")).module
    edited = replace(
        compiled,
        metadata={
            **dict(compiled.metadata),
            "architecture": "Qwen3ForCausalLM",
            "output_boundary": "logits_fp32_and_greedy_token",
        },
    )

    artifact = write_artifact(
        edited,
        tmp_path / "edited",
        target="nvidia-sm90",
        emit_executable=True,
    )
    source = (artifact / "generated_kernels.py").read_text()
    assert "_flagmega_elementwise_add" in source
    assert isinstance(load_runtime(artifact), GeneratedTirSingleTensorModule)
