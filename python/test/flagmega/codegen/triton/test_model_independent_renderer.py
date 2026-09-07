# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace
from pathlib import Path
import re

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.codegen import triton as triton_codegen
from triton.flagmega.codegen.triton.candidates import (
    default_triton_candidate_registry,
)
from triton.flagmega.codegen.triton.renderer import renderer_registry
from triton.flagmega.codegen.triton.templates import (
    KernelTemplateSpec,
    TritonTemplateRegistry,
)
from triton.flagmega.compiler import Compiler
from triton.flagmega.runtime import (
    GeneratedTirSingleTensorModule,
    load as load_runtime,
    package_registry,
)
from triton.flagmega.targets.nvidia import sm90_triton_implementation_model


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
    assert any(
        call["family"] == "elementwise" for call in first_plan["calls"]
    )

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
    assert (
        (first_package / "generated_kernels.py").read_text()
        == (second_package / "generated_kernels.py").read_text()
    )


def test_codegen_catalog_and_package_boundary_have_no_model_identity():
    model = sm90_triton_implementation_model()
    identifiers = {
        *default_triton_candidate_registry().op_names,
        *(implementation.id for implementation in model.implementations),
        *(implementation.family for implementation in model.implementations),
        *(implementation.variant for implementation in model.implementations),
    }
    assert not any("qwen" in value.lower() for value in identifiers)
    assert tuple((spec.name, spec.kind) for spec in renderer_registry.specs) == (
        ("bufferized-tir", "tir_call_graph/v1"),
    )
    assert tuple((spec.kind, spec.target) for spec in package_registry.specs) == (
        ("elementwise/v2", "nvidia-sm90"),
        ("elementwise_add/v1", "nvidia-sm90"),
        ("tir_call_graph/v1", "nvidia-sm90"),
    )
    codegen_root = Path(triton_codegen.__file__).parent
    sources = tuple(
        path.relative_to(codegen_root).as_posix()
        for path in codegen_root.rglob("*")
        if path.suffix in {".py", ".jinja"}
    )
    assert not any("qwen" in path.lower() for path in sources)


def test_every_architecture_template_is_reachable_from_catalog_roots():
    registry = TritonTemplateRegistry()
    model = sm90_triton_implementation_model()
    roots = {
        registry.resolve(KernelTemplateSpec(
            value.family,
            value.variant,
            "nvidia",
            "sm90",
        ))
        for value in model.implementations
    }
    reference = re.compile(
        r'''(?:include|extends|import|from)\s+["']([^"']+)["']'''
    )
    closure: set[str] = set()
    pending = list(roots)
    while pending:
        relative = pending.pop()
        if relative in closure:
            continue
        closure.add(relative)
        source = (registry.root / relative).read_text(encoding="utf-8")
        pending.extend(reference.findall(source))

    architecture_templates = {
        path.relative_to(registry.root).as_posix()
        for path in (
            registry.root / "kernels"
        ).glob("*/platforms/nvidia/sm90/*.py.jinja")
    }
    assert architecture_templates <= closure


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
