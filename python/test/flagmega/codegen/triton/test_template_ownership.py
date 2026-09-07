# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega.codegen.triton.templates import (
    KernelTemplateSpec,
    TritonTemplateRegistry,
)
from triton.flagmega.targets.nvidia.implementations import (
    sm90_triton_implementation_model,
)


def test_every_registered_tir_implementation_template_owns_wrapper_rendering():
    registry = TritonTemplateRegistry()
    model = sm90_triton_implementation_model()
    resolved = {
        registry.resolve(
            KernelTemplateSpec(
                implementation.family,
                implementation.variant,
                "nvidia",
                "sm90",
            )
        )
        for implementation in model.implementations
    }

    assert resolved
    for relative in sorted(resolved):
        source = (registry.root / relative).read_text(encoding="utf-8")
        assert "render_calls is defined" in source, relative


def test_call_graph_entrypoint_contains_only_schedule_and_resource_construction():
    registry = TritonTemplateRegistry()
    source = (
        registry.root / "entrypoints/call_graph.py.jinja"
    ).read_text(encoding="utf-8")

    assert "call.family" not in source
    assert "{% for call in render_calls %}" not in source
    assert source.count("{% for handoff in pipeline_schedule.handoffs %}") == 1
    assert "pipeline_role_events" in source
    assert "tle.gpu.warp_specialize" in source


def test_codegen_sources_do_not_depend_on_model_identity():
    root = TritonTemplateRegistry().root.parent
    offenders = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".jinja"}:
            if "qwen" in path.read_text(encoding="utf-8").lower():
                offenders.append(path.relative_to(root).as_posix())

    assert offenders == []
