# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Independent programs retain their JIT identity after an unrelated edit."""

from dataclasses import replace
import ast
import json

from jinja2 import ChoiceLoader, DictLoader
import pytest
from python.test.flagmega.codegen.triton.test_kernel_programs import graph
from triton.flagmega.artifacts import load_artifact, write_artifact
from triton.flagmega.artifacts import module as artifact_module
from triton.flagmega.compiler import Compiler
from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry
from triton.flagmega.runtime import load
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule as Runtime
from triton.flagmega.selection import apply_plan, override_plan
from triton.flagmega.targets import NvidiaSm90Target
from triton.flagmega.errors import ArtifactError


def compile_pair(first_variant):
    class Target(NvidiaSm90Target):
        def propose_tir(self, module):
            module = super().propose_tir(module)
            points = [point for point in module.selection_points
                      if point.kind == "tir" and point.owner in module.node_map
                      and module.node_map[point.owner].op == "ntt.packed_matmul"]
            assert len(points) == 2
            choices = [(points[0].id, first_variant),
                       (points[1].id, "tir.dense_matmul.packed_k_major_mma")]
            module = apply_plan(module, override_plan(
                module, choices, policy="test.program-source-cache",
                rationale="Change the first operator while preserving the second program."))
            return replace(module, metadata={**module.metadata, "kernel_program_partition": (
                dict(symbol="program_first", calls=(points[0].owner,), num_warps=8),
                dict(symbol="program_second", calls=(points[1].owner,), num_warps=4),
            )})

    compiler = Compiler()
    compiler.target = Target()
    return compiler.compile(graph(packed=True)).module


def write_edited_pair(tmp_path, monkeypatch):
    module = compile_pair("tir.dense_matmul.packed_k_major_mma")
    before = write_artifact(module, tmp_path / "before", target="nvidia-sm90", emit_executable=True)
    original_init = TritonTemplateRegistry.__init__

    def templates(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        loader = self.environment.loader
        source = loader.get_source(self.environment, "module.py.jinja")[0]
        prefix = ("{% if not programs and symbol == 'program_first' %}"
                  "# Extra source locations in the first operator namespace.\n"
                  "# The second operator's code and dependencies are unchanged.\n"
                  "{% endif %}")
        self.environment.loader = ChoiceLoader([DictLoader({"module.py.jinja": prefix + source}), loader])

    monkeypatch.setattr(TritonTemplateRegistry, "__init__", templates)
    after = write_artifact(module, tmp_path / "after", target="nvidia-sm90", emit_executable=True)
    return before, after


def test_only_edited_program_recompiles(tmp_path, monkeypatch):
    before, after = write_edited_pair(tmp_path, monkeypatch)
    original, changed = load(before), load(after)
    first, second = original.kernels
    edited_first, same_second = changed.kernels
    assert first.cache_key != edited_first.cache_key
    assert second.src == same_second.src
    assert second.cache_key == same_second.cache_key



def test_program_templates_render_once(tmp_path, monkeypatch):
    from triton.flagmega.codegen.triton import render_triton_package
    from triton.flagmega.codegen.triton.tir_package import describe_tir_package, render_tir_package
    from triton.flagmega.codegen.triton.renderer import RENDERER_VERSION

    module = compile_pair("tir.dense_matmul.packed_k_major_mma")
    original_init = TritonTemplateRegistry.__init__
    visits = []

    def templates(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        get_template = self.environment.get_template

        def lookup(name, *args, **kwargs):
            if isinstance(name, str) and name.startswith("kernels/"):
                visits.append(name)
            return get_template(name, *args, **kwargs)

        self.environment.get_template = lookup

    monkeypatch.setattr(TritonTemplateRegistry, "__init__", templates)
    render_tir_package(describe_tir_package(module), RENDERER_VERSION)
    expected = len(visits)
    assert expected > 0
    visits.clear()
    render_triton_package(module, tmp_path)
    assert len(visits) == expected


def test_program_schedule_transforms_run_only_in_the_kernel_namespace(tmp_path, monkeypatch):
    from triton.flagmega.codegen.triton import render_triton_package

    original_render = TritonTemplateRegistry.render
    transformed = []

    def render(self, name, context):
        if name == "module.py.jinja":
            # A schedule transform belongs to a concrete kernel namespace.
            # Replaying it while assembling wrappers would change it twice.
            assert not context.get("programs"), "kernel transform called for a package wrapper"
            for event in context["entry_events"]:
                if event.get("kind") == "tir.kernel_call":
                    assert event.get("participation_active", "True") == "True"
                    event["participation_active"] = "(tl.program_id(0) == 0)"
                    transformed.append(event["call"])
        return original_render(self, name, context)

    monkeypatch.setattr(TritonTemplateRegistry, "render", render)
    package = render_triton_package(compile_pair("tir.dense_matmul.packed_k_major_mma"), tmp_path)
    assert len(transformed) == len(set(transformed)) == 2
    for record in package["program_sources"]:
        source = (tmp_path / record["source"]).read_text()
        assert "(tl.program_id(0) == 0)" in source


def test_descriptor_override_keeps_program_preparation(tmp_path, monkeypatch):
    from triton.flagmega.codegen.triton.registry import PackageRendererSpec
    from triton.flagmega.codegen.triton.renderer import renderer_registry

    module = compile_pair("tir.dense_matmul.packed_k_major_mma")
    original_init = TritonTemplateRegistry.__init__
    visits = []

    def templates(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        get_template = self.environment.get_template

        def lookup(name, *args, **kwargs):
            if isinstance(name, str) and name.startswith("kernels/"):
                visits.append(name)
            return get_template(name, *args, **kwargs)

        self.environment.get_template = lookup

    monkeypatch.setattr(TritonTemplateRegistry, "__init__", templates)
    before = write_artifact(module, tmp_path / "before", target="nvidia-sm90", emit_executable=True)
    expected = len(visits)
    assert expected > 0
    visits.clear()
    spec = renderer_registry.resolve(module)
    # Existing descriptor extensions retain the registered source renderer
    # through the original five-argument registration API.
    wrapped = PackageRendererSpec(spec.name, spec.kind, spec.matches,
                                  lambda module: spec.describe(module), spec.render)
    monkeypatch.setitem(renderer_registry._specs, spec.name, wrapped)
    after = write_artifact(module, tmp_path / "after", target="nvidia-sm90", emit_executable=True)
    assert len(visits) == expected
    assert [k.cache_key for k in load(before).kernels] == [k.cache_key for k in load(after).kernels]


def test_mixed_mesh_namespaces_keep_program_template_flags(tmp_path):
    from triton.flagmega.codegen.triton.tir_package import describe_tir_package, render_tir_package
    from triton.flagmega.codegen.triton.renderer import RENDERER_VERSION

    descriptor = describe_tir_package(compile_pair("tir.dense_matmul.packed_k_major_mma"))
    first, second = descriptor["programs"]
    descriptor["programs"] = ({**first, "mesh_context": {}}, {**second, "shared_silu": True})
    source = ast.parse(render_tir_package(descriptor, RENDERER_VERSION))
    builder = next(node for node in source.body if isinstance(node, ast.FunctionDef)
                   and node.name == "_flagmega_build_program_second")
    assert any(isinstance(node, ast.FunctionDef) and node.name == "_flagmega_silu" for node in builder.body)


def test_device_reuses_unchanged_program_compilation(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")
    before, after = write_edited_pair(tmp_path, monkeypatch)
    values = {
        "x": (torch.arange(128, device="cuda").reshape(1, 128) % 17 - 8).bfloat16() / 16,
        "a": torch.ones((8, 8, 8, 2, 8), device="cuda", dtype=torch.bfloat16) / 8,
        "b": torch.ones((8, 4, 8, 2, 8), device="cuda", dtype=torch.bfloat16) / 4,
    }
    guards = [torch.full((n + 32,), -123., device="cuda", dtype=torch.bfloat16) for n in (64, 32)]
    outputs = [value[16:-16].reshape(1, -1, 8) for value in guards]
    hashes = []
    for artifact in (before, after):
        runtime = load(artifact, device="cuda:0")
        results = iter(outputs)
        arguments = tuple(next(results) if spec["role"] == "result" else values[spec["value"]]
                          for spec in runtime.external_arguments)
        try:
            Runtime.prepare(runtime, *arguments)
            hashes.append(tuple(prepared.compiled_kernel.hash for prepared in runtime._prepared))
            assert all(report["spill_bytes"] == 0 for report in runtime.resource_report)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.graph(capture := torch.cuda.CUDAGraph(), stream=stream):
                Runtime.run_into(runtime, *arguments)
            for mutation in range(3):
                values["x"].add_(1 / 16)
                capture.replay()
                torch.cuda.synchronize()
                for name, output in zip(("a", "b"), outputs, strict=True):
                    weight = values[name].permute(0, 3, 4, 1, 2).reshape(128, -1)
                    gold = (values["x"].double() @ weight.double()).bfloat16()
                    assert torch.equal(output.reshape(1, -1), gold)
                assert all(bool((guard[:16] == -123).all() & (guard[-16:] == -123).all())
                           for guard in guards)
        finally:
            runtime.close()
    assert hashes[0][0] != hashes[1][0]
    assert hashes[0][1] == hashes[1][1]


@pytest.mark.parametrize("cause", ["bytes", "section", "digest", "exports", "path",
                                   "alias_duplicate", "alias_cycle", "alias_missing"])
def test_program_files_retain_artifact_integrity_checks(tmp_path, cause):
    artifact = write_artifact(compile_pair("tir.dense_matmul.packed_k_major_mma"),
                              tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    path = artifact / "artifact.json"
    manifest = json.loads(path.read_text())
    record = manifest["codegen"]["program_sources"][0]
    if cause == "bytes":
        source = artifact / record["source"]
        source.write_text(source.read_text() + "\n# corrupt source\n")
    elif cause == "section":
        manifest["sections"] = [section for section in manifest["sections"]
                                if section["name"] != f"codegen.program.{record['symbol']}"]
    elif cause == "digest":
        record["source_sha256"] = "0" * 64
    elif cause == "exports":
        record["symbols"] = []
    elif cause == "path":
        record["source"] = "../outside.py"
    elif cause == "alias_duplicate":
        manifest["codegen"]["program_entry_aliases"] = {"program_first": "program_second"}
    elif cause == "alias_cycle":
        manifest["codegen"]["program_entry_aliases"] = {"alias_a": "alias_b", "alias_b": "alias_a"}
    else:
        manifest["codegen"]["program_entry_aliases"] = {"alias_a": "missing_entry"}
    path.write_text(json.dumps(manifest))
    with pytest.raises(ArtifactError):
        load(artifact)


def test_legacy_single_source_artifact_loads_and_refreshes(tmp_path, monkeypatch):
    render = artifact_module.render_triton_package

    def legacy(*args, **kwargs):
        package = render(*args, **kwargs)
        return {name: value for name, value in package.items()
                if name not in {"program_source_layout", "program_sources", "program_entry_aliases"}}

    monkeypatch.setattr(artifact_module, "render_triton_package", legacy)
    artifact = write_artifact(compile_pair("tir.dense_matmul.packed_k_major_mma"),
                              tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    before, module = load_artifact(artifact)
    original = load(artifact)
    assert len(original.kernels) == 2
    assert all(kernel.fn.__code__.co_filename == str(artifact / before["codegen"]["source"])
               for kernel in original.kernels)
    ir_stat = (artifact / before["final_ir"]).stat()
    monkeypatch.setattr(artifact_module, "render_triton_package", render)
    artifact_module.refresh_artifact_codegen(artifact)
    after, refreshed = load_artifact(artifact)
    runtime = load(artifact)
    assert refreshed.semantic_hash == module.semantic_hash
    assert len(after["codegen"]["program_sources"]) == len(runtime.kernels) == 2
    assert len({kernel.fn.__code__.co_filename for kernel in runtime.kernels}) == 2
    assert (artifact / before["final_ir"]).stat() == ir_stat


def test_refresh_rejects_a_corrupt_unreferenced_program_file(tmp_path):
    artifact = write_artifact(compile_pair("tir.dense_matmul.packed_k_major_mma"),
                              tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    manifest_path = artifact / "artifact.json"
    before = manifest_path.read_bytes()
    record = json.loads(before)["codegen"]["program_sources"][0]
    destination = artifact / "codegen" / "programs" / record["source_sha256"] / f"{record['symbol']}.py"
    destination.parent.mkdir(parents=True)
    destination.write_text("# corrupt unused source at the next refresh destination\n")
    with pytest.raises(ArtifactError, match="program source"):
        artifact_module.refresh_artifact_codegen(artifact)
    assert manifest_path.read_bytes() == before
    load(artifact)


@pytest.mark.parametrize("packed", [False, True])
def test_split_sources_preserve_shared_entry_identity(tmp_path, packed):
    from python.test.flagmega.codegen.triton.test_function_programs import compile_repeated_stages

    module = compile_repeated_stages(packed=packed)
    artifact = write_artifact(module, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact)
    assert len(runtime.codegen["program_sources"]) == 6
    assert len(runtime.kernels) == 12
    for stage in range(6):
        assert runtime.kernels[stage] is runtime.kernels[stage + 6]


@pytest.mark.parametrize("split", [False, True])
def test_shared_entry_type_annotations_have_a_capture_scope(tmp_path, split):
    import triton.language as tl
    from python.test.flagmega.codegen.triton.pipelines.conftest import _compile_pipeline_module
    from python.test.flagmega.codegen.triton.test_kernel_programs import import_source
    from triton.flagmega.codegen.triton import render_triton_package

    module = _compile_pipeline_module(reusable=True, worker_depth=4, program_dag=True,
                                     implementation="tir.dense_matmul.tensor_descriptor_gemv")
    package = render_triton_package(module, tmp_path)
    source = package["program_sources"][0]["source"] if split else package["source"]
    kernel = getattr(import_source(tmp_path / source), "program_0")
    assert kernel.fn.__annotations__["_flagmega_call_offsets"] == tl.int64
    assert kernel.get_capture_scope().get("tl") is tl
