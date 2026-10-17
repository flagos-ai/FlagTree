# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Kernel iteration reuses an artifact's unchanged IR and readonly assets."""

import json
from dataclasses import replace

import pytest
import torch
from jinja2 import ChoiceLoader, DictLoader
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import load_artifact, write_artifact
from triton.flagmega.artifacts import module as artifact_module
from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry
from triton.flagmega.compiler import Compiler
from triton.flagmega.errors import ArtifactError, CodegenError


def artifact(tmp_path):
    class Graph(fm.Module):
        def forward(self):
            kind = fm.tensor_type("float32", (17,))
            value = self.input("value", kind, id="value")
            weight = self.weight("weight", kind, source="checkpoint", key="weight")
            self.function("main", (value,), (fm.F.math.add(value, weight),))

    class Source:
        def load_tensor(self, key, *, device="cpu"):
            assert key == "weight"
            return torch.arange(17, dtype=torch.float32, device=device)

    module = Compiler().compile(Graph(dialect="high_level", stage="imported", entry="main").build()).module
    return write_artifact(module, tmp_path / "artifact", target="nvidia-sm90", checkpoint=Source(),
                          emit_executable=True)


def test_refresh_uses_normal_templates_without_repacking_or_rewriting_ir(tmp_path, monkeypatch):
    root = artifact(tmp_path)
    before, module = load_artifact(root)
    preserved = {path: (path.stat(), path.read_bytes()) for path in (*root.joinpath("assets").rglob("*"),
                                                                  *root.joinpath("ir").rglob("*")) if path.is_file()}
    original = TritonTemplateRegistry.__init__

    def templates(self, *args, **kwargs):
        original(self, *args, **kwargs)
        loader = self.environment.loader
        source = loader.get_source(self.environment, "module.py.jinja")[0]
        self.environment.loader = ChoiceLoader([DictLoader({"module.py.jinja": "# refreshed template\n" + source}),
                                                loader])

    monkeypatch.setattr(TritonTemplateRegistry, "__init__", templates)
    monkeypatch.setattr(artifact_module, "pack_rdata", lambda *a, **k: pytest.fail("unchanged assets were repacked"))
    monkeypatch.setattr(artifact_module, "emit_module", lambda *a, **k: pytest.fail("unchanged IR was rewritten"))
    assert artifact_module.refresh_artifact_codegen(root) == root.resolve()
    after, reloaded = load_artifact(root)
    assert reloaded.semantic_hash == module.semantic_hash
    assert after["rdata"] == before["rdata"]
    assert after["codegen"]["source_sha256"] != before["codegen"]["source_sha256"]
    assert after["codegen"]["source"].startswith("codegen/")
    assert (root / before["codegen"]["source"]).is_file()
    for path, (stat, payload) in preserved.items():
        current = path.stat()
        assert (current.st_ino, current.st_size, current.st_mtime_ns) == (stat.st_ino, stat.st_size, stat.st_mtime_ns)
        assert path.read_bytes() == payload


def test_failed_refresh_keeps_the_previous_loadable_manifest(tmp_path, monkeypatch):
    root = artifact(tmp_path)
    old_manifest = (root / "artifact.json").read_bytes()

    def fail(*args, **kwargs):
        raise CodegenError("test rendering failure")

    monkeypatch.setattr(artifact_module, "render_triton_package", fail)
    with pytest.raises(CodegenError, match="test rendering failure"):
        artifact_module.refresh_artifact_codegen(root)
    assert (root / "artifact.json").read_bytes() == old_manifest
    load_artifact(root)


def test_refresh_rejects_modified_ir_before_rendering(tmp_path, monkeypatch):
    root = artifact(tmp_path)
    manifest = json.loads((root / "artifact.json").read_text())
    path = root / manifest["final_ir"]
    path.write_text(path.read_text() + "\n# changed IR\n")
    monkeypatch.setattr(artifact_module, "render_triton_package", lambda *a, **k: pytest.fail("unchecked IR rendered"))
    with pytest.raises(ArtifactError, match="ir.python"):
        artifact_module.refresh_artifact_codegen(root)


def test_refresh_preserves_the_readonly_payload_verification_gate(tmp_path):
    root = artifact(tmp_path)
    image = root / "assets/rdata.bin"
    with image.open("r+b") as stream:
        first = stream.read(1)
        stream.seek(0)
        stream.write(bytes((first[0] ^ 1,)))
    artifact_module.refresh_artifact_codegen(root)
    with pytest.raises(ArtifactError, match="digest|hash"):
        load_artifact(root)


def test_kernel_schedule_edit_reuses_the_closed_readonly_snapshot(tmp_path):
    root = artifact(tmp_path)
    before, original = load_artifact(root)
    index = root / "assets/rdata.index.json"
    old_index = index.read_bytes()
    old_image = (root / "assets/rdata.bin").stat()

    def change_schedule(module):
        result = []
        for definition in module.kernel_definitions:
            dispatch = definition.dispatch
            if dispatch.microkernel.family == "elementwise":
                microkernel = replace(dispatch.microkernel, parameters={**dispatch.microkernel.parameters,
                                                                        "elements_per_program": 128})
                definition = replace(definition, dispatch=replace(dispatch, microkernel=microkernel))
            result.append(definition)
        return result

    artifact_module.refresh_artifact_codegen(root, kernel_rewrite=change_schedule)
    after, edited = load_artifact(root)
    assert edited.semantic_hash != original.semantic_hash
    assert after["semantic_hash"] == edited.semantic_hash
    assert after["rdata"] == before["rdata"]
    assert after["final_ir"] != before["final_ir"]
    assert (root / before["final_ir"]).is_file()
    assert index.read_bytes() == old_index
    new_image = (root / "assets/rdata.bin").stat()
    assert (new_image.st_ino, new_image.st_mtime_ns) == (old_image.st_ino, old_image.st_mtime_ns)
    assert any(d.dispatch.microkernel.parameters.get("elements_per_program") == 128 for d in edited.kernel_definitions)


def test_kernel_refresh_rejects_an_edit_to_the_buffer_signature(tmp_path):
    root = artifact(tmp_path)
    before = (root / "artifact.json").read_bytes()

    def change_signature(module):
        first, *rest = module.kernel_definitions
        return (replace(first, parameters=first.parameters[:-1]), *rest)

    with pytest.raises(ArtifactError, match="kernel.*(signature|ABI)"):
        artifact_module.refresh_artifact_codegen(root, kernel_rewrite=change_signature)
    assert (root / "artifact.json").read_bytes() == before
