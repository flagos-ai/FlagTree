# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Independent namespaces avoid repeating unrelated program metadata."""

import ast
from copy import deepcopy

import pytest

from python.test.flagmega.codegen.triton.test_program_source_cache import compile_pair
from triton.flagmega.codegen.triton import render_triton_package
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.errors import ArtifactError
from triton.flagmega.runtime import load
from triton.flagmega.runtime.kernel_programs import verified_kernel_programs


def _program_metadata(path):
    module = ast.parse(path.read_text())
    declaration = next(node for node in module.body if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name)
                               and target.id == "FLAGMEGA_KERNEL_PROGRAMS"
                               for target in node.targets))
    return ast.literal_eval(declaration.value)


def test_independent_sources_carry_only_their_own_invocations(tmp_path):
    module = compile_pair("tir.dense_matmul.packed_k_major_mma")
    package = render_triton_package(module, tmp_path)
    assert len(package["kernels"]) == len(package["program_sources"]) == 2
    assert _program_metadata(tmp_path / package["source"]) == package["kernels"]
    for record in package["program_sources"]:
        expected = [invocation for invocation in package["kernels"]
                    if invocation.get("entry_symbol", invocation["symbol"]) == record["symbol"]]
        assert _program_metadata(tmp_path / record["source"]) == expected


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "unrelated", "warps"])
def test_scoped_source_metadata_remains_exactly_verified(tmp_path, corruption):
    module = compile_pair("tir.dense_matmul.packed_k_major_mma")
    artifact = write_artifact(module, tmp_path, target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact)
    assert len(verified_kernel_programs(runtime.codegen, runtime.kernels, module)) == 2
    namespace = runtime.kernels[0].fn.__globals__
    literal = deepcopy(namespace["FLAGMEGA_KERNEL_PROGRAMS"])
    if corruption == "missing":
        literal.clear()
    elif corruption == "duplicate":
        literal.append(deepcopy(literal[0]))
    elif corruption == "unrelated":
        literal.append(deepcopy(runtime.codegen["kernels"][1]))
    else:
        literal[0]["num_warps"] = 4
    namespace["FLAGMEGA_KERNEL_PROGRAMS"] = literal
    with pytest.raises(ArtifactError, match="launch resources"):
        verified_kernel_programs(runtime.codegen, runtime.kernels, module)


def test_v1_full_metadata_keeps_jit_identity_and_runtime_verification(tmp_path, monkeypatch):
    from triton.flagmega.codegen.triton import renderer
    from triton.flagmega.codegen.triton.program_sources import LEGACY_PROGRAM_SOURCE_LAYOUT

    module = compile_pair("tir.dense_matmul.packed_k_major_mma")
    scoped = load(write_artifact(module, tmp_path / "scoped", target="nvidia-sm90", emit_executable=True))
    original = renderer.render_program_source_files

    def legacy(descriptor, version, render, destination, source):
        def full_metadata(context, version):
            return render({**context, "kernels": descriptor["kernels"]}, version)
        result = original(descriptor, version, full_metadata, destination, source)
        return {**result, "program_source_layout": LEGACY_PROGRAM_SOURCE_LAYOUT}

    monkeypatch.setattr(renderer, "render_program_source_files", legacy)
    full = load(write_artifact(module, tmp_path / "full", target="nvidia-sm90", emit_executable=True))
    assert [kernel.cache_key for kernel in full.kernels] == [kernel.cache_key for kernel in scoped.kernels]
    assert len(verified_kernel_programs(full.codegen, full.kernels, module)) == 2
    declared_v1 = {**scoped.codegen, "program_source_layout": LEGACY_PROGRAM_SOURCE_LAYOUT}
    with pytest.raises(ArtifactError, match="launch resources"):
        verified_kernel_programs(declared_v1, scoped.kernels, module)
