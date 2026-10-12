# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The C++ backend's package: what it emits, what it refuses, and what it runs.

The execution tests need an Iluvatar GPU and the COREX device compiler, so they
skip elsewhere. The render tests do not, and they are the ones that catch a
template or ABI regression.
"""

import ctypes
import json

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.codegen.cpp import (
    describe_cpp_package,
    render_cpp_source_package,
    toolchain_available,
)
from triton.flagmega.compiler import Compiler
from triton.flagmega.errors import CodegenError
from triton.flagmega.options import CompileOptions

TARGET = "iluvatar-bi-v150"

requires_toolchain = pytest.mark.skipif(
    not toolchain_available(TARGET),
    reason="The COREX device compiler is required to build a C++ package",
)


def _elementwise_module(op: str = "math.add") -> fm.IRModule:

    class Graph(fm.Module):

        def __init__(self):
            super().__init__(dialect="high_level", stage="imported", entry="main")

        def forward(self):
            input_type = fm.tensor_type("bfloat16", (3, 17))
            lhs = self.input("lhs", input_type, id="lhs")
            if op == "math.add":
                rhs = self.input("rhs", input_type, id="rhs")
                output = fm.F.math.add(lhs, rhs, name="output")
                inputs = (lhs, rhs)
            else:
                output = fm.F.math.silu(lhs, name="output")
                inputs = (lhs,)
            self.function("main", inputs, (output,))

    return Graph().build()


def _compiled(op: str = "math.add") -> fm.IRModule:
    return Compiler(CompileOptions(target=TARGET)).compile(
        _elementwise_module(op)
    ).module


def _iluvatar_device():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or not getattr(torch, "corex", False):
        pytest.skip("An Iluvatar COREX GPU is required")
    return torch


def test_describe_reuses_the_shared_tir_descriptor():
    """The C++ descriptor extends the Triton one rather than re-deriving it.

    Both backends must address the same bytes, so the buffer plan, call ABI and
    index algebra are shared. A second derivation could drift.
    """

    from triton.flagmega.codegen.triton import describe_tir_package

    module = _compiled()
    shared = describe_tir_package(module)
    cpp = describe_cpp_package(module)

    assert cpp["render_calls"] == shared["render_calls"]
    assert cpp["runtime_binding"] == shared["runtime_binding"]
    assert cpp["grid"] == shared["grid"]
    # And adds only what C++ emission needs.
    assert cpp["schema"] == "flagmega.cpp-package-descriptor/v1"
    assert cpp["block_threads"] > 0
    assert set(cpp["argument_types"]) >= set(cpp["signature_arguments"])


def test_argument_types_come_from_the_buffer_plan(tmp_path):
    """Every pointer parameter is typed, and pools stay byte pointers.

    An untyped fallback here is not a cosmetic issue: typing a bf16 buffer as
    `unsigned char *` compiles cleanly and then reads one byte at a time.
    """

    descriptor = describe_cpp_package(_compiled())
    types = descriptor["argument_types"]

    assert types["lhs"] == {"type": "ftt::bfloat16_t", "pointer": True}
    assert types["rhs"] == {"type": "ftt::bfloat16_t", "pointer": True}
    assert types["output"] == {"type": "ftt::bfloat16_t", "pointer": True}
    # A byte pool is retyped per use by the renderer's own pointer casts.
    assert types["block_local_data"] == {"type": "unsigned char", "pointer": True}


def test_renders_the_launch_abi_and_kernel_bodies(tmp_path):
    package = render_cpp_source_package(_compiled(), tmp_path)
    source = (tmp_path / "generated_kernels.cpp").read_text("utf-8")

    assert package["kind"] == "tir_call_graph_cpp/v1"
    assert package["source"] == "generated_kernels.cpp"
    # The three C entry points the Python runtime calls through ctypes.
    assert "int flagmega_launch(void **arguments, void *stream)" in source
    assert "int flagmega_argument_count()" in source
    assert "int flagmega_abi_version()" in source
    assert f"FTT_GLOBAL void {package['symbol']}(" in source
    # The kernel itself, and the FTT library it was written against.
    assert "#include <ftt/ftt.h>" in source
    assert "// flagmega-kernel: elementwise/add" in source


def test_refuses_a_kernel_without_a_cpp_template(monkeypatch):
    """An unimplemented kernel must raise, not silently fall back to Triton.

    A package is entirely C++ or entirely Triton. A mixed package would be a
    surprising thing to hand someone who asked for one backend.
    """

    from triton.flagmega.codegen.cpp import templates

    monkeypatch.setattr(
        templates.CppTemplateRegistry, "available", lambda self, spec: False
    )
    with pytest.raises(CodegenError, match="no template for kernel"):
        describe_cpp_package(_compiled())


def test_refuses_a_warp_specialized_pipeline():
    """A producer/consumer schedule has no C++ implementation yet.

    Emitting it as a plain sequential body would compute the right answer by
    the wrong schedule, so it raises instead.
    """

    module = _compiled()
    descriptor = describe_cpp_package(module)
    assert descriptor["pipeline_schedule"] is None

    from triton.flagmega.codegen.cpp import renderer

    with pytest.raises(CodegenError, match="warp-specialized"):
        renderer._reject_unsupported_constructs(
            {**descriptor, "pipeline_schedule": {"stages": [{}]}}, module
        )


@requires_toolchain
def test_artifact_records_the_backend_and_verifies_its_library(tmp_path):
    """A C++ artifact's library is a hashed section, like its source."""

    artifact = write_artifact(
        _compiled(),
        tmp_path / "artifact",
        target=TARGET,
        emit_executable=True,
        codegen_backend="cpp",
    )
    manifest = json.loads((artifact / "artifact.json").read_text("utf-8"))

    assert manifest["codegen_backend"] == "cpp"
    sections = {value["name"]: value for value in manifest["sections"]}
    assert sections["codegen.source"]["kind"] == "codegen.cpp-source/v1"
    assert sections["codegen.library"]["kind"] == "codegen.cpp-library/v1"
    assert manifest["codegen"]["library_sha256"] == sections["codegen.library"]["sha256"]

    # load_artifact re-verifies every section hash.
    from triton.flagmega.artifacts import load_artifact

    reloaded, _ = load_artifact(artifact)
    assert reloaded["codegen_backend"] == "cpp"


@requires_toolchain
def test_rejects_a_tampered_library(tmp_path):
    artifact = write_artifact(
        _compiled(),
        tmp_path / "artifact",
        target=TARGET,
        emit_executable=True,
        codegen_backend="cpp",
    )
    library = artifact / "libflagmega_generated.so"
    library.write_bytes(library.read_bytes() + b"\0")

    from triton.flagmega.artifacts import load_artifact
    from triton.flagmega.errors import ArtifactError

    with pytest.raises(ArtifactError):
        load_artifact(artifact)


@requires_toolchain
def test_built_library_agrees_with_the_manifest_signature(tmp_path):
    """The library reports its own argument count and ABI version.

    This is what catches a library rebuilt from a different artifact: the
    Python side would otherwise pass the wrong number of pointers.
    """

    artifact = write_artifact(
        _compiled(),
        tmp_path / "artifact",
        target=TARGET,
        emit_executable=True,
        codegen_backend="cpp",
    )
    manifest = json.loads((artifact / "artifact.json").read_text("utf-8"))
    library = ctypes.CDLL(str(artifact / "libflagmega_generated.so"))
    library.flagmega_argument_count.restype = ctypes.c_int
    library.flagmega_abi_version.restype = ctypes.c_int

    expected = len(manifest["codegen"]["runtime_binding"]["signature"])
    assert library.flagmega_argument_count() == expected
    assert library.flagmega_abi_version() == 1


@pytest.mark.parametrize("op", ["math.add", "math.silu"])
@requires_toolchain
def test_executes_bit_exactly_through_the_artifact_runtime(tmp_path, op):
    torch = _iluvatar_device()
    artifact = write_artifact(
        _compiled(op),
        tmp_path / "artifact",
        target=TARGET,
        emit_executable=True,
        codegen_backend="cpp",
    )
    from triton.flagmega.runtime import load

    module = load(artifact, device="cuda:0")
    lhs = torch.randn(3, 17, dtype=torch.bfloat16, device="cuda")
    if op == "math.add":
        rhs = torch.randn(3, 17, dtype=torch.bfloat16, device="cuda")
        module.prepare(lhs, rhs)
        result = module.run(lhs, rhs)
        expected = lhs + rhs
    else:
        module.prepare(lhs)
        result = module.run(lhs)
        expected = (lhs.float() * torch.sigmoid(lhs.float())).to(torch.bfloat16)
    torch.cuda.synchronize()

    assert torch.equal(result, expected)
