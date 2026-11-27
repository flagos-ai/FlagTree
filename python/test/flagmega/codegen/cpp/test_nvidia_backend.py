# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""CPU checks for NVIDIA C++ emission, admission, and its scratch ABI."""

import ctypes
from pathlib import Path
import shutil
import subprocess

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.codegen.cpp.build import nvidia_toolchain, iluvatar_bi_v150_toolchain
from triton.flagmega.codegen.cpp.renderer import describe_cpp_package, render_cpp_source_package
from triton.flagmega.compiler import Compiler
from triton.flagmega.errors import ArtifactError, CodegenError, RuntimeContractError
from triton.flagmega.options import CompileOptions
from triton.flagmega.runtime.cpp_module import _CppPackageLibrary, _CppLaunchMixin


def test_nvcc_driver_uses_cuda_target_and_host_pic_flags(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_HOME", str(tmp_path))
    toolchain = nvidia_toolchain("nvidia-sm120")
    command = toolchain.command(Path("kernel.cpp"), Path("kernel.so"))
    assert command[0] == str(tmp_path / "bin/nvcc")
    assert command[1:4] == ("-x", "cu", "-arch=sm_120")
    assert "-DFTT_ARCH_NVIDIA" in command
    assert command[command.index("-fPIC") - 1] == "-Xcompiler"
    assert "ivcore" not in command
    corex = iluvatar_bi_v150_toolchain().command(Path("kernel.cpp"), Path("kernel.so"))
    assert "--cuda-gpu-arch=ivcore11" in corex
    assert "-DFTT_ARCH_ILUVATAR" in corex
    assert "-Xcompiler" not in corex


@pytest.fixture(scope="module")
def compiled_nvidia():
    from python.test.flagmega.codegen.cpp.test_cpp_package import _elementwise_module
    return Compiler(CompileOptions(target="nvidia-sm120", bufferize_opt_level="fast")).compile(
                                       _elementwise_module()).module


@pytest.fixture(scope="module")
def compiled_distributed_nvidia():
    placement = fm.Placement((2, 2), "yx", "bb")
    class Graph(fm.Module):
        def forward(self):
            tensor = fm.tensor_type("bfloat16", (4, 64))
            distributed = fm.DistributedType(tensor, (fm.SBP.broadcast(),
                                                     fm.SBP.split_contiguous((0,), 2)), placement)
            lhs = self.input("lhs", tensor)
            rhs = self.input("rhs", tensor)
            left = fm.F.distributed.boxing(lhs, distributed)
            right = fm.F.distributed.boxing(rhs, distributed)
            output = fm.F.math.add(left, right, name="output")
            self.function("main", (lhs, rhs), (output,))
    graph = Graph(dialect="high_level", stage="distributed", entry="main",
                  metadata={"auto_distribution": {"placement": placement.to_data()}}).build()
    return Compiler(CompileOptions(target="nvidia-sm120", bufferize_opt_level="fast")).compile(graph).module


def test_cuda_descriptor_uses_target_geometry_and_explicit_scratch(compiled_distributed_nvidia, monkeypatch):
    monkeypatch.delenv("FLAGMEGA_CPP_BLOCK_THREADS", raising=False)
    descriptor = describe_cpp_package(compiled_distributed_nvidia)
    assert descriptor["warp_size"] == 32
    assert descriptor["block_threads"] == 32 * descriptor["num_warps"]
    assert descriptor["launch_abi_version"] == 2
    assert descriptor["launch_resource_limits"]["compute_capability"] == [12, 0]
    assert descriptor["distributed_entry"]
    scratch = descriptor["launch_scratch"]
    assert scratch["count"] == descriptor["grid_barrier_counter_slots"]
    assert scratch["nbytes"] == scratch["count"] * 4
    assert scratch["reset"] == "per-launch-stream"


@pytest.mark.parametrize("threads", ["0", "33", "2048", "not-an-int"])
def test_invalid_cuda_block_size_fails_before_emission(compiled_nvidia, monkeypatch, threads):
    monkeypatch.setenv("FLAGMEGA_CPP_BLOCK_THREADS", threads)
    with pytest.raises(CodegenError, match="block_threads|must be an integer"):
        describe_cpp_package(compiled_nvidia)


def test_cuda_source_checks_occupancy_and_forwards_runtime_owned_counters(compiled_distributed_nvidia, tmp_path):
    render_cpp_source_package(compiled_distributed_nvidia, tmp_path)
    source = (tmp_path / "generated_kernels.cpp").read_text()
    assert "cudaFuncGetAttributes" in source
    assert "cudaOccupancyMaxActiveBlocksPerMultiprocessor" in source
    assert "cudaErrorCooperativeLaunchTooLarge" in source
    assert "cudaLaunchCooperativeKernel" in source
    assert "unsigned int *ftt_grid_barrier" in source
    assert "cudaMemsetAsync(scratch, 0," in source
    assert "launch_arguments[" in source and "= &scratch;" in source
    assert "_ftt_grid_barrier_counters" not in source
    assert "cudaStreamIsCapturing" not in source
    assert "const ftt::dim_t block_index = ftt::arch::block_id();" in source


def test_cpp_runtime_registered_for_cuda_targets():
    from triton.flagmega.runtime.loader import package_registry
    from triton.flagmega.runtime.cpp_module import create_cpp_tir_runtime
    for target in ("nvidia-sm89", "nvidia-sm90", "nvidia-sm120", "iluvatar-bi-v150"):
        assert package_registry.resolve("tir_call_graph_cpp/v1", target).factory is create_cpp_tir_runtime


def test_cpp_readonly_loading_preserves_shared_rdata_keyword():
    class Base:
        def _load_rdata(self, torch, device, expected_nbytes, *, shared_rdata=None):
            assert shared_rdata is sentinel
            return expected_nbytes
    class Runtime(_CppLaunchMixin, Base):
        codegen = {}
    sentinel = object()
    assert Runtime()._load_rdata(None, "cuda:0", 16, shared_rdata=sentinel) == 16


@pytest.fixture
def fake_cuda_library(tmp_path):
    if shutil.which("c++") is None:
        pytest.skip("A host C++ compiler is required")
    source = tmp_path / "fake.cpp"
    source.write_text('''
extern "C" int flagmega_abi_version() { return 2; }
extern "C" int flagmega_argument_count() { return 1; }
static void *saved_scratch;
extern "C" int flagmega_launch(void **args, void *stream, void *scratch) {
    saved_scratch = scratch; return 0;
}
extern "C" void *get_scratch() { return saved_scratch; }
static int admission_status;
extern "C" void set_status(int value) { admission_status = value; }
extern "C" int flagmega_resource_info(int *values) {
    int report[] = {36, 32, 4096, 0, 2, 1024};
    for (int index = 0; index < 6; ++index) values[index] = report[index];
    return admission_status;
}
''')
    library = tmp_path / "fake.so"
    subprocess.run(["c++", "-shared", "-fPIC", str(source), "-o", str(library)], check=True)
    return library


def test_cpp_v2_library_validates_admission_and_passes_separate_scratch(fake_cuda_library):
    library = _CppPackageLibrary(fake_cuda_library, expected_arguments=1, abi_version=2,
                                 resource_admission=True)
    report = library.validate_resources()
    assert report["available_sm_count"] == 36 and report["resident_blocks_per_sm"] == 2
    library.launch((0x1000,), 0x2000, scratch=0x3000)
    get_scratch = library._library.get_scratch
    get_scratch.restype = ctypes.c_void_p
    assert get_scratch() == 0x3000
    library._library.set_status(720)  # CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE
    with pytest.raises(RuntimeContractError, match="resource admission.*720"):
        library.validate_resources()


def test_cpp_v1_runtime_rejects_v2_library(fake_cuda_library):
    with pytest.raises(ArtifactError, match="ABI version 2"):
        _CppPackageLibrary(fake_cuda_library, expected_arguments=1)


@pytest.mark.parametrize("call,message", [
    ({"family": "matmul_glu", "weight_encoding": "IQ4_XS", "gate_clamp_max": "1.0"}, "projection clamps"),
    ({"family": "gdn_recurrent", "b_weight_encoding": "Q4_K", "projection_dim": 128}, "256-element blocks"),
    ({"family": "embedding", "weight_encoding": "unsupported"}, "weight encoding"),
])
def test_unsupported_numerical_contracts_fail_closed(compiled_nvidia, call, message):
    from triton.flagmega.codegen.cpp.renderer import _reject_unsupported_constructs
    with pytest.raises(CodegenError, match=message):
        _reject_unsupported_constructs({"render_calls": [call]}, compiled_nvidia)


def test_runtime_instances_own_separate_scratch_and_preserve_it_until_close(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    import sys
    allocations = []
    def allocate(shape, **kwargs):
        tensor = SimpleNamespace(data_ptr=lambda: 0x1000 * (len(allocations) + 1))
        # Freeze the address rather than consulting the later allocation count.
        address = 0x1000 * (len(allocations) + 1)
        tensor.data_ptr = lambda: address
        allocations.append(tensor)
        return tensor
    torch = SimpleNamespace(cuda=SimpleNamespace(device=lambda device: nullcontext()),
                            empty=allocate, int32="int32")
    monkeypatch.setitem(sys.modules, "torch", torch)
    launches = []
    library = SimpleNamespace(validate_resources=lambda: {},
                              launch=lambda pointers, stream, scratch: launches.append((stream, scratch)))
    class Base:
        def _require_loaded(self): pass
        def _require_prepared(self): assert self.prepared
        def _validate_external_arguments(self, arguments): pass
        def _mark_prepared(self): pass
        def _release_resources(self): self._prepared = None
    class Runtime(_CppLaunchMixin, Base):
        def __init__(self):
            self._prepared = None
            self.prepare_count = 0
            self._device = "cuda:0"
            self._pool_values = {}
            self.runtime_binding = {"pools": ()}
            self.codegen = {"grid": [4, 1, 1], "block_threads": 128, "distributed_entry": True,
                            "grid_barrier_counter_slots": 1,
                            "launch_scratch": {"kind": "grid-barrier-counters/v1", "count": 1,
                                               "nbytes": 4, "dtype": "int32", "reset": "per-launch-stream"}}
        def _cpp_library(self): return library
    first, second = Runtime(), Runtime()
    first._prepare_external(())
    second._prepare_external(())
    assert first._cpp_scratch is not second._cpp_scratch
    assert len(allocations) == 2
    first._prepare_external(())
    assert len(allocations) == 2
    first._launch_external((), stream=17)
    second._launch_external((), stream=18)
    assert launches == [(17, 0x1000), (18, 0x2000)]
    with pytest.raises(RuntimeContractError, match="first CUDA stream"):
        first._launch_external((), stream=18)
    first._release_resources()
    assert not first.prepared and first._cpp_scratch is None


def test_normal_cuda_artifact_builds_and_verifies_without_device_execution(compiled_distributed_nvidia, tmp_path):
    from triton.flagmega.codegen.cpp.build import toolchain_available
    if not toolchain_available("nvidia-sm120"):
        pytest.skip("NVCC is required for this small compilation test")
    from triton.flagmega.artifacts import write_artifact, load_artifact
    artifact = write_artifact(compiled_distributed_nvidia, tmp_path / "artifact", target="nvidia-sm120",
                              emit_executable=True, codegen_backend="cpp")
    manifest, _ = load_artifact(artifact)
    assert manifest["codegen_backend"] == "cpp"
    assert manifest["codegen"]["launch_scratch"]["nbytes"] > 0
    library = _CppPackageLibrary(artifact / manifest["codegen"]["library"],
                                 expected_arguments=len(manifest["codegen"]["runtime_binding"]["signature"]),
                                 abi_version=2, resource_admission=True)
    assert library.abi_version == 2  # Query ABI only; no CUDA resource query or launch.


def test_cpp_register_cap_default_command_and_explicit_nvcc_flag(monkeypatch, tmp_path):
    from triton.flagmega.codegen.cpp.build import FTT_INCLUDE_DIR
    monkeypatch.setenv("CUDA_HOME", str(tmp_path))
    tc = nvidia_toolchain("nvidia-sm120")
    source, library = Path("kernel.cpp"), Path("kernel.so")
    baseline = (str(tmp_path / "bin/nvcc"), "-x", "cu", "-arch=sm_120",
                "-std=c++20", "-O3", "-Xcompiler", "-fPIC", "-shared",
                "-DFTT_ARCH_NVIDIA", f"-I{FTT_INCLUDE_DIR}",
                str(source), "-o", str(library))
    assert tc.command(source, library) == baseline
    assert tc.command(source, library, max_registers_per_thread=None) == baseline
    capped = tc.command(source, library, max_registers_per_thread=168)
    assert capped.count("--maxrregcount=168") == 1
    assert tuple(x for x in capped if x != "--maxrregcount=168") == baseline
    assert not any("fmad" in x for x in capped)
    with pytest.raises(CodegenError, match="NVIDIA"):
        iluvatar_bi_v150_toolchain().command(source, library, max_registers_per_thread=168)


@pytest.mark.parametrize("value", [True, False, 0, -1, 256, 168.0, "168", "bad"])
def test_cpp_register_cap_invalid_value_fails_before_build(monkeypatch, tmp_path, value):
    from triton.flagmega.codegen.cpp import build
    monkeypatch.setattr(build, "resolve_toolchain", lambda target: pytest.fail("compiler resolved"))
    with pytest.raises(CodegenError, match="integer between 1 and 255"):
        build.build_cpp_package(tmp_path / "missing.cpp", tmp_path / "new/out.so",
                                target="nvidia-sm120", max_registers_per_thread=value)
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("backend,target,executable", [
    ("triton", "nvidia-sm120", True), ("cpp", "iluvatar-bi-v150", True),
    ("cpp", "nvidia-sm120", False), ("cpp", "unknown", True),
])
def test_cpp_register_cap_artifact_rejects_before_verify_pack_or_output(
        monkeypatch, tmp_path, backend, target, executable):
    from triton.flagmega.artifacts import module as artifacts
    monkeypatch.setattr(artifacts, "verify_module", lambda module: pytest.fail("module verified"))
    monkeypatch.setattr(artifacts, "pack_rdata", lambda *args, **kwargs: pytest.fail("rdata packed"))
    with pytest.raises(CodegenError, match="executable NVIDIA"):
        artifacts.write_artifact(object(), tmp_path / "new", target=target,
                                 codegen_backend=backend, emit_executable=executable,
                                 cpp_max_registers=168)
    assert not (tmp_path / "new").exists()


def test_cpp_register_cap_package_rejects_before_render(tmp_path):
    from triton.flagmega.codegen.cpp.package import render_cpp_package_artifact
    with pytest.raises(CodegenError, match="NVIDIA"):
        render_cpp_package_artifact(object(), tmp_path / "new", target="iluvatar-bi-v150",
                                    cpp_max_registers=168)
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("cap", [None, 168])
def test_cpp_register_cap_normal_nvcc_artifact_integrity(compiled_nvidia, tmp_path, cap):
    import hashlib
    from triton.flagmega.artifacts import write_artifact, load_artifact
    from triton.flagmega.errors import ArtifactError
    from triton.flagmega.codegen.cpp.build import toolchain_available
    if not toolchain_available("nvidia-sm120"):
        pytest.skip("NVCC required; no device execution")
    output = write_artifact(compiled_nvidia, tmp_path / "artifact", target="nvidia-sm120",
                            emit_executable=True, codegen_backend="cpp", cpp_max_registers=cap)
    manifest, loaded = load_artifact(output)
    codegen = manifest["codegen"]
    assert loaded.semantic_hash == compiled_nvidia.semantic_hash
    library_path = output / codegen["library"]
    assert codegen["library_sha256"] == hashlib.sha256(library_path.read_bytes()).hexdigest()
    assert codegen["build_options"] == {"max_registers_per_thread": cap}
    provenance = codegen["build_provenance"]
    if cap is None:
        assert not any(arg.startswith("--maxrregcount") for arg in provenance["argv"])
    else:
        assert provenance["argv"].count("--maxrregcount=168") == 1
    assert "-O3" in provenance["argv"] and "-arch=sm_120" in provenance["argv"]
    assert provenance["actual_exit_code"] == 0
    assert "nvcc" in provenance["compiler"]["version"]
    assert provenance["compiler"]["sha256"] == hashlib.sha256(
        Path(provenance["compiler"]["path"]).read_bytes()).hexdigest()
    # ABI query is host-only; no CUDA resource/device query or launch.
    library = _CppPackageLibrary(library_path,
        expected_arguments=len(codegen["runtime_binding"]["signature"]),
        abi_version=codegen["launch_abi_version"])
    assert library.abi_version == codegen["launch_abi_version"]
    with library_path.open("ab") as stream:
        stream.write(b"integrity-test")
    with pytest.raises(ArtifactError):
        load_artifact(output)
