"""Query the actual COREX driver rather than infer admission from ELF notes."""

import os
from pathlib import Path
import subprocess

import pytest
import triton
import triton.language as tl
from triton.compiler.compiler import ASTSource, GPUTarget


@pytest.fixture(scope="module")
def executable(tmp_path_factory):
    root = Path(os.environ.get("COREX_HOME", "/usr/local/corex"))
    compiler = root / "bin" / "clang++"
    if not compiler.is_file():
        pytest.skip("requires the COREX C++ compiler")
    output = tmp_path_factory.mktemp("kernel-resource-cpp") / "probe"
    source = Path(__file__).parent / "cuda" / "kernel_resource_probe.cpp"
    result = subprocess.run(
        [str(compiler), "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
         str(source), f"-I{root / 'include'}", f"-L{root / 'lib64'}", "-lcuda",
         f"-Wl,-rpath,{root / 'lib64'}", "-o", str(output)],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return output


@pytest.mark.parametrize("arguments", [
    [], ["--unknown"], ["kernel", "symbol", "0", "0"],
    ["kernel", "symbol", "-1", "0"], ["kernel", "symbol", "256", "-1"],
    ["kernel", "symbol", "256", "0x40"], ["kernel", "symbol", "2147483648", "0"],
])
def test_resource_probe_rejects_invalid_arguments(executable, arguments):
    result = subprocess.run([str(executable), *arguments], capture_output=True, text=True, timeout=30)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "GPU_UNAVAILABLE" not in result.stderr


def test_resource_probe_checks_input_before_driver(executable, tmp_path):
    missing = tmp_path / "missing.cubin"
    result = subprocess.run([str(executable), str(missing), "symbol", "3072", "18168"],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert "cannot read cubin" in result.stderr
    empty = tmp_path / "empty.cubin"
    empty.touch()
    result = subprocess.run([str(executable), str(empty), "symbol", "3072", "18168"],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert "cubin is empty" in result.stderr


def test_resource_probe_rejects_non_elf_before_driver(executable, tmp_path):
    invalid = tmp_path / "invalid.cubin"
    invalid.write_bytes(b"not an ELF cubin" * 8)
    result = subprocess.run([str(executable), str(invalid), "symbol", "3072", "18168"],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert "not a little-endian ELF64" in result.stderr
    assert "GPU_UNAVAILABLE" not in result.stderr


@triton.jit
def _resource_probe_kernel(out):
    tl.store(out, 42)


def test_resource_probe_kernel_queries_without_launch(executable, tmp_path):
    compiled = triton.compile(ASTSource(_resource_probe_kernel, signature={"out": "*i32"}),
                              target=GPUTarget("corex", 71, 64), options={"num_warps": 4})
    cubin = tmp_path / "kernel.cubin"
    cubin.write_bytes(compiled.asm["cubin"])
    result = subprocess.run([str(executable), str(cubin), compiled.name, "256", str(compiled.metadata.shared)],
                            capture_output=True, text=True, timeout=60)
    if result.returncode == 77 and "GPU_UNAVAILABLE" in result.stderr:
        pytest.skip(result.stderr.strip())
    assert result.returncode == 0, result.stdout + result.stderr
    fields = dict(line.split("=", 1) for line in result.stdout.splitlines())
    assert fields["query_only"] == "1"
    assert fields["module_loading"] == "0"
    assert fields["kernel"] == compiled.name
    assert int(fields["num_regs"]) > 0
    assert fields["requested_threads"] == "256"
    assert fields["threads_within_device"] == fields["threads_within_function"] == "1"
    assert int(fields["occupancy_blocks_per_sm"]) > 0
