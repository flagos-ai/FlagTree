"""Check SME directly through the vendor CUDA C++ toolchain, without Triton."""

import os
from pathlib import Path
import subprocess

import pytest


PROBE = Path(__file__).parent / "cuda" / "sme_contract_probe.cu"


@pytest.fixture(scope="module")
def sdk():
    root = Path(os.environ.get("COREX_HOME", "/usr/local/corex"))
    compiler = root / "bin" / "clang++"
    if not compiler.is_file():
        pytest.skip("requires the COREX CUDA C++ compiler")
    return root, compiler


def _compile(sdk, output, *flags):
    root, compiler = sdk
    result = subprocess.run(
        [str(compiler), "-x", "ivcore", "--offload-arch=ivcore11",
         f"--cuda-path={root}", "-std=c++17", "-O2", *flags,
         str(PROBE), "-o", str(output)],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_sme_cuda_cpp_shape_codegen(sdk, tmp_path):
    assembly = tmp_path / "sme_contract_probe.s"
    result = _compile(sdk, assembly, "--cuda-device-only", "-S")
    instructions = assembly.read_text()
    for shape in ("1x1b64", "1x4b64", "1x8b64", "4x1b64", "8x1b64", "16x1b64"):
        assert f"sl_sme_load_{shape} " in instructions
    assert "g2scnt(0)" in instructions
    assert "sl_sme_load_16x1b64_rowxfb16" in instructions
    assert "SME gOffset-Imm field is not aligned to 64bytes" in result.stderr


@pytest.fixture(scope="module")
def executable(sdk, tmp_path_factory):
    root, _ = sdk
    output = tmp_path_factory.mktemp("sme-cuda-cpp") / "sme_contract_probe"
    _compile(sdk, output, f"-L{root / 'lib64'}", "-lcudart",
             f"-Wl,-rpath,{root / 'lib64'}")
    return output


def test_sme_cuda_cpp_legal_runtime(executable):
    result = subprocess.run([str(executable), "legal"], capture_output=True,
                            text=True, timeout=60)
    if result.returncode == 77 and "GPU_UNAVAILABLE" in result.stderr:
        pytest.skip(result.stderr.strip())
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("PASS ") == 12
