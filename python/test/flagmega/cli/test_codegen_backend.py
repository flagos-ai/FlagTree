# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
import pytest

from triton.flagmega import cli


@pytest.mark.parametrize("command", ["compile", "resume", "replay"])
def test_executable_commands_select_cpp_explicitly(command):
    args = cli._parser().parse_args([command, "--input", "input.py", "--output", "artifact",
                                     "--target", "nvidia-sm120", "--codegen-backend", "cpp"])
    assert args.codegen_backend == "cpp"
    defaults = cli._parser().parse_args([command, "--input", "input.py", "--output", "artifact"])
    assert defaults.codegen_backend == "triton"


def test_cli_passes_backend_to_artifact_writer(monkeypatch, tmp_path):
    from types import SimpleNamespace
    source = SimpleNamespace(metadata={}, semantic_hash="source")
    selected = SimpleNamespace(metadata={}, semantic_hash="selected")
    monkeypatch.setattr(cli, "load_module", lambda path: source)
    fake_compiler = SimpleNamespace(compile=lambda source, stop_after: SimpleNamespace(module=selected, reports=[]),
                                    options=SimpleNamespace(effective_dump_flags=0),
                                    diagnostics=SimpleNamespace(dumper=None))
    monkeypatch.setattr(cli, "Compiler", lambda options: fake_compiler)
    monkeypatch.setattr(cli, "_summary", lambda module: {})
    written = {}
    def write(module, output, **kwargs):
        written.update(kwargs)
        return output
    monkeypatch.setattr(cli, "write_artifact", write)
    assert cli.main(["compile", "--input", "input.py", "--output", str(tmp_path),
                     "--codegen-backend", "cpp", "--target", "nvidia-sm120", "--emit-executable"]) == 0
    assert written["codegen_backend"] == "cpp" and written["emit_executable"]


@pytest.mark.parametrize("command", ["compile", "resume", "replay"])
@pytest.mark.parametrize("extra", [
    [], ["--codegen-backend", "cpp"],
    ["--codegen-backend", "cpp", "--emit-executable", "--target", "iluvatar-bi-v150"],
])
def test_cpp_register_cap_cli_rejects_before_input_or_compiler(monkeypatch, tmp_path, command, extra):
    monkeypatch.setattr(cli, "load_module", lambda path: pytest.fail("input loaded"))
    monkeypatch.setattr(cli, "Compiler", lambda options: pytest.fail("compiler created"))
    output = tmp_path / "artifact"
    assert cli.main([command, "--input", "missing.py", "--output", str(output),
                     "--cpp-max-registers", "168", *extra]) != 0
    assert not output.exists()


@pytest.mark.parametrize("value", ["0", "256", "-1", "true", "168.5"])
def test_cpp_register_cap_cli_invalid_values(tmp_path, value):
    argv = ["compile", "--input", "missing.py", "--output", str(tmp_path / "new"),
            "--cpp-max-registers", value, "--codegen-backend", "cpp", "--emit-executable"]
    if value in ["true", "168.5"]:
        with pytest.raises(SystemExit) as error:
            cli.main(argv)
        assert error.value.code == 2
    else:
        assert cli.main(argv) != 0
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("command", ["compile", "resume", "replay"])
def test_cpp_register_cap_real_cli_normal_nvcc_propagation(tmp_path, command):
    from triton.flagmega.codegen.cpp.build import toolchain_available
    if not toolchain_available("nvidia-sm120"):
        pytest.skip("NVCC required; no device execution")
    from python.test.flagmega.codegen.cpp.test_cpp_package import _elementwise_module
    from triton.flagmega.compiler import Compiler
    from triton.flagmega.options import CompileOptions
    from triton.flagmega.ir import emit_module
    from triton.flagmega.artifacts import load_artifact
    module = Compiler(CompileOptions(target="nvidia-sm120", bufferize_opt_level="fast",
                                    codegen_backend="cpp")).compile(
        _elementwise_module()).module
    source = emit_module(module, tmp_path / "input.py")
    output = tmp_path / "artifact"
    assert cli.main([command, "--input", str(source), "--output", str(output),
                     "--codegen-backend", "cpp", "--target", "nvidia-sm120",
                     "--emit-executable", "--cpp-max-registers", "168"]) == 0
    manifest, loaded = load_artifact(output)
    assert loaded.semantic_hash == module.semantic_hash
    assert manifest["codegen"]["build_options"] == {"max_registers_per_thread": 168}
    assert manifest["codegen"]["build_provenance"]["argv"].count("--maxrregcount=168") == 1
