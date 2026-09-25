"""Compiler compatibility boundaries; SDK execution is mocked, no device required."""

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def compiler():
    return pytest.importorskip("triton.backends.enflame.compiler")


@pytest.mark.parametrize("arch,mode,message,retry", [
    ("gcu300", "debugger|config={}", "ld.lld: error: relocation R_X against fabs(float)", True),
    ("gcu300", "", "ld.lld: error: relocation R_X against fabs(float)", False),
    ("gcu300", "profiler", "ld.lld: error: relocation R_X against fabs(float)", False),
    ("gcu400", "debugger", "ld.lld: error: relocation R_X against fabs(float)", False),
    ("gcu300", "debugger", "other compiler error", False),
    ("gcu300", "debugger", "ld.lld: error: relocation R_X against other(float)", False),
])
def test_fabs_fallback_boundaries(compiler, monkeypatch, tmp_path, arch, mode, message, retry):
    user_lib = tmp_path / "user.bc"
    user_lib.write_bytes(b"user library")
    options = SimpleNamespace(arch=arch, instrumentation_mode=mode, max_shared=10, max_local=10, max_dsm=10,
                              extern_libs=(("user", str(user_lib)), ))
    mod = ('module attributes {gcu.shared_memory_size = 0, gcu.local_memory_size = 0, '
           'gcu.dsm_memory_size = 0, targets = [#gcu.target<arch = "gcu300", link = ["user.bc"]>]} {}')
    metadata = {"tle_raw": False}
    calls = []
    original_error = RuntimeError(message)

    def compile_ir(ir, *args):
        calls.append(ir)
        if len(calls) == 1:
            raise original_error
        output = next(arg.split("=", 1)[1] for arg in args if arg.startswith("--output="))
        Path(output).write_bytes(b"compiled")

    def link_ir(ir, *args):
        assert "targets =" not in ir
        assert f" l={user_lib}" in args[0]
        compat = Path(args[0].split(" l=")[-1]).read_text()
        assert "@_Z4fabsf" in compat
        assert "volatile" in compat
        return "linked IR"

    monkeypatch.setattr(compiler.toolkit, "compile", compile_ir)
    monkeypatch.setattr(compiler.toolkit, "gcu_compiler_opt", link_ir)
    if retry:
        assert compiler.make_fatbin(mod, metadata, options) == b"compiled"
        assert metadata["debug_math_compat"] == "scalar_fabs"
        assert calls == [mod, "linked IR"]
    else:
        with pytest.raises(RuntimeError) as error:
            compiler.make_fatbin(mod, metadata, options)
        assert error.value is original_error
        assert calls == [mod]
        assert "debug_math_compat" not in metadata


@pytest.mark.parametrize("mode,payload,enabled,expected", [
    ("debugger|config={\"debug_record_level\":1}", 16, False, True),
    ("debugger", 0, False, False),
    ("", 16, False, False),
    ("profiler", 16, False, False),
    ("debugger", 16, True, True),
])
def test_payload_i64_options_do_not_mutate_input(compiler, mode, payload, enabled, expected):

    @dataclass(frozen=True)
    class Options:
        instrumentation_mode: str
        enable_i64: bool

    options = Options(mode, enabled)
    metadata = {"debug_full_dump_payload_bytes_per_instance": payload}
    result = compiler._debug_codegen_options(options, metadata)
    assert result.enable_i64 is expected
    assert options.enable_i64 is enabled
    if expected != enabled:
        assert result is not options
        assert metadata["enable_i64"] is True
    else:
        assert result is options
        assert "enable_i64" not in metadata
