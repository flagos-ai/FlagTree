"""Regression tests for the enable_i64 pass option.

make_llir appends ``enable_i64=true`` to ``--convert-gpu-to-gcu`` whenever
``options.enable_i64`` is set, and the option is only meaningful to a toolkit
whose pass-options parser knows it. See
https://github.com/flagos-ai/FlagTree/issues/1233.
"""
import importlib
import importlib.util
import types

import pytest
import torch

import triton
import triton.language as tl

if importlib.util.find_spec("triton.backends.enflame") is None:
    import triton_gcu.triton

from triton.compiler.compiler import make_backend


def _backend_and_modules():
    """The active GCU backend, its compiler module, and the module's toolkit."""
    target = triton.runtime.driver.active.get_current_target()
    backend = make_backend(target)
    # The backend lives either in triton.backends.enflame or in triton_gcu,
    # depending on how the wheel was built, so ask the backend which one it is.
    compiler = importlib.import_module(type(backend).__module__)
    return backend, compiler, compiler.toolkit


@pytest.mark.parametrize("env, expected", [("0", True), ("1", False)])
def test_parse_options_is_idempotent(env, expected, monkeypatch):
    """Parsing the resolved options again must not change enable_i64.

    Triton parses the options twice -- JITFunction._pack_args parses the kernel's
    keyword arguments, then _do_compile hands the resolved dataclass back to
    compile() as a dict -- so the second parse has to agree with the first. It
    used to flip, because the environment default was derived only when the key
    was already present, and the resolved dict is exactly what has it present.
    """
    backend, _, _ = _backend_and_modules()
    monkeypatch.setenv("ENABLE_I64_CHECK", env)

    first = backend.parse_options(dict())
    assert first.enable_i64 is expected
    assert backend.parse_options(first.__dict__).enable_i64 is expected


def test_explicit_enable_i64_is_honoured(monkeypatch):
    """ENABLE_I64=False stays False, whatever ENABLE_I64_CHECK says."""
    backend, _, _ = _backend_and_modules()
    monkeypatch.setenv("ENABLE_I64_CHECK", "0")

    resolved = backend.parse_options({"ENABLE_I64": False})
    assert resolved.enable_i64 is False
    assert backend.parse_options(resolved.__dict__).enable_i64 is False


def _probe_a_toolkit(tmp_path, monkeypatch, toolkit, name, script):
    """Run the probe against a fake gcu-compiler-opt with the given script body."""
    bin_dir = tmp_path / name
    bin_dir.mkdir(exist_ok=True)
    tool = bin_dir / "gcu-compiler-opt"
    tool.write_text("#!/bin/sh\n" + script)
    tool.chmod(0o755)
    monkeypatch.setattr(toolkit, "TOOLKIT_PATH", str(bin_dir))
    toolkit.toolkit_supports_enable_i64.cache_clear()
    return toolkit.toolkit_supports_enable_i64()


def test_enable_i64_option_follows_the_toolkit(tmp_path, monkeypatch, capsys):
    """The option is dropped for a toolkit whose parser does not know it."""
    _, compiler, toolkit = _backend_and_modules()
    monkeypatch.delenv("TRITON_GCU_ENABLE_I64", raising=False)
    monkeypatch.setattr(compiler, "_warned_enable_i64_dropped", False)

    try:
        # what tops1.9.10 answers
        assert _probe_a_toolkit(tmp_path, monkeypatch, toolkit, "old",
                                "echo '<Pass-Options-Parser>: no such option enable_i64' >&2\nexit 1\n") is False
        # a toolkit that knows the option, on a module it does not mind
        assert _probe_a_toolkit(tmp_path, monkeypatch, toolkit, "new", "exit 0\n") is True
        # a toolkit that knows the option but rejects the empty module: the exit
        # status is not what the probe asks about
        assert _probe_a_toolkit(tmp_path, monkeypatch, toolkit, "empty",
                                "echo 'error: no gpu module' >&2\nexit 1\n") is True
    finally:
        toolkit.toolkit_supports_enable_i64.cache_clear()

    # ... and the string make_llir appends follows the probe. The warning is
    # printed once: a serving process compiles many kernels.
    asks_for_i64 = types.SimpleNamespace(enable_i64=True, arch="gcu300")
    monkeypatch.setattr(toolkit, "toolkit_supports_enable_i64", lambda: False)
    assert compiler._enable_i64_pass_option(asks_for_i64) == ""
    assert "does not support enable_i64" in capsys.readouterr().out
    assert compiler._enable_i64_pass_option(asks_for_i64) == ""
    assert capsys.readouterr().out == ""

    monkeypatch.setattr(toolkit, "toolkit_supports_enable_i64", lambda: True)
    assert compiler._enable_i64_pass_option(asks_for_i64) == " enable_i64=true"
    assert compiler._enable_i64_pass_option(types.SimpleNamespace(enable_i64=False, arch="gcu300")) == ""


@pytest.mark.parametrize("arch", ["gcu400", "gcu410"])
def test_only_gcu300_is_probed(arch, monkeypatch):
    """Later targets always know the option, so they are not probed for it."""
    _, compiler, toolkit = _backend_and_modules()

    def _unexpected():
        raise AssertionError(f"{arch} must not be probed for enable_i64")

    monkeypatch.setattr(toolkit, "toolkit_supports_enable_i64", _unexpected)
    assert compiler._enable_i64_pass_option(types.SimpleNamespace(enable_i64=True, arch=arch)) == " enable_i64=true"
    assert compiler._enable_i64_pass_option(types.SimpleNamespace(enable_i64=False, arch=arch)) == ""


def test_enable_i64_probe_can_be_overridden(monkeypatch):
    """TRITON_GCU_ENABLE_I64 answers for the probe when it is set."""
    _, _, toolkit = _backend_and_modules()
    monkeypatch.setenv("TRITON_GCU_ENABLE_I64", "0")
    toolkit.toolkit_supports_enable_i64.cache_clear()
    try:
        assert toolkit.toolkit_supports_enable_i64() is False
    finally:
        toolkit.toolkit_supports_enable_i64.cache_clear()


def test_i64_kernel_compiles_and_runs(device):
    """The reproducer from the issue, with the flag on and off.

    enable_i64 is emitted for every kernel, so a toolkit that cannot parse it
    failed every compile on tops1.9.10, whether or not the kernel used 64-bit
    integers.
    """

    @triton.jit
    def add_kernel(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr, ENABLE_I64: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask), mask=mask)

    # One program per BLOCK-wide tile. The reproducer in the issue used a single
    # program spanning four tiles and got garbage on gcu300 (768/1024
    # mismatched, values ~1e38 -- loads never landed); a grid of cdiv tiles is
    # the idiom every other kernel test in this suite uses, and it is not what
    # the option is about.
    n = 1024
    x = torch.randn(n, device=device, dtype=torch.float32)
    y = torch.randn(n, device=device, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(n, META['BLOCK']), )
    for flag in (False, True):
        out = torch.empty(n, device=device, dtype=torch.float32)
        add_kernel[grid](x, y, out, n, BLOCK=256, ENABLE_I64=flag)
        torch.testing.assert_close(out, x + y)
