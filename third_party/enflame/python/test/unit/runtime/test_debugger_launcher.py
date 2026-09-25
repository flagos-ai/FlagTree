"""Check the debugger gateway and launch ABI without compiling device code."""

from contextlib import contextmanager
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.mark.parametrize("enabled,hidden", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("num_args", [0, 2])
def test_debugger_gateway_and_hidden_arguments(monkeypatch, enabled, hidden, num_args):
    backend = pytest.importorskip("triton.backends.enflame.backend")
    calls = []
    signatures = []
    control = object()
    user_args = tuple(object() for _ in range(num_args))
    metadata = SimpleNamespace(arch="gcu300", debug_enabled=enabled, debug_launch_hidden_arg=hidden)
    source = SimpleNamespace(constants={}, signature={i: "*i32" for i in range(num_args)})

    @contextmanager
    def gateway(kind, meta, grid, stream, launch_metadata, args):
        assert (kind, meta, grid, stream, launch_metadata, args) == ("gcu", metadata, (1, 2, 3), 17, None, user_args)
        calls.append("enter")
        try:
            yield (control, ) if hidden else ()
        finally:
            calls.append("finalize")

    def native_launch(*args, **kwargs):
        calls.append(args)
        return "launched"

    def generate(constants, signature, arch, **kwargs):
        signatures.append(dict(signature))
        return "source"

    flagtree = ModuleType("flagtree")
    flagtree._flagprism = SimpleNamespace(debugger_launch_context=gateway)
    monkeypatch.setitem(sys.modules, "flagtree", flagtree)
    monkeypatch.setattr(backend, "generate_launcher", generate)
    monkeypatch.setattr(backend, "compile_module_from_src", lambda *args: SimpleNamespace(launch=native_launch))
    monkeypatch.setattr(backend, "wrap_handle_tensordesc", lambda launch, signature: launch)
    launcher = backend.GcuLauncher(source, metadata)
    args = (1, 2, 3, 17, 0, (), None, None, None) + user_args
    assert launcher(*args) == "launched"
    expected_signature = dict(source.signature)
    if hidden:
        expected_signature[num_args] = "*i8"
    assert signatures == [expected_signature]
    native_args = args + ((control, ) if hidden else ())
    assert calls == (["enter", native_args, "finalize"] if enabled else [native_args])
