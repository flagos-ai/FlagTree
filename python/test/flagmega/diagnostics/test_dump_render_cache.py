# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import triton.flagmega.diagnostics.dump as dump_module
from triton.flagmega import ir as fm
from triton.flagmega.diagnostics import DumpFlags, DumpManager


def _two_function_module():
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    value_type = fm.tensor_type("float32", (8,))
    for name in ("main", "helper"):
        argument = builder.var(f"{name}_argument", value_type, id=f"{name}_argument")
        result = builder.call("math.silu", (argument,), value_type, id=f"{name}_result")
        builder.function(name, (argument,), (result,))
    return builder.build(entry="main")


def test_identical_module_boundaries_reuse_verified_function_render(tmp_path, monkeypatch):
    module = _two_function_module()
    manager = DumpManager(tmp_path, DumpFlags.PASS_IR)
    calls = []
    original = dump_module.module_source

    def observe(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(dump_module, "module_source", observe)
    manager.root.dump_module(module, "Before", category=DumpFlags.PASS_IR)
    manager.root.dump_module(module, "Repeated", category=DumpFlags.PASS_IR)

    assert len(calls) == len(module.functions)
    for function in ("main", "helper"):
        assert (tmp_path / "Before" / f"{function}.py").read_bytes() == (
            tmp_path / "Repeated" / f"{function}.py"
        ).read_bytes()
        assert fm.load_module(tmp_path / "Repeated").semantic_hash == module.semantic_hash
