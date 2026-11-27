# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.diagnostics import DumpFlags, DumpManager
from triton.flagmega.errors import IRVerificationError

from .helpers import scheduled_nested_module


def test_execution_schedule_is_real_editable_python_ir(tmp_path):
    module = scheduled_nested_module()
    path = fm.emit_module(module, tmp_path / "scheduled.py")
    source = path.read_text(encoding="utf-8")
    script = path.with_suffix(".script").read_text(encoding="utf-8")

    assert "self.execution_function(" in source
    assert "T.execution_function(" in source
    assert "T.prim_function_call(" in source
    assert "memory_pools=" in source
    assert 'T.ExecutionFunc("main"' in script
    assert "T.Call('nested', @worker" in script
    assert "DependsOn: ['prepared']" in script
    loaded = fm.load_module(path)
    assert loaded.semantic_hash == module.semantic_hash
    assert (
        fm.execution_calls_of(loaded.execution_function_map["main"])[1].memory_pools
        == fm.execution_calls_of(module.execution_function_map["main"])[1].memory_pools
    )


def test_execution_schedule_function_directory_is_dumpable_and_mergeable(tmp_path):
    module = scheduled_nested_module()
    dumper = DumpManager(tmp_path, DumpFlags.PASS_IR).root

    dumped = dumper.dump_module(module, "After", category=DumpFlags.PASS_IR)

    assert dumped is not None
    assert {path.stem for path in dumped.directory.glob("*.py")} == {
        "main", "worker",
    }
    assert fm.load_module(dumped.directory) == module


def test_execution_schedule_mismatch_is_rejected_outside_dump_fragment():
    module = scheduled_nested_module()
    worker = module.function_map["worker"]
    mismatched = replace(module, functions=(worker,), entry="worker")

    with pytest.raises(
        IRVerificationError,
        match="ExecutionFunction set must exactly cover graph functions",
    ):
        fm.verify_module(mismatched)


def test_execution_dump_fragment_requires_complete_schedule_signatures():
    module = scheduled_nested_module()
    worker = module.function_map["worker"]
    forged_fragment = replace(
        module,
        functions=(worker,),
        entry="worker",
        metadata={
            **module.metadata,
            "_dump_function_fragment": True,
            "function_signatures": {"worker": {}},
        },
    )

    with pytest.raises(
        IRVerificationError,
        match="ExecutionFunction set must exactly cover graph functions",
    ):
        fm.verify_module(forged_fragment)
