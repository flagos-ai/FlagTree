# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.passes.tir import materialize_memory_synchronization

from .test_materialize_memory_synchronization import _plan
from ..execution.helpers import scheduled_nested_module


def test_barrier_is_python_ir_and_readable_script(tmp_path):
    module = materialize_memory_synchronization(
        scheduled_nested_module(), _plan()
    )
    path = fm.emit_module(module, tmp_path / "synchronized.py")
    source = path.read_text(encoding="utf-8")
    script = path.with_suffix(".script").read_text(encoding="utf-8")

    assert "T.barrier(" in source
    assert "T.synchronization_range(" in source
    assert "T.Barrier(Chip" in script
    assert fm.load_module(path).semantic_hash == module.semantic_hash
