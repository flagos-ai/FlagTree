# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Per-function checkpoints cannot collapse distinct valid IR symbols."""

from urllib.parse import unquote

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.diagnostics import DumpFlags, DumpManager


@pytest.mark.parametrize("names", [
    ("main", "decode_layer", "decode_layer.layout_1"),
    ("a/b", "a?b", "a_b", "a%2Fb"),
    ("worker", ".worker", "worker.", ".."),
    ("层1", "层2", "_1", "_2"),
])
def test_dump_filename_is_injective_and_directory_resume_preserves_symbols(tmp_path, names):
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    for index, name in enumerate(names):
        value = builder.var(f"arg_{index}", fm.tensor_type("float32", (8,)), id=f"arg_{index}")
        result = builder.call("math.silu", (value,), value.type, id=f"result_{index}")
        builder.function(name, (value,), (result,))
    module = fm.verify_module(builder.build(entry=names[0]))
    manager = DumpManager(tmp_path, DumpFlags.PASS_IR)
    for boundary in ("Before", "After"):
        dump = manager.root.dump_module(module, boundary, category=DumpFlags.PASS_IR)
        paths = [value.checkpoint for value in dump.functions]
        assert len(set(paths)) == len(names)
        assert all(value.parent == tmp_path / boundary for value in paths)
        assert all(not value.name.startswith(".") for value in paths)
        for name, path in zip(names, paths, strict=True):
            assert unquote(path.stem) == name
            assert path.with_suffix(".il").exists()
        assert fm.load_module(dump.directory) == module
    assert (tmp_path / "Before" / "main.py").exists() == ("main" in names)
