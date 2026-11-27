# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega import pattern_match as pm


def _distributed_view_module():
    placement = fm.Placement((8,), "b", "b")
    tensor = fm.tensor_type("bfloat16", [1, 128])
    distributed = fm.DistributedType(
        tensor,
        (fm.SBP.broadcast(), fm.SBP.split_block_cyclic((0,), 16)),
        placement,
    )

    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="distributed", entry="main")

        def forward(self):
            source = self.input("source", tensor, id="source")
            view = fm.F.distributed.sharded_view(source, distributed, name="view")
            output = fm.F.distributed.boxing(view, tensor, name="output")
            self.function("main", (source,), (output,))

    return Graph().build()


def test_distributed_ops_are_real_f_calls_and_python_round_trip(tmp_path):
    module = fm.verify_module(_distributed_view_module())
    checkpoint = fm.emit_module(module, tmp_path / "distributed.py")
    source = checkpoint.read_text(encoding="utf-8")
    loaded = fm.load_module(checkpoint)

    assert "F.distributed.sharded_view(" in source
    assert "F.distributed.boxing(" in source
    assert "fm.SBP.split(" in source
    assert loaded == module
    assert loaded.semantic_hash == module.semantic_hash


def test_pattern_match_has_handwritten_distributed_functionals():
    module = _distributed_view_module()
    pattern = pm.F.distributed.is_boxing(
        pm.F.distributed.is_sharded_view(call_name="view"),
        call_name="boxing",
    )
    match = pm.try_match_root(module.node_map["output"], pattern, module)

    assert match is not None
    assert match["view"].id == "view"
    assert match["boxing"].id == "output"
