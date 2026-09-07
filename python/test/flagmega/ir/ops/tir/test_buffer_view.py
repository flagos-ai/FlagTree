# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm


def test_buffer_view_is_editable_python_ir_and_preserves_distributed_type(tmp_path):
    tensor = fm.tensor_type("int32", (16,))
    distributed = fm.DistributedType(
        tensor,
        (fm.SBP.broadcast(),),
        fm.Placement((2, 4), "yx", "bb"),
    )

    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="bufferized_tir", stage="bufferized_tir", entry="main")

        def forward(self):
            source = self.input("source", tensor, id="source")
            view = fm.F.tir.buffer_view(source, distributed, name="view")
            self.function("main", (source,), (view,))

    module = Graph().build()
    path = fm.emit_module(module, tmp_path / "view.py")
    loaded = fm.load_module(path)

    assert loaded.node_map["view"].op == "tir.buffer_view"
    assert loaded.node_map["view"].type == distributed
    assert loaded.semantic_hash == module.semantic_hash
