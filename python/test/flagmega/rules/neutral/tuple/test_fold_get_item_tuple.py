# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.rules import DataflowRewriter
from triton.flagmega.rules.neutral import fold_get_item_tuple_rule


class _LiteralTupleProjection(fm.Module):
    def __init__(self):
        super().__init__(dialect="ntt", stage="packed", entry="main")

    def forward(self):
        lhs = self.input("lhs", fm.tensor_type("float32", (2, 3)), id="lhs")
        rhs = self.input("rhs", fm.tensor_type("bfloat16", (5,)), id="rhs")
        pair = fm.F.builtin.tuple(lhs, rhs, name="pair")
        projected = fm.F.tensors.get_item(pair, 1, name="projected")
        self.function("main", (lhs, rhs), (projected,))


def test_get_item_of_literal_tuple_redirects_to_selected_field():
    rewritten = DataflowRewriter((fold_get_item_tuple_rule(),)).rewrite(
        _LiteralTupleProjection().build()
    )

    assert rewritten.function_map["main"].outputs == ("rhs",)
    assert tuple(node.id for node in rewritten.nodes) == ("lhs", "rhs")


def test_get_item_of_non_literal_tuple_is_not_folded():
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="ntt", stage="packed", entry="main")

        def forward(self):
            source = self.input(
                "source",
                fm.TupleType((
                    fm.tensor_type("float32", (2,)),
                    fm.tensor_type("float32", (3,)),
                )),
                id="source",
            )
            projected = fm.F.tensors.get_item(source, 0, name="projected")
            self.function("main", (source,), (projected,))

    module = Graph().build()

    rewritten = DataflowRewriter((fold_get_item_tuple_rule(),)).rewrite(module)

    assert rewritten.semantic_hash == module.semantic_hash
