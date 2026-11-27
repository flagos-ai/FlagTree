# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.egraph import EGraph


def test_hash_consing_uses_canonical_ir_attribute_encoding():
    graph = EGraph()
    tensor = fm.tensor_type("float32", (2, 4))
    source = fm.Node("source", "builtin.var", (), tensor, attrs={"name": "source"})
    graph.add_node(source, original=True)
    boxed = fm.Node(
        "boxed",
        "distributed.boxing",
        ("source",),
        tensor,
        attrs={"new_type": tensor},
    )

    class_id = graph.add_node(boxed, original=True)

    assert graph.class_view(class_id).nodes[0].op == "distributed.boxing"


def test_same_spelled_variables_remain_distinct_identity_leaves():
    graph = EGraph()
    tensor = fm.tensor_type("float32", (4,))
    lhs = fm.Node("lhs", "builtin.var", (), tensor, attrs={"name": "value"})
    rhs = fm.Node("rhs", "builtin.var", (), tensor, attrs={"name": "value"})

    lhs_class = graph.add_node(lhs, original=True)
    rhs_class = graph.add_node(rhs, original=True)

    assert graph.find(lhs_class) != graph.find(rhs_class)


def test_nullary_constants_with_different_result_types_are_not_congruent():
    graph = EGraph()
    narrow = fm.Node(
        "narrow",
        "builtin.splat_const",
        (),
        fm.tensor_type("bfloat16", (8,)),
        attrs={"value": 0.0},
    )
    wide = fm.Node(
        "wide",
        "builtin.splat_const",
        (),
        fm.tensor_type("bfloat16", (32,)),
        attrs={"value": 0.0},
    )

    narrow_class = graph.add_node(narrow, original=True)
    wide_class = graph.add_node(wide, original=True)

    assert graph.find(narrow_class) != graph.find(wide_class)
