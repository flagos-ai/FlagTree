# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A redirect must rebuild only the nodes that name the redirected value.

`_redirect` renames one value across the module. Reconstructing every node to
do it re-canonicalizes each node's attrs and metadata, and a rewrite sweep
redirects once per rule hit, so the untouched majority dominated the cost: on
the 45-layer GLM-5.3 graph a single redirect rebuilt 2569 nodes to rename a
value one node referenced, and
`PostFunctionBoundaryPackPropagation` spent 1392 s there.

Rebuilding selectively has to stay observationally identical, including for a
value with wide fan-out and for function parameters and outputs.
"""

from dataclasses import replace

from triton.flagmega import ir as fm
from triton.flagmega.rules.rewriter import _redirect


def _module():
    """A graph where one value feeds many consumers and a function output."""

    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    source = builder.var("source", fm.tensor_type("float32", (4, )), id="source")
    target = builder.var("target", fm.tensor_type("float32", (4, )), id="target")
    # Several consumers of `source`, plus nodes that never mention it.  Built
    # through the builder's node API so the fixture needs no Module context.
    consumers = tuple(
        builder.node(
            op="tensors.cast",
            type=fm.tensor_type("float32", (4, )),
            inputs=(source, ),
            id=f"consumer_{index}",
            attrs={"dtype": "float32"},
        )
        for index in range(6)
    )
    bystanders = tuple(
        builder.node(
            op="tensors.cast",
            type=fm.tensor_type("float32", (4, )),
            inputs=(target, ),
            id=f"bystander_{index}",
            attrs={"dtype": "float32"},
        )
        for index in range(6)
    )
    builder.function("main", (source, target), (*consumers, *bystanders, source))
    return builder.build(entry="main")


def _redirect_rebuilding_everything(module, source, target):
    """The previous implementation, kept here as the equivalence oracle."""

    nodes = tuple(
        replace(node, inputs=tuple(target if v == source else v for v in node.inputs))
        for node in module.nodes
        if node.id != source
    )
    functions = tuple(
        replace(
            function,
            parameters=tuple(target if v == source else v for v in function.parameters),
            outputs=tuple(target if v == source else v for v in function.outputs),
        )
        for function in module.functions
    )
    points = tuple(
        replace(point, owner=target) if point.owner == source else point
        for point in module.selection_points
    )
    return replace(module, nodes=nodes, functions=functions, selection_points=points)


def test_selective_redirect_matches_rebuilding_every_node():
    """The targeted rebuild must be observationally identical.

    Compares nodes, functions, selection points and the semantic hash, so a
    divergence in canonicalized attrs or metadata would show up rather than
    hiding behind equal ids.
    """

    module = _module()
    expected = _redirect_rebuilding_everything(module, "source", "target")
    actual = _redirect(module, "source", "target")

    assert actual.nodes == expected.nodes
    assert actual.functions == expected.functions
    assert actual.selection_points == expected.selection_points
    assert actual.semantic_hash == expected.semantic_hash


def test_untouched_nodes_keep_their_identity():
    """A node that does not name the value must not be reconstructed.

    This is the property that makes the redirect cheap, and it is invisible to
    an equality check: rebuilding every node still compares equal. Identity
    pins it directly.
    """

    module = _module()
    before = {node.id: node for node in module.nodes}
    after = _redirect(module, "source", "target")

    rebuilt = [
        node.id for node in after.nodes if node is not before.get(node.id)
    ]
    assert rebuilt, "the consumers of the redirected value must be rebuilt"
    assert all(
        "source" in before[node_id].inputs for node_id in rebuilt
    ), f"rebuilt a node that never named the value: {rebuilt}"
    assert all(
        node is before[node.id]
        for node in after.nodes
        if "source" not in before[node.id].inputs
    ), "a node that does not name the value was reconstructed"


def test_redirect_rewrites_function_parameters_and_outputs():
    """The value also appears in the function signature, which must follow."""

    after = _redirect(_module(), "source", "target")
    function = after.function_map["main"]
    assert "source" not in function.parameters
    assert "source" not in function.outputs
    assert function.parameters.count("target") == 2
    assert function.outputs[-1] == "target"
