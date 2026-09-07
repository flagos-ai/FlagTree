# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

from triton.flagmega import ir as fm
from triton.flagmega.rules import DataflowRewriter, RewriteRule


def test_iteration_limit_counts_fixed_point_sweeps_not_matching_nodes():
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    values = tuple(
        builder.var(f"value_{index}", fm.tensor_type("float32", (1,)), id=f"value_{index}")
        for index in range(40)
    )
    builder.function("main", values, values)
    module = builder.build(entry="main")

    rule = RewriteRule(
        "mark",
        lambda node, _: not bool(node.metadata.get("marked", False)),
        lambda node, _: replace(node, metadata={**dict(node.metadata), "marked": True}),
    )
    rewritten = DataflowRewriter(
        (rule,), max_iterations=2, remove_unused=False).rewrite(module)

    assert len(rewritten.nodes) == 40
    assert all(node.metadata["marked"] is True for node in rewritten.nodes)
