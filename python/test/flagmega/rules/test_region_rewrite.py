# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.rules import DataflowRewriter, RewriteResult, RewriteRule


def graph():

    class Graph(fm.Module):

        def forward(self):
            x = self.input("x", fm.tensor_type("float32", (2, )))
            a = fm.F.math.add(x, x, name="a")
            b = fm.F.math.add(a, x, name="b")
            c = fm.F.math.add(a, b, name="c")
            self.function("main", (x, ), (c, ))

    return Graph(dialect="high_level", stage="imported", entry="main").build()


def test_region_relocates_shared_value_and_updates_consumer_atomically():
    source = graph()
    rule = RewriteRule(
        "region", lambda n, m: n.id == "b" and not n.metadata,
        lambda n, m: RewriteResult(replace(n, metadata={"done": True}), prefix_nodes=(m.node_map["a"], ), removed_ids=(
            "a", ), extra_replacements=(replace(m.node_map["c"], inputs=("b", "b")), )))
    result = DataflowRewriter((rule, )).rewrite(source)
    assert [n.id for n in result.nodes] == ["n0", "a", "b", "c"]
    assert result.node_map["c"].inputs == ("b", "b")


@pytest.mark.parametrize("removed,extra", [(("n0", ), ()), (("absent", ), ()), (("b", ), ())])
def test_region_rejects_dangling_or_invalid_removals(removed, extra):
    rule = RewriteRule("invalid", lambda n, m: n.id == "b",
                       lambda n, m: RewriteResult(n, removed_ids=removed, extra_replacements=extra))
    with pytest.raises(IRVerificationError):
        DataflowRewriter((rule, )).rewrite(graph())


def test_region_edits_are_not_silently_ignored_by_egraph():
    from triton.flagmega import pattern_match as pm
    from triton.flagmega.egraph import EGraphRewriter

    rule = RewriteRule("region", pm.F.math.is_add(),
                       lambda result, module: RewriteResult(result.root, removed_ids=("a",)))
    with pytest.raises(IRVerificationError, match="Region edits require DataflowRewriter"):
        EGraphRewriter((rule,)).rewrite(graph())
