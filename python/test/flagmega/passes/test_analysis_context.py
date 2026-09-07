from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.passes import (
    FunctionalPass,
    PassManager,
    current_pass_context,
)


def _module():
    builder = fm.IRBuilder(dialect="high_level", stage="unit")
    value = builder.var("value", fm.tensor_type("float32", [1]), id="value")
    builder.function("main", [value], [value])
    return builder.build(entry="main")


class NodeCountAnalysis:
    name = "node_count"

    def __init__(self):
        self.calls = []

    def analyze(self, module, function=None):
        self.calls.append((module.semantic_hash, function))
        return len(module.nodes)


def test_run_pass_context_lazily_memoizes_and_promotes_preserved_analysis():
    provider = NodeCountAnalysis()
    seen = []

    def first(module):
        context = current_pass_context()
        seen.append((context.manager_name, context.pass_name, context.pass_index))
        assert context.require_analysis("node_count") == 1
        assert context.require_analysis("node_count") == 1
        return replace(module, metadata={"iteration": 1})

    def second(module):
        assert current_pass_context().require_analysis("node_count") == 1
        return module

    result = (
        PassManager("analysis")
        .register_analysis(provider)
        .add(FunctionalPass("first", first, preserves=frozenset({"node_count"})))
        .add(FunctionalPass("second", second, preserves=frozenset({"node_count"})))
        .run(_module())
    )

    assert provider.calls == [(result.executions[0].input_semantic_hash, None)]
    assert seen == [("analysis", "first", 0)]
    assert result.invalidated_analyses == ()


def test_nonpreserving_pass_invalidates_real_cached_analysis():
    provider = NodeCountAnalysis()

    def query(module):
        current_pass_context().require_analysis("node_count")
        return module

    result = (
        PassManager("analysis")
        .register_analysis(provider)
        .add(FunctionalPass("query", query))
        .run(_module())
    )

    assert result.invalidated_analyses == ("node_count",)


def test_pass_context_is_not_visible_outside_execution():
    with pytest.raises(RuntimeError, match="No FlagMega pass"):
        current_pass_context()
