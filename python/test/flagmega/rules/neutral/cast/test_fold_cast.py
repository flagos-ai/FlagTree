# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.passes.target_independent import decompose_complex_ops
from triton.flagmega.rules import DataflowRewriter
from triton.flagmega.rules.neutral.fold_cast import fold_cast_rule


def graph(source="float32", intermediate="bfloat16", *, shared=False, output=None):

    class Graph(fm.Module):

        def forward(self):
            x = self.input("x", fm.tensor_type(source, (2, 32)), id="x")
            first = fm.F.tensors.cast(x, intermediate, name="first")
            last = fm.F.tensors.cast(first, output or source, name="last")
            self.function("main", (x, ), (last, first) if shared else (last, ))

    return Graph(dialect="high_level", stage="imported", entry="main").build()


@pytest.mark.parametrize("source,intermediate", [
    ("float32", "bfloat16"),
    ("bfloat16", "float32"),
    ("float32", "float8_e4m3fn"),
    ("bfloat16", "bfloat16"),
])
def test_float_round_trip_is_eliminated_in_target_independent(source, intermediate):
    result = decompose_complex_ops(graph(source, intermediate))
    assert result.functions[0].outputs == ("x", )
    assert not any(node.op == "tensors.cast" for node in result.nodes)


def test_observable_intermediate_survives():
    result = DataflowRewriter((fold_cast_rule(), )).rewrite(graph(shared=True))
    assert result.functions[0].outputs == ("x", "first")
    assert result.node_map["first"].op == "tensors.cast"


@pytest.mark.parametrize("source,intermediate,output", [
    ("float32", "int32", "float32"),
    ("int32", "float32", "int32"),
    ("float32", "bfloat16", "float8_e4m3fn"),
])
def test_does_not_remove_integer_conversion_or_non_round_trip(source, intermediate, output):
    result = DataflowRewriter((fold_cast_rule(), )).rewrite(graph(source, intermediate, output=output))
    assert result.node_map["last"].op == "tensors.cast"
