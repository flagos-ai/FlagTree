# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.rules import DataflowRewriter
from triton.flagmega.rules.neutral.fold_matmul_cast import fold_matmul_cast_rule
from triton.flagmega.passes.target_independent import decompose_complex_ops


def graph(shared=False, exported=False, dtype="bfloat16"):

    class Graph(fm.Module):

        def forward(self):
            lhs = self.input("lhs", fm.tensor_type(dtype, (2, 64)))
            rhs = self.input("rhs", fm.tensor_type(dtype, (32, 64)))
            projection = fm.F.math.matmul(lhs, rhs, transpose_b=True, name="projection")
            wide = fm.F.tensors.cast(projection, "float32", name="wide")
            outputs = [wide]
            if shared:
                outputs.append(fm.F.math.silu(projection))
            if exported:
                outputs.append(projection)
            self.function("main", (lhs, rhs), outputs)

    return Graph(dialect="high_level", stage="imported", entry="main").build()


@pytest.mark.parametrize("pipeline", [False, True])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_private_matmul_widening_becomes_explicit_f32_output(pipeline, dtype):
    source = graph(dtype=dtype)
    result = decompose_complex_ops(source) if pipeline else DataflowRewriter(
        (fold_matmul_cast_rule(), )).rewrite(source)
    assert not any(node.op == "tensors.cast" for node in result.nodes)
    output = result.node_map[result.functions[0].outputs[0]]
    assert output.op == "math.matmul"
    assert output.attrs["output_data_type"] == "float32"
    assert output.type == fm.tensor_type("float32", (2, 32))


@pytest.mark.parametrize("shared,exported", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_other_consumers_keep_the_original_matmul_precision(shared, exported, dtype):
    source = graph(shared, exported, dtype)
    result = DataflowRewriter((fold_matmul_cast_rule(), )).rewrite(source)
    assert result.semantic_hash == source.semantic_hash
