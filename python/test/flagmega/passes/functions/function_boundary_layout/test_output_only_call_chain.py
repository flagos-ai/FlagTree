# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Output-only ABI changes must rewrite all call arguments, not only new inputs."""

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes.functions.function_boundary_layout import propagate_function_boundary_layouts


@pytest.mark.parametrize("tuple_result", [False, True])
def test_changed_output_view_is_resolved_in_unchanged_call_input(tuple_result):

    class Graph(fm.Module):

        def forward(self):
            packed = fm.tensor_type(fm.vector_type("float32", (8, )), (1, 4))
            logical = fm.tensor_type(fm.vector_type("float32", (4, )), (1, 8))
            x = self.input("x", packed)
            computed = fm.F.math.vectorized_unary(x, unary_op="silu")
            view = fm.F.tensors.bitcast(computed, logical.dtype)
            outputs = (view, computed) if tuple_result else (view, )
            self.function("worker", (x, ), outputs, attrs={"reusable": True})
            value = self.input("value", packed)
            result_type = fm.TupleType((logical, packed)) if tuple_result else logical
            current = value
            for index in range(3):
                call = fm.F.builtin.call(current, callee="worker", result_type=result_type, name=f"call{index}")
                field = fm.F.tensors.get_item(call, 0) if tuple_result else call
                current = fm.F.tensors.bitcast(field, packed.dtype, name=f"packed{index}")
            self.function("main", (value, ), (current, ))

    original = Graph(dialect="ntt", stage="packed", entry="main").build()
    result = propagate_function_boundary_layouts(original)
    result = propagate_function_boundary_layouts(result)
    fm.verify_module(result)
    calls = [node for node in result.nodes if node.op == "builtin.call"]
    assert len(calls) == 3
    for call in calls:
        assert result.node_map[call.inputs[0]].type == result.node_map[result.function_map["worker"].parameters[0]].type
    feeds = {"value": torch.linspace(-2, 2, 32).reshape(1, 4, 8)}
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(evaluator.run(result, feeds), evaluator.run(original, feeds), rtol=0, atol=0)
