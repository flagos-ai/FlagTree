"""Ordinary eager execution releases consumed intermediates; traces retain them."""

import weakref

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import TorchEvaluator


@pytest.mark.parametrize("trace", [False, True])
def test_consumed_weights_do_not_accumulate_during_untraced_run(trace):
    torch = pytest.importorskip("torch")

    class Graph(fm.Module):
        def forward(self):
            first = self.weight("first", fm.tensor_type("float32", (32,)), id="first", source="unit", key="first")
            second = self.weight("second", fm.tensor_type("float32", (32,)), id="second", source="unit", key="second")
            a = fm.F.math.reduce_sum(first, axes=(0,), keep_dims=True)
            b = fm.F.math.reduce_sum(second, axes=(0,), keep_dims=True)
            output = fm.F.math.add(a, b)
            self.function("main", (), (output,))

    class Resolver:
        def resolve(self, node):
            if node.attrs["key"] == "second":
                assert (self.first() is not None) is trace
            value = torch.ones(32)
            if node.attrs["key"] == "first":
                self.first = weakref.ref(value)
            return value

    module = Graph(dialect="high_level", stage="imported", entry="main").build()
    evaluator = TorchEvaluator(Resolver())
    result = evaluator.run_result(module, {}) if trace else evaluator.run(module, {})
    output = result.outputs if trace else result
    assert output[0].item() == 64
    if trace:
        assert result.trace["first"].numel() == 32
