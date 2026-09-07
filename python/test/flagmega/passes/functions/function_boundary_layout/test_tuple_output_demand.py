# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes.functions import propagate_function_boundary_layouts


class _TupleProducerWithPackedDemand(fm.Module):
    def __init__(self):
        super().__init__(dialect="ntt", stage="packed", entry="main")

    def forward(self):
        tensor_type = fm.tensor_type("float32", (4, 16))
        parameter = self.input("parameter", tensor_type, id="parameter")
        first = fm.F.math.silu(parameter, name="first")
        second = fm.F.math.silu(first, name="second")

        value = self.input("value", tensor_type, id="value")
        call = fm.F.builtin.call(
            value,
            result_type=fm.TupleType((tensor_type, tensor_type)),
            callee="producer",
            name="call",
        )
        item0 = fm.F.tensors.get_item(call, 0, name="item0")
        item1 = fm.F.tensors.get_item(call, 1, name="item1")
        packed0 = fm.F.tensors.pack(
            item0, (4,), axes=(1,), name="packed0"
        )
        packed1 = fm.F.tensors.pack(
            item1, (4,), axes=(1,), name="packed1"
        )
        result = fm.F.math.vectorized_binary(
            packed0, packed1, binary_op="add", name="result"
        )
        self.function("main", (value,), (result,))
        self.function(
            "producer", (parameter,), (first, second), attrs={"reusable": True}
        )


def test_tuple_get_item_pack_demands_specialize_each_callee_output():
    original = _TupleProducerWithPackedDemand().build()
    rewritten = propagate_function_boundary_layouts(original)

    producer = rewritten.function_map["producer"]
    assert all(
        rewritten.node_map[output].op == "tensors.pack"
        for output in producer.outputs
    )
    assert "packed0" not in rewritten.node_map
    assert "packed1" not in rewritten.node_map
    assert rewritten.node_map["item0.boundary_packed"].op == "builtin.get_item"
    assert rewritten.node_map["item1.boundary_packed"].op == "builtin.get_item"

    value = torch.randn(4, 16)
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(
        evaluator.run(rewritten, {"value": value})[0],
        evaluator.run(original, {"value": value})[0],
    )
