# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes.functions import (
    propagate_function_boundary_layouts,
)


class _RepeatedPackedLayer(fm.Module):
    def __init__(self):
        super().__init__(dialect="ntt", stage="packed", entry="main")

    def forward(self):
        logical_type = fm.tensor_type("float32", (4, 16))
        layer_input = self.input(
            "layer_input", logical_type, id="layer_input"
        )
        packed = fm.F.tensors.pack(
            layer_input, (4,), axes=(1,), name="layer_pack"
        )
        computed = fm.F.math.vectorized_unary(
            packed, unary_op="silu", name="layer_compute"
        )
        layer_output = fm.F.tensors.unpack(
            computed, axes=(1,), name="layer_unpack"
        )

        value = self.input("value", logical_type, id="value")
        call0 = fm.F.builtin.call(
            value,
            result_type=logical_type,
            callee="layer",
            name="call0",
        )
        call1 = fm.F.builtin.call(
            call0,
            result_type=logical_type,
            callee="layer",
            name="call1",
        )
        self.function("main", (value,), (call1,))
        self.function(
            "layer",
            (layer_input,),
            (layer_output,),
            attrs={"reusable": True, "noinline": True},
        )


def test_repeated_calls_share_one_packed_boundary_without_intermediate_repack():
    original = _RepeatedPackedLayer().build()
    rewritten = propagate_function_boundary_layouts(original)

    layer = rewritten.function_map["layer"]
    assert isinstance(rewritten.node_map[layer.parameters[0]].type.dtype, fm.VectorType)
    assert isinstance(rewritten.node_map[layer.outputs[0]].type.dtype, fm.VectorType)
    assert sum(node.op == "tensors.pack" for node in rewritten.nodes) == 1
    assert sum(node.op == "tensors.unpack" for node in rewritten.nodes) == 1

    value = torch.randn(4, 16)
    evaluator = TorchEvaluator(DictWeightResolver({}))
    expected = evaluator.run(original, {"value": value})[0]
    actual = evaluator.run(rewritten, {"value": value})[0]
    torch.testing.assert_close(actual, expected)
