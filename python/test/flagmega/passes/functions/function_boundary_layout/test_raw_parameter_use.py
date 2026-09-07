# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes.functions import (
    function_nodes,
    propagate_function_boundary_layouts,
)


class _PackedLayerWithRawUse(fm.Module):
    def __init__(self):
        super().__init__(dialect="ntt", stage="packed", entry="main")

    def forward(self):
        logical_type = fm.tensor_type("float32", (4, 16))
        parameter = self.input("parameter", logical_type, id="parameter")
        packed = fm.F.tensors.pack(parameter, (4,), axes=(1,), name="packed")
        raw = fm.F.math.silu(parameter, name="raw")
        raw_packed = fm.F.tensors.pack(raw, (4,), axes=(1,), name="raw_packed")
        summed = fm.F.math.vectorized_binary(
            packed, raw_packed, binary_op="add", name="summed"
        )
        result = fm.F.tensors.unpack(summed, axes=(1,), name="result")

        value = self.input("value", logical_type, id="value")
        call = fm.F.builtin.call(
            value,
            result_type=logical_type,
            callee="layer",
            name="call",
        )
        self.function("main", (value,), (call,))
        self.function("layer", (parameter,), (result,), attrs={"reusable": True})


def test_input_pack_is_hoisted_and_raw_parameter_use_gets_inverse_restore():
    original = _PackedLayerWithRawUse().build()
    rewritten = propagate_function_boundary_layouts(original)

    layer = rewritten.function_map["layer"]
    parameter = rewritten.node_map[layer.parameters[0]]
    assert isinstance(parameter.type.dtype, fm.VectorType)

    owned = {node.id: node for node in function_nodes(rewritten, layer)}
    restores = [
        node
        for node in owned.values()
        if node.op == "tensors.unpack" and node.inputs == (parameter.id,)
    ]
    assert len(restores) == 1
    assert owned["raw"].inputs == (restores[0].id,)
    assert sum(node.op == "tensors.pack" for node in owned.values()) == 1

    value = torch.randn(4, 16)
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(
        evaluator.run(rewritten, {"value": value})[0],
        evaluator.run(original, {"value": value})[0],
    )
