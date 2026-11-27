# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes.functions import (
    function_nodes,
    propagate_function_boundary_layouts,
)


class _NestedLayoutLayer(fm.Module):
    def __init__(self):
        super().__init__(dialect="ntt", stage="packed", entry="main")

    def forward(self):
        logical_type = fm.tensor_type("float32", (4, 16))
        parameter = self.input("parameter", logical_type, id="parameter")
        packed4 = fm.F.tensors.pack(
            parameter, (4,), axes=(1,), name="packed4"
        )
        packed2 = fm.F.tensors.pack(
            packed4, (2,), axes=(1,), name="packed2"
        )
        unpacked2 = fm.F.tensors.unpack(
            packed2, axes=(1,), name="unpacked2"
        )
        result = fm.F.tensors.unpack(
            unpacked2, axes=(1,), name="result"
        )

        value = self.input("value", logical_type, id="value")
        call = fm.F.builtin.call(
            value, result_type=logical_type, callee="layer", name="call"
        )
        self.function("main", (value,), (call,))
        self.function("layer", (parameter,), (result,), attrs={"reusable": True})


def test_nested_pack_unpack_chain_moves_across_boundary_to_fixed_point():
    original = _NestedLayoutLayer().build()
    rewritten = propagate_function_boundary_layouts(original)

    layer = rewritten.function_map["layer"]
    parameter_type = rewritten.node_map[layer.parameters[0]].type
    assert isinstance(parameter_type.dtype, fm.VectorType)
    assert parameter_type.dtype.lanes == (2, 4)
    assert not {
        node.op
        for node in function_nodes(rewritten, layer)
    } & {"tensors.pack", "tensors.unpack"}

    main_ops = [node.op for node in function_nodes(rewritten, rewritten.function_map["main"])]
    assert main_ops.count("tensors.pack") == 2
    assert main_ops.count("tensors.unpack") == 2

    value = torch.randn(4, 16)
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(
        evaluator.run(rewritten, {"value": value})[0],
        evaluator.run(original, {"value": value})[0],
    )
