# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes.functions import (
    function_nodes,
    post_function_boundary_pack_propagation,
    propagate_function_boundary_layouts,
)
from triton.flagmega.targets import NvidiaSm90Target


class _PackedLayerCalledWithScalarUnary(fm.Module):
    def __init__(self):
        super().__init__(dialect="ntt", stage="packed", entry="main")

    def forward(self):
        tensor_type = fm.tensor_type("float32", (4, 16))
        parameter = self.input("parameter", tensor_type, id="parameter")
        packed = fm.F.tensors.pack(
            parameter, (4,), axes=(1,), name="layer_pack"
        )
        result = fm.F.tensors.unpack(
            packed, axes=(1,), name="layer_unpack"
        )

        value = self.input("value", tensor_type, id="value")
        scalar = fm.F.math.silu(value, name="scalar_silu")
        call = fm.F.builtin.call(
            scalar, result_type=tensor_type, callee="layer", name="call"
        )
        self.function("main", (value,), (call,))
        self.function("layer", (parameter,), (result,), attrs={"reusable": True})


def test_post_boundary_egraph_pushes_caller_pack_through_scalar_unary():
    original = _PackedLayerCalledWithScalarUnary().build()
    boundary = propagate_function_boundary_layouts(original)
    rewritten = post_function_boundary_pack_propagation(
        boundary, NvidiaSm90Target())

    main_nodes = {
        node.id: node
        for node in function_nodes(rewritten, rewritten.function_map["main"])
    }
    call = next(node for node in main_nodes.values() if node.op == "builtin.call")
    vector_unary = main_nodes[call.inputs[0]]
    assert vector_unary.op == "math.vectorized_unary"
    assert main_nodes[vector_unary.inputs[0]].op == "tensors.pack"
    assert sum(node.op == "math.silu" for node in main_nodes.values()) == 0

    value = torch.randn(4, 16)
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(
        evaluator.run(rewritten, {"value": value})[0],
        evaluator.run(original, {"value": value})[0],
    )
