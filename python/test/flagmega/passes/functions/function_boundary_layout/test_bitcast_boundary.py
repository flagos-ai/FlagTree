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


class _BitcastOutputLayer(fm.Module):
    def __init__(self):
        super().__init__(dialect="ntt", stage="packed", entry="main")

    def forward(self):
        vector = fm.VectorType(fm.DType.FLOAT32, (4,))
        packed_type = fm.tensor_type(vector, (2, 4))
        logical_type = fm.tensor_type("float32", (2, 16))
        parameter = self.input("parameter", packed_type, id="parameter")
        logical = fm.F.tensors.bitcast(
            parameter, fm.DType.FLOAT32, name="logical"
        )
        value = self.input("value", packed_type, id="value")
        call = fm.F.builtin.call(
            value, result_type=logical_type, callee="layer", name="call"
        )
        self.function("main", (value,), (call,))
        self.function("layer", (parameter,), (logical,), attrs={"reusable": True})


class _BitcastOutputWithCallerPack(fm.Module):
    def __init__(self):
        super().__init__(dialect="ntt", stage="packed", entry="main")

    def forward(self):
        vector = fm.VectorType(fm.DType.FLOAT32, (4,))
        packed_type = fm.tensor_type(vector, (2, 4))
        logical_type = fm.tensor_type("float32", (2, 16))
        parameter = self.input("packed_parameter", packed_type, id="packed_parameter")
        logical = fm.F.tensors.bitcast(
            parameter, fm.DType.FLOAT32, name="layer_bitcast"
        )
        value = self.input("packed_value", packed_type, id="packed_value")
        call = fm.F.builtin.call(
            value,
            result_type=logical_type,
            callee="packed_layer",
            name="packed_call",
        )
        result = fm.F.tensors.pack(
            call, (4,), axes=(1,), name="caller_pack"
        )
        self.function("main", (value,), (result,))
        self.function(
            "packed_layer", (parameter,), (logical,), attrs={"reusable": True}
        )


def test_output_bitcast_is_hoisted_to_the_caller_compatibility_view():
    original = _BitcastOutputLayer().build()
    rewritten = propagate_function_boundary_layouts(original)

    layer = rewritten.function_map["layer"]
    assert not any(
        node.op == "tensors.bitcast"
        for node in function_nodes(rewritten, layer)
    )
    raw = rewritten.node_map["call.boundary_packed"]
    assert raw.op == "builtin.call"
    assert raw.type == original.node_map["parameter"].type
    logical = rewritten.node_map["call"]
    assert logical.op == "tensors.bitcast"
    assert logical.inputs == (raw.id,)

    value = torch.randn(2, 4, 4)
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(
        evaluator.run(rewritten, {"value": value})[0],
        evaluator.run(original, {"value": value})[0],
    )


def test_post_boundary_egraph_folds_caller_pack_of_hoisted_bitcast():
    original = _BitcastOutputWithCallerPack().build()
    boundary = propagate_function_boundary_layouts(original)
    rewritten = post_function_boundary_pack_propagation(
        boundary, NvidiaSm90Target())

    main = rewritten.function_map["main"]
    main_nodes = function_nodes(rewritten, main)
    assert not any(
        node.op in {"tensors.pack", "tensors.bitcast"}
        for node in main_nodes
    )
    assert rewritten.node_map[main.outputs[0]].op == "builtin.call"

    value = torch.randn(2, 4, 4)
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(
        evaluator.run(rewritten, {"packed_value": value})[0],
        evaluator.run(original, {"packed_value": value})[0],
    )
