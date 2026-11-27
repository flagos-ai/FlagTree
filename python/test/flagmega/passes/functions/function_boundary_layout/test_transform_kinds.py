# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes.functions import (
    function_nodes,
    propagate_function_boundary_layouts,
)


def _assert_same(module, rewritten, feeds):
    evaluator = TorchEvaluator(DictWeightResolver({}))
    expected = evaluator.run(module, feeds)[0]
    actual = evaluator.run(rewritten, feeds)[0]
    torch.testing.assert_close(actual, expected)


class _InputUnpackLayer(fm.Module):
    def __init__(self):
        super().__init__(dialect="ntt", stage="packed", entry="main")

    def forward(self):
        packed_type = fm.tensor_type(
            fm.VectorType(fm.DType.FLOAT32, (4,)), (4, 4)
        )
        logical_type = fm.tensor_type("float32", (4, 16))
        parameter = self.input("parameter", packed_type, id="parameter")
        unpacked = fm.F.tensors.unpack(
            parameter, axes=(1,), name="unpacked"
        )
        computed = fm.F.math.silu(unpacked, name="computed")
        value = self.input("value", packed_type, id="value")
        call = fm.F.builtin.call(
            value, result_type=logical_type, callee="layer", name="call"
        )
        self.function("main", (value,), (call,))
        self.function("layer", (parameter,), (computed,), attrs={"reusable": True})


class _InputAndOutputUnpackLayer(fm.Module):
    def __init__(self):
        super().__init__(dialect="ntt", stage="packed", entry="main")

    def forward(self):
        packed_type = fm.tensor_type(
            fm.VectorType(fm.DType.FLOAT32, (4,)), (4, 4)
        )
        logical_type = fm.tensor_type("float32", (4, 16))
        parameter = self.input("both_parameter", packed_type, id="both_parameter")
        unpacked = fm.F.tensors.unpack(
            parameter, axes=(1,), name="both_unpacked"
        )
        value = self.input("both_value", packed_type, id="both_value")
        call = fm.F.builtin.call(
            value,
            result_type=logical_type,
            callee="both_layer",
            name="both_call",
        )
        self.function("main", (value,), (call,))
        self.function(
            "both_layer", (parameter,), (unpacked,), attrs={"reusable": True}
        )


def test_input_unpack_is_propagated_as_a_logical_function_abi():
    original = _InputUnpackLayer().build()
    rewritten = propagate_function_boundary_layouts(original)

    layer = rewritten.function_map["layer"]
    assert rewritten.node_map[layer.parameters[0]].type.dtype == fm.DType.FLOAT32
    assert "unpacked" not in rewritten.node_map
    boundary = rewritten.node_map["call.arg0.boundary_pack"]
    assert boundary.op == "tensors.unpack"
    assert rewritten.node_map["call"].inputs == (boundary.id,)

    _assert_same(original, rewritten, {"value": torch.randn(4, 4, 4)})


def test_same_unpack_at_input_and_output_restores_original_value_inside_callee():
    original = _InputAndOutputUnpackLayer().build()
    rewritten = propagate_function_boundary_layouts(original)

    layer = rewritten.function_map["both_layer"]
    assert rewritten.node_map[layer.parameters[0]].type.dtype == fm.DType.FLOAT32
    output = rewritten.node_map[layer.outputs[0]]
    assert output.op == "tensors.pack"
    assert output.metadata["boundary_layout"] == "raw_parameter_restore"
    assert "both_unpacked" not in rewritten.node_map

    _assert_same(original, rewritten, {"both_value": torch.randn(4, 4, 4)})


class _InputPermuteWithRawUse(fm.Module):
    def __init__(self):
        super().__init__(dialect="ntt", stage="packed", entry="main")

    def forward(self):
        source_type = fm.tensor_type("float32", (2, 3))
        transformed_type = fm.tensor_type("float32", (3, 2))
        parameter = self.input("parameter", source_type, id="parameter")
        permuted = fm.F.tensors.permute(parameter, (1, 0), name="permuted")
        transformed = fm.F.math.silu(permuted, name="transformed")
        raw = fm.F.math.silu(parameter, name="raw")
        raw_permuted = fm.F.tensors.permute(raw, (1, 0), name="raw_permuted")
        result = fm.F.math.add(transformed, raw_permuted, name="result")

        value = self.input("value", source_type, id="value")
        call = fm.F.builtin.call(
            value,
            result_type=transformed_type,
            callee="layer",
            name="call",
        )
        self.function("main", (value,), (call,))
        self.function("layer", (parameter,), (result,), attrs={"reusable": True})


def test_input_permute_is_hoisted_and_raw_use_gets_inverse_permute():
    original = _InputPermuteWithRawUse().build()
    rewritten = propagate_function_boundary_layouts(original)

    layer = rewritten.function_map["layer"]
    assert rewritten.node_map[layer.parameters[0]].type.shape == fm.tensor_type(
        "float32", (3, 2)
    ).shape
    owned = {node.id: node for node in function_nodes(rewritten, layer)}
    restores = [
        node
        for node in owned.values()
        if node.metadata.get("boundary_layout") == "raw_parameter_restore"
    ]
    assert len(restores) == 1
    assert restores[0].op == "tensors.permute"
    assert restores[0].attrs == {"axes": (1, 0)}
    assert owned["raw"].inputs == (restores[0].id,)

    _assert_same(original, rewritten, {"value": torch.randn(2, 3)})


class _CallerPermuteDemand(fm.Module):
    def __init__(self):
        super().__init__(dialect="ntt", stage="packed", entry="main")

    def forward(self):
        source_type = fm.tensor_type("float32", (2, 3))
        parameter = self.input("parameter", source_type, id="parameter")
        computed = fm.F.math.silu(parameter, name="computed")
        value = self.input("value", source_type, id="value")
        call = fm.F.builtin.call(
            value, result_type=source_type, callee="layer", name="call"
        )
        result = fm.F.tensors.permute(call, (1, 0), name="caller_permute")
        self.function("main", (value,), (result,))
        self.function("layer", (parameter,), (computed,), attrs={"reusable": True})


def test_caller_permute_demand_moves_into_callee_output():
    original = _CallerPermuteDemand().build()
    rewritten = propagate_function_boundary_layouts(original)

    layer = rewritten.function_map["layer"]
    output = rewritten.node_map[layer.outputs[0]]
    assert output.op == "tensors.permute"
    assert output.inputs == ("computed",)
    assert "caller_permute" not in rewritten.node_map

    _assert_same(original, rewritten, {"value": torch.randn(2, 3)})
