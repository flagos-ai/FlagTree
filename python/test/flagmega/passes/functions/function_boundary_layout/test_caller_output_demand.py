# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes.functions import propagate_function_boundary_layouts


class _CallerPackedOutput(fm.Module):
    def __init__(self):
        super().__init__(dialect="ntt", stage="packed", entry="main")

    def forward(self):
        logical_type = fm.tensor_type("float32", (4, 16))
        parameter = self.input("parameter", logical_type, id="parameter")
        computed = fm.F.math.silu(parameter, name="computed")

        value = self.input("value", logical_type, id="value")
        call = fm.F.builtin.call(
            value,
            result_type=logical_type,
            callee="layer",
            name="call",
        )
        packed = fm.F.tensors.pack(call, (4,), axes=(1,), name="caller_pack")
        self.function("main", (value,), (packed,))
        self.function("layer", (parameter,), (computed,), attrs={"reusable": True})


class _CallerPackedOutputWithLogicalConsumer(fm.Module):
    def __init__(self):
        super().__init__(dialect="ntt", stage="packed", entry="main")

    def forward(self):
        logical_type = fm.tensor_type("float32", (4, 16))
        parameter = self.input("mixed_parameter", logical_type, id="mixed_parameter")
        computed = fm.F.math.silu(parameter, name="mixed_computed")

        value = self.input("mixed_value", logical_type, id="mixed_value")
        call = fm.F.builtin.call(
            value,
            result_type=logical_type,
            callee="mixed_layer",
            name="mixed_call",
        )
        packed = fm.F.tensors.pack(
            call, (4,), axes=(1,), name="mixed_caller_pack"
        )
        logical = fm.F.math.silu(call, name="logical_consumer")
        logical_packed = fm.F.tensors.pack(
            logical, (4,), axes=(1,), name="logical_packed"
        )
        result = fm.F.math.vectorized_binary(
            packed, logical_packed, binary_op="add", name="mixed_result"
        )
        self.function("main", (value,), (result,))
        self.function(
            "mixed_layer", (parameter,), (computed,), attrs={"reusable": True}
        )


def test_caller_pack_demand_specializes_callee_output_and_removes_caller_pack():
    original = _CallerPackedOutput().build()
    rewritten = propagate_function_boundary_layouts(original)

    layer = rewritten.function_map["layer"]
    layer_output = rewritten.node_map[layer.outputs[0]]
    assert layer_output.op == "tensors.pack"
    assert layer_output.inputs == ("computed",)
    assert isinstance(layer_output.type.dtype, fm.VectorType)

    main = rewritten.function_map["main"]
    assert main.outputs == ("call.boundary_packed",)
    assert rewritten.node_map[main.outputs[0]].op == "builtin.call"

    value = torch.randn(4, 16)
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(
        evaluator.run(rewritten, {"value": value})[0],
        evaluator.run(original, {"value": value})[0],
    )


def test_caller_pack_demand_retains_one_logical_restore_for_other_consumers():
    original = _CallerPackedOutputWithLogicalConsumer().build()
    rewritten = propagate_function_boundary_layouts(original)

    assert "mixed_caller_pack" not in rewritten.node_map
    raw_call = rewritten.node_map["mixed_call.boundary_packed"]
    assert raw_call.op == "builtin.call"
    restore = rewritten.node_map["mixed_call"]
    assert restore.op == "tensors.unpack"
    assert restore.inputs == (raw_call.id,)
    assert rewritten.node_map["logical_consumer"].inputs == (restore.id,)

    value = torch.randn(4, 16)
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(
        evaluator.run(rewritten, {"mixed_value": value})[0],
        evaluator.run(original, {"mixed_value": value})[0],
    )
