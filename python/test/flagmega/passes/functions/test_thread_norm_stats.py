# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes.functions import thread_norm_stats_across_function_boundaries


class _RepeatedLayer(fm.Module):
    def __init__(self):
        super().__init__(dialect="nn", stage="packed", entry="main")

    def forward(self):
        value_type = fm.tensor_type("float32", (2, 8))
        parameter_type = fm.tensor_type("float32", (8,))
        token_type = fm.tensor_type("int32", (1,))

        layer_input = self.input("layer_input", value_type, id="layer_input")
        layer_token = self.input("layer_token", token_type, id="layer_token")
        layer_scale = self.input("layer_scale", parameter_type, id="layer_scale")
        layer_bias = self.input("layer_bias", parameter_type, id="layer_bias")
        layer_stats = fm.F.nn.norm_stats(
            layer_input, axis=1, use_mean=False, name="layer_stats")
        layer_output = fm.F.nn.norm_apply(
            layer_input,
            layer_stats,
            layer_scale,
            layer_bias,
            axis=1,
            epsilon=1e-6,
            use_mean=False,
            name="layer_output",
        )

        value = self.input("value", value_type, id="value")
        token = self.input("token", token_type, id="token")
        scale = self.input("scale", parameter_type, id="scale")
        bias = self.input("bias", parameter_type, id="bias")
        result_type = fm.TupleType((value_type, token_type))
        call0 = fm.F.builtin.call(
            value, token, scale, bias,
            result_type=result_type, callee="layer", name="call0")
        value0 = fm.F.tensors.get_item(call0, 0, name="value0")
        token0 = fm.F.tensors.get_item(call0, 1, name="token0")
        call1 = fm.F.builtin.call(
            value0, token0, scale, bias,
            result_type=result_type, callee="layer", name="call1")
        value1 = fm.F.tensors.get_item(call1, 0, name="value1")
        token1 = fm.F.tensors.get_item(call1, 1, name="token1")
        final_stats = fm.F.nn.norm_stats(
            value1, axis=1, use_mean=False, name="final_stats")
        final_output = fm.F.nn.norm_apply(
            value1,
            final_stats,
            scale,
            bias,
            axis=1,
            epsilon=1e-6,
            use_mean=False,
            name="final_output",
        )
        self.function(
            "layer",
            (layer_input, layer_token, layer_scale, layer_bias),
            (layer_output, layer_token),
            attrs={"reusable": True, "noinline": True},
        )
        self.function("main", (value, token, scale, bias), (final_output, token1))


def test_threads_seed_recurrent_and_final_statistics_without_changing_entry_abi():
    original = _RepeatedLayer().build()
    rewritten = thread_norm_stats_across_function_boundaries(original)

    assert rewritten.function_map["main"].parameters == original.function_map["main"].parameters
    assert rewritten.function_map["main"].outputs == original.function_map["main"].outputs
    layer = rewritten.function_map["layer"]
    assert len(layer.parameters) == 5
    assert len(layer.outputs) == 3
    assert rewritten.node_map["layer_stats"].op == "nn.bind_norm_stats"
    assert rewritten.node_map["layer_stats"].inputs == (
        "layer_input", "layer_input.norm_stats")

    call0 = rewritten.node_map["call0"]
    call1 = rewritten.node_map["call1"]
    assert len(call0.inputs) == len(call1.inputs) == 5
    assert rewritten.node_map[call0.inputs[-1]].op == "nn.norm_stats"
    threaded = rewritten.node_map[call1.inputs[-1]]
    assert threaded.op == "builtin.get_item"
    assert threaded.inputs == ("call0",)
    assert threaded.attrs["index"] == 2
    assert rewritten.node_map["final_stats"].op == "builtin.get_item"
    assert rewritten.node_map["final_stats"].inputs == ("call1",)
    assert rewritten.node_map["final_stats"].attrs["index"] == 2

    feeds = {
        "value": torch.randn(2, 8),
        "token": torch.tensor([7], dtype=torch.int32),
        "scale": torch.randn(8),
        "bias": torch.randn(8),
    }
    evaluator = TorchEvaluator(DictWeightResolver({}))
    expected = evaluator.run(original, feeds)
    actual = evaluator.run(rewritten, feeds)
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


def test_does_not_thread_when_state_output_is_ambiguous():
    class Ambiguous(fm.Module):
        def __init__(self):
            super().__init__(dialect="nn", stage="packed", entry="main")

        def forward(self):
            value_type = fm.tensor_type("float32", (2, 8))
            parameter_type = fm.tensor_type("float32", (8,))
            layer_input = self.input("layer_input", value_type)
            scale = self.input("scale", parameter_type)
            bias = self.input("bias", parameter_type)
            stats = fm.F.nn.norm_stats(layer_input, axis=1, use_mean=False)
            output = fm.F.nn.norm_apply(
                layer_input, stats, scale, bias,
                axis=1, epsilon=1e-6, use_mean=False)
            value = self.input("value", value_type)
            call = fm.F.builtin.call(
                value, scale, bias,
                result_type=fm.TupleType((value_type, value_type)),
                callee="layer",
            )
            first = fm.F.tensors.get_item(call, 0)
            second = fm.F.tensors.get_item(call, 1)
            self.function("layer", (layer_input, scale, bias), (output, output))
            self.function("main", (value, scale, bias), (first, second))

    module = Ambiguous().build()
    assert thread_norm_stats_across_function_boundaries(module) == module
