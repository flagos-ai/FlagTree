# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A producer's newly exposed Partial result must reach the norm consumer."""

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler


def _module(*, return_stats=False, bind_stats=False):
    placement = fm.Placement((2, 4), "yx", "bb")
    broad = fm.SBP.broadcast()
    value_type = fm.DistributedType(fm.tensor_type("float32", (1, 32)), (broad, broad), placement)
    stats_type = fm.DistributedType(fm.tensor_type("float32", (1, 1, 1)), (broad,) * 3, placement)
    partial_type = replace(stats_type, partial=fm.SBP.partial((0, 1)))
    producer_type = replace(value_type, axis_policies=(broad, fm.SBP.split_contiguous((0, 1))))
    parameter_type = fm.DistributedType(fm.tensor_type("float32", (32,)), (broad,), placement)

    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="nn", stage="qkv_combine_lowered", entry="main")

        def forward(self):
            producer_input = self.input("producer_input", producer_type, id="producer_input")
            producer_stats = fm.F.nn.norm_stats(producer_input, axis=1, use_mean=False, name="producer_stats")
            materialized = fm.F.distributed.boxing(producer_stats, stats_type, name="materialized")
            self.function("producer", (producer_input,), (materialized,), attrs={"reusable": True})

            layer_value = self.input("layer_value", value_type, id="layer_value")
            layer_stats = self.input("layer_stats", stats_type, id="layer_stats")
            layer_scale = self.input("layer_scale", parameter_type, id="layer_scale")
            layer_bias = self.input("layer_bias", parameter_type, id="layer_bias")
            stats = fm.F.nn.bind_norm_stats(
                layer_value, layer_stats, axis=1, use_mean=False, name="bound_stats",
            ) if bind_stats else layer_stats
            normalized = fm.F.nn.norm_apply(
                layer_value, stats, layer_scale, layer_bias,
                axis=1, epsilon=1e-6, use_mean=False, name="normalized",
            )
            outputs = (normalized, layer_stats) if return_stats else (normalized,)
            self.function("layer", (layer_value, layer_stats, layer_scale, layer_bias), outputs,
                          attrs={"reusable": True, "noinline": True})

            value = self.input("value", value_type, id="value")
            partial = self.input("partial", producer_type, id="partial")
            scale = self.input("scale", parameter_type, id="scale")
            bias = self.input("bias", parameter_type, id="bias")
            produced = fm.F.builtin.call(partial, callee="producer", result_type=stats_type, name="produced")
            result_type = fm.TupleType((value_type, stats_type)) if return_stats else value_type
            first = fm.F.builtin.call(value, produced, scale, bias, callee="layer", result_type=result_type, name="first")
            second = fm.F.builtin.call(value, produced, scale, bias, callee="layer", result_type=result_type, name="second")
            self.function("main", (value, partial, scale, bias), (first, second))

    return Graph().build(), partial_type, stats_type


@pytest.mark.parametrize("bind_stats", [False, True])
def test_new_boundary_boxing_is_sunk_before_tir_lowering(bind_stats):
    module, partial, _ = _module(bind_stats=bind_stats)
    result = Compiler().compile(module, stop_after="lower-vectorization-contracts").module

    layer = result.function_map[str(result.node_map["first"].attrs["callee"])]
    assert result.node_map[layer.parameters[1]].type == partial
    assert result.node_map["second"].attrs["callee"] == layer.name
    for name in ("first", "second"):
        stats = result.node_map[result.node_map[name].inputs[1]]
        assert stats.type == partial
        assert stats.op == "builtin.call"
    assert any(node.op == "distributed.boxing" and node.inputs == (layer.parameters[1],)
               for node in result.nodes)


def test_stats_return_is_an_additional_use_and_keeps_materialized_abi():
    module, _, materialized = _module(return_stats=True)
    result = Compiler().compile(module, stop_after="lower-vectorization-contracts").module
    layer = result.function_map[str(result.node_map["first"].attrs["callee"])]
    assert result.node_map[layer.parameters[1]].type == materialized
