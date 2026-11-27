# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

from triton.flagmega import ir as fm
from triton.flagmega.passes.functions import (
    function_nodes,
    sink_norm_stats_boxing_across_function_boundaries,
)


def _types(*, local_axes: tuple[int, ...] | None = None):
    placement = fm.Placement((4, 8), "yx", "bb")
    value_tensor = fm.tensor_type("float32", (2, 32))
    stats_tensor = fm.tensor_type("float32", (1, 2, 1))
    parameter_tensor = fm.tensor_type("float32", (32,))
    broadcast = fm.SBP.broadcast()
    input_type = fm.DistributedType(
        value_tensor,
        (broadcast, broadcast),
        placement,
    )
    materialized_type = fm.DistributedType(
        stats_tensor,
        (broadcast, broadcast, broadcast),
        placement,
    )
    parameter_type = fm.DistributedType(
        parameter_tensor,
        (broadcast,),
        placement,
    )
    if local_axes is None:
        return input_type, materialized_type, parameter_type, None
    split = fm.SBP.split_contiguous(local_axes)
    local_input_type = fm.DistributedType(
        value_tensor,
        (broadcast, split),
        placement,
    )
    split_parameter_type = fm.DistributedType(
        parameter_tensor,
        (split,),
        placement,
    )
    return input_type, materialized_type, split_parameter_type, local_input_type


def _direct_module(*, additional_consumer=False, second_axes=(0, 1)):
    input_type, materialized_type, parameter_type, _ = _types()
    partial0 = replace(materialized_type, partial=fm.SBP.partial((0, 1)))
    partial1 = replace(materialized_type, partial=fm.SBP.partial(second_axes))

    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="nn", stage="distributed", entry="main")

        def forward(self):
            layer_input = self.input("layer_input", input_type, id="layer_input")
            layer_stats = self.input("layer_stats", materialized_type, id="layer_stats")
            layer_scale = self.input("layer_scale", parameter_type, id="layer_scale")
            layer_bias = self.input("layer_bias", parameter_type, id="layer_bias")
            normalized = fm.F.nn.norm_apply(
                layer_input,
                layer_stats,
                layer_scale,
                layer_bias,
                axis=1,
                epsilon=1e-6,
                use_mean=False,
                name="normalized",
            )

            input0 = self.input("input0", input_type, id="input0")
            input1 = self.input("input1", input_type, id="input1")
            stats0 = self.input("stats0", partial0, id="stats0")
            stats1 = self.input("stats1", partial1, id="stats1")
            scale = self.input("scale", parameter_type, id="scale")
            bias = self.input("bias", parameter_type, id="bias")
            boxed0 = fm.F.distributed.boxing(
                stats0, materialized_type, name="boxed0")
            boxed1 = fm.F.distributed.boxing(
                stats1, materialized_type, name="boxed1")
            result_type = (
                fm.TupleType((input_type, materialized_type))
                if additional_consumer else input_type
            )
            call0 = fm.F.builtin.call(
                input0, boxed0, scale, bias,
                result_type=result_type,
                callee="layer",
                name="call0",
            )
            call1 = fm.F.builtin.call(
                input1, boxed1, scale, bias,
                result_type=result_type,
                callee="layer",
                name="call1",
            )
            outputs = (normalized, layer_stats) if additional_consumer else (normalized,)
            self.function(
                "layer",
                (layer_input, layer_stats, layer_scale, layer_bias),
                outputs,
                attrs={"reusable": True, "noinline": True},
            )
            self.function(
                "main",
                (input0, input1, stats0, stats1, scale, bias),
                (call0, call1),
            )

    return Graph().build(), materialized_type, partial0, partial1


def test_sinks_consistent_partial_boxing_into_single_norm_apply_consumer():
    module, materialized, partial, _ = _direct_module()
    rewritten = sink_norm_stats_boxing_across_function_boundaries(module)

    layer = rewritten.function_map["layer"]
    assert rewritten.node_map[layer.parameters[1]].type == partial
    apply = next(
        node for node in rewritten.nodes
        if node.op == "nn.norm_apply" and node.id == "normalized"
    )
    boxing = rewritten.node_map[apply.inputs[1]]
    assert boxing.op == "distributed.boxing"
    assert boxing.inputs == (layer.parameters[1],)
    assert boxing.type == materialized
    assert rewritten.node_map["call0"].inputs[1] == "stats0"
    assert rewritten.node_map["call1"].inputs[1] == "stats1"
    assert "boxed0" not in rewritten.node_map
    assert "boxed1" not in rewritten.node_map


def test_specializes_reusable_function_for_different_partial_signatures():
    module, materialized, partial0, partial1 = _direct_module(second_axes=(0,))
    rewritten = sink_norm_stats_boxing_across_function_boundaries(module)

    variants = tuple(
        function for function in rewritten.functions
        if function.name.startswith("layer")
    )
    assert tuple(function.name for function in variants) == (
        "layer",
        "layer_norm_stats_layout_1",
    )
    assert tuple(
        rewritten.node_map[function.parameters[1]].type for function in variants
    ) == (partial0, partial1)
    assert rewritten.node_map["call0"].attrs["callee"] == "layer"
    assert rewritten.node_map["call1"].attrs["callee"] == "layer_norm_stats_layout_1"
    for function in variants:
        body = {node.id for node in function_nodes(rewritten, function)}
        boxings = tuple(
            node for node in rewritten.nodes
            if node.id in body and node.op == "distributed.boxing"
        )
        assert len(boxings) == 1
        assert boxings[0].type == materialized
        assert boxings[0].inputs == (function.parameters[1],)


def test_keeps_boundary_when_stats_parameter_has_another_consumer():
    module, materialized, _, _ = _direct_module(additional_consumer=True)
    rewritten = sink_norm_stats_boxing_across_function_boundaries(module)

    assert rewritten == module
    assert rewritten.node_map["layer_stats"].type == materialized


def test_converts_initial_materialized_seed_from_callee_local_sharded_view():
    input_type, materialized_type, parameter_type, local_input_type = _types(
        local_axes=(0, 1)
    )
    assert local_input_type is not None
    partial_type = replace(materialized_type, partial=fm.SBP.partial((0, 1)))

    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="nn", stage="distributed", entry="main")

        def forward(self):
            layer_input = self.input("layer_input", input_type, id="layer_input")
            layer_stats = self.input("layer_stats", materialized_type, id="layer_stats")
            layer_scale = self.input("layer_scale", parameter_type, id="layer_scale")
            layer_bias = self.input("layer_bias", parameter_type, id="layer_bias")
            local_input = fm.F.distributed.sharded_view(
                layer_input, local_input_type, name="local_input")
            bound = fm.F.nn.bind_norm_stats(
                layer_input,
                layer_stats,
                axis=1,
                use_mean=False,
                name="bound_stats",
            )
            output = fm.F.nn.norm_apply(
                local_input,
                bound,
                layer_scale,
                layer_bias,
                axis=1,
                epsilon=1e-6,
                use_mean=False,
                name="output",
            )

            input0 = self.input("input0", input_type, id="input0")
            input1 = self.input("input1", input_type, id="input1")
            recurrent = self.input("recurrent", partial_type, id="recurrent")
            scale = self.input("scale", parameter_type, id="scale")
            bias = self.input("bias", parameter_type, id="bias")
            seed = fm.F.nn.norm_stats(
                input0, axis=1, use_mean=False, name="seed")
            boxed = fm.F.distributed.boxing(
                recurrent, materialized_type, name="boxed")
            call0 = fm.F.builtin.call(
                input0, seed, scale, bias,
                result_type=local_input_type,
                callee="layer",
                name="call0",
            )
            call1 = fm.F.builtin.call(
                input1, boxed, scale, bias,
                result_type=local_input_type,
                callee="layer",
                name="call1",
            )
            self.function(
                "layer",
                (layer_input, layer_stats, layer_scale, layer_bias),
                (output,),
                attrs={"reusable": True, "noinline": True},
            )
            self.function(
                "main",
                (input0, input1, recurrent, scale, bias),
                (call0, call1),
            )

    rewritten = sink_norm_stats_boxing_across_function_boundaries(Graph().build())
    layer = rewritten.function_map["layer"]
    assert rewritten.node_map[layer.parameters[1]].type == partial_type
    call0 = rewritten.node_map["call0"]
    call1 = rewritten.node_map["call1"]
    seed = rewritten.node_map[call0.inputs[1]]
    assert seed.op == "nn.norm_stats"
    assert seed.type == partial_type
    assert rewritten.node_map[seed.inputs[0]].op == "distributed.sharded_view"
    assert call1.inputs[1] == "recurrent"
    assert "seed" not in rewritten.node_map
    assert "boxed" not in rewritten.node_map
