# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes.auto_distributed import (
    PyNttDistributedReshardRealizationPolicy,
)
from triton.flagmega.passes.functions import (
    function_nodes,
    propagate_post_auto_distributed_function_boundary_layouts,
)


def _types():
    placement = fm.Placement((2, 4), "yx", "bb")
    tensor = fm.tensor_type("float32", (1, 64))
    source = fm.DistributedType(
        tensor,
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((0, 1))),
        placement,
    )
    target = fm.DistributedType(
        tensor,
        (fm.SBP.broadcast(), fm.SBP.broadcast()),
        placement,
    )
    return source, target


def test_caller_output_demand_moves_reshard_into_callee_and_restores_raw_use():
    source, target = _types()

    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="ntt", stage="distributed", entry="main")

        def forward(self):
            layer_input = self.input("layer_input", source, id="layer_input")
            layer_output = fm.F.math.silu(layer_input, name="layer_output")

            value = self.input("value", source, id="value")
            call = fm.F.builtin.call(
                value, result_type=source, callee="layer", name="call")
            demanded = fm.F.distributed.boxing(call, target, name="demanded")
            raw = fm.F.math.silu(call, name="raw")
            self.function("main", (value,), (demanded, raw))
            self.function(
                "layer",
                (layer_input,),
                (layer_output,),
                attrs={"reusable": True, "noinline": True},
            )

    original = Graph().build()
    rewritten = propagate_post_auto_distributed_function_boundary_layouts(
        original,
        PyNttDistributedReshardRealizationPolicy(),
    )

    layer = rewritten.function_map["layer"]
    assert rewritten.node_map[layer.outputs[0]].type == target
    body_adapters = tuple(
        node for node in function_nodes(rewritten, layer)
        if node.op in {"distributed.boxing", "distributed.sharded_view"}
    )
    assert len(body_adapters) == 1
    assert body_adapters[0].type == target

    assert "demanded" not in rewritten.node_map
    raw = rewritten.node_map["raw"]
    restore = rewritten.node_map[raw.inputs[0]]
    assert restore.id == "call"
    assert restore.type == source
    assert restore.op == "distributed.sharded_view"
    raw_call = rewritten.node_map[restore.inputs[0]]
    assert raw_call.op == "builtin.call"
    assert raw_call.type == target
    assert rewritten.function_map["main"].outputs[0] == raw_call.id

    value = torch.randn(1, 64)
    evaluator = TorchEvaluator(DictWeightResolver({}))
    actual = evaluator.run(rewritten, {"value": value})
    expected = evaluator.run(original, {"value": value})
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


def test_tuple_caller_output_demand_retypes_only_selected_port():
    source, target = _types()

    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="ntt", stage="distributed", entry="main")

        def forward(self):
            layer_input = self.input("layer_input", source, id="layer_input")
            first = fm.F.math.silu(layer_input, name="first")
            second = fm.F.math.silu(layer_input, name="second")

            value = self.input("value", source, id="value")
            call = fm.F.builtin.call(
                value,
                result_type=fm.TupleType((source, source)),
                callee="layer",
                name="call",
            )
            projected = fm.F.tensors.get_item(call, 0, name="projected")
            demanded = fm.F.distributed.boxing(
                projected, target, name="demanded")
            self.function("main", (value,), (demanded,))
            self.function(
                "layer",
                (layer_input,),
                (first, second),
                attrs={"reusable": True, "noinline": True},
            )

    rewritten = propagate_post_auto_distributed_function_boundary_layouts(
        Graph().build(),
        PyNttDistributedReshardRealizationPolicy(),
    )

    layer = rewritten.function_map["layer"]
    assert rewritten.node_map[layer.outputs[0]].type == target
    assert rewritten.node_map[layer.outputs[1]].type == source
    raw_call = next(
        node for node in rewritten.nodes
        if node.op == "builtin.call" and node.attrs["callee"] == "layer"
    )
    assert raw_call.type == fm.TupleType((target, source))
    assert "demanded" not in rewritten.node_map
    main_output = rewritten.node_map[rewritten.function_map["main"].outputs[0]]
    assert main_output.op == "builtin.get_item"
    assert main_output.type == target
