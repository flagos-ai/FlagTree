# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

from triton.flagmega import ir as fm
from triton.flagmega.passes.auto_distributed import (
    PyNttDistributedReshardRealizationPolicy,
)
from triton.flagmega.passes.functions import (
    propagate_post_auto_distributed_function_boundary_layouts,
)


def _module(*, mixed_consumer: bool):
    placement = fm.Placement((2, 4), "yx", "bb")
    tensor = fm.tensor_type("float32", (1, 1, 1))
    materialized = fm.DistributedType(
        tensor,
        (fm.SBP.broadcast(), fm.SBP.broadcast(), fm.SBP.broadcast()),
        placement,
    )
    partial = replace(materialized, partial=fm.SBP.partial((0, 1)))

    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="ntt", stage="distributed", entry="main")

        def forward(self):
            layer_input = self.input("layer_input", partial, id="layer_input")
            value = self.input("value", partial, id="value")
            call = fm.F.builtin.call(
                value, result_type=partial, callee="layer", name="call")
            demanded = fm.F.distributed.boxing(
                call, materialized, name="demanded")
            outputs = (demanded, call) if mixed_consumer else (demanded,)
            self.function("main", (value,), outputs)
            self.function(
                "layer",
                (layer_input,),
                (layer_input,),
                attrs={"reusable": True, "noinline": True},
            )

    return Graph().build(), partial, materialized


def test_absorbs_irreversible_partial_materialization_when_all_consumers_match():
    module, partial, materialized = _module(mixed_consumer=False)
    rewritten = propagate_post_auto_distributed_function_boundary_layouts(
        module,
        PyNttDistributedReshardRealizationPolicy(),
    )

    layer = rewritten.function_map["layer"]
    output = rewritten.node_map[layer.outputs[0]]
    assert output.op == "distributed.boxing"
    assert rewritten.node_map[output.inputs[0]].type == partial
    assert output.type == materialized
    assert "demanded" not in rewritten.node_map
    main_output = rewritten.node_map[rewritten.function_map["main"].outputs[0]]
    assert main_output.op == "builtin.call"
    assert main_output.type == materialized


def test_keeps_partial_abi_when_an_unmaterialized_consumer_exists():
    module, partial, materialized = _module(mixed_consumer=True)
    rewritten = propagate_post_auto_distributed_function_boundary_layouts(
        module,
        PyNttDistributedReshardRealizationPolicy(),
    )

    assert rewritten.function_map["layer"].outputs == ("layer_input",)
    assert rewritten.node_map["call"].type == partial
    assert rewritten.node_map["demanded"].type == materialized
