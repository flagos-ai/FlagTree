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


def test_internal_output_reshard_moves_to_each_caller_compatibility_view():
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

    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="ntt", stage="distributed", entry="main")

        def forward(self):
            parameter = self.input("parameter", source, id="parameter")
            computed = fm.F.math.silu(parameter, name="computed")
            boxed = fm.F.distributed.boxing(computed, target, name="boxed")

            value = self.input("value", source, id="value")
            call = fm.F.builtin.call(
                value, result_type=target, callee="layer", name="call")
            self.function("main", (value,), (call,))
            self.function(
                "layer",
                (parameter,),
                (boxed,),
                attrs={"reusable": True, "noinline": True},
            )

    original = Graph().build()
    rewritten = propagate_post_auto_distributed_function_boundary_layouts(
        original,
        PyNttDistributedReshardRealizationPolicy(),
    )

    layer = rewritten.function_map["layer"]
    assert layer.outputs == ("computed",)
    assert not any(
        node.op in {"distributed.boxing", "distributed.sharded_view"}
        for node in function_nodes(rewritten, layer)
    )
    raw_call = rewritten.node_map["call.distributed_boundary_abi"]
    assert raw_call.op == "builtin.call"
    assert raw_call.type == source
    logical_call = rewritten.node_map["call"]
    assert logical_call.op in {
        "distributed.boxing", "distributed.sharded_view"}
    assert logical_call.inputs == (raw_call.id,)
    assert logical_call.type == target

    value = torch.randn(1, 64)
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(
        evaluator.run(rewritten, {"value": value})[0],
        evaluator.run(original, {"value": value})[0],
    )
