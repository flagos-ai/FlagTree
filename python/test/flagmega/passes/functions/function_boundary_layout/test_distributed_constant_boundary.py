# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes.auto_distributed import (
    PyNttDistributedReshardRealizationPolicy,
)
from triton.flagmega.passes.functions import (
    propagate_post_auto_distributed_function_boundary_layouts,
)


def _types():
    placement = fm.Placement((2, 2), "yx", "bb")
    tensor = fm.tensor_type("float32", (4, 16))
    broadcast = fm.DistributedType(
        tensor,
        (fm.SBP.broadcast(), fm.SBP.broadcast()),
        placement,
    )
    split = fm.DistributedType(
        tensor,
        (
            fm.SBP.broadcast(),
            fm.SBP.split_contiguous((0, 1), 4),
        ),
        placement,
    )
    return tensor, broadcast, split


def _constant_weight_module():
    tensor, broadcast, split = _types()

    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="ntt", stage="distributed", entry="main")

        def forward(self):
            parameter = self.input("weight", broadcast, id="layer_weight")
            local_weight = fm.F.distributed.boxing(
                parameter,
                split,
                name="layer_weight.local",
            )
            weight = self.weight(
                "layers.0.weight",
                tensor,
                source="unit.safetensors",
                key="layers.0.weight",
                id="weight",
            )
            distributed_weight = fm.F.distributed.sharded_view(
                weight,
                broadcast,
                name="weight.broadcast",
            )
            call = fm.F.builtin.call(
                distributed_weight,
                result_type=split,
                callee="layer",
                name="call",
            )
            self.function("main", (), (call,))
            self.function(
                "layer",
                (parameter,),
                (local_weight,),
                attrs={"reusable": True, "noinline": True},
            )

    return Graph().build()


def test_constant_boundary_boxing_becomes_caller_sharded_view():
    original = _constant_weight_module()
    rewritten = propagate_post_auto_distributed_function_boundary_layouts(
        original,
        PyNttDistributedReshardRealizationPolicy(),
    )

    assert "layer_weight.local" not in rewritten.node_map
    parameter = rewritten.node_map[rewritten.function_map["layer"].parameters[0]]
    assert parameter.type == _types()[2]
    adapter = rewritten.node_map[rewritten.node_map["call"].inputs[0]]
    assert adapter.op == "distributed.sharded_view"
    assert adapter.inputs == ("weight.broadcast",)
    assert adapter.type == parameter.type
    assert not any(node.op == "distributed.boxing" for node in rewritten.nodes)

    value = torch.arange(64, dtype=torch.float32).reshape(4, 16)
    resolver = DictWeightResolver({"layers.0.weight": value})
    expected = TorchEvaluator(resolver).run(original, {})[0]
    actual = TorchEvaluator(resolver).run(rewritten, {})[0]
    torch.testing.assert_close(actual, expected)


def test_mixed_parameter_uses_get_an_explicit_logical_restore():
    _, broadcast, split = _types()

    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="ntt", stage="distributed", entry="main")

        def forward(self):
            parameter = self.input("value", broadcast, id="layer_value")
            local = fm.F.distributed.boxing(parameter, split, name="local")
            direct = fm.F.math.silu(parameter, name="direct")
            value = self.input("value", broadcast, id="value")
            call = fm.F.builtin.call(
                value,
                result_type=fm.TupleType((split, broadcast)),
                callee="layer",
                name="call",
            )
            self.function("main", (value,), (call,))
            self.function("layer", (parameter,), (local, direct))

    module = Graph().build()
    rewritten = propagate_post_auto_distributed_function_boundary_layouts(
        module,
        PyNttDistributedReshardRealizationPolicy(),
    )

    assert rewritten != module
    assert rewritten.node_map["layer_value"].type == split
    assert "local" not in rewritten.node_map
    direct = rewritten.node_map["direct"]
    restore = rewritten.node_map[direct.inputs[0]]
    assert restore.op in {
        "distributed.boxing", "distributed.sharded_view"}
    assert restore.type == broadcast
    assert restore.inputs == ("layer_value",)
