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
    split_columns = fm.DistributedType(
        tensor,
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((0, 1))),
        placement,
    )
    split_rows = fm.DistributedType(
        tensor,
        (fm.SBP.split_contiguous((0,)), fm.SBP.broadcast()),
        placement,
    )
    return broadcast, split_columns, split_rows


def _mixed_raw_use_module():
    broadcast, split_columns, _ = _types()

    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="ntt", stage="distributed", entry="main")

        def forward(self):
            parameter = self.input("layer_value", broadcast, id="layer_value")
            local = fm.F.distributed.boxing(
                parameter, split_columns, name="local")
            direct = fm.F.math.silu(parameter, name="direct")

            value = self.input("value", broadcast, id="value")
            call = fm.F.builtin.call(
                value,
                result_type=fm.TupleType((split_columns, broadcast)),
                callee="layer",
                name="call",
            )
            self.function("main", (value,), (call,))
            self.function(
                "layer",
                (parameter,),
                (local, direct),
                attrs={"reusable": True, "noinline": True},
            )

    return Graph().build()


def test_promotes_distributed_parameter_and_restores_only_raw_uses():
    broadcast, split_columns, _ = _types()
    original = _mixed_raw_use_module()
    rewritten = propagate_post_auto_distributed_function_boundary_layouts(
        original,
        PyNttDistributedReshardRealizationPolicy(),
    )

    layer = rewritten.function_map["layer"]
    parameter = rewritten.node_map[layer.parameters[0]]
    assert parameter.type == split_columns
    assert "local" not in rewritten.node_map
    assert layer.outputs[0] == parameter.id

    direct = rewritten.node_map["direct"]
    restore = rewritten.node_map[direct.inputs[0]]
    assert restore.op in {
        "distributed.boxing", "distributed.sharded_view"}
    assert restore.inputs == (parameter.id,)
    assert restore.type == broadcast
    assert restore.metadata["boundary_layout"] == "raw_parameter_restore"

    call = rewritten.node_map["call"]
    caller_adapter = rewritten.node_map[call.inputs[0]]
    assert caller_adapter.op == "distributed.boxing"
    assert caller_adapter.type == split_columns

    value = torch.randn(4, 16)
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(
        evaluator.run(rewritten, {"value": value})[0],
        evaluator.run(original, {"value": value})[0],
    )


def test_constant_alias_candidate_promotes_parameter_and_restores_raw_uses():
    broadcast, split_columns, _ = _types()

    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="ntt", stage="distributed", entry="main")

        def forward(self):
            parameter = self.input("layer_weight", broadcast, id="layer_weight")
            local = fm.F.distributed.boxing(
                parameter, split_columns, name="local")
            direct = fm.F.math.silu(parameter, name="direct")
            weight = self.weight(
                "weight",
                broadcast.tensor,
                source="unit.safetensors",
                key="weight",
                id="weight",
            )
            distributed_weight = fm.F.distributed.sharded_view(
                weight, broadcast, name="weight.broadcast")
            call = fm.F.builtin.call(
                distributed_weight,
                result_type=fm.TupleType((split_columns, broadcast)),
                callee="layer",
                name="call",
            )
            self.function("main", (), (call,))
            self.function(
                "layer",
                (parameter,),
                (local, direct),
                attrs={"reusable": True, "noinline": True},
            )

    original = Graph().build()
    rewritten = propagate_post_auto_distributed_function_boundary_layouts(
        original,
        PyNttDistributedReshardRealizationPolicy(),
    )

    layer = rewritten.function_map["layer"]
    parameter = rewritten.node_map[layer.parameters[0]]
    assert parameter.type == split_columns
    assert "local" not in rewritten.node_map
    restore = rewritten.node_map[rewritten.node_map["direct"].inputs[0]]
    assert restore.op in {
        "distributed.boxing", "distributed.sharded_view"}
    assert restore.inputs == (parameter.id,)
    assert restore.type == broadcast
    caller_adapter = rewritten.node_map[rewritten.node_map["call"].inputs[0]]
    assert caller_adapter.op == "distributed.sharded_view"
    assert caller_adapter.type == split_columns

    value = torch.randn(4, 16)
    resolver = DictWeightResolver({"weight": value})
    torch.testing.assert_close(
        TorchEvaluator(resolver).run(rewritten, {})[0],
        TorchEvaluator(resolver).run(original, {})[0],
    )


def test_multiple_boxing_targets_choose_most_frequent_and_restore_tuple_at_caller():
    broadcast, split_columns, split_rows = _types()

    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="ntt", stage="distributed", entry="main")

        def forward(self):
            parameter = self.input("layer_value", broadcast, id="layer_value")
            column0 = fm.F.distributed.boxing(
                parameter, split_columns, name="column0")
            column1 = fm.F.distributed.boxing(
                parameter, split_columns, name="column1")
            row = fm.F.distributed.boxing(parameter, split_rows, name="row")

            weight = self.weight(
                "weight",
                broadcast.tensor,
                source="unit.safetensors",
                key="weight",
                id="weight",
            )
            value = fm.F.distributed.sharded_view(
                weight, broadcast, name="weight.broadcast")
            call = fm.F.builtin.call(
                value,
                result_type=fm.TupleType(
                    (split_columns, split_columns, split_rows)),
                callee="layer",
                name="call",
            )
            self.function("main", (), (call,))
            self.function(
                "layer",
                (parameter,),
                (column0, column1, row),
                attrs={"reusable": True, "noinline": True},
            )

    rewritten = propagate_post_auto_distributed_function_boundary_layouts(
        Graph().build(),
        PyNttDistributedReshardRealizationPolicy(),
    )

    layer = rewritten.function_map["layer"]
    parameter = rewritten.node_map[layer.parameters[0]]
    assert parameter.type == split_columns
    assert layer.outputs == (parameter.id, parameter.id, parameter.id)
    assert "column0" not in rewritten.node_map
    assert "column1" not in rewritten.node_map
    assert "row" not in rewritten.node_map

    raw_call = next(
        node for node in rewritten.nodes
        if node.op == "builtin.call" and node.attrs["callee"] == "layer"
    )
    assert raw_call.type == fm.TupleType(
        (split_columns, split_columns, split_columns))
    logical_tuple = rewritten.node_map[
        rewritten.function_map["main"].outputs[0]]
    assert logical_tuple.op == "builtin.tuple"
    assert logical_tuple.type == fm.TupleType(
        (split_columns, split_columns, split_rows))
    row_restore = rewritten.node_map[logical_tuple.inputs[2]]
    assert row_restore.op in {
        "distributed.boxing", "distributed.sharded_view"}
    assert row_restore.type == split_rows
    assert rewritten.node_map[row_restore.inputs[0]].type == split_columns
    assert not any(
        node.metadata.get("boundary_layout") == "raw_parameter_restore"
        for node in rewritten.nodes
    )
