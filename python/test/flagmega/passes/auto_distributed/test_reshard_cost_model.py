# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from dataclasses import replace
from triton.flagmega.targets import NvidiaSm90Target
from triton.flagmega.passes.auto_distributed import (
    DistributedReshardCostModel,
    DistributedReshardRealization,
    DistributedReshardRealizationContext,
    DistributedReshardSourceKind,
    DistributedReshardUsageKind,
)


def _types():
    tensor = fm.tensor_type("bfloat16", (1, 2048))
    placement = fm.Placement((8, 16), "yx", "bb")
    split = fm.DistributedType(
        tensor,
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((0, 1), 16)),
        placement,
    )
    broadcast = fm.DistributedType(
        tensor,
        (fm.SBP.broadcast(), fm.SBP.broadcast()),
        placement,
    )
    return tensor, split, broadcast


def test_internal_widening_sharded_view_costs_one_grid_synchronization():
    _, split, broadcast = _types()
    model = DistributedReshardCostModel(grid_synchronization_cost=2200)
    context = DistributedReshardRealizationContext(
        split,
        broadcast,
        DistributedReshardSourceKind.INTERNAL,
        DistributedReshardUsageKind.INTERNAL,
    )

    assert model.realization_cost(
        context, DistributedReshardRealization.SHARDED_VIEW
    ) == 2200


def test_local_subview_and_terminal_output_alias_are_zero_cost():
    _, split, broadcast = _types()
    model = DistributedReshardCostModel(grid_synchronization_cost=2200)
    local = DistributedReshardRealizationContext(
        broadcast,
        split,
        DistributedReshardSourceKind.INTERNAL,
        DistributedReshardUsageKind.INTERNAL,
    )
    terminal = DistributedReshardRealizationContext(
        split,
        broadcast,
        DistributedReshardSourceKind.INTERNAL,
        DistributedReshardUsageKind.PROGRAM_OUTPUT,
    )

    assert model.realization_cost(
        local, DistributedReshardRealization.SHARDED_VIEW
    ) == 0
    assert model.realization_cost(
        terminal, DistributedReshardRealization.SHARDED_VIEW
    ) == 0


def test_logical_constant_view_does_not_charge_runtime_synchronization():
    tensor, split, _ = _types()
    model = DistributedReshardCostModel(grid_synchronization_cost=2200)
    context = DistributedReshardRealizationContext(
        tensor,
        split,
        DistributedReshardSourceKind.CONSTANT,
        DistributedReshardUsageKind.INTERNAL,
    )

    assert model.realization_cost(
        context, DistributedReshardRealization.SHARDED_VIEW
    ) == 0


def test_materialized_stats_cost_depends_on_collective_fan_in():
    placement = fm.Placement((2, 8), "yx", "bb")
    tensor = fm.tensor_type("float32", (1, 1, 16, 1))
    output = fm.DistributedType(tensor, (fm.SBP.broadcast(),) * 4, placement)
    target = NvidiaSm90Target()
    costs = []
    for axis in (0, 1):
        source = replace(output, partial=fm.SBP.partial((axis,), fm.ReduceOp.SUM))
        context = DistributedReshardRealizationContext(
            source, output, DistributedReshardSourceKind.INTERNAL, DistributedReshardUsageKind.INTERNAL,
        )
        costs.append(target.distributed_reshard_cost_model().realization_cost(context, DistributedReshardRealization.BOXING))
    assert costs[1] > costs[0]
    assert costs[0] >= target.distributed_operation_cost_model().grid_synchronization_cycles
