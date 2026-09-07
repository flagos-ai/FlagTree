# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
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
