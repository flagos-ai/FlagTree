# Copyright 2026- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.passes.auto_distributed import (
    DistributedReshardRealization,
    DistributedReshardRealizationContext,
    DistributedReshardSourceKind,
    DistributedReshardUsageKind,
    PyNttDistributedReshardRealizationPolicy,
)
from triton.flagmega.passes.auto_distributed.reshard import reshard_step_cost
from triton.flagmega.passes.auto_distributed.reshard_cost import DistributedReshardCostModel


def _types(axes):
    placement = fm.Placement((8, 16), "yx", "bb")
    tensor = fm.tensor_type("bfloat16", (1, 256))
    broadcast = fm.DistributedType(
        tensor, (fm.SBP.broadcast(), fm.SBP.broadcast()), placement
    )
    exclusive = fm.DistributedType(
        tensor,
        broadcast.axis_policies,
        placement,
        exclusive=fm.SBP.exclusive(axes),
    )
    return broadcast, exclusive


@pytest.mark.parametrize("axes,active_owners", [((0,), 16), ((1,), 8), ((0, 1), 1)])
def test_exclusive_sbp_tracks_selected_mesh_axes(axes, active_owners):
    _, value = _types(axes)

    assert value.exclusive.axes == axes
    assert fm.exclusive_owner_count(value) == 128 // active_owners
    assert not fm.is_fully_replicated(value)
    assert fm.type_from_data(value.to_data()) == value


def test_exclusive_transition_is_a_sharded_view_with_sync_cost():
    broadcast, exclusive = _types((0,))
    context = DistributedReshardRealizationContext(
        broadcast,
        exclusive,
        DistributedReshardSourceKind.INTERNAL,
        DistributedReshardUsageKind.INTERNAL,
    )

    assert PyNttDistributedReshardRealizationPolicy().classify(context) \
        is DistributedReshardRealization.SHARDED_VIEW
    assert DistributedReshardCostModel(grid_synchronization_cost=2200).realization_cost(
        context, DistributedReshardRealization.SHARDED_VIEW
    ) == 2200


def test_same_exclusive_policy_is_zero_cost():
    _, exclusive = _types((1,))

    assert reshard_step_cost(exclusive, exclusive) == 0


def test_exclusive_transition_uses_only_the_selected_mesh_group():
    broadcast, exclusive = _types((0,))

    assert fm.exclusive_transition_axes(broadcast, exclusive) == (0,)
    assert fm.exclusive_transition_axes(exclusive, broadcast) == (0,)


@pytest.mark.parametrize('op,attrs', [
    ('tensors.pad', {'pad_end': (0, 4), 'pad_value': 0.0}),
    ('tensors.slice_to_shape', {'shape': (1, 253)}),
])
@pytest.mark.parametrize('vector_lanes', [(), (4,)])
def test_shape_extent_changes_preserve_exclusive_value_ownership(op, attrs, vector_lanes):
    placement = fm.Placement((4, 16), 'cx', 'cb')
    dtype = fm.vector_type('bfloat16', vector_lanes) if vector_lanes else 'bfloat16'
    source = fm.DistributedType(fm.tensor_type(dtype, (1, 256)),
                                (fm.SBP.broadcast(), fm.SBP.broadcast()), placement,
                                exclusive=fm.SBP.exclusive((0,), (2,)))
    node = fm.Node('source', 'builtin.var', (), source, attrs={'name': 'source'})
    result = fm.get_definition(op).prepare((node,), attrs).result_type
    assert result.exclusive == source.exclusive
    assert result.partial is None
    assert result.placement == placement
    assert result.tensor.dtype == dtype
