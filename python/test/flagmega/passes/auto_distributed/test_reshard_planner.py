# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.passes.auto_distributed import DistributedReshardPlanner, can_box


def _types():
    tensor = fm.tensor_type("bfloat16", [1, 128])
    placement = fm.Placement((8,), "b", "b")
    broadcast = fm.DistributedType(
        tensor, (fm.SBP.broadcast(), fm.SBP.broadcast()), placement)
    split = fm.DistributedType(
        tensor, (fm.SBP.broadcast(), fm.SBP.split_contiguous((0,))), placement)
    partial = fm.DistributedType(
        tensor,
        (fm.SBP.broadcast(), fm.SBP.broadcast()),
        placement,
        partial=fm.SBP.partial((0,)),
    )
    return tensor, broadcast, split, partial


def test_direct_reshard_is_preferred_when_boxing_supports_it():
    _, broadcast, split, _ = _types()

    plans = DistributedReshardPlanner.plan(broadcast, split)

    assert len(plans) == 1
    assert plans[0].step_types == (split,)


def test_complete_device_partial_can_handoff_to_the_collective_mesh_before_reduction():
    from dataclasses import replace
    tensor = fm.tensor_type(fm.vector_type('bfloat16', (2, 4)), (1, 512))
    source = fm.DistributedType(tensor, (fm.SBP.broadcast(),) * 2,
        fm.Placement((4, 1, 1), 'cyx', 'cbb'), partial=fm.SBP.partial((0,)))
    target = fm.DistributedType(tensor, (fm.SBP.broadcast(),) * 2,
        fm.Placement((4, 8, 16), 'cyx', 'cbb'))
    assert not DistributedReshardPlanner.plan(source, target, max_hops=1)
    plans = DistributedReshardPlanner.plan(source, target, max_hops=2)
    assert plans and plans[0].step_types == (replace(source, placement=target.placement), target)


def test_cross_mesh_component_handoff_does_not_ignore_device_identity_or_fragmentation():
    tensor = fm.tensor_type('float32', (1, 4096))
    source_place = fm.Placement((4, 4, 16), 'cyx', 'cbb')
    source = fm.DistributedType(tensor, (fm.SBP.broadcast(), fm.SBP.split_contiguous((1, 2))),
        source_place, partial=fm.SBP.partial((0,)))
    target = fm.DistributedType(tensor, (fm.SBP.broadcast(),) * 2, fm.Placement((4, 8, 16), 'cyx', 'cbb'))
    assert not DistributedReshardPlanner.plan(source, target)
    complete = fm.DistributedType(tensor, (fm.SBP.broadcast(),) * 2,
        fm.Placement((4, 1, 1), 'cyx', 'cbb'), partial=fm.SBP.partial((0,)))
    for place in (fm.Placement((8, 8, 16), 'cyx', 'cbb'), fm.Placement((4, 8, 16), 'dyx', 'cbb')):
        target = fm.DistributedType(tensor, (fm.SBP.broadcast(),) * 2, place)
        assert not DistributedReshardPlanner.plan(complete, target)


def test_component_handoff_cannot_bypass_the_target_edge_policy():
    tensor = fm.tensor_type('float32', (1, 4096))
    source = fm.DistributedType(tensor, (fm.SBP.broadcast(),) * 2,
        fm.Placement((4, 1, 1), 'cyx', 'cbb'), partial=fm.SBP.partial((0,)))
    target = fm.DistributedType(tensor, (fm.SBP.broadcast(),) * 2,
        fm.Placement((4, 8, 16), 'cyx', 'cbb'))
    # A logical component proof does not authorize physical backing or a
    # collective that the target cannot realize. Reject either stage.
    for reject_handoff in (True, False):
        def supported(lhs, rhs):
            forbidden = lhs.placement != rhs.placement if reject_handoff else rhs.partial is None
            return not forbidden and can_box(lhs, rhs)
        assert not DistributedReshardPlanner.plan(source, target, supported)


def test_planner_builds_bounded_partial_broadcast_path_when_direct_edge_is_illegal():
    _, broadcast, split, partial = _types()

    def only_staged(source, target):
        if source == partial and target == split:
            return False
        return can_box(source, target)

    plans = DistributedReshardPlanner.plan(partial, split, only_staged, max_hops=3)

    assert plans
    assert all(len(plan.step_types) <= 3 for plan in plans)
    assert any(broadcast in plan.step_types for plan in plans)


def test_planner_keeps_direct_and_partial_reduce_scatter_programs_for_global_search():
    tensor = fm.tensor_type("bfloat16", [16, 16])
    placement = fm.Placement((2, 4), "bd", "bb")
    source = fm.DistributedType(
        tensor,
        (fm.SBP.broadcast(), fm.SBP.broadcast()),
        placement,
        partial=fm.SBP.partial((0, 1)),
    )
    target = fm.DistributedType(
        tensor,
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((0,), 8)),
        placement,
    )

    plans = DistributedReshardPlanner.plan(source, target)

    assert any(plan.step_types == (target,) for plan in plans)
    staged = next(plan for plan in plans if len(plan.step_types) == 2)
    intermediate = staged.step_types[0]
    assert isinstance(intermediate, fm.DistributedType)
    assert intermediate.partial is None
    assert intermediate.axis_policies[0] == fm.SBP.split_contiguous((1,), 4)
    assert intermediate.axis_policies[1] == target.axis_policies[1]


def test_planner_rejects_logically_different_tensors():
    _, _, split, _ = _types()
    other = fm.tensor_type("bfloat16", [1, 64])

    assert DistributedReshardPlanner.plan(other, split) == ()


def test_partial_reduce_scatter_skips_tensor_axes_smaller_than_the_placement():
    tensor = fm.tensor_type("bfloat16", [1, 1024])
    placement = fm.Placement((8,), "b", "b")
    source = fm.DistributedType(
        tensor,
        (fm.SBP.broadcast(), fm.SBP.broadcast()),
        placement,
        partial=fm.SBP.partial((0,)),
    )
    target = fm.DistributedType(
        tensor,
        (fm.SBP.broadcast(), fm.SBP.broadcast()),
        placement,
    )

    plans = DistributedReshardPlanner.plan(source, target)

    intermediates = {
        plan.step_types[0]
        for plan in plans
        if len(plan.step_types) == 2
    }
    assert intermediates
    assert all(isinstance(value, fm.DistributedType) for value in intermediates)
    assert all(isinstance(value.axis_policies[0], fm.SBPBroadCast) for value in intermediates)
    assert any(isinstance(value.axis_policies[1], fm.SBPSplit) for value in intermediates)
