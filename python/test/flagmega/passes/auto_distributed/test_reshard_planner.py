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
