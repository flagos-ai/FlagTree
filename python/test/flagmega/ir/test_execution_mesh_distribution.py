# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Execution edits preserve chip ownership while changing local participants."""
from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.ir.execution_mesh import placement_for_execution_mesh
from triton.flagmega.ir.distributed_type import (
    chip_data_type, chip_shared_backing_type, chip_view_error,
    is_component_shard_subview,
)


def test_fixed_execution_edit_preserves_device_axes():
    placement = fm.Placement((4, 8, 16), 'dyx', 'cbb')
    changed = placement_for_execution_mesh(placement, fm.ExecutionMesh.fixed((4, 16)))
    assert changed == fm.Placement((4, 4, 16), 'dyx', 'cbb')
    assert placement_for_execution_mesh(placement, None) is placement


def test_dynamic_data_placement_contains_no_runtime_cta_count():
    placement = fm.Placement((4, 8, 16), 'dyx', 'cbb')
    changed = placement_for_execution_mesh(placement, fm.ExecutionMesh.dynamic())
    assert changed == fm.Placement((4, 1, 1), 'dyx', 'cbb')


def test_chip_projection_preserves_tp_and_vector_payload():
    tensor = fm.tensor_type(fm.vector_type('bfloat16', (4,)), (4096,))
    source = fm.DistributedType(tensor, (fm.SBP.split_contiguous((0, 1, 2), 8),),
                                fm.Placement((4, 8, 16), 'dyx', 'cbb'))
    assert chip_data_type(source) == fm.DistributedType(
        tensor, (fm.SBP.split_contiguous((0,), 1024),), fm.Placement((4,), 'd', 'c'))


def test_cross_mesh_alias_requires_the_same_chip_region():
    tensor = fm.tensor_type('float32', (4096,))
    source = fm.DistributedType(tensor, (fm.SBP.split_contiguous((0, 1, 2), 8),),
                                fm.Placement((4, 8, 16), 'dyx', 'cbb'))
    good = fm.DistributedType(tensor, (fm.SBP.split_contiguous((0, 1, 2), 16),),
                              fm.Placement((4, 4, 16), 'dyx', 'cbb'))
    bad = fm.DistributedType(tensor, (fm.SBP.broadcast(),), good.placement)
    assert chip_view_error(source, good) is None
    assert chip_view_error(source, bad) is not None
    assert fm.sharded_view_error(source, good) is None


def test_block_only_split_becomes_a_complete_chip_value():
    tensor = fm.tensor_type('float32', (4096,))
    source = fm.DistributedType(tensor, (fm.SBP.split_contiguous((0, 1), 32),),
                                fm.Placement((8, 16), 'yx', 'bb'))
    assert chip_data_type(source) == tensor


@pytest.mark.parametrize('policy', [fm.SBP.partial((1,)), fm.SBP.exclusive((1,), (0,))])
def test_unpublished_block_partial_or_exclusive_is_not_relabelled_b(policy):
    tensor = fm.tensor_type('float32', (4096,))
    placement = fm.Placement((4, 8), 'db', 'cb')
    field = {'partial' if isinstance(policy, fm.SBPPartial) else 'exclusive': policy}
    source = fm.DistributedType(tensor, (fm.SBP.broadcast(),), placement, **field)
    target = fm.DistributedType(tensor, (fm.SBP.broadcast(),), fm.Placement((4, 1), 'db', 'cb'))
    assert chip_data_type(source) is None
    assert chip_view_error(source, target) is not None


def test_cross_device_partial_is_preserved():
    tensor = fm.tensor_type('float32', (4096,))
    source = fm.DistributedType(tensor, (fm.SBP.broadcast(),), fm.Placement((4, 8), 'db', 'cb'),
                                partial=fm.SBP.partial((0,)))
    assert chip_data_type(source) == fm.DistributedType(
        tensor, (fm.SBP.broadcast(),), fm.Placement((4,), 'd', 'c'), partial=fm.SBP.partial((0,)))
    target = fm.DistributedType(tensor, (fm.SBP.broadcast(),), fm.Placement((4, 1), 'db', 'cb'),
                                partial=fm.SBP.partial((0,)))
    assert chip_view_error(source, target) is None


@pytest.mark.parametrize('bad_policy', [None, fm.SBP.partial((0,), 'max')])
def test_cross_mesh_component_view_cannot_publish_or_change_a_device_partial(bad_policy):
    tensor = fm.tensor_type('float32', (4096,))
    source = fm.DistributedType(tensor, (fm.SBP.broadcast(),), fm.Placement((4, 8), 'db', 'cb'),
                                partial=fm.SBP.partial((0,)))
    target = fm.DistributedType(tensor, (fm.SBP.broadcast(),), fm.Placement((4, 1), 'db', 'cb'),
                                partial=bad_policy)
    assert chip_view_error(source, target) is not None


@pytest.mark.parametrize('ownership', ['split', 'partial', 'exclusive'])
def test_complete_chip_backing_preserves_device_contract_under_a_new_mesh(ownership):
    tensor = fm.tensor_type(fm.vector_type('bfloat16', (2, 4)), (4096,))
    source_place = fm.Placement((4, 1, 1), 'dyx', 'cbb')
    target_place = fm.Placement((4, 8, 16), 'dyx', 'cbb')
    policy = fm.SBP.split_contiguous((0,), 1024) if ownership == 'split' else fm.SBP.broadcast()
    fields = {'partial': fm.SBP.partial((0,))} if ownership == 'partial' else (
        {'exclusive': fm.SBP.exclusive((0,), (2,))} if ownership == 'exclusive' else {})
    source = fm.DistributedType(tensor, (policy,), source_place, **fields)
    backing = chip_shared_backing_type(source, target_place)
    assert backing is not None
    assert backing.tensor == tensor and backing.placement == target_place
    assert chip_data_type(backing) == chip_data_type(source)
    assert backing.partial == source.partial and backing.exclusive == source.exclusive
    assert fm.block_owner_count(backing) == 1
    assert fm.local_shape(backing) == fm.local_shape(source)


@pytest.mark.parametrize('reason', ['fragmented', 'block_partial', 'block_exclusive', 'device_count', 'device_name'])
def test_incomplete_or_different_device_images_cannot_reuse_a_chip_backing(reason):
    tensor = fm.tensor_type('float32', (4096,))
    source_place = fm.Placement((4, 4, 16), 'dyx', 'cbb')
    source = fm.DistributedType(tensor, (fm.SBP.broadcast(),), source_place, partial=fm.SBP.partial((0,)))
    target_place = fm.Placement((4, 8, 16), 'dyx', 'cbb')
    if reason == 'fragmented':
        source = replace(source, axis_policies=(fm.SBP.split_contiguous((1, 2)),))
    elif reason == 'block_partial':
        source = replace(source, partial=fm.SBP.partial((1,)))
    elif reason == 'block_exclusive':
        source = replace(source, partial=None, exclusive=fm.SBP.exclusive((1,), (0,)))
    elif reason == 'device_count':
        target_place = fm.Placement((8, 8, 16), 'dyx', 'cbb')
    else:
        target_place = fm.Placement((4, 8, 16), 'cyx', 'cbb')
    assert chip_shared_backing_type(source, target_place) is None


@pytest.mark.parametrize('change', ['publish', 'reduction', 'owner', 'widen'])
def test_component_subview_cannot_change_partial_semantics_or_widen_geometry(change):
    tensor = fm.tensor_type('float32', (4096,))
    placement = fm.Placement((4, 8, 16), 'dyx', 'cbb')
    source = fm.DistributedType(tensor, (fm.SBP.split_contiguous((2,)),), placement,
                                partial=fm.SBP.partial((0,)))
    target = replace(source, axis_policies=(fm.SBP.split_contiguous((2, 1)),))
    assert is_component_shard_subview(source, target)
    if change == 'publish':
        target = replace(target, partial=None)
    elif change == 'reduction':
        target = replace(target, partial=fm.SBP.partial((0,), 'max'))
    elif change == 'owner':
        # The parent itself has no split over this axis, so use a complete
        # component to isolate ownership from the geometry rejection.
        source = replace(source, axis_policies=(fm.SBP.broadcast(),))
        target = replace(source, partial=fm.SBP.partial((1,)))
    else:
        target = replace(source, axis_policies=(fm.SBP.broadcast(),))
    assert not is_component_shard_subview(source, target)


def test_component_subview_preserves_the_exact_device_exclusive_owner():
    tensor = fm.tensor_type('float32', (4096,))
    placement = fm.Placement((4, 8, 16), 'dyx', 'cbb')
    source = fm.DistributedType(tensor, (fm.SBP.broadcast(),), placement,
                                exclusive=fm.SBP.exclusive((0,), (2,)))
    target = source
    assert is_component_shard_subview(source, target)
    assert not is_component_shard_subview(source, replace(target, exclusive=fm.SBP.exclusive((0,), (1,))))
