# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.ir import distributed_type as distributed_type_module
from triton.flagmega.ir import local_shard as local_shard_module


def _types():
    tensor = fm.tensor_type("bfloat16", (1, 2048))
    placement = fm.Placement((8, 16), "yx", "bb")
    split_y = fm.DistributedType(
        tensor,
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((0,), 256)),
        placement,
    )
    split_yx = fm.DistributedType(
        tensor,
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((0, 1), 16)),
        placement,
    )
    split_x = fm.DistributedType(
        tensor,
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((1,), 128)),
        placement,
    )
    return split_y, split_yx, split_x


def test_contiguous_split_refinement_is_a_same_owner_local_subview():
    split_y, split_yx, _ = _types()

    assert fm.is_local_shard_subview(split_y, split_yx)


def test_contiguous_split_refinement_proof_is_cached(monkeypatch):
    split_y, _, split_x = _types()
    original = local_shard_module.uniform_contiguous_owner_regions
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    distributed_type_module.is_local_shard_subview.cache_clear()
    monkeypatch.setattr(local_shard_module, "uniform_contiguous_owner_regions", counted)
    assert not fm.is_local_shard_subview(split_y, split_x)
    assert not fm.is_local_shard_subview(split_y, split_x)
    assert calls == 2
    distributed_type_module.is_local_shard_subview.cache_clear()


def test_contiguous_split_coarsening_or_sibling_ownership_is_not_a_subview():
    split_y, split_yx, split_x = _types()

    assert not fm.is_local_shard_subview(split_yx, split_y)
    assert not fm.is_local_shard_subview(split_y, split_x)


def test_split_to_broadcast_coarsening_does_not_enter_symbolic_proof(monkeypatch):
    split_y, _, _ = _types()
    broadcast = fm.DistributedType(
        split_y.tensor,
        (fm.SBP.broadcast(), fm.SBP.broadcast()),
        split_y.placement,
    )

    def unexpected_descriptor(*args, **kwargs):
        raise AssertionError("split-to-broadcast is a structural coarsening")

    distributed_type_module.is_local_shard_subview.cache_clear()
    monkeypatch.setattr(
        local_shard_module,
        "local_shard_descriptor",
        unexpected_descriptor,
    )
    try:
        assert not fm.is_local_shard_subview(split_y, broadcast)
    finally:
        distributed_type_module.is_local_shard_subview.cache_clear()


def test_larger_cyclic_capacity_is_rejected_before_symbolic_owner_binding(monkeypatch):
    tensor = fm.tensor_type("float32", (128,))
    placement = fm.Placement((8,), "x", "b")
    source = fm.DistributedType(tensor, (fm.SBP.split_block_cyclic((0,), 16),), placement)
    target = fm.DistributedType(tensor, (fm.SBP.split_block_cyclic((0,), 32),), placement)

    def unexpected(*args, **kwargs):
        raise AssertionError("capacity disproves containment without binding owners")

    distributed_type_module.is_local_shard_subview.cache_clear()
    monkeypatch.setattr(local_shard_module, "local_shard_descriptor", unexpected)
    try:
        assert not fm.is_local_shard_subview(source, target)
    finally:
        distributed_type_module.is_local_shard_subview.cache_clear()


def test_noncontiguous_first_stage_is_rejected_without_owner_arithmetic(monkeypatch):
    tensor = fm.tensor_type("float32", (1024,))
    placement = fm.Placement((8,), "x", "b")
    source = fm.DistributedType(tensor, (fm.SBP.split_block_cyclic((0,), 16),), placement)
    target = fm.DistributedType(tensor, (fm.SBP.split_contiguous((0,), 16),), placement)

    def unexpected(*args, **kwargs):
        raise AssertionError("a cyclic stage spanning multiple blocks is not a rectangular region")

    distributed_type_module.is_local_shard_subview.cache_clear()
    monkeypatch.setattr(local_shard_module, "local_shard_descriptor", unexpected)
    try:
        assert not fm.is_local_shard_subview(source, target)
    finally:
        distributed_type_module.is_local_shard_subview.cache_clear()


def test_uniform_sibling_intervals_are_decided_without_symbolic_simplification(monkeypatch):
    source, _, target = _types()

    def unexpected(*args, **kwargs):
        raise AssertionError("fixed uniform shard origins are linear in bounded owner coordinates")

    distributed_type_module.is_local_shard_subview.cache_clear()
    monkeypatch.setattr(local_shard_module, "local_shard_descriptor", unexpected)
    try:
        assert not fm.is_local_shard_subview(source, target)
    finally:
        distributed_type_module.is_local_shard_subview.cache_clear()


def test_uniform_interval_proof_matches_every_concrete_owner():
    from itertools import product

    tensor = fm.tensor_type("float32", (72,))
    placement = fm.Placement((2, 3), "yx", "bb")
    policies = (
        fm.SBP.broadcast(),
        fm.SBP.split_contiguous((0,), 6),
        fm.SBP.split_contiguous((1,), 4),
        fm.SBP.split_contiguous((0, 1), 1),
        fm.SBP.split(fm.SplitStage.contiguous((0,), 6), fm.SplitStage.contiguous((1,), 2)),
        fm.SBP.split_block_cyclic((0, 1), 12),
    )
    types = tuple(fm.DistributedType(tensor, (policy,), placement) for policy in policies)
    for source, target in product(types, repeat=2):
        expected = True
        for coordinate in product(range(2), range(3)):
            parent = local_shard_module.local_shard_descriptor(source, coordinate).contiguous_region
            child = local_shard_module.local_shard_descriptor(target, coordinate).contiguous_region
            start = child.offset[0].fixed_value - parent.offset[0].fixed_value
            expected &= 0 <= start <= parent.shape[0].fixed_value - child.shape[0].fixed_value
        assert fm.is_local_shard_subview(source, target) == expected


def test_dynamic_broadcast_axis_does_not_block_static_split_proof(monkeypatch):
    tensor = fm.tensor_type("float32", (fm.dim("B", 1, 8), 128))
    placement = fm.Placement((2, 2), "yx", "bb")
    source = fm.DistributedType(tensor, (fm.SBP.broadcast(), fm.SBP.split_contiguous((0,), 64)), placement)
    target = fm.DistributedType(tensor, (fm.SBP.broadcast(), fm.SBP.split_contiguous((1,), 64)), placement)

    def unexpected(*args, **kwargs):
        raise AssertionError("identical symbolic broadcast extents cancel exactly")

    distributed_type_module.is_local_shard_subview.cache_clear()
    monkeypatch.setattr(local_shard_module, "local_shard_descriptor", unexpected)
    try:
        assert not fm.is_local_shard_subview(source, target)
    finally:
        distributed_type_module.is_local_shard_subview.cache_clear()


def test_ragged_owner_origins_disprove_containment_without_tail_algebra(monkeypatch):
    tensor = fm.tensor_type("float32", (16,))
    placement = fm.Placement((8, 16), "yx", "bb")
    source = fm.DistributedType(tensor, (fm.SBP.split_block_cyclic((0, 1), 1),), placement)
    target = fm.DistributedType(tensor, (fm.SBP.split_contiguous((1,), 1),), placement)

    def unexpected(*args, **kwargs):
        raise AssertionError("negative owner-origin difference disproves containment even at ragged tails")

    distributed_type_module.is_local_shard_subview.cache_clear()
    monkeypatch.setattr(local_shard_module, "local_shard_descriptor", unexpected)
    try:
        assert not fm.is_local_shard_subview(source, target)
    finally:
        distributed_type_module.is_local_shard_subview.cache_clear()
