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
    original = local_shard_module.local_shard_descriptor
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    distributed_type_module.is_local_shard_subview.cache_clear()
    monkeypatch.setattr(local_shard_module, "local_shard_descriptor", counted)
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
