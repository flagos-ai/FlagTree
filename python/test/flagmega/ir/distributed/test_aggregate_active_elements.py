# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.ir.local_shard import aggregate_active_elements


@pytest.mark.parametrize("mesh", [(3,), (3, 2), (3, 2, 4)])
def test_aggregate_counts_active_elements_and_unused_mesh_replicas(mesh):
    from math import prod
    value = fm.DistributedType(
        fm.tensor_type(fm.vector_type("bfloat16", 8), (12, 5)),
        (fm.SBP.split_contiguous((0,), 5), fm.SBP.broadcast()),
        fm.Placement(mesh, "xyz"[:len(mesh)], "b" * len(mesh)))
    assert aggregate_active_elements(value) == 12 * 5 * prod(mesh[1:])


@pytest.mark.parametrize("ownership,expected", [(None, 240), ("partial", 240), ("exclusive", 60)])
def test_aggregate_respects_exclusive_owners_and_partial_replicas(ownership, expected):
    value = fm.DistributedType(
        fm.tensor_type("float32", (5, 6)), (fm.SBP.broadcast(),) * 2, fm.Placement((4, 2), "xy", "bb"),
        partial=fm.SBP.partial((0,)) if ownership == "partial" else None,
        exclusive=fm.SBP.exclusive((0,), (2,)) if ownership == "exclusive" else None)
    assert aggregate_active_elements(value) == expected


def test_dynamic_active_traffic_is_unknown_not_capacity_estimate():
    value = fm.DistributedType(
        fm.tensor_type("float32", (fm.dim("tokens", minimum=1, maximum=16),)),
        (fm.SBP.split_block_cyclic((0,), 2),), fm.Placement((4,), "x", "b"))
    assert aggregate_active_elements(value) is None
