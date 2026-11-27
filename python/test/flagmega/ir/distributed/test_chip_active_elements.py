# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

from itertools import product
from math import prod

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.ir.local_shard import max_chip_active_elements


@pytest.mark.parametrize("levels", ("bbb", "cbb", "dbb", "bcb", "cdb"))
@pytest.mark.parametrize("layout", ("broadcast", "cyclic", "staged", "multidimensional", "empty"))
def test_chip_active_traffic_matches_exhaustive_physical_owner_oracle(levels, layout):
    placement = fm.Placement((4, 3, 2), "xyz", levels)
    b = fm.SBP.broadcast()
    shape, policies = {
        "broadcast": ((17, 5), (b, b)),
        "cyclic": ((17, 5), (fm.SBP.split_block_cyclic((1, 0), 3), b)),
        "staged": ((20, 5), (fm.SBP.split(
            fm.SplitStage.contiguous((0,), 5), fm.SplitStage.block_cyclic((1,), 2)), b)),
        "multidimensional": ((5, 3), (fm.SBP.split_block_cyclic((0,), 3),
                                       fm.SBP.split_contiguous((1,), 2))),
        "empty": ((1, 0), (fm.SBP.split_block_cyclic((0, 1), 4), b)),
    }[layout]
    value = fm.DistributedType(
        fm.tensor_type(fm.vector_type("bfloat16", 8), shape), policies, placement,
    )
    per_chip = {}
    for owner in product(*(range(size) for size in placement.hierarchy)):
        chip = tuple(coordinate for coordinate, level in zip(owner, levels, strict=True) if level != "b")
        active = fm.local_shard_descriptor(value, owner).active_shape
        per_chip[chip] = per_chip.get(chip, 0) + prod(extent.fixed_value for extent in active)
    assert max_chip_active_elements(value) == max(per_chip.values())


@pytest.mark.parametrize("levels", ("bbb", "cbb", "dbb", "bcb", "cdb"))
@pytest.mark.parametrize("axes", ((0,), (1,), (0, 2), (0, 1, 2)))
def test_exclusive_chip_and_block_owners_match_exhaustive_oracle(levels, axes):
    placement = fm.Placement((4, 3, 2), "xyz", levels)
    value = fm.DistributedType(
        fm.tensor_type("float32", (17,)), (fm.SBP.broadcast(),), placement,
        exclusive=fm.SBP.exclusive(axes, (1,) * len(axes)),
    )
    per_chip = {}
    for owner in product(*(range(size) for size in placement.hierarchy)):
        if any(owner[axis] != 1 for axis in axes):
            continue
        chip = tuple(coordinate for coordinate, level in zip(owner, levels, strict=True) if level != "b")
        per_chip[chip] = per_chip.get(chip, 0) + 17
    assert max_chip_active_elements(value) == max(per_chip.values())


def test_busiest_chip_is_not_the_average_of_ragged_shards():
    value = fm.DistributedType(
        fm.tensor_type("float32", (5,)), (fm.SBP.split_block_cyclic((0,), 2),),
        fm.Placement((4, 2), "cx", "cb"),
    )
    # Per-chip traffic is [4, 4, 2, 0], whose average is not its critical path.
    assert max_chip_active_elements(value) == 4


def test_dynamic_chip_traffic_is_unknown():
    value = fm.DistributedType(
        fm.tensor_type("float32", (fm.dim("tokens", minimum=1, maximum=16),)),
        (fm.SBP.split_block_cyclic((0, 1), 2),), fm.Placement((4, 2), "cx", "cb"),
    )
    assert max_chip_active_elements(value) is None
