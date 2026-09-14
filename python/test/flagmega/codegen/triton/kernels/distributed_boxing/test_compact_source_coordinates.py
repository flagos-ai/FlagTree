# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Routed collective addresses invert the shared LocalShardDescriptor contract."""

from itertools import product
from types import SimpleNamespace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.kernel_call_renderers import _compact_source_coordinates
from triton.flagmega.ir.local_shard import local_shard_descriptor


@pytest.mark.parametrize("policy", [
    fm.SBP.split_contiguous((0, 1), 5),
    fm.SBP.split_block_cyclic((1, 0), 3),
    fm.SBP.split(fm.SplitStage.contiguous((0,), 20), fm.SplitStage.block_cyclic((1,), 3)),
    fm.SBP.split(fm.SplitStage.block_cyclic((1,), 6), fm.SplitStage.contiguous((0,), 6)),
])
def test_inverse_matches_staged_ragged_owner_maps(policy):
    placement = fm.Placement((2, 4, 2), "xyz", "bbb")
    value_type = fm.DistributedType(
        fm.tensor_type("float32", (40,)), (policy,), placement,
    )
    local, owner = _compact_source_coordinates(
        {"distributed_type": value_type.to_data(), "logical_shape": (40,)}, ("global_index",)
    )
    seen = set()
    for x, y in product(range(2), range(4)):
        z = 0
        shard = local_shard_descriptor(value_type, (x, y, z)).axes[0]
        for index in range(shard.active_extent.fixed_value):
            logical = shard.map_local_to_global(index).fixed_value
            scope = {"global_index": logical, "tl": SimpleNamespace(minimum=min, maximum=max)}
            assert eval(local[0], {"__builtins__": {}}, scope) == index
            assert eval(owner, {"__builtins__": {}}, scope) == (x * 4 + y) * 2 + z
            seen.add(logical)
    assert seen == set(range(40))
