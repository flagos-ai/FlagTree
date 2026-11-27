# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega.codegen.triton.kernel_call_renderers import (
    _distributed_unique_writer_active, )


def _abi(*axis_policies):
    return {"distributed_type": {
        "placement": {"hierarchy": (8, 16)},
        "axis_policies": axis_policies,
    }}


def test_broadcast_stateful_result_elects_one_grid_owner():
    broadcast = {"kind": "broadcast"}

    assert _distributed_unique_writer_active(_abi(broadcast, broadcast)) == "(shard_y == 0) & (shard_x == 0)"


def test_stateful_result_elects_one_replica_per_distinct_shard():
    broadcast = {"kind": "broadcast"}
    split_y = {
        "kind": "split",
        "stages": ({"hierarchy_axes": (0, )}, ),
    }

    # One writer for each of the 8 split shards.  Which column is elected is a
    # placement choice (owners are spread so that values splitting different
    # axes of one call do not pile onto the same block), so assert the count.
    expression = _distributed_unique_writer_active(_abi(broadcast, split_y))
    owners = [(shard_y, shard_x)
              for shard_y in range(8)
              for shard_x in range(16)
              if eval(  # noqa: S307 - rendered predicate over fixed local names
                  expression,
                  {"__builtins__": {}},
                  {"shard_y": shard_y, "shard_x": shard_x},
              )]
    assert len(owners) == 8
    assert len({shard_y for shard_y, _ in owners}) == 8
