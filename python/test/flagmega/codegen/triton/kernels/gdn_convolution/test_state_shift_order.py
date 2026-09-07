# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega.codegen.triton.kernel_call_renderers import (
    _distributed_unique_writer_active,
)


def _abi(*axis_policies):
    return {
        "distributed_type": {
            "placement": {"hierarchy": (8, 16)},
            "axis_policies": axis_policies,
        }
    }


def test_broadcast_stateful_result_elects_one_grid_owner():
    broadcast = {"kind": "broadcast"}

    assert _distributed_unique_writer_active(
        _abi(broadcast, broadcast)
    ) == "(shard_y == 0) & (shard_x == 0)"


def test_stateful_result_elects_one_replica_per_distinct_shard():
    broadcast = {"kind": "broadcast"}
    split_y = {
        "kind": "split",
        "stages": ({"hierarchy_axes": (0,)},),
    }

    assert _distributed_unique_writer_active(
        _abi(broadcast, split_y)
    ) == "(shard_x == 0)"
