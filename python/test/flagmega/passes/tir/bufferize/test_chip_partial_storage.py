# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A cooperative chip partial has one complete component, not CTA fragments."""

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.passes.tir.bufferize.planner import _distributed_storage_kind

from python.test.flagmega.ir.ops.nn.test_partitioned_linear import graph


def test_complete_chip_partial_is_stored_once_in_each_device_pool():
    module = Compiler().compile(graph(k=2048, n=128, lanes=(2, 4), input_lanes=(2, 4),
        placement=fm.Placement((4, 8, 16), "cyx", "cbb"),
        k_policy=fm.SBP.split_contiguous((0,)), materialize=True)).module
    plan = fm.verify_buffer_plan(module)
    result = plan.buffer_map["projection"]
    assert result.distributed_type.partial == fm.SBPPartial((0,))
    assert result.storage == "workspace"
    assert result.distributed_storage_kind == fm.DistributedBufferStorageKind.COMPACT_PER_CHIP
    assert result.component_shape == (1, 16)
    assert result.nbytes == 256


@pytest.mark.parametrize("scope,partial_axis,role,chip_images,expected", [
    (fm.MemorySharingScope.CHIP, 0, "value", True, fm.DistributedBufferStorageKind.COMPACT_PER_CHIP),
    (fm.MemorySharingScope.CHIP, 1, "value", True, fm.DistributedBufferStorageKind.COMPACT_PER_OWNER),
    (fm.MemorySharingScope.BLOCK, 0, "value", True, fm.DistributedBufferStorageKind.COMPACT_LOCAL),
    (fm.MemorySharingScope.CHIP, 0, "function_argument", True, fm.DistributedBufferStorageKind.COMPACT_PER_OWNER),
    (fm.MemorySharingScope.CHIP, 0, "value", False, fm.DistributedBufferStorageKind.COMPACT_PER_OWNER),
])
def test_scope_and_partial_owners_select_distinct_physical_representations(scope, partial_axis, role, chip_images, expected):
    placement = fm.Placement((4, 8, 16), "cyx", "cbb")
    kind = fm.DistributedType(fm.tensor_type("bfloat16", (1, 128)), (fm.SBP.broadcast(),) * 2,
                              placement, partial=fm.SBPPartial((partial_axis,)))
    actual = _distributed_storage_kind(kind, "workspace", role, function_pool=True,
        function_pool_sharing_scope=scope, rdata_chip_sharding=chip_images)
    assert actual == expected


@pytest.mark.parametrize("block_axes,expected", [((), 0), ((1,), 2), ((2,), 5), ((1, 2), 37)])
def test_remote_component_projection_ignores_chip_and_broadcast_block_replicas(block_axes, expected):
    from triton.flagmega.codegen.triton.kernel_call_renderers import _physical_component_owner

    placement = fm.Placement((4, 8, 16), "cyx", "cbb")
    policy = fm.SBP.split_contiguous(block_axes) if block_axes else fm.SBP.broadcast()
    kind = fm.DistributedType(fm.tensor_type("bfloat16", (1, 4096)), (fm.SBP.broadcast(), policy),
                              placement, partial=fm.SBPPartial((0,)))
    abi = {"storage_kind": "compact_per_chip", "distributed_type": kind.to_data()}
    expression = _physical_component_owner(abi, ("3", "2", "5"), placement.hierarchy)
    assert eval(expression, {"__builtins__": {}}) == expected


def test_chip_compact_writer_populates_every_partial_device_component():
    from triton.flagmega.codegen.triton.kernel_call_renderers import _canonical_writer_active

    placement = fm.Placement((4, 8, 16), "cyx", "cbb")
    kind = fm.DistributedType(fm.tensor_type("float32", (1, 4096)),
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((2,))), placement, partial=fm.SBPPartial((0,)))
    abi = {"storage_kind": "compact_per_chip", "distributed_type": kind.to_data()}
    predicate = _canonical_writer_active(abi)
    for chip in range(4):
        for replica in range(8):
            for owner in range(16):
                active = eval(predicate, {"__builtins__": {}},
                              {"shard_coord0": chip, "shard_coord1": replica, "shard_coord2": owner})
                assert active == (replica == 0), (chip, replica, owner, predicate)
