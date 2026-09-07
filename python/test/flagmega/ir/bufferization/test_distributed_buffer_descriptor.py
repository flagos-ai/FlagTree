# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm


def _distributed_type():
    placement = fm.Placement((2, 4), "yx", "bb")
    return fm.DistributedType(
        fm.tensor_type("bfloat16", (64, 32)),
        (fm.SBP.split_contiguous((0, 1)), fm.SBP.broadcast()),
        placement,
    )


def test_compact_per_owner_descriptor_separates_component_span_from_backing():
    distributed = _distributed_type()
    component_bytes = 8 * 32 * 2
    physical = fm.PhysicalBuffer(
        "workspace:distributed", "workspace", component_bytes * 8, 256
    )
    descriptor = fm.BufferDescriptor(
        "sharded",
        fm.DType.BFLOAT16,
        (64, 32),
        (32, 1),
        "workspace",
        256,
        fm.MemSpan(physical, 0, component_bytes),
        distributed_type=distributed,
        distributed_storage_kind=fm.DistributedBufferStorageKind.COMPACT_PER_OWNER,
    )

    assert descriptor.local_shape == (8, 32)
    assert descriptor.component_shape == (8, 32)
    assert descriptor.nbytes == component_bytes
    assert descriptor.mem_span.buffer.nbytes == component_bytes * 8
    assert descriptor.physical_access_span == fm.MemSpan(physical, 0, component_bytes * 8)

    restored = fm.BufferDescriptor.from_data(
        descriptor.to_data(), {physical.id: physical}
    )
    assert restored == descriptor


def test_canonical_global_descriptor_exposes_logical_storage_component():
    distributed = _distributed_type()
    global_bytes = 64 * 32 * 2
    physical = fm.PhysicalBuffer(
        "workspace:global", "workspace", global_bytes, 256
    )
    descriptor = fm.BufferDescriptor(
        "global",
        fm.DType.BFLOAT16,
        (64, 32),
        (32, 1),
        "workspace",
        256,
        fm.MemSpan(physical, 0, global_bytes),
        distributed_type=distributed,
        distributed_storage_kind=fm.DistributedBufferStorageKind.CANONICAL_GLOBAL,
    )

    assert descriptor.local_shape == (8, 32)
    assert descriptor.component_shape == (64, 32)
    assert descriptor.physical_access_span is descriptor.mem_span


def test_partial_descriptor_rejects_collapsed_canonical_storage():
    placement = fm.Placement((2, 2), "yx", "bb")
    partial = fm.DistributedType(
        fm.tensor_type("float32", (1,)), (fm.SBP.broadcast(),),
        placement, fm.SBP.partial((0, 1)),
    )
    physical = fm.PhysicalBuffer("external:stats", "external", 4, 4)

    with pytest.raises(ValueError, match="Partial buffer.*independent owner components"):
        fm.BufferDescriptor(
            "stats", fm.DType.FLOAT32, (1,), (1,), "return", 4,
            fm.MemSpan(physical, 0, 4), distributed_type=partial,
            distributed_storage_kind=fm.DistributedBufferStorageKind.CANONICAL_GLOBAL,
        )


def test_replicated_local_descriptor_exposes_one_complete_logical_replica():
    distributed = _distributed_type()
    global_bytes = 64 * 32 * 2
    physical = fm.PhysicalBuffer(
        "block_local:replica", "block_local_data", global_bytes, 256
    )
    descriptor = fm.BufferDescriptor(
        "replica",
        fm.DType.BFLOAT16,
        (64, 32),
        (32, 1),
        "block_local_data",
        256,
        fm.MemSpan(physical, 0, global_bytes),
        distributed_type=distributed,
        distributed_storage_kind=fm.DistributedBufferStorageKind.REPLICATED_LOCAL,
    )

    assert descriptor.local_shape == (8, 32)
    assert descriptor.component_shape == (64, 32)
    assert descriptor.nbytes == global_bytes
    assert descriptor.physical_access_span is descriptor.mem_span
    assert fm.BufferDescriptor.from_data(
        descriptor.to_data(), {physical.id: physical}
    ) == descriptor


def test_compact_per_owner_descriptor_rejects_missing_owner_components():
    distributed = _distributed_type()
    component_bytes = 8 * 32 * 2
    physical = fm.PhysicalBuffer(
        "workspace:too_small", "workspace", component_bytes * 7, 256
    )

    with pytest.raises(ValueError, match="every owner component"):
        fm.BufferDescriptor(
            "sharded",
            fm.DType.BFLOAT16,
            (64, 32),
            (32, 1),
            "workspace",
            256,
            fm.MemSpan(physical, 0, component_bytes),
            distributed_type=distributed,
            distributed_storage_kind=fm.DistributedBufferStorageKind.COMPACT_PER_OWNER,
        )


def test_compact_local_refined_view_retains_its_parent_shard_backing():
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
    physical = fm.PhysicalBuffer(
        "block_local:parent", "block_local_data", 256 * 2, 256
    )
    descriptor = fm.BufferDescriptor(
        "refined",
        fm.DType.BFLOAT16,
        (1, 2048),
        (256, 1),
        "block_local_data",
        256,
        fm.MemSpan(physical, 0, 256 * 2),
        distributed_type=split_yx,
        distributed_storage_kind=fm.DistributedBufferStorageKind.COMPACT_LOCAL,
        distributed_backing_type=split_y,
    )

    assert descriptor.local_shape == (1, 16)
    assert descriptor.component_shape == (1, 256)
    assert descriptor.storage_distributed_type == split_y
    assert fm.BufferDescriptor.from_data(
        descriptor.to_data(), {physical.id: physical}
    ) == descriptor
