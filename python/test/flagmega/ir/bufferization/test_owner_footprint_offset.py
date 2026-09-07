# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""All-owner access spans retain relative offsets and exclude allocation padding."""

import pytest

from triton.flagmega import ir as fm


def _descriptor(physical_bytes, offset):
    placement = fm.Placement((2, 3), "yx", "bb")
    typ = fm.DistributedType(fm.tensor_type("float32", (48,)),
                             (fm.SBP.split_contiguous((0, 1)),), placement)
    physical = fm.PhysicalBuffer("allocation", "workspace", physical_bytes, 32, 1024)
    return fm.BufferDescriptor("view", fm.DType.FLOAT32, (48,), (1,), "workspace", 32,
                                fm.MemSpan(physical, offset, 32), distributed_type=typ,
                                distributed_storage_kind=fm.DistributedBufferStorageKind.COMPACT_PER_OWNER)


def test_owner_footprint_preserves_component_origin_and_not_allocator_padding():
    descriptor = _descriptor(512, 64)
    footprint = descriptor.physical_access_span
    assert descriptor.nbytes == 32
    assert footprint.offset == 1088
    assert footprint.nbytes == 192
    restored = fm.BufferDescriptor.from_data(descriptor.to_data(), {"allocation": footprint.buffer})
    assert restored.physical_access_span == footprint


def test_owner_footprint_cannot_overrun_after_a_nonzero_component_origin():
    with pytest.raises(ValueError, match="every owner component"):
        _descriptor(192, 32)
