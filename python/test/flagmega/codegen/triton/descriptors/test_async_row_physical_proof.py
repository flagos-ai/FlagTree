# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The copy proof follows physical storage, not the spelling of a split."""

from math import prod

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.call_abi import describe_local_buffer_abi
from triton.flagmega.codegen.triton.row_transfer import validate_async_row_copy
from triton.flagmega.errors import CodegenError


def _descriptor(policy, storage, *, origin=0, alignment=256):
    placement = fm.Placement((2, 3), "ab", "bb")
    dtype = fm.DistributedType(fm.tensor_type("bfloat16", (1, 6144)),
                               (fm.SBP.broadcast(), policy), placement)
    physical = fm.PhysicalBuffer("storage", "workspace", 16384, alignment)
    nbytes = (12288 if storage.exposes_logical_coordinates else
              2 * prod(axis.local_capacity.fixed_value for axis in fm.local_shard_descriptor(dtype, (0, 0)).axes))
    return fm.BufferDescriptor("input", fm.DType.BFLOAT16, (1, 6144), (6144, 1),
                               "workspace", 1, fm.MemSpan(physical, origin, nbytes),
                               distributed_type=dtype, distributed_storage_kind=storage)


@pytest.mark.parametrize("origin, expected", ((0, 256), (2, 2), (16, 16), (1024, 256)))
def test_view_alignment_includes_memspan_alias_offset(origin, expected):
    descriptor = _descriptor(fm.SBP.broadcast(), fm.DistributedBufferStorageKind.CANONICAL_GLOBAL, origin=origin)
    abi = describe_local_buffer_abi(descriptor)
    assert abi["alignment_bytes"] == expected


@pytest.mark.parametrize("storage, supported", ((fm.DistributedBufferStorageKind.CANONICAL_GLOBAL, False),
                                                (fm.DistributedBufferStorageKind.COMPACT_LOCAL, True)))
def test_cyclic_logical_rows_need_compact_physical_storage(storage, supported):
    descriptor = _descriptor(fm.SBP.split_block_cyclic((0,), 1), storage)
    abi = describe_local_buffer_abi(descriptor)
    # This isolated ABI test supplies the explicit prepared-pool contract;
    # no BufferPlan scheduling or runtime is needed to test the proof.
    abi.update(pooled=True, pool_scope_stride_bytes=0)
    if supported:
        validate_async_row_copy(abi, 1024)
    else:
        with pytest.raises(CodegenError, match="unit-stride physical"):
            validate_async_row_copy(abi, 1024)


def test_contiguous_canonical_shards_preserve_aligned_owner_origins():
    descriptor = _descriptor(fm.SBP.split_contiguous((0,), 3072), fm.DistributedBufferStorageKind.CANONICAL_GLOBAL)
    abi = describe_local_buffer_abi(descriptor)
    abi.update(pooled=True, pool_scope_stride_bytes=0)
    validate_async_row_copy(abi, 1024)


def test_inactive_boundary_owner_cannot_use_unmasked_copy():
    descriptor = _descriptor(fm.SBP.split_contiguous((0,), 4096), fm.DistributedBufferStorageKind.CANONICAL_GLOBAL)
    abi = describe_local_buffer_abi(descriptor)
    abi.update(pooled=True, pool_scope_stride_bytes=0)
    with pytest.raises(CodegenError, match="fully active static rows"):
        validate_async_row_copy(abi, 1024)
