# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Dense Torch storage for a persistent tensor's planned external buffer."""

from math import prod

from triton.flagmega.errors import ArtifactError
from triton.flagmega.ir import tensor_type
from triton.flagmega.ir.distributed_storage import DistributedBufferStorageKind
from triton.flagmega.ir.distributed_type import block_owner_count, placement_owner_count


def tensor_state_storage_type(descriptor):
    """Preserve component order instead of allocating the logical global shape.

    The convenience state objects own contiguous tensors. A compact external
    allocation is an owner-major array of dense components. Subviews and
    padded component strides need a different storage/view contract and are
    rejected rather than silently interpreting their bytes as dense state.
    """
    if descriptor.distributed_backing_type is not None or descriptor.byte_offset:
        raise ArtifactError(f"Tensor state {descriptor.id!r} requires a complete external storage root.")
    shape = descriptor.component_shape
    strides = []
    stride = 1
    for size in reversed(shape):
        strides.append(stride)
        stride *= size
    if descriptor.strides != tuple(reversed(strides)):
        raise ArtifactError(f"Tensor state {descriptor.id!r} requires dense component strides.")
    if descriptor.distributed_storage_kind in {
        DistributedBufferStorageKind.COMPACT_PER_OWNER,
        DistributedBufferStorageKind.COMPACT_PER_CHIP,
    }:
        if descriptor.component_stride_bytes != prod(shape) * descriptor.dtype.itemsize:
            raise ArtifactError(f"Tensor state {descriptor.id!r} requires a dense owner stride.")
        kind = descriptor.distributed_type
        assert kind is not None
        owners = (block_owner_count(kind) if descriptor.distributed_storage_kind
                  is DistributedBufferStorageKind.COMPACT_PER_CHIP else placement_owner_count(kind))
        shape = (owners, *shape)
    result = tensor_type(descriptor.dtype, shape)
    if prod(shape) * descriptor.dtype.itemsize != descriptor.physical_access_span.nbytes:
        raise ArtifactError(f"Tensor state {descriptor.id!r} physical shape does not cover its buffer ABI.")
    return result


__all__ = ["tensor_state_storage_type"]
