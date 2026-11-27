# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A refined view shares the owner's parent component; it adds no replicas."""
import pytest

from triton.flagmega import ir as fm
from triton.flagmega.ir.bufferization.descriptor import BufferDescriptor
from triton.flagmega.ir.tir.buffer import Buffer
from triton.flagmega.errors import IRSchemaError


@pytest.mark.parametrize("parent_split", [False, True])
@pytest.mark.parametrize("padding", [0, 256])
def test_compact_chip_view_capacity_and_effect_span_use_parent_owners(parent_split, padding):
    mesh = fm.Placement((2, 2, 2), "cyx", "cbb")
    tensor = fm.tensor_type(fm.vector_type("bfloat16", (8,)), (2, 1, 16))
    broad = fm.SBP.broadcast()
    parent = fm.DistributedType(tensor, (fm.SBP.split_contiguous((0,), 1), broad,
        fm.SBP.split_contiguous((1,), 8) if parent_split else broad), mesh)
    child = fm.DistributedType(tensor, (fm.SBP.split_contiguous((0,), 1), broad,
        fm.SBP.split_contiguous((1, 2), 4)), mesh)
    component_bytes = (8 if parent_split else 16) * 16
    stride = component_bytes + padding
    owners = 2 if parent_split else 1
    allocation_bytes = 32 + stride * (owners - 1) + component_bytes
    physical = fm.PhysicalBuffer("parent", "external", allocation_bytes, alignment=16)
    view = BufferDescriptor("refined", tensor.dtype, (2, 1, 16),
        ((8 if parent_split else 16), (8 if parent_split else 16), 1), "parameter", 16,
        fm.MemSpan(physical, 32, component_bytes), "refined",
        distributed_type=child, distributed_storage_kind=fm.DistributedBufferStorageKind.COMPACT_PER_CHIP,
        distributed_backing_type=parent, owner_stride_bytes=stride)
    assert view.physical_access_span.nbytes == allocation_bytes - 32
    assert view.storage_distributed_type == parent
    restored = BufferDescriptor.from_data(view.to_data(), {physical.id: physical})
    assert restored == view
    assert restored.physical_access_span.byte_offset == 32
    assert restored.physical_access_span.nbytes == allocation_bytes - 32
    # The editable TIR boundary must count the same parent components as the
    # descriptor rather than inventing one complete parent for each reader.
    tir_buffer = Buffer("refined", tensor.dtype, view.mem_span, tensor.shape,
        tuple(fm.dim(stride) for stride in view.strides), child,
        view.distributed_storage_kind, parent, stride)
    assert tir_buffer.component_stride_bytes == stride
    if parent_split:
        # Each pointer-visible span fits, but the final parent's component
        # overflows by one byte once the nonzero pointer origin is included.
        short = fm.PhysicalBuffer("short_parent", "external", allocation_bytes - 1, alignment=16)
        with pytest.raises(IRSchemaError, match="backing must contain"):
            Buffer("refined", tensor.dtype, fm.MemSpan(short, 32, component_bytes),
                tensor.shape, tuple(fm.dim(value) for value in view.strides), child,
                view.distributed_storage_kind, parent, stride)
