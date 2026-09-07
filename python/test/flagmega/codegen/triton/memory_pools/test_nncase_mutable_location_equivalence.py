# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Address-equivalence rules for nncase mutable memory locations."""

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.call_abi import describe_local_buffer_abi
from triton.flagmega.codegen.triton.physical_access import emit_buffer_pointer
from triton.flagmega.ir.bufferization import FunctionBufferPlan


def _distributed_type():
    return fm.DistributedType(
        fm.tensor_type("bfloat16", (8, 16)),
        (fm.SBP.broadcast(), fm.SBP.split_block_cyclic((0,), 4)),
        fm.Placement((2, 4), "yx", "bb"),
    )


def _descriptor(storage_kind, *, memory_space="workspace", start=512):
    distributed = _distributed_type()
    component_elements = 8 * 8
    owner_count = 8
    physical_elements = (
        component_elements * owner_count
        if storage_kind is fm.DistributedBufferStorageKind.COMPACT_PER_OWNER
        else 8 * 16
    )
    physical = fm.PhysicalBuffer(
        f"{memory_space}:value",
        memory_space,
        physical_elements * 2,
        256,
        start,
        "main",
    )
    return fm.BufferDescriptor(
        "value",
        fm.DType.BFLOAT16,
        (8, 16),
        (16, 1),
        memory_space,
        256,
        fm.MemSpan(
            physical,
            0,
            (
                component_elements * 2
                if not storage_kind.exposes_logical_coordinates
                else 8 * 16 * 2
            ),
        ),
        function="main",
        distributed_type=distributed,
        distributed_storage_kind=storage_kind,
    )


def test_canonical_data_uses_allocation_offset_without_owner_stride():
    abi = describe_local_buffer_abi(
        _descriptor(fm.DistributedBufferStorageKind.CANONICAL_GLOBAL)
    )

    assert abi["component_stride_scalar_elements"] == 0
    assert emit_buffer_pointer(abi, "workspace") == (
        "(workspace.to(tl.pointer_type(tl.bfloat16)) + 256)"
    )


def test_compact_chip_data_uses_allocation_local_component_stride():
    abi = describe_local_buffer_abi(
        _descriptor(fm.DistributedBufferStorageKind.COMPACT_PER_OWNER)
    )

    assert abi["component_stride_scalar_elements"] == 64
    assert emit_buffer_pointer(abi, "workspace") == (
        "((workspace.to(tl.pointer_type(tl.bfloat16)) + 256) "
        "+ (shard_index) * 64)"
    )


def test_block_local_data_uses_pool_scope_stride_without_double_owner_stride():
    descriptor = _descriptor(
        fm.DistributedBufferStorageKind.COMPACT_LOCAL,
        memory_space="block_local_data",
        start=0,
    )
    space = fm.MemorySpace(
        "block_local_data",
        "device",
        256,
        1 << 20,
        fm.AllocationStrategy.SAT,
        allocation_scope=fm.MemoryAllocationScope.FUNCTION,
        sharing_scope=fm.MemorySharingScope.BLOCK,
    )
    function = FunctionBufferPlan(
        "main",
        (),
        (),
        memory_pools=(
            fm.FunctionMemoryPool(
                "block_local_data", 256, 256, (descriptor.physical_id,)
            ),
        ),
    )
    plan = fm.BufferPlan(
        (descriptor,),
        (descriptor.mem_span.buffer,),
        (space,),
        (function,),
        workspace_bytes=0,
        rdata_bytes=0,
        alignment=256,
        entry_inputs=(),
        entry_outputs=(),
        default_workspace="block_local_data",
    )

    abi = describe_local_buffer_abi(descriptor, plan, pool_scope_count=8)

    assert abi["component_stride_scalar_elements"] == 0
    assert abi["pool_scope_stride_bytes"] == 256
    assert emit_buffer_pointer(abi, "block_pool") == (
        "(block_pool.to(tl.pointer_type(tl.bfloat16)) "
        "+ (shard_index) * 128)"
    )
