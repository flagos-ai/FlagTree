# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.distributed_abi import (
    KernelExecutionKind,
    distributed_buffer_abi,
    kernel_execution_kind,
)
from triton.flagmega.codegen.triton.call_abi import describe_local_buffer_abi
from triton.flagmega.codegen.triton.physical_access import (
    emit_active_extent,
    emit_local_scalar_offset,
    emit_owner_base,
)
from triton.flagmega.codegen.triton.kernel_call_renderers import (
    _same_suffix_local_mapping,
)


def _descriptor(storage_kind=fm.DistributedBufferStorageKind.CANONICAL_GLOBAL):
    partial = (
        None
        if storage_kind.exposes_logical_coordinates
        else fm.SBP.partial((1,))
    )
    distributed = fm.DistributedType(
        fm.tensor_type("bfloat16", (8, 16)),
        (fm.SBP.broadcast(), fm.SBP.split_block_cyclic((0,), 4)),
        fm.Placement((2, 4), "yx", "bb"),
        partial=partial,
    )
    local_elements = 8 * 8
    global_elements = 8 * 16
    per_owner = storage_kind is fm.DistributedBufferStorageKind.COMPACT_PER_OWNER
    component_elements = (
        global_elements if storage_kind.exposes_logical_coordinates else local_elements
    )
    backing_elements = local_elements * 8 if per_owner else component_elements
    physical = fm.PhysicalBuffer("workspace:value", "workspace", backing_elements * 2, 256)
    return fm.BufferDescriptor(
        "value",
        fm.DType.BFLOAT16,
        (8, 16),
        (16, 1),
        "workspace",
        256,
        fm.MemSpan(physical, 0, component_elements * 2),
        distributed_type=distributed,
        distributed_storage_kind=storage_kind,
    )


def test_ordinary_distributed_kernel_is_local_shard_execution():
    assert kernel_execution_kind(
        "math.packed_dense_matmul", {"requires": ("mma_v3",)}
    ) is KernelExecutionKind.LOCAL_SHARD


@pytest.mark.parametrize(
    "distributed",
    (
        fm.DistributedType(
            fm.tensor_type("bfloat16", (8, 16)),
            (fm.SBP.broadcast(), fm.SBP.broadcast()),
            fm.Placement((2, 4), "yx", "bb"),
        ),
        fm.DistributedType(
            fm.tensor_type("bfloat16", (8, 16)),
            (fm.SBP.broadcast(), fm.SBP.split_contiguous((0,), 8)),
            fm.Placement((2, 4), "yx", "bb"),
        ),
        fm.DistributedType(
            fm.tensor_type("bfloat16", (8, 16)),
            (fm.SBP.broadcast(), fm.SBP.split_block_cyclic((0, 1), 4)),
            fm.Placement((2, 4), "yx", "bb"),
        ),
        fm.DistributedType(
            fm.tensor_type("bfloat16", (8, 16)),
            (fm.SBP.broadcast(), fm.SBP.broadcast()),
            fm.Placement((2, 4), "yx", "bb"),
            partial=fm.SBP.partial((1,)),
        ),
    ),
)
def test_ordinary_execution_kind_does_not_branch_on_distributed_type(distributed):
    assert kernel_execution_kind(
        "math.add", {"distributed_type": distributed.to_data()}
    ) is KernelExecutionKind.LOCAL_SHARD


def test_grid_synchronization_does_not_turn_local_kernel_into_collective():
    assert kernel_execution_kind(
        "nn.rms_norm", {"requires": ("cooperative_grid", "grid_sync")}
    ) is KernelExecutionKind.SYNCHRONIZED_LOCAL


def test_boxing_fact_marks_only_explicit_layout_transition_collective():
    assert kernel_execution_kind(
        "distributed.boxing", {"collective_semantics": "gather-reduce-scatter"}
    ) is KernelExecutionKind.COLLECTIVE


def test_canonical_global_buffer_maps_local_indices_through_sbp_descriptor():
    abi = distributed_buffer_abi(_descriptor(), owner_index=4)

    assert abi.owner_coordinates == (1, 0)
    assert abi.local_capacity_shape == (8, 8)
    assert abi.active_shape == (8, 8)
    assert abi.component_offset_elements == 0
    assert tuple(abi.storage_axis_coordinate(1, index) for index in range(8)) == (
        4, 5, 6, 7, 12, 13, 14, 15,
    )
    assert abi.partial_group_owner_indices == ()


def test_compact_per_owner_buffer_uses_owner_stride_and_dense_local_indices():
    abi = distributed_buffer_abi(
        _descriptor(fm.DistributedBufferStorageKind.COMPACT_PER_OWNER),
        owner_index=5,
    )

    assert abi.owner_coordinates == (1, 1)
    assert abi.component_offset_elements == 5 * 8 * 8
    assert tuple(abi.storage_axis_coordinate(1, index) for index in range(8)) == tuple(range(8))
    assert abi.logical_axis_coordinate(1, 0) == 4
    # Partial axis 1 varies while split axis 0 remains fixed. Its components
    # require distinct storage, unlike a materialized canonical tensor.
    assert abi.partial_group_owner_indices == (4, 5, 6, 7)


def test_render_abi_preserves_local_domain_and_complete_block_cyclic_mapping():
    abi = describe_local_buffer_abi(_descriptor())

    assert abi["logical_shape"] == (8, 16)
    assert abi["local_capacity_shape"] == (8, 8)
    assert abi["active_shape_expressions"] == ("8", "8")
    assert abi["coordinate_space"] == "canonical_global"
    assert abi["logical_coordinate_expressions"][0] == "local_coord_0"
    expression = abi["logical_coordinate_expressions"][1]
    assert "shard_coord_0" in expression
    assert "local_coord_1" in expression
    assert "//" in expression
    assert "%" in expression


def test_plain_tensor_abi_exposes_the_complete_logical_coordinate_space():
    physical = fm.PhysicalBuffer("input:value", "input", 4096, 16)
    descriptor = fm.BufferDescriptor(
        "value",
        fm.DType.BFLOAT16,
        (1, 2048),
        (2048, 1),
        "input",
        16,
        fm.MemSpan(physical, 0, 4096),
    )

    abi = describe_local_buffer_abi(descriptor)

    assert abi["distributed_type"] is None
    assert abi["logical_shape"] == abi["local_capacity_shape"]
    assert abi["coordinate_space"] == "canonical_global"


def test_render_abi_uses_owner_stride_only_for_compact_per_owner_storage():
    abi = describe_local_buffer_abi(
        _descriptor(fm.DistributedBufferStorageKind.COMPACT_PER_OWNER)
    )

    assert abi["coordinate_space"] == "local"
    assert abi["component_stride_elements"] == 64
    # Logical coordinates remain available to semantics such as block-scale
    # lookup even though the data pointer itself addresses a compact shard.
    assert "shard_coord_0" in abi["logical_coordinate_expressions"][1]


def test_replicated_local_buffer_uses_global_coordinates_without_owner_stride():
    descriptor = _descriptor(fm.DistributedBufferStorageKind.REPLICATED_LOCAL)
    abi = describe_local_buffer_abi(descriptor)
    owner = distributed_buffer_abi(descriptor, owner_index=4)

    assert abi["storage_kind"] == "replicated_local"
    assert abi["coordinate_space"] == "canonical_global"
    assert abi["component_stride_elements"] == 0
    assert owner.component_offset_elements == 0
    assert tuple(owner.storage_axis_coordinate(1, index) for index in range(8)) == (
        4, 5, 6, 7, 12, 13, 14, 15,
    )


def test_physical_access_maps_local_coordinates_only_for_canonical_storage():
    canonical = describe_local_buffer_abi(_descriptor())
    compact = describe_local_buffer_abi(
        _descriptor(fm.DistributedBufferStorageKind.COMPACT_PER_OWNER)
    )

    canonical_offset = emit_local_scalar_offset(canonical, ("row", "column"))
    compact_offset = emit_local_scalar_offset(compact, ("row", "column"))

    assert "shard_y" in canonical_offset
    assert "column" in canonical_offset
    assert "shard_y" not in compact_offset
    assert emit_owner_base(canonical, "buffer") == "buffer"
    assert emit_owner_base(compact, "buffer") == "(buffer + (shard_index) * 64)"
    assert emit_active_extent(canonical, 1) == "8"


def test_refined_compact_view_addresses_relative_to_parent_shard_origin():
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

    owner = distributed_buffer_abi(descriptor, owner_index=3 * 16 + 5)
    abi = describe_local_buffer_abi(descriptor)

    assert owner.local_capacity_shape == (1, 16)
    assert owner.storage_axis_coordinate(1, 0) == 5 * 16
    assert owner.storage_axis_coordinate(1, 15) == 5 * 16 + 15
    assert abi["schema"] == "flagmega.local-buffer-abi/v2"
    assert abi["coordinate_space"] == "parent_shard_local"
    assert abi["local_capacity_shape"] == (1, 16)
    assert abi["component_stride_elements"] == 0
    offset = emit_local_scalar_offset(abi, ("row", "column"))
    assert "shard_x" in offset
    assert "shard_y" not in offset
    assert "column" in offset


def test_vector_buffer_abi_exposes_scalar_pointer_strides():
    distributed = fm.DistributedType(
        fm.tensor_type(fm.VectorType(fm.DType.FLOAT32, (4,)), (8, 16)),
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((0,), 8)),
        fm.Placement((2,), "x", "b"),
    )
    component_bytes = 8 * 8 * 4 * fm.DType.FLOAT32.itemsize
    physical = fm.PhysicalBuffer(
        "workspace:vector", "workspace", component_bytes * 2, 256
    )
    buffer = fm.BufferDescriptor(
        "vector",
        fm.VectorType(fm.DType.FLOAT32, (4,)),
        (8, 16),
        (16, 1),
        "workspace",
        256,
        fm.MemSpan(physical, 0, component_bytes),
        distributed_type=distributed,
        distributed_storage_kind=fm.DistributedBufferStorageKind.COMPACT_PER_OWNER,
    )

    abi = describe_local_buffer_abi(buffer)

    assert abi["scalar_lane_shape"] == (4,)
    assert abi["scalar_storage_strides"] == (64, 4)
    assert abi["component_stride_elements"] == 64
    assert abi["component_stride_scalar_elements"] == 256
    assert emit_local_scalar_offset(
        abi, ("outer", "inner"), lane_coordinate="lane"
    ) == "(outer) * 64 + (inner) * 4 + (lane)"


def test_broadcast_suffix_operand_uses_the_same_owner_local_mapping():
    result = {
        "local_capacity_shape": (1, 8),
        "logical_coordinate_expressions": (
            "local_coord_0",
            "local_coord_1 + shard_coord_0 * 8",
        ),
    }
    scale = {
        "local_capacity_shape": (8,),
        "logical_coordinate_expressions": (
            "local_coord_0 + shard_coord_0 * 8",
        ),
    }
    wrong_owner = {
        "local_capacity_shape": (8,),
        "logical_coordinate_expressions": (
            "local_coord_0 + shard_coord_1 * 8",
        ),
    }

    assert _same_suffix_local_mapping(scale, result)
    assert not _same_suffix_local_mapping(wrong_owner, result)
