# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.errors import RuntimeContractError
from triton.flagmega.ir.bufferization import (
    BufferPlan,
    CallBufferBinding,
    CallMemoryPoolBinding,
    FunctionBufferPlan,
    FunctionMemoryPool,
)
from triton.flagmega.runtime import (
    materialize_memory_pool_views,
    materialize_workspace_views,
    memory_pool_view_specs,
    workspace_view_specs,
)


def _descriptor(name, dtype, shape, physical, *, function):
    strides = []
    stride = 1
    for extent in reversed(shape):
        strides.append(stride)
        stride *= extent
    return fm.BufferDescriptor(
        name,
        dtype,
        shape,
        tuple(reversed(strides)),
        "workspace",
        16,
        fm.MemSpan(physical, fm.dim(0), fm.dim(stride * dtype.itemsize)),
        source_node=name,
        live_start=0,
        live_end=0,
        function=function,
    )


def test_workspace_views_resolve_nested_call_frames_and_vector_lanes():
    main_storage = fm.PhysicalBuffer(
        "main-storage", "workspace", fm.dim(16), 16, fm.dim(0), "main", 0, 0
    )
    call_frame = fm.PhysicalBuffer(
        "call-frame", "workspace", fm.dim(64), 16, fm.dim(32), "main", 1, 1
    )
    child_storage = fm.PhysicalBuffer(
        "child-storage", "workspace", fm.dim(8), 16, fm.dim(16), "child", 0, 0
    )
    main_value = _descriptor(
        "main_value", fm.DType.INT32, (4,), main_storage, function="main"
    )
    child_value = _descriptor(
        "child_value", fm.vector_type("bfloat16", 2), (2,), child_storage,
        function="child",
    )
    call = CallBufferBinding(
        "invoke_child", "main", "child", (), (), "call-frame", 32, 32
    )
    plan = BufferPlan(
        buffers=(main_value, child_value),
        physical_buffers=(main_storage, call_frame, child_storage),
        memory_spaces=(),
        functions=(
            FunctionBufferPlan(
                "main", (), (), 96, 16, ("main-storage", "call-frame"),
                calls=(call,), values=(("main_value", ("main_value",)),),
            ),
            FunctionBufferPlan(
                "child", (), (), 32, 16, ("child-storage",),
                values=(("child_value", ("child_value",)),),
            ),
        ),
        workspace_bytes=96,
        rdata_bytes=0,
        alignment=16,
        entry_inputs=(),
        entry_outputs=(),
    )

    specs = workspace_view_specs(plan, entry="main")
    by_key = {value.key: value for value in specs}
    assert by_key["main_value"].byte_offset == 0
    assert by_key["main_value"].shape == (4,)
    assert by_key["invoke_child::child_value"].byte_offset == 48
    assert by_key["invoke_child::child_value"].shape == (2, 2)
    assert by_key["invoke_child::child_value"].strides == (2, 1)

    arena = torch.zeros(96, dtype=torch.uint8)
    views = materialize_workspace_views(arena, specs)
    views["main_value"].fill_(7)
    views["invoke_child::child_value"].fill_(3)

    assert arena[:16].view(torch.int32).tolist() == [7, 7, 7, 7]
    assert arena[48:56].view(torch.bfloat16).tolist() == [3, 3, 3, 3]
    cloned = materialize_workspace_views(arena, specs, clone=True)
    arena.zero_()
    assert cloned["main_value"].tolist() == [7, 7, 7, 7]


def test_workspace_specs_reject_a_call_frame_outside_the_entry_arena():
    call_frame = fm.PhysicalBuffer(
        "call-frame", "workspace", fm.dim(4), 4, fm.dim(8), "main", 0, 0,
        "call_memory_pool",
    )
    child_storage = fm.PhysicalBuffer(
        "child-storage", "workspace", fm.dim(4), 4, fm.dim(0), "child", 0, 0
    )
    child_value = _descriptor(
        "child_value", fm.DType.FLOAT32, (1,), child_storage, function="child"
    )
    plan = BufferPlan(
        buffers=(child_value,),
        physical_buffers=(call_frame, child_storage),
        memory_spaces=(),
        functions=(
            FunctionBufferPlan(
                "main", (), (), 8, 4, ("call-frame",),
                calls=(CallBufferBinding(
                    "bad", "main", "child", (), (), "call-frame", 8, 4
                ),),
            ),
            FunctionBufferPlan(
                "child", (), (), 4, 4, ("child-storage",),
                values=(("child_value", ("child_value",)),),
            ),
        ),
        workspace_bytes=8,
        rdata_bytes=0,
        alignment=4,
        entry_inputs=(),
        entry_outputs=(),
    )

    with pytest.raises(RuntimeContractError, match="exceeds the entry arena"):
        workspace_view_specs(plan, entry="main")


def test_workspace_views_preserve_compact_per_owner_components():
    placement = fm.Placement((2, 2), "yx", "bb")
    distributed_type = fm.DistributedType(
        fm.tensor_type("bfloat16", (4,)),
        (fm.SBP.split_contiguous((0,), 2),),
        placement,
    )
    physical = fm.PhysicalBuffer(
        "owners", "workspace", fm.dim(16), 4, fm.dim(0), "main", 0, 0
    )
    descriptor = fm.BufferDescriptor(
        "shards",
        fm.DType.BFLOAT16,
        (4,),
        (1,),
        "workspace",
        4,
        fm.MemSpan(physical, fm.dim(0), fm.dim(4)),
        source_node="shards",
        live_start=0,
        live_end=0,
        function="main",
        distributed_type=distributed_type,
        distributed_storage_kind=fm.DistributedBufferStorageKind.COMPACT_PER_OWNER,
    )
    plan = BufferPlan(
        buffers=(descriptor,),
        physical_buffers=(physical,),
        memory_spaces=(),
        functions=(FunctionBufferPlan(
            "main", (), (), 16, 4, ("owners",),
            values=(("shards", ("shards",)),),
        ),),
        workspace_bytes=16,
        rdata_bytes=0,
        alignment=4,
        entry_inputs=(),
        entry_outputs=(),
    )

    specs = workspace_view_specs(plan, entry="main")
    assert specs[0].shape == (4, 2)
    assert specs[0].strides == (2, 1)
    arena = torch.zeros(16, dtype=torch.uint8)
    view = materialize_workspace_views(arena, specs)["shards"]
    view.copy_(torch.arange(8, dtype=torch.bfloat16).reshape(4, 2))
    assert arena.view(torch.bfloat16).tolist() == list(range(8))


def test_block_pool_diagnostics_resolve_call_frames_for_every_physical_scope():
    child_storage = fm.PhysicalBuffer(
        "child-block", "block_data", fm.dim(4), 4, fm.dim(0), "child", 0, 0
    )
    call_frame = fm.PhysicalBuffer(
        "call-block", "block_data", fm.dim(4), 4, fm.dim(16), "main", 0, 0,
        "call_memory_pool",
    )
    descriptor = fm.BufferDescriptor(
        "child_value",
        fm.DType.FLOAT32,
        (1,),
        (1,),
        "block_data",
        4,
        fm.MemSpan(child_storage, fm.dim(0), fm.dim(4)),
        source_node="child_value",
        live_start=0,
        live_end=0,
        function="child",
    )
    plan = BufferPlan(
        buffers=(descriptor,),
        physical_buffers=(child_storage, call_frame),
        memory_spaces=(),
        functions=(
            FunctionBufferPlan(
                "main",
                (),
                (),
                memory_pools=(
                    FunctionMemoryPool("workspace", 0, 4, ()),
                    FunctionMemoryPool("block_data", 32, 4, ("call-block",)),
                ),
                calls=(CallBufferBinding(
                    "invoke_child",
                    "main",
                    "child",
                    (),
                    (),
                    memory_pools=(
                        CallMemoryPoolBinding("block_data", "call-block", 16, 4),
                    ),
                ),),
            ),
            FunctionBufferPlan(
                "child",
                (),
                (),
                memory_pools=(
                    FunctionMemoryPool("workspace", 0, 4, ()),
                    FunctionMemoryPool("block_data", 4, 4, ("child-block",)),
                ),
                values=(("child_value", ("child_value",)),),
            ),
        ),
        workspace_bytes=0,
        rdata_bytes=0,
        alignment=4,
        entry_inputs=(),
        entry_outputs=(),
    )

    specs = memory_pool_view_specs(
        plan, entry="main", memory_space="block_data"
    )
    assert specs[0].byte_offset == 16
    arena = torch.zeros(64, dtype=torch.uint8)
    view = materialize_memory_pool_views(
        arena, specs, scope_count=2, scope_nbytes=32
    )["invoke_child::child_value"]
    assert view.shape == (2, 1)
    view.copy_(torch.tensor([[3.0], [7.0]], dtype=torch.float32))
    assert arena[16:20].view(torch.float32).item() == 3.0
    assert arena[48:52].view(torch.float32).item() == 7.0
