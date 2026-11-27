# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.passes.tir.bufferize import BufferizationOptions
from dataclasses import replace


def test_block_scoped_workspace_allocates_one_compact_local_component_per_scope():
    placement = fm.Placement((2, 4), "yx", "bb")
    value_type = fm.DistributedType(
        fm.tensor_type("bfloat16", (8, 16)),
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((0,), 8)),
        placement,
    )
    builder = fm.IRBuilder(dialect="tir", stage="selected_tir")
    source = builder.var("source", value_type, id="source")
    temporary = builder.call(
        "test.unary", (source,), value_type, id="temporary"
    )
    output = builder.call(
        "test.unary", (temporary,), value_type, id="output"
    )
    builder.function("main", (source,), (output,))

    plan = fm.make_buffer_plan(
        builder.build(entry="main"),
        options=BufferizationOptions.generic(),
    )
    descriptor = plan.buffer_map["temporary"]

    assert plan.memory_space_map["workspace"].sharing_scope is fm.MemorySharingScope.BLOCK
    assert descriptor.distributed_storage_kind is fm.DistributedBufferStorageKind.COMPACT_LOCAL
    assert descriptor.component_shape == (8, 8)
    assert descriptor.nbytes == 8 * 8 * 2
    assert descriptor.mem_span.buffer.nbytes == 8 * 8 * 2
    assert plan.function_memory_space_bytes("main", "workspace") == 256


def test_workspace_pool_identity_is_target_declared_not_hard_coded():
    generic = BufferizationOptions.generic()
    spaces = tuple(
        replace(space, name="data") if space.name == "workspace" else space
        for space in generic.memory_spaces
    )
    options = BufferizationOptions(spaces, workspace="data")
    value_type = fm.tensor_type("float32", (16,))
    builder = fm.IRBuilder(dialect="tir", stage="selected_tir")
    source = builder.var("source", value_type, id="source")
    temporary = builder.call(
        "test.unary", (source,), value_type, id="temporary"
    )
    output = builder.call(
        "test.unary", (temporary,), value_type, id="output"
    )
    builder.function("main", (source,), (output,))

    module = builder.build(entry="main")
    plan = fm.make_buffer_plan(module, options=options)
    descriptor = plan.buffer_map["temporary"]
    buffered = replace(
        module,
        stage="allocated_tir",
        dialect="bufferized_tir",
        metadata={**module.metadata, "buffer_plan": plan.to_data()},
    )

    assert descriptor.storage == "data"
    assert descriptor.mem_span.buffer.memory_space == "data"
    assert plan.workspace_memory_space.name == "data"
    assert plan.function_memory_space_bytes("main", "data") == 256
    assert fm.verify_buffer_plan(buffered) == plan
