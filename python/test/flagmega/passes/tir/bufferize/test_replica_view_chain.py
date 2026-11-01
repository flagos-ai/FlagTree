# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Changing a private replica's semantic owners must retain its allocation."""

from dataclasses import replace

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.passes.tir.bufferize import BufferizationOptions


@pytest.mark.parametrize("lanes", [(), (8,), (2, 4)])
@pytest.mark.parametrize("scope", [fm.MemorySharingScope.BLOCK, fm.MemorySharingScope.CHIP])
def test_reshaped_replica_keeps_private_backing_when_block_owners_change(lanes, scope):
    mesh = fm.Placement((4, 8, 16), "cyx", "cbb")
    dtype = fm.vector_type("bfloat16", lanes) if lanes else "bfloat16"
    tensor = fm.tensor_type(dtype, (1, 128))
    broadcast = fm.DistributedType(tensor, (fm.SBP.broadcast(),) * 2, mesh)
    first_type = replace(broadcast, axis_policies=(fm.SBP.broadcast(), fm.SBP.split_contiguous((1,), 16)))
    reshaped_type = fm.DistributedType(fm.tensor_type(dtype, (1, 1, 128)),
        (fm.SBP.broadcast(), fm.SBP.broadcast(), fm.SBP.split_contiguous((1,), 16)), mesh)
    final_type = replace(reshaped_type, axis_policies=(fm.SBP.broadcast(), fm.SBP.broadcast(),
        fm.SBP.split_contiguous((1, 2), 1)))
    builder = fm.IRBuilder(dialect="tir", stage="selected_tir")
    value = builder.var("value", broadcast, id="value")
    produced = builder.call("tir.kernel", (value,), broadcast, id="produced")
    first = builder.call("tir.buffer_view", (produced,), first_type, id="first",
                         attrs={"alias_kind": "sharded_view"})
    reshaped = builder.call("tir.buffer_view", (first,), reshaped_type, id="reshaped",
                            attrs={"alias_kind": "reshape"})
    final = builder.call("tir.buffer_view", (reshaped,), final_type, id="final",
                         attrs={"alias_kind": "sharded_view"})
    output = builder.call("tir.kernel", (final,), final_type, id="output")
    builder.function("main", (value,), (output,))
    options = BufferizationOptions.generic()
    options = replace(options, memory_spaces=tuple(
        replace(space, sharing_scope=scope) if space.name == "workspace" else space
        for space in options.memory_spaces))
    plan = fm.make_buffer_plan(builder.build(entry="main"), options=options)
    source = plan.buffer_map["produced"]
    private = scope is fm.MemorySharingScope.BLOCK
    source_kind = (fm.DistributedBufferStorageKind.COMPACT_LOCAL if private else
                   fm.DistributedBufferStorageKind.CANONICAL_GLOBAL)
    view_kind = (fm.DistributedBufferStorageKind.REPLICATED_LOCAL if private else
                 fm.DistributedBufferStorageKind.CANONICAL_GLOBAL)
    assert source.distributed_storage_kind is source_kind
    for name in ("first", "reshaped", "final"):
        view = plan.buffer_map[name]
        assert view.distributed_storage_kind is view_kind
        assert view.mem_span.must_alias(source.mem_span)
        assert view.mem_span.nbytes == source.nbytes
        assert view.storage == source.storage
