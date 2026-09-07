# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Equal SBP policies do not imply equal owners after physical relocation."""

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.ir.bufferization import AllocationStrategy, MemoryAllocationScope, MemorySharingScope, MemorySpace
from triton.flagmega.passes.tir.bufferize import BufferizationOptions
from triton.flagmega.passes.tir.bufferize.synchronization import _hazard_requirement, _node_accesses


def _access(*, cyclic=False):
    placement = fm.Placement((2, 4), "yx", "bb")
    policy = fm.SBP.split_block_cyclic((0, 1), 1) if cyclic else fm.SBP.split_contiguous((0, 1))
    distributed = fm.DistributedType(fm.tensor_type("float32", (128,)), (policy,), placement)
    options = BufferizationOptions((
        MemorySpace("workspace", "device", 64, 4096, AllocationStrategy.SAT,
                    allocation_scope=MemoryAllocationScope.FUNCTION, sharing_scope=MemorySharingScope.CHIP),
        MemorySpace("rdata", "readonly_device", 64, 4096, AllocationStrategy.LINEAR,
                    allocation_scope=MemoryAllocationScope.MODULE, sharing_scope=MemorySharingScope.CHIP),
        MemorySpace("external", "external", 1, 4096, AllocationStrategy.EXTERNAL,
                    allocation_scope=MemoryAllocationScope.EXTERNAL, sharing_scope=MemorySharingScope.CHIP),
    ))
    builder = fm.IRBuilder(dialect="tir", stage="selected_tir")
    source = builder.var("source", distributed, id="source")
    produced = builder.call("tir.kernel", (source,), distributed, id="produced")
    output = builder.call("tir.kernel", (produced,), distributed, id="output")
    builder.function("main", (source,), (output,))
    module = builder.build(entry="main")
    plan = fm.make_buffer_plan(module, options=options)
    bindings = dict(plan.function_map["main"].values)
    return next(value for value in _node_accesses(module, plan, "main", produced, bindings)
                if value.buffer == bindings["produced"][0])


@pytest.mark.parametrize("mode", ("raw", "war"))
@pytest.mark.parametrize("shift", (64, 256))
def test_shifted_components_cannot_reuse_sbp_only_block_proof(mode, shift):
    original = _access()
    read = replace(original, mode="read", effect=fm.MemoryEffect.READ)
    write = replace(original, node="reuse", buffer="reuse", offset=original.offset + shift)
    previous, current = (write, read) if mode == "raw" else (read, write)
    # E.g. after a 64-byte shift, the new owner 0 writes the old owner 1's
    # bytes. A block barrier cannot synchronize those different CTAs.
    assert _hazard_requirement(previous, current)[0] == "grid"


@pytest.mark.parametrize("mode", ("raw", "war"))
def test_different_owner_strides_cannot_reuse_sbp_only_block_proof(mode):
    original = _access()
    read = replace(original, mode="read", effect=fm.MemoryEffect.READ)
    write = replace(original, node="reuse", buffer="reuse", nbytes=original.nbytes * 2)
    previous, current = (write, read) if mode == "raw" else (read, write)
    assert _hazard_requirement(previous, current)[0] == "grid"


def test_same_component_origin_and_stride_keep_the_block_proof():
    write = _access()
    read = replace(write, node="consumer", mode="read", effect=fm.MemoryEffect.READ)
    assert _hazard_requirement(write, read)[0] == "block"


@pytest.mark.parametrize("mode", ("raw", "war"))
def test_canonical_and_compact_cyclic_storage_have_different_physical_owners(mode):
    original = _access(cyclic=True)
    read = replace(original, mode="read", effect=fm.MemoryEffect.READ)
    write = replace(original, node="reuse", buffer="reuse",
                    distributed_storage_kind=fm.DistributedBufferStorageKind.CANONICAL_GLOBAL)
    previous, current = (write, read) if mode == "raw" else (read, write)
    assert _hazard_requirement(previous, current)[0] == "grid"
