# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.ir import DType, IRBuilder, make_buffer_plan, tensor_type
from triton.flagmega.ir.bufferization import (
    AllocationStrategy,
    MemoryAllocationScope,
    MemorySharingScope,
    MemorySpace,
)
from triton.flagmega.passes.tir import materialize_kernel_prim_functions
from triton.flagmega.passes.tir.bufferize import (
    BufferizationOptions,
    plan_memory_synchronization,
)


def _workspace_options(scope: MemorySharingScope) -> BufferizationOptions:
    maximum = (1 << 31) - 1
    return BufferizationOptions((
        MemorySpace(
            "workspace", "device", 64, maximum, AllocationStrategy.SAT,
            allocation_scope=MemoryAllocationScope.FUNCTION,
            sharing_scope=scope,
        ),
        MemorySpace(
            "rdata", "readonly_device", 64, maximum, AllocationStrategy.LINEAR,
            allocation_scope=MemoryAllocationScope.MODULE,
            sharing_scope=MemorySharingScope.CHIP,
        ),
        MemorySpace(
            "external", "external", 1, maximum, AllocationStrategy.EXTERNAL,
            allocation_scope=MemoryAllocationScope.EXTERNAL,
            sharing_scope=MemorySharingScope.CHIP,
        ),
    ))


def test_memory_synchronization_is_derived_from_concrete_raw_ranges():
    builder = IRBuilder(dialect="tir", stage="selected_tir")
    value_type = tensor_type(DType.BFLOAT16, [1, 128])
    source = builder.var("source", value_type, id="source")
    produced = builder.call("tir.kernel", [source], value_type, id="produced")
    output = builder.call("tir.kernel", [produced], value_type, id="output")
    builder.function("main", [source], [output])
    module = builder.build(entry="main")

    synchronization = plan_memory_synchronization(module, make_buffer_plan(module))

    assert len(synchronization.events) == 1
    event = synchronization.events[0]
    assert (event.after, event.before, event.scope) == ("produced", "output", "block")
    assert event.hazards == ("WRITE->READ",)
    assert event.ranges[0].storage == "workspace"
    assert event.ranges[0].nbytes == 256


def test_memory_synchronization_does_not_serialize_independent_ranges():
    builder = IRBuilder(dialect="tir", stage="selected_tir")
    value_type = tensor_type(DType.BFLOAT16, [1, 128])
    lhs = builder.var("lhs", value_type, id="lhs")
    rhs = builder.var("rhs", value_type, id="rhs")
    first = builder.call("tir.kernel", [lhs], value_type, id="first")
    second = builder.call("tir.kernel", [rhs], value_type, id="second")
    builder.function("main", [lhs, rhs], [first, second])
    module = builder.build(entry="main")

    synchronization = plan_memory_synchronization(module, make_buffer_plan(module))
    assert synchronization.events == ()


def test_same_distributed_owner_map_uses_block_scope():
    builder = IRBuilder(dialect="tir", stage="selected_tir")
    tensor = tensor_type(DType.BFLOAT16, [1, 128])
    distributed = fm.DistributedType(
        tensor,
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((0, 1))),
        fm.Placement((2, 4), "yx", "bb"),
    )
    source = builder.var("source", distributed, id="source")
    produced = builder.call("tir.kernel", [source], distributed, id="produced")
    output = builder.call("tir.kernel", [produced], distributed, id="output")
    builder.function("main", [source], [output])
    module = builder.build(entry="main")

    synchronization = plan_memory_synchronization(
        module,
        make_buffer_plan(module, options=_workspace_options(MemorySharingScope.CHIP)),
    )

    assert len(synchronization.events) == 1
    assert synchronization.events[0].scope == "block"


def test_collective_access_keeps_chip_scope():
    builder = IRBuilder(dialect="tir", stage="selected_tir")
    tensor = tensor_type(DType.BFLOAT16, [1, 128])
    distributed = fm.DistributedType(
        tensor,
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((0, 1))),
        fm.Placement((2, 4), "yx", "bb"),
    )
    source = builder.var("source", distributed, id="source")
    produced = builder.call("tir.kernel", [source], distributed, id="produced")
    boxed = builder.call(
        "tir.kernel",
        [produced],
        distributed,
        id="boxed",
        attrs={
            "semantic_op": "distributed.boxing",
            "facts": {"collective_semantics": "all_gather"},
        },
    )
    builder.function("main", [source], [boxed])
    module = builder.build(entry="main")

    synchronization = plan_memory_synchronization(
        module,
        make_buffer_plan(module, options=_workspace_options(MemorySharingScope.CHIP)),
    )

    assert len(synchronization.events) == 1
    assert synchronization.events[0].scope == "grid"


def test_collective_result_in_replicated_block_storage_needs_only_block_barrier():
    builder = IRBuilder(dialect="tir", stage="selected_tir")
    value_type = tensor_type(DType.BFLOAT16, [1, 128])
    source = builder.var("source", value_type, id="source")
    collective = builder.call(
        "tir.kernel",
        [source],
        value_type,
        id="collective",
        attrs={
            "semantic_op": "distributed.boxing",
            "facts": {"collective_semantics": "tensor_load"},
        },
    )
    consumed = builder.call(
        "tir.kernel", [collective], value_type, id="consumed"
    )
    builder.function("main", [source], [consumed])
    module = builder.build(entry="main")

    synchronization = plan_memory_synchronization(
        module,
        make_buffer_plan(
            module, options=_workspace_options(MemorySharingScope.BLOCK)
        ),
    )

    assert len(synchronization.events) == 1
    assert synchronization.events[0].scope == "block"


def test_materialized_collective_dispatch_keeps_chip_scope():
    builder = IRBuilder(dialect="semantic_tir", stage="selected_tir")
    tensor = tensor_type(DType.BFLOAT16, [1, 128])
    source = builder.var("source", tensor, id="source")
    produced = builder.call(
        "tir.kernel",
        [source],
        tensor,
        id="produced",
        attrs={
            "semantic_op": "math.silu",
            "candidate": "tir.silu.local",
            "parameters": {"family": "elementwise", "variant": "silu"},
            "facts": {},
            "semantic_attrs": {},
        },
    )
    boxed = builder.call(
        "tir.kernel",
        [produced],
        tensor,
        id="boxed",
        attrs={
            "semantic_op": "distributed.boxing",
            "candidate": "tir.distributed_boxing.gather_reduce_scatter",
            "parameters": {
                "family": "distributed_boxing",
                "variant": "gather_reduce_scatter",
            },
            "facts": {
                "collective_semantics": "gather_reduce_scatter",
                "requires": ("cooperative_grid", "grid_sync"),
            },
            "semantic_attrs": {},
        },
    )
    builder.function("main", [source], [boxed])
    module = materialize_kernel_prim_functions(builder.build(entry="main"))

    synchronization = plan_memory_synchronization(
        module,
        make_buffer_plan(module, options=_workspace_options(MemorySharingScope.CHIP)),
    )

    assert len(synchronization.events) == 1
    assert synchronization.events[0].scope == "grid"


def test_reference_effect_keeps_chip_scope():
    builder = IRBuilder(dialect="tir", stage="selected_tir")
    state_type = fm.RefType(
        "state", (("cache", tensor_type(DType.BFLOAT16, [2, 8])),)
    )
    state = builder.var("state", state_type, id="state")
    first = builder.call(
        "tir.kernel",
        [state],
        state_type,
        id="first",
        effect=fm.effect("read_write", "state"),
    )
    second = builder.call(
        "tir.kernel",
        [first],
        state_type,
        id="second",
        effect=fm.effect("read_write", "state"),
    )
    builder.function("main", [state], [second])
    module = builder.build(entry="main")

    synchronization = plan_memory_synchronization(module, make_buffer_plan(module))

    assert len(synchronization.events) == 1
    assert synchronization.events[0].scope == "grid"
