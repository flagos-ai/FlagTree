# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.ir.bufferization import (
    AllocationStrategy,
    MemoryAllocationScope,
    MemorySharingScope,
    MemorySpace,
)
from triton.flagmega.passes.tir.bufferize import (
    BufferizationOptions,
    plan_memory_synchronization,
)


def _chip_workspace_options() -> BufferizationOptions:
    maximum = (1 << 31) - 1
    return BufferizationOptions((
        MemorySpace(
            "workspace", "device", 64, maximum, AllocationStrategy.SAT,
            allocation_scope=MemoryAllocationScope.FUNCTION,
            sharing_scope=MemorySharingScope.CHIP,
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


def _kernel(
    name: str,
    value_type: fm.IRType,
    input_effect: fm.MemoryEffect,
    *,
    requires: tuple[str, ...] = (),
) -> fm.PrimFunction:
    input_parameter = fm.PrimParameter(
        "input", value_type, fm.PrimParameterRole.INPUT
    )
    output_parameter = fm.PrimParameter(
        "result", value_type, fm.PrimParameterRole.OUTPUT
    )
    dispatch = fm.KernelDispatch(
        semantic_op=f"test.{name}",
        candidate=f"test.{name}",
        parameters={"family": "test", "variant": name},
        facts={"requires": requires} if requires else {},
        arguments=("input",),
        outputs=("result",),
        reads=("input",),
        writes=("result",),
        memory_effects=(("input", input_effect), ("result", fm.MemoryEffect.WRITE)),
    )
    return fm.PrimFunction(
        name,
        "triton",
        (input_parameter, output_parameter),
        fm.Sequential((dispatch,)),
        fm.Return((fm.ReturnBinding(fm.ValueRef("result", value_type), "result"),)),
    )


def _typed_kernel(
    name: str,
    input_type: fm.IRType,
    output_type: fm.IRType,
) -> fm.PrimFunction:
    input_parameter = fm.PrimParameter(
        "input", input_type, fm.PrimParameterRole.INPUT
    )
    output_parameter = fm.PrimParameter(
        "result", output_type, fm.PrimParameterRole.OUTPUT
    )
    dispatch = fm.KernelDispatch(
        semantic_op=f"test.{name}",
        candidate=f"test.{name}",
        parameters={"family": "test", "variant": name},
        arguments=("input",),
        outputs=("result",),
        reads=("input",),
        writes=("result",),
        memory_effects=(
            ("input", fm.MemoryEffect.READ),
            ("result", fm.MemoryEffect.WRITE),
        ),
    )
    return fm.PrimFunction(
        name,
        "triton",
        (input_parameter, output_parameter),
        fm.Sequential((dispatch,)),
        fm.Return(
            (fm.ReturnBinding(fm.ValueRef("result", output_type), "result"),)
        ),
    )


def test_partial_owner_raw_uses_only_its_mesh_axis_group():
    placement = fm.Placement((8, 16), "yx", "bb")
    tensor = fm.tensor_type("float32", (1, 128))
    partial = fm.DistributedType(
        tensor,
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((1,))),
        placement,
        partial=fm.SBP.partial((0,)),
    )
    producer = _kernel("producer", partial, fm.MemoryEffect.READ)
    consumer = _kernel(
        "consumer", partial, fm.MemoryEffect.READ.across_partial_owners()
    )
    builder = fm.IRBuilder(dialect="tir", stage="selected_tir")
    source = builder.var("source", partial, id="source")
    produced = builder.call(
        "tir.call", (source,), partial, id="produced",
        attrs={"callee": producer.name},
    )
    consumed = builder.call(
        "tir.call", (produced,), partial, id="consumed",
        attrs={"callee": consumer.name},
    )
    builder.prim_function(producer)
    builder.prim_function(consumer)
    builder.function("main", (source,), (consumed,))
    module = builder.build(entry="main")

    plan = plan_memory_synchronization(
        module,
        fm.make_buffer_plan(module, options=_chip_workspace_options()),
    )

    assert len(plan.events) == 1
    event = plan.events[0]
    assert (event.after, event.before) == ("produced", "consumed")
    assert event.scope == "grid"
    assert event.axis_group_axes == (0,)


def test_synchronized_local_kernel_does_not_widen_partial_owner_hazard():
    """Internal grid synchronization is not an all-owner publication effect."""

    placement = fm.Placement((8, 16), "yx", "bb")
    tensor = fm.tensor_type("float32", (1, 128))
    partial = fm.DistributedType(
        tensor,
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((1,))),
        placement,
        partial=fm.SBP.partial((0,)),
    )
    producer = _kernel("synchronized_producer", partial, fm.MemoryEffect.READ)
    consumer = _kernel(
        "synchronized_consumer",
        partial,
        fm.MemoryEffect.READ.across_partial_owners(),
        requires=("cooperative_grid", "grid_sync"),
    )
    builder = fm.IRBuilder(dialect="tir", stage="selected_tir")
    source = builder.var("source", partial, id="source")
    produced = builder.call(
        "tir.call", (source,), partial, id="produced",
        attrs={"callee": producer.name},
    )
    consumed = builder.call(
        "tir.call", (produced,), partial, id="consumed",
        attrs={"callee": consumer.name},
    )
    builder.prim_function(producer)
    builder.prim_function(consumer)
    builder.function("main", (source,), (consumed,))
    module = builder.build(entry="main")

    event = plan_memory_synchronization(
        module,
        fm.make_buffer_plan(module, options=_chip_workspace_options()),
    ).events[0]

    assert event.scope == "grid"
    assert event.axis_group_axes == (0,)


def test_sat_reuse_keeps_both_war_and_waw_pending_after_a_block_barrier():
    """Reading a dead view does not discharge its earlier chip-visible write."""

    placement = fm.Placement((8, 16), "yx", "bb")
    tensor = fm.tensor_type("float32", (1, 128))
    reader_type = fm.DistributedType(
        tensor,
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((1,))),
        placement,
    )
    writer_type = fm.DistributedType(
        tensor,
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((1, 0))),
        placement,
    )
    scalar_type = fm.tensor_type("float32", ())
    builder = fm.IRBuilder(dialect="tir", stage="selected_tir")
    source = builder.var("source", reader_type, id="source")
    resident = builder.call(
        "tir.kernel", (source,), reader_type, id="resident"
    )
    observed = builder.call(
        "tir.kernel", (resident,), scalar_type, id="observed"
    )
    one = builder.call(
        "tir.scalar_const", (), scalar_type, id="one", attrs={"value": 1.0}
    )
    reused = builder.call("tir.kernel", (one,), writer_type, id="reused")
    result = builder.call("tir.kernel", (reused,), writer_type, id="result")
    builder.function("main", (source,), (observed, result))
    module = builder.build(entry="main")
    plan = fm.make_buffer_plan(module, options=_chip_workspace_options())

    assert plan.buffer_map["resident"].offset == plan.buffer_map["reused"].offset
    event = next(
        value for value in plan_memory_synchronization(module, plan).events
        if value.before == "reused"
    )

    assert event.hazards == ("READ->WRITE", "WRITE->WRITE")
    assert event.scope == "grid"
    assert event.axis_group_axes == ()


def test_vector_reinterpret_prefix_coarsening_uses_removed_axis_group():
    """Granularity is compared in bytes across vector/scalar buffer views."""

    placement = fm.Placement((8, 16), "yx", "bb")
    vector_tensor = fm.tensor_type(fm.vector_type("bfloat16", (8,)), (1, 256))
    scalar_tensor = fm.tensor_type("bfloat16", (1, 2048))
    fine = fm.DistributedType(
        vector_tensor,
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((0, 1), 2)),
        placement,
    )
    fine_scalar = fm.DistributedType(
        scalar_tensor,
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((0, 1), 16)),
        placement,
    )
    coarse = fm.DistributedType(
        scalar_tensor,
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((0,), 256)),
        placement,
    )
    producer = _typed_kernel("vector_producer", fine, fine)
    consumer = _typed_kernel("coarse_consumer", coarse, coarse)
    builder = fm.IRBuilder(dialect="tir", stage="selected_tir")
    source = builder.var("source", fine, id="source")
    produced = builder.call(
        "tir.call", (source,), fine, id="produced", attrs={"callee": producer.name}
    )
    unpacked = builder.call(
        "tir.buffer_view",
        (produced,),
        fine_scalar,
        id="unpacked",
        attrs={"alias_kind": "vector_reinterpret"},
    )
    coarse_view = builder.call(
        "distributed.sharded_view",
        (unpacked,),
        coarse,
        id="coarse_view",
        attrs={"new_type": coarse},
    )
    consumed = builder.call(
        "tir.call",
        (coarse_view,),
        coarse,
        id="consumed",
        attrs={"callee": consumer.name},
    )
    builder.prim_function(producer)
    builder.prim_function(consumer)
    builder.function("main", (source,), (consumed,))
    module = builder.build(entry="main")
    event = plan_memory_synchronization(
        module,
        fm.make_buffer_plan(module, options=_chip_workspace_options()),
    ).events[0]

    assert (event.after, event.before) == ("produced", "consumed")
    assert event.scope == "grid"
    assert event.axis_group_axes == (1,)


def test_vector_reinterpret_equivalent_split_does_not_widen_axis_group():
    """The same byte split expressed in vector/scalar units is one owner map."""

    placement = fm.Placement((8, 16), "yx", "bb")
    vector_tensor = fm.tensor_type(fm.vector_type("bfloat16", (8,)), (1, 256))
    scalar_tensor = fm.tensor_type("bfloat16", (1, 2048))
    vector_split = fm.DistributedType(
        vector_tensor,
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((0,), 32)),
        placement,
    )
    scalar_split = fm.DistributedType(
        scalar_tensor,
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((0,), 256)),
        placement,
    )
    producer = _typed_kernel("vector_split_producer", vector_split, vector_split)
    consumer = _typed_kernel("scalar_split_consumer", scalar_split, scalar_split)
    builder = fm.IRBuilder(dialect="tir", stage="selected_tir")
    source = builder.var("source", vector_split, id="source")
    produced = builder.call(
        "tir.call", (source,), vector_split, id="produced", attrs={"callee": producer.name}
    )
    unpacked = builder.call(
        "tir.buffer_view",
        (produced,),
        scalar_split,
        id="unpacked",
        attrs={"alias_kind": "vector_reinterpret"},
    )
    consumed = builder.call(
        "tir.call",
        (unpacked,),
        scalar_split,
        id="consumed",
        attrs={"callee": consumer.name},
    )
    builder.prim_function(producer)
    builder.prim_function(consumer)
    builder.function("main", (source,), (consumed,))
    module = builder.build(entry="main")
    event = plan_memory_synchronization(
        module,
        fm.make_buffer_plan(module, options=_chip_workspace_options()),
    ).events[0]

    assert event.scope == "grid"
    assert event.axis_group_axes == (1,)


def test_full_physical_axis_group_is_canonicalized_to_full_grid():
    placement = fm.Placement((8,), "x", "b")
    tensor = fm.tensor_type("float32", (1, 128))
    partial = fm.DistributedType(
        tensor,
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((0,))),
        placement,
        partial=fm.SBP.partial((0,)),
    )
    producer = _kernel("producer_full", partial, fm.MemoryEffect.READ)
    consumer = _kernel(
        "consumer_full", partial,
        fm.MemoryEffect.READ.across_partial_owners(),
    )
    builder = fm.IRBuilder(dialect="tir", stage="selected_tir")
    source = builder.var("source", partial, id="source")
    produced = builder.call(
        "tir.call", (source,), partial, id="produced",
        attrs={"callee": producer.name},
    )
    consumed = builder.call(
        "tir.call", (produced,), partial, id="consumed",
        attrs={"callee": consumer.name},
    )
    builder.prim_function(producer)
    builder.prim_function(consumer)
    builder.function("main", (source,), (consumed,))
    module = builder.build(entry="main")

    event = plan_memory_synchronization(
        module,
        fm.make_buffer_plan(module, options=_chip_workspace_options()),
    ).events[0]

    assert event.scope == "grid"
    assert event.axis_group_axes == ()
