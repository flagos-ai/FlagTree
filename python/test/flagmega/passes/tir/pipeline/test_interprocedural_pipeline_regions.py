# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

from triton.flagmega import ir as fm
from triton.flagmega.passes.tir import lower_transfer_pipeline_regions

from .helpers import module, pipeline_dispatch, prim_function


def _shared_for_call(function: str, call_id: str, dispatch):
    result = []
    for buffer in dispatch.shared_workspace_buffers:
        physical = replace(
            buffer.mem_span.buffer,
            id=f"execution:{function}:{call_id}:{buffer.name}",
            function=function,
        )
        result.append(replace(
            buffer,
            name=f"{call_id}.{buffer.name}",
            mem_span=fm.T.mem_span(
                physical,
                start=buffer.mem_span.start,
                size=buffer.mem_span.size,
            ),
        ))
    return tuple(result)


def _call(call_id, callee, source, output, shared, *, transfer=True):
    return fm.T.prim_function_call(
        call_id,
        callee,
        arguments=(fm.T.prim_call_binding("source", source),),
        results=(fm.T.prim_call_binding("output", output),),
        shared_workspace_buffers=shared,
        transfer_sources=(source,) if transfer else (),
        reads=(source,),
        writes=(output,),
    )


def _interprocedural_module(*, disjoint: bool = False):
    dispatch = pipeline_dispatch(
        "pipeline_kernel", 0, shared_start=0
    )
    kernel = prim_function("pipeline_kernel", (dispatch,))
    callee_call = _call(
        "callee_kernel",
        kernel.name,
        "callee_source",
        "callee_output",
        _shared_for_call("worker", "callee_kernel", dispatch),
    )
    worker = fm.T.execution_function(
        "worker",
        ("callee_source",),
        ("callee_output",),
        fm.T.sequential((callee_call,)),
        attrs={"transfer_source_parameters": ("callee_source",)},
    )

    first_shared = _shared_for_call("main", "first", dispatch)
    second_shared = _shared_for_call("main", "second", dispatch)
    if disjoint:
        second_shared = tuple(
            replace(
                buffer,
                mem_span=fm.T.mem_span(
                    replace(buffer.mem_span.buffer, start=64),
                    start=buffer.mem_span.start,
                    size=buffer.mem_span.size,
                ),
            )
            for buffer in second_shared
        )
    first = _call(
        "first", "worker", "main_source", "first_output", first_shared
    )
    second = _call(
        "second", "worker", "main_source", "main_output", second_shared
    )
    main = fm.T.execution_function(
        "main",
        ("main_source",),
        ("main_output",),
        fm.T.sequential((first, second)),
    )
    return replace(
        module(kernel),
        execution_functions=(worker, main),
    )


def test_pipeline_bearing_callee_is_propagated_to_each_caller_site():
    result = lower_transfer_pipeline_regions(_interprocedural_module())
    worker = result.execution_function_map["worker"]
    main = result.execution_function_map["main"]

    assert isinstance(worker.body.fields[0], fm.ProducerConsumerRegion)
    assert isinstance(main.body.fields[0], fm.ProducerConsumerRegion)
    region = main.body.fields[0]
    stages = [
        value for value in region.consume_body.fields
        if isinstance(value, fm.PipelineStage)
    ]
    assert [value.operation.call_id for value in stages] == ["first", "second"]
    assert [
        value.stage_id for value in region.consume_body.fields
        if isinstance(value, fm.PipelineDrain)
    ] == ["main_transfer_stage_0"]


def test_repeated_pipeline_callee_with_disjoint_shared_ranges_does_not_drain():
    result = lower_transfer_pipeline_regions(
        _interprocedural_module(disjoint=True)
    )
    region = result.execution_function_map["main"].body.fields[0]

    assert not any(
        isinstance(value, fm.PipelineDrain)
        for value in region.consume_body.fields
    )


def test_block_barrier_releases_mutable_source_to_pipeline_producer():
    dispatch = pipeline_dispatch(
        "pipeline_kernel", 0, shared_start=0
    )
    kernel = prim_function("pipeline_kernel", (dispatch,))
    writer = replace(
        _call(
            "writer",
            "ordinary_kernel",
            "update",
            "source",
            (),
            transfer=False,
        ),
        writes=("source",),
    )
    pipeline = _call(
        "pipeline",
        kernel.name,
        "source",
        "output",
        _shared_for_call("main", "pipeline", dispatch),
    )
    barrier = fm.T.barrier(
        fm.T.BarrierScope.BLOCK,
        ("writer",),
        "pipeline",
    )
    main = fm.T.execution_function(
        "main",
        ("update",),
        ("output",),
        fm.T.sequential((writer, barrier, pipeline)),
    )
    result = lower_transfer_pipeline_regions(replace(
        module(kernel), execution_functions=(main,)
    ))
    region = result.execution_function_map["main"].body.fields[0]
    consumer = region.consume_body.fields
    producer = region.produce_body.fields
    consumer_handoff = next(
        value for value in consumer if isinstance(value, fm.PipelineHandoff)
    )
    producer_handoff = next(
        value for value in producer if isinstance(value, fm.PipelineHandoff)
    )

    assert consumer.index(consumer_handoff) == consumer.index(barrier) + 1
    assert producer_handoff.handoff_id == consumer_handoff.handoff_id
    assert producer.index(producer_handoff) + 1 == next(
        index for index, value in enumerate(producer)
        if isinstance(value, fm.PipelineStage)
    )


def test_barrier_after_later_predecessor_releases_all_prior_sequential_writes():
    dispatch = pipeline_dispatch("pipeline_kernel", 0, shared_start=0)
    kernel = prim_function("pipeline_kernel", (dispatch,))
    first_writer = replace(
        _call(
            "first_writer", "ordinary_kernel", "first_update", "first_source",
            (), transfer=False,
        ),
        writes=("first_source",),
    )
    second_writer = replace(
        _call(
            "second_writer", "ordinary_kernel", "second_update", "second_source",
            (), transfer=False,
        ),
        writes=("second_source",),
    )
    pipeline = _call(
        "pipeline", kernel.name, "first_source", "output",
        _shared_for_call("main", "pipeline", dispatch),
    )
    barrier = fm.T.barrier(
        fm.T.BarrierScope.CHIP,
        ("second_writer",),
        "pipeline",
    )
    main = fm.T.execution_function(
        "main",
        ("first_update", "second_update"),
        ("output",),
        fm.T.sequential((first_writer, second_writer, barrier, pipeline)),
    )

    result = lower_transfer_pipeline_regions(replace(
        module(kernel), execution_functions=(main,)
    ))
    region = result.execution_function_map["main"].body.fields[0]
    consumer_handoff = next(
        value for value in region.consume_body.fields
        if isinstance(value, fm.PipelineHandoff)
    )
    producer_handoff = next(
        value for value in region.produce_body.fields
        if isinstance(value, fm.PipelineHandoff)
    )

    assert region.consume_body.fields.index(consumer_handoff) == 3
    assert producer_handoff.handoff_id == consumer_handoff.handoff_id


def test_structured_ordinary_execution_owner_hands_shared_range_to_stage():
    dispatch = pipeline_dispatch("pipeline_kernel", 0, shared_start=0)
    kernel = prim_function("pipeline_kernel", (dispatch,))
    shared = _shared_for_call("main", "shared", dispatch)
    ordinary = _call(
        "ordinary", "ordinary_kernel", "update", "temporary", shared,
        transfer=False,
    )
    conditional = fm.T.if_then_else(
        fm.T.immediate(True, fm.tensor_type("bool", ())),
        fm.T.sequential((ordinary,)),
    )
    pipeline = _call(
        "pipeline", kernel.name, "source", "output", shared,
    )
    main = fm.T.execution_function(
        "main",
        ("update", "source"),
        ("output",),
        fm.T.sequential((conditional, pipeline)),
    )

    result = lower_transfer_pipeline_regions(replace(
        module(kernel), execution_functions=(main,)
    ))
    region = result.execution_function_map["main"].body.fields[0]
    consumer_handoff = next(
        value for value in region.consume_body.fields
        if isinstance(value, fm.PipelineHandoff)
    )
    producer_handoff = next(
        value for value in region.produce_body.fields
        if isinstance(value, fm.PipelineHandoff)
    )

    assert region.consume_body.fields[0] is conditional
    assert region.consume_body.fields.index(consumer_handoff) == 1
    assert producer_handoff.handoff_id == consumer_handoff.handoff_id
