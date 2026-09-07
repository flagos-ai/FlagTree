# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

from triton.flagmega import ir as fm
from triton.flagmega.passes.tir import lower_transfer_pipeline_regions

from .helpers import module, pipeline_dispatch, prim_function


def _region(result):
    function = result.prim_functions[0]
    return function.body.fields[0]


def test_single_pipeline_dispatch_gets_explicit_producer_and_consumer_roles():
    dispatch = pipeline_dispatch("kernel", 0, shared_start=0)

    result = lower_transfer_pipeline_regions(
        module(prim_function("kernel", (dispatch,)))
    )
    region = _region(result)

    assert isinstance(region, fm.ProducerConsumerRegion)
    assert [type(value) for value in region.produce_body.fields] == [fm.PipelineStage]
    assert [type(value) for value in region.consume_body.fields] == [fm.PipelineStage]
    assert region.produce_body.fields[0].operation is dispatch
    assert region.consume_body.fields[0].operation is dispatch


def test_non_pipeline_function_is_structurally_unchanged():
    dispatch = pipeline_dispatch(
        "kernel", 0, shared_start=0, pipelined=False
    )
    original = module(prim_function("kernel", (dispatch,)))

    assert lower_transfer_pipeline_regions(original) is original


def test_overlapping_pipeline_stages_drain_previous_shared_owner():
    first = pipeline_dispatch("kernel", 0, shared_start=0)
    second = pipeline_dispatch("kernel", 1, shared_start=32)

    region = _region(lower_transfer_pipeline_regions(
        module(prim_function("kernel", (first, second)))
    ))

    producer_drains = [
        value.stage_id for value in region.produce_body.fields
        if isinstance(value, fm.PipelineDrain)
    ]
    consumer_drains = [
        value.stage_id for value in region.consume_body.fields
        if isinstance(value, fm.PipelineDrain)
    ]
    assert producer_drains == ["kernel_transfer_stage_0"]
    assert consumer_drains == producer_drains


def test_disjoint_pipeline_stages_do_not_add_synchronization():
    first = pipeline_dispatch("kernel", 0, shared_start=0)
    second = pipeline_dispatch("kernel", 1, shared_start=64)

    region = _region(lower_transfer_pipeline_regions(
        module(prim_function("kernel", (first, second)))
    ))

    assert not any(isinstance(value, fm.PipelineDrain) for value in region.consume_body.fields)
    assert not any(isinstance(value, fm.PipelineHandoff) for value in region.consume_body.fields)


def test_ordinary_shared_owner_hands_reused_range_to_pipeline_producer():
    ordinary = pipeline_dispatch(
        "kernel", 0, shared_start=0, pipelined=False
    )
    pipeline = pipeline_dispatch("kernel", 1, shared_start=32)

    region = _region(lower_transfer_pipeline_regions(
        module(prim_function("kernel", (ordinary, pipeline)))
    ))

    consumer_handoff = next(
        value for value in region.consume_body.fields
        if isinstance(value, fm.PipelineHandoff)
    )
    producer_handoff = next(
        value for value in region.produce_body.fields
        if isinstance(value, fm.PipelineHandoff)
    )
    assert consumer_handoff.handoff_id == producer_handoff.handoff_id
    assert region.consume_body.fields.index(consumer_handoff) == 1
    assert region.produce_body.fields.index(producer_handoff) == 0


def test_structured_ordinary_shared_owner_is_an_atomic_consumer_boundary():
    ordinary = pipeline_dispatch(
        "kernel", 0, shared_start=0, pipelined=False
    )
    conditional = fm.T.if_then_else(
        fm.T.immediate(True, fm.tensor_type("bool", ())),
        fm.T.sequential((ordinary,)),
    )
    pipeline = pipeline_dispatch("kernel", 1, shared_start=32)
    function = prim_function("kernel", ())
    function = replace(
        function,
        body=fm.T.sequential((conditional, pipeline)),
    )

    region = _region(lower_transfer_pipeline_regions(module(function)))
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
