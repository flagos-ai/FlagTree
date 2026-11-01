# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Producer source dependencies, rather than consumer work, require barriers."""

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.passes.tir import lower_transfer_pipeline_regions
from triton.flagmega.passes.tir.lower_transfer_pipeline_regions import _producer_needs_source_barriers

from .helpers import module, pipeline_dispatch, prim_function
from .test_interprocedural_pipeline_regions import _call, _shared_for_call


def _source_graph(scope, *, source_written=False, write_after=False):
    dispatch = pipeline_dispatch("weight_kernel", 0, shared_start=0)
    kernel = prim_function("weight_kernel", (dispatch,))
    writer = _call("writer", "ordinary", "update",
                   "source" if source_written else "activation", (), transfer=False)
    stage = _call("weight_load", kernel.name, "source", "output",
                  _shared_for_call("main", "weight_load", dispatch))
    publication = fm.T.barrier(scope, ("writer",), "weight_load")
    completion = fm.T.barrier(fm.BarrierScope.CHIP, ("weight_load",), "done")
    fields = ((stage, completion, writer, publication) if write_after
              else (writer, publication, stage, completion))
    main = fm.T.execution_function("main", ("update", "source"), ("output",),
                                   fm.T.sequential(fields))
    return replace(module(kernel), execution_functions=(main,))


@pytest.mark.parametrize("scope", (fm.BarrierScope.BLOCK, fm.BarrierScope.CHIP, fm.BarrierScope.DEVICE))
@pytest.mark.parametrize("write_after", (False, True))
def test_readonly_producer_does_not_wait_on_consumer_barriers(scope, write_after):
    graph = _source_graph(scope, write_after=write_after)
    region = lower_transfer_pipeline_regions(graph).execution_function_map["main"].body.fields[0]
    assert not any(isinstance(value, fm.Barrier) for value in region.produce_body.fields)
    assert [value for value in region.consume_body.fields if isinstance(value, fm.Barrier)] == [
        value for value in graph.execution_function_map["main"].body.fields if isinstance(value, fm.Barrier)
    ]
    assert sum(isinstance(value, fm.PipelineStage) for value in region.produce_body.fields) == 1


@pytest.mark.parametrize("scope", (fm.BarrierScope.BLOCK, fm.BarrierScope.CHIP, fm.BarrierScope.DEVICE))
@pytest.mark.parametrize("write_after", (False, True))
def test_mutable_source_retains_its_partition_publication_order(scope, write_after):
    graph = _source_graph(scope, source_written=True, write_after=write_after)
    region = lower_transfer_pipeline_regions(graph).execution_function_map["main"].body.fields[0]
    expected = [value for value in graph.execution_function_map["main"].body.fields if isinstance(value, fm.Barrier)]
    assert [value for value in region.produce_body.fields if isinstance(value, fm.Barrier)] == expected
    if not write_after:
        handoff, = [value for value in region.produce_body.fields if isinstance(value, fm.PipelineHandoff)]
        stage, = [value for value in region.produce_body.fields if isinstance(value, fm.PipelineStage)]
        assert region.produce_body.fields.index(handoff) + 1 == region.produce_body.fields.index(stage)
        assert region.consume_body.fields.index(handoff) == region.consume_body.fields.index(expected[0]) + 1


@pytest.mark.parametrize("offset", (0, 8, 16, 32))
@pytest.mark.parametrize("write_after", (False, True))
def test_alias_and_loop_tail_writes_require_source_barriers(offset, write_after):
    physical = fm.T.physical_buffer("global", "workspace", 64, 16, function="main")
    spans = {"written": fm.T.mem_span(physical, start=0, size=16),
             "source": fm.T.mem_span(physical, start=offset, size=16)}
    writer = fm.KernelInvoke("writer", "ordinary",
                            results=(fm.PrimCallBinding("output", "written"),), writes=("written",))
    reader = fm.KernelInvoke("reader", "pipeline",
                            arguments=(fm.PrimCallBinding("input", "source"),),
                            transfer_sources=("source",), reads=("source",))
    stage = fm.PipelineStage("read_stage", reader)
    order = (reader, writer) if write_after else (writer, reader)
    assert _producer_needs_source_barriers(order, {id(reader): stage}, spans) is (offset < 16)
