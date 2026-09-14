# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.microkernels.materialization import validate_transfer_pipeline
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.passes.tir import lower_transfer_pipeline_regions

from .helpers import module, pipeline_dispatch, prim_function
from .test_interprocedural_pipeline_regions import _call, _shared_for_call


def _indexed_pipeline(metadata_index=1):
    dispatch = pipeline_dispatch("kernel", 1, shared_start=0)
    pipeline = replace(dispatch.microkernel.transfer_pipeline,
                       producer_read_argument_indices=(metadata_index,))
    dispatch = replace(dispatch, arguments=("source", "route_ids"), reads=("source", "route_ids"),
                       microkernel=replace(dispatch.microkernel, transfer_pipeline=pipeline))
    function = prim_function("kernel", (dispatch,))
    function = replace(function, parameters=(function.parameters[0],
                       fm.T.prim_parameter("route_ids", fm.tensor_type("int64", (9,)),
                                           fm.T.PrimParameterRole.INPUT, alignment_bytes=8),
                       function.parameters[1]))
    return function, dispatch


def test_index_read_is_a_dependency_but_not_an_aligned_transfer():
    function, dispatch = _indexed_pipeline()
    validate_transfer_pipeline(function, dispatch, dispatch.microkernel, stage="test")
    ordinary = pipeline_dispatch("kernel", 0, shared_start=128)
    ordinary = replace(ordinary, outputs=("route_ids",), writes=("route_ids",))
    function = replace(function, body=fm.T.sequential((ordinary, dispatch)))
    result = lower_transfer_pipeline_regions(module(function))
    region = result.prim_functions[0].body.fields[0]
    consumer = [value for value in region.consume_body.fields if isinstance(value, fm.PipelineHandoff)]
    producer = [value for value in region.produce_body.fields if isinstance(value, fm.PipelineHandoff)]
    assert len(consumer) == len(producer) == 1
    assert consumer[0].handoff_id == producer[0].handoff_id
    assert region.produce_body.fields[1] == producer[0]


@pytest.mark.parametrize("has_barrier", (False, True))
def test_router_write_handoff_requires_a_real_publication_barrier(has_barrier):
    function, dispatch = _indexed_pipeline()
    writer = _call("router", "ordinary_kernel", "source", "ids", (), transfer=False)
    invocation = fm.T.prim_function_call(
        "experts", function.name,
        (fm.T.prim_call_binding("source", "source"), fm.T.prim_call_binding("route_ids", "ids")),
        (fm.T.prim_call_binding("output", "output"),),
        shared_workspace_buffers=_shared_for_call("main", "experts", dispatch),
        transfer_sources=("source", "ids"), reads=("source", "ids"), writes=("output",))
    barrier = fm.T.barrier(fm.T.BarrierScope.BLOCK, ("router",), "experts")
    main = fm.T.execution_function("main", ("source",), ("output",),
                                  fm.T.sequential((writer, *((barrier,) if has_barrier else ()), invocation)))
    graph = replace(module(function), execution_functions=(main,))
    if not has_barrier:
        with pytest.raises(IRVerificationError, match="without an effective Block barrier"):
            lower_transfer_pipeline_regions(graph)
        return
    region = lower_transfer_pipeline_regions(graph).execution_function_map["main"].body.fields[0]
    handoff = next(value for value in region.produce_body.fields if isinstance(value, fm.PipelineHandoff))
    assert region.consume_body.fields.index(handoff) == region.consume_body.fields.index(barrier) + 1


@pytest.mark.parametrize("kind", ("out_of_range", "unread", "written"))
def test_invalid_producer_read_dependency_is_rejected(kind):
    function, dispatch = _indexed_pipeline(2 if kind == "out_of_range" else 1)
    if kind == "unread":
        dispatch = replace(dispatch, reads=("source",))
    if kind == "written":
        dispatch = replace(dispatch, writes=(*dispatch.writes, "route_ids"))
    with pytest.raises(IRVerificationError, match="producer read operand"):
        validate_transfer_pipeline(function, dispatch, dispatch.microkernel, stage="test")
