# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The scheduling domain is the function, not individual kernel definitions."""

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.passes.tir import lower_transfer_pipeline_regions
from .helpers import pipeline_dispatch, prim_function
from .test_interprocedural_pipeline_regions import _shared_for_call


@pytest.mark.parametrize("overlap", [False, True])
def test_adjacent_ops_share_one_region_and_drain_only_for_actual_shared_interference(overlap):
    definitions, invokes = [], []
    for index in range(2):
        dispatch = pipeline_dispatch(f"op{index}", index, shared_start=0 if overlap else index * 64)
        signature = prim_function(f"op{index}", (dispatch,))
        definitions.append(fm.T.kernel_definition(signature.name, signature.module_kind,
                                                  signature.parameters, dispatch, signature.results))
        invokes.append(fm.T.kernel_invoke(f"invoke{index}", signature.name,
                        arguments=(fm.T.prim_call_binding("source", "x"),),
                        results=(fm.T.prim_call_binding("output", f"y{index}"),),
                        shared_workspace_buffers=_shared_for_call("main", f"invoke{index}", dispatch),
                        transfer_sources=("x",), reads=("x",), writes=(f"y{index}",)))
    module = fm.IRModule("bufferized_tir", "synchronized_tir", (), (fm.Function("main", (), ()),), "main",
                         kernel_definitions=tuple(definitions), execution_functions=(
                             fm.T.execution_function("main", ("x",), ("y0", "y1"), fm.T.sequential(tuple(invokes))),))
    result = lower_transfer_pipeline_regions(module)
    assert result.kernel_definitions == module.kernel_definitions
    assert not result.prim_functions
    region, = result.execution_functions[0].body.fields
    assert isinstance(region, fm.ProducerConsumerRegion)
    assert all(not isinstance(node, fm.ProducerConsumerRegion) for node in region.consume_body.fields)
    assert [stage.operation for stage in region.consume_body.fields if isinstance(stage, fm.PipelineStage)] == invokes
    drains = [node for node in region.consume_body.fields if isinstance(node, fm.PipelineDrain)]
    assert len(drains) == int(overlap)
