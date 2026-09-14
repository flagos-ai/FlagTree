# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""P/C read dependencies follow physical subspans, not just logical names."""

from collections import defaultdict

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.passes.tir.lower_transfer_pipeline_regions import _add_execution_source_handoffs


@pytest.mark.parametrize("offset", (0, 8, 16, 32))
@pytest.mark.parametrize("barrier_position", ("none", "before_call", "before_intervening_call"))
def test_source_subspan_overlap_controls_handoff(offset, barrier_position):
    backing = fm.T.physical_buffer("ids", "workspace", 64, 16, function="main")
    spans = {"written": fm.T.mem_span(backing, start=0, size=16),
             "alias_read": fm.T.mem_span(backing, start=offset, size=16)}
    writer = fm.KernelInvoke("writer", "router", results=(fm.PrimCallBinding("result", "written"),), writes=("written",))
    unrelated = fm.KernelInvoke("unrelated", "elementwise", results=(fm.PrimCallBinding("result", "other"),), writes=("other",))
    consumer = fm.KernelInvoke("consumer", "indexed", arguments=(fm.PrimCallBinding("ids", "alias_read"),),
                               transfer_sources=("alias_read",), reads=("alias_read",))
    stage = fm.PipelineStage("indexed_stage", consumer)
    barrier = fm.T.barrier(fm.T.BarrierScope.BLOCK, ("writer",),
                           "unrelated" if barrier_position == "before_intervening_call" else "consumer")
    order = (writer, *((barrier,) if barrier_position == "before_intervening_call" else ()), unrelated,
             *((barrier,) if barrier_position == "before_call" else ()), consumer)
    after, before = defaultdict(list), defaultdict(list)
    args = (order, {id(consumer): stage}, after, before, "main", spans)
    if offset < 16 and barrier_position == "none":
        with pytest.raises(IRVerificationError, match="without an effective Block barrier"):
            _add_execution_source_handoffs(*args)
    else:
        _add_execution_source_handoffs(*args)
        if offset < 16:
            assert len(after[id(barrier)]) == len(before[id(consumer)]) == 1
            assert after[id(barrier)][0] == before[id(consumer)][0]
        else:
            assert not after and not before


def test_alias_of_pipeline_output_is_released_after_its_consumer():
    backing = fm.T.physical_buffer("ids", "workspace", 64, 16, function="main")
    producer = fm.KernelInvoke("producer", "first", results=(fm.PrimCallBinding("result", "written"),), writes=("written",))
    consumer = fm.KernelInvoke("consumer", "second", arguments=(fm.PrimCallBinding("ids", "alias"),),
                               transfer_sources=("alias",), reads=("alias",))
    stages = {id(call): fm.PipelineStage(call.call_id, call) for call in (producer, consumer)}
    after, before = defaultdict(list), defaultdict(list)
    _add_execution_source_handoffs((producer, consumer), stages, after, before, "main",
                                  {"written": fm.T.mem_span(backing),
                                   "alias": fm.T.mem_span(backing, start=16, size=16)})
    assert len(after[id(producer)]) == len(before[id(consumer)]) == 1
    assert after[id(producer)][0] == before[id(consumer)][0]


def test_source_handoff_precedes_independent_consumer_work():
    writer = fm.KernelInvoke("writer", "router", results=(fm.PrimCallBinding("result", "ids"),), writes=("ids",))
    independent = fm.KernelInvoke("independent", "compute")
    consumer = fm.KernelInvoke("consumer", "indexed", arguments=(fm.PrimCallBinding("ids", "ids"),),
                               transfer_sources=("ids",), reads=("ids",))
    publication = fm.T.barrier(fm.T.BarrierScope.BLOCK, ("writer",), "independent")
    later = fm.T.barrier(fm.T.BarrierScope.CHIP, ("independent",), "consumer")
    after, before = defaultdict(list), defaultdict(list)
    _add_execution_source_handoffs((writer, publication, independent, later, consumer),
                                  {id(consumer): fm.PipelineStage("stage", consumer)}, after, before, "main", {})
    assert after[id(publication)] == before[id(consumer)]
    assert len(after[id(publication)]) == 1
    assert not after[id(later)]


@pytest.mark.parametrize("first_axes,required_axes", (((), ()), ((0,), (1,)), ((0,), (0, 1))))
def test_source_handoff_waits_for_sufficient_owner_scope(first_axes, required_axes):
    placement = fm.Placement(hierarchy=(2, 4), name="yx", hierarchy_levels="bb")
    writer = fm.KernelInvoke("writer", "router", results=(fm.PrimCallBinding("result", "ids"),), writes=("ids",))
    consumer = fm.KernelInvoke("consumer", "indexed", arguments=(fm.PrimCallBinding("ids", "ids"),),
                               transfer_sources=("ids",), reads=("ids",))
    first = fm.Barrier(fm.BarrierScope.CHIP if first_axes else fm.BarrierScope.BLOCK,
                       ("writer",), "consumer", axis_group_axes=first_axes)
    publication = fm.Barrier(fm.BarrierScope.CHIP, ("writer",), "consumer", axis_group_axes=required_axes)
    after, before = defaultdict(list), defaultdict(list)
    _add_execution_source_handoffs((writer, first, publication, consumer),
                                  {id(consumer): fm.PipelineStage("stage", consumer)}, after, before, "main", {},
                                  {"consumer": (("writer", ("grid", required_axes, placement)),)})
    assert after[id(publication)] == before[id(consumer)]
    assert len(after[id(publication)]) == 1
    assert not after[id(first)]


def test_later_local_writer_does_not_hide_an_unpublished_chip_source():
    first = fm.KernelInvoke("first", "remote", results=(fm.PrimCallBinding("result", "a"),), writes=("a",))
    second = fm.KernelInvoke("second", "local", results=(fm.PrimCallBinding("result", "b"),), writes=("b",))
    consumer = fm.KernelInvoke("consumer", "indexed", arguments=(fm.PrimCallBinding("a", "a"), fm.PrimCallBinding("b", "b")),
                               transfer_sources=("a", "b"), reads=("a", "b"))
    barrier = fm.T.barrier(fm.T.BarrierScope.BLOCK, ("second",), "consumer")
    with pytest.raises(IRVerificationError, match="publication barrier"):
        _add_execution_source_handoffs((first, second, barrier, consumer),
                                      {id(consumer): fm.PipelineStage("stage", consumer)},
                                      defaultdict(list), defaultdict(list), "main", {},
                                      {"consumer": (("first", ("grid", (), None)),
                                                    ("second", ("block", (), None)))})
