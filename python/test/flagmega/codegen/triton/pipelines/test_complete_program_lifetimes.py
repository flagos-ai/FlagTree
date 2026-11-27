# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A whole-entry program keeps synchronization owned by the physical schedule."""

from copy import deepcopy
from types import SimpleNamespace
import ast

import pytest

from triton.flagmega.codegen.triton.kernel_programs import describe_kernel_programs
from triton.flagmega.codegen.triton.pipeline_programs import project_pipeline_schedule
from triton.flagmega.errors import CodegenError


def complete_entry():
    ordinary = dict(kind="tir.kernel_call", call="ordinary", parameters={})
    pipeline = dict(kind="pipeline_kernel_call", call="pipeline", parameters={})
    handoff = dict(kind="handoff", endpoint="reuse_writer", role="consumer")
    drain = dict(kind="drain", endpoints=["tile_reader"], stage_id="stage")
    events = [dict(kind="barrier", scope="block"), ordinary, handoff,
              dict(kind="barrier", scope="block"), pipeline, drain]
    schedule = dict(
        stages=[dict(id="stage", call="pipeline",
                     channels=[dict(reader_name="tile_reader", writer_name="tile_writer")],
                     consumer_workspaces=[], auxiliary_consumer=None,
                     workspaces=[dict(offset_bytes=0, nbytes=64,
                                      alignment_bytes=16, allocation_alignment_bytes=16)])],
        handoffs=[dict(id="reuse", pipe_name="reuse_pipe", semantic=True,
                       repeated=False, wrap_around=False,
                       writer_name="reuse_writer", reader_name="reuse_reader")],
        consumer_events=events,
        producer_events=[dict(kind="handoff", endpoint="reuse_reader", role="producer"),
                         dict(kind="pipeline_kernel_call", call="pipeline"),
                         {**drain, "endpoints": ["tile_writer"]}],
        auxiliary_events=[], consumer_endpoints=["tile_reader", "reuse_writer"],
        producer_endpoints=["tile_writer", "reuse_reader"], auxiliary_endpoints=[],
        trip_scoped_closes=[], trip_scoped_drains={},
        shared_arena_required_nbytes=64, shared_arena_nbytes=64,
        shared_arena_alignment_bytes=16,
    )
    programs = [dict(symbol="whole", calls=["ordinary", "pipeline"], num_warps=4)]
    descriptor = dict(entry_events=events, pipeline_schedule=schedule,
                      render_calls=[ordinary, pipeline], device_functions=[],
                      grid=[1, 1, 1], dynamic_argument_indices=[0],
                      distributed_entry=False, grid_barrier_axis_groups=[], runtime_binding={})
    module = SimpleNamespace(entry="main", metadata={"kernel_program_partition": programs})
    return module, dict(programs=programs), descriptor


def test_whole_entry_keeps_root_handoff_drain_endpoints_and_shared_allocation():
    module, plan, descriptor = complete_entry()
    before = deepcopy(descriptor)
    result = describe_kernel_programs(module, plan, descriptor)
    assert descriptor == before
    program, = result["programs"]
    # Kernel entry replaces only the incoming local fence. The internal
    # handoff, barrier, drain and the entire producer lifetime survive.
    expected = {**before["pipeline_schedule"], "consumer_events": before["entry_events"][1:]}
    assert program["pipeline_schedule"] == expected
    assert program["entry_events"] == expected["consumer_events"]
    assert result["kernels"][0]["calls"] == ["ordinary", "pipeline"]


def test_root_handoff_cannot_be_split_between_independent_programs():
    module, plan, descriptor = complete_entry()
    programs = [dict(symbol="first", calls=["ordinary"], num_warps=4),
                dict(symbol="second", calls=["pipeline"], num_warps=4)]
    module.metadata["kernel_program_partition"] = programs
    plan["programs"] = programs
    with pytest.raises(CodegenError, match="handoff|complete executable"):
        describe_kernel_programs(module, plan, descriptor)


@pytest.mark.parametrize("change", ["call", "handoff", "drain", "barrier"])
def test_complete_entry_preservation_rejects_a_changed_lifetime(change):
    _, _, descriptor = complete_entry()
    schedule = descriptor["pipeline_schedule"]
    segment = deepcopy(schedule["consumer_events"][1:])
    if change == "call":
        segment[0]["call"] = "different"
    else:
        segment.pop(next(i for i, e in enumerate(segment) if e["kind"] == change))
    with pytest.raises(CodegenError, match="complete entry"):
        project_pipeline_schedule(None, schedule, segment, complete_entry=True)


def test_complete_entry_preserves_loop_carried_and_auxiliary_lifetimes():
    _, _, descriptor = complete_entry()
    schedule = descriptor["pipeline_schedule"]
    schedule["handoffs"][0].update(repeated=True, wrap_around=True)
    schedule["trip_scoped_closes"] = [dict(endpoint="reuse_writer")]
    schedule["trip_scoped_drains"] = {"consumer": [dict(endpoints=["tile_reader"])]}
    schedule["auxiliary_endpoints"] = ["aux_reader"]
    schedule["auxiliary_events"] = [dict(kind="pipeline_kernel_call", call="pipeline")]
    schedule["stages"][0]["auxiliary_consumer"] = dict(warps=1)
    body = schedule["consumer_events"][1:]
    schedule["consumer_events"] = [dict(kind="loop", loop_var="iv", events=body)]
    before = deepcopy(schedule)
    projected = project_pipeline_schedule(None, schedule, schedule["consumer_events"], complete_entry=True)
    assert projected == before
    assert schedule == before


@pytest.mark.parametrize("reusable,depth", [(False, 0), (True, 0), (True, 1)])
@pytest.mark.parametrize("auxiliary", [False, True])
def test_normal_complete_program_preserves_all_physical_roles(
    compile_pipeline_module, reusable, depth, auxiliary,
):
    from triton.flagmega.codegen.triton.kernel_programs import _incoming_completion
    from triton.flagmega.codegen.triton.tir_package import describe_tir_package, render_tir_package

    implementation = "tir.dense_matmul.tensor_descriptor_smem_pipeline_" + (
        "aux_gemv" if auxiliary else "gemv")
    module = compile_pipeline_module(reusable=reusable, wrapper_depth=depth,
                                     implementation=implementation, whole_program=True)
    package = describe_tir_package(module)
    original = package["pipeline_schedule"]
    program, = package["programs"]
    expected = {**original, "consumer_events": _incoming_completion(original["consumer_events"])}
    assert program["pipeline_schedule"] == expected
    source = render_tir_package(package, "unit")
    ast.parse(source)
    assert source.count("tle.gpu.warp_specialize(") == 1
    assert source.count("_flagmega_shared_arena = tle.gpu.alloc(") == 1
    if auxiliary:
        assert program["pipeline_schedule"]["auxiliary_endpoints"]
        assert "auxiliary_consumer" in source
