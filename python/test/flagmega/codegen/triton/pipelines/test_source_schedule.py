# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega.codegen.triton.tir_package import (
    describe_tir_package,
    render_tir_package,
)


def test_direct_region_generates_both_roles_and_typed_shared_alias(
    compile_pipeline_module,
):
    package = describe_tir_package(compile_pipeline_module(reusable=False))
    schedule = package["pipeline_schedule"]
    source = render_tir_package(package, "unit")

    compile(source, "pipeline.py", "exec")
    assert schedule["schema"] == "flagmega.triton-pipeline-source-schedule/v3"
    assert schedule["shared_arena_nbytes"] == 8192
    assert schedule["stages"][0]["workspaces"][0]["shape"] == (2, 16, 128)
    assert schedule["producer_events"][-1]["role"] == "producer"
    assert schedule["consumer_events"][-1]["role"] == "consumer"
    assert "tle.gpu.copy(" in source
    assert "tle.gpu.local_ptr(" in source
    assert "tle.gpu.warp_specialize(" in source
    assert "alias=_flagmega_shared_arena" in source
    assert "qwen" not in source.lower()


def test_reusable_calls_share_one_wrapper_but_keep_per_call_pipes_and_descriptors(
    compile_pipeline_module,
):
    package = describe_tir_package(compile_pipeline_module(reusable=True))
    schedule = package["pipeline_schedule"]
    source = render_tir_package(package, "unit")

    assert len(schedule["stages"]) == 2
    assert len({
        channel["pipe_name"]
        for stage in schedule["stages"]
        for channel in stage["channels"]
    }) == 2
    assert {value["workspaces"][0]["offset_bytes"] for value in schedule["stages"]} == {0}
    assert any(value["kind"] == "drain" for value in schedule["consumer_events"])
    assert any(value["kind"] == "drain" for value in schedule["producer_events"])
    assert len(package["host_tensor_descriptor_specs"]) == 2
    assert source.count("def _flagmega_worker_call_0_projection__producer(") == 1
    assert source.count("def _flagmega_worker_call_0_projection__consumer(") == 1
    assert source.count("tle.pipe(\n        capacity=2") == 2
    assert source.count(".pipe.wait_drained()") == 2


def test_nested_pipeline_exposes_only_endpoints_not_drained_inside_callee(
    compile_pipeline_module,
):
    package = describe_tir_package(compile_pipeline_module(
        reusable=True,
        worker_depth=2,
    ))
    schedule = package["pipeline_schedule"]
    source = render_tir_package(package, "unit")

    definitions = {value["function"]: value for value in schedule["device_functions"]}

    def drains(events, role, bindings):
        for event in events:
            if event["kind"] == "drain":
                yield from (bindings.get(value, value) for value in event["endpoints"])
            elif event["kind"] == "function_call":
                child = definitions[event["callee"]]["schedule"]
                actuals = {formal: bindings.get(actual, actual)
                           for formal, actual in event["endpoint_bindings"].items()}
                yield from drains(child[role], role, actuals)

    drain_endpoints = [endpoint for role in ("consumer_events", "producer_events")
                       for endpoint in drains(schedule[role], role, {})]
    # Each pipe endpoint has one lifetime and therefore exactly one drain.
    # The first inner stage is drained before the second stage reuses its
    # Shared range; an outer drain must expose only the callee's still-live
    # second endpoint rather than draining the first endpoint again.
    assert len(drain_endpoints) == 6
    assert len(drain_endpoints) == len(set(drain_endpoints))
    # Two drains in the shared worker body plus two residual drains in main;
    # executing the worker twice still drains the six distinct actual endpoints.
    assert source.count(".pipe.wait_drained()") == 4
