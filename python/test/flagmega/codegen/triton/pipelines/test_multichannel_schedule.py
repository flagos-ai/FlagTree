# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from copy import deepcopy
from dataclasses import replace

from triton.flagmega.codegen.triton.kernel_call_renderers import (
    _bind_transfer_pipeline_parameters,
)
from triton.flagmega.codegen.triton.pipeline_source import (
    _Context,
    _PipelineSourceBuilder,
)
from triton.flagmega.codegen.triton.tir_package import (
    describe_tir_package,
    render_tir_package,
)
from triton.flagmega.ir import KernelInvoke
from triton.flagmega.ir.tir import iter_tir_children


def _pipeline_call(module):
    pending = [module.execution_function_map[module.entry].body]
    while pending:
        value = pending.pop()
        if isinstance(value, KernelInvoke) and value.shared_workspace_buffers:
            return value
        pending.extend(iter_tir_children(value))
    raise AssertionError("compiled fixture has no transfer-pipeline call")


def _workspace_views(source, names):
    size = source.mem_span.size.fixed_value
    result = []
    for index, name in enumerate(names):
        physical = replace(
            source.mem_span.buffer,
            id=f"{source.mem_span.buffer.id}.{name}",
            start=index * size,
        )
        result.append(replace(
            source,
            name=name,
            mem_span=replace(source.mem_span, buffer=physical),
        ))
    return tuple(result)


def _replace_pipeline_event_arguments(events, role, endpoints):
    for event in events:
        if event.get("kind") != "pipeline_kernel_call":
            continue
        original = str(event["arguments"])
        _, separator, remainder = original.partition(", ")
        assert separator
        event["arguments"] = ", ".join((*endpoints, remainder))
        event["role"] = role


def test_multiple_channels_bundled_fields_and_consumer_workspace_render(
    compile_pipeline_module,
):
    module = compile_pipeline_module(reusable=False)
    package = describe_tir_package(module)
    call = _pipeline_call(module)
    source_workspace = call.shared_workspace_buffers[0]
    workspaces = _workspace_views(
        source_workspace,
        ("key_stage", "projection_left", "projection_right", "scratch"),
    )

    encoded = deepcopy(next(
        value for value in package["render_calls"]
        if value["call"] == call.call_id
    ))
    workspace_abi = []
    for original, actual in zip(
        (encoded["shared_workspaces"][0],) * len(workspaces),
        workspaces,
        strict=True,
    ):
        value = dict(original)
        value.update({
            "name": actual.name,
            "offset_bytes": actual.mem_span.absolute_start.fixed_value,
            "physical_buffer": actual.mem_span.buffer.id,
        })
        workspace_abi.append(value)
    encoded["shared_workspaces"] = workspace_abi
    encoded["transfer_pipeline"] = {
        "channels": [
            {
                "name": "key",
                "source_argument_indices": [1],
                "shared_workspace_indices": [0],
                "source_alignment_bytes": 16,
            },
            {
                "name": "projection",
                "source_argument_indices": [0, 1],
                "shared_workspace_indices": [1, 2],
                "source_alignment_bytes": 16,
            },
        ],
        "consumer_shared_workspace_indices": [3],
        "auxiliary_consumer": None,
    }
    encoded["pipeline_channel_field_names"] = {
        "key": ("weight",),
        "projection": ("left", "right"),
    }
    _bind_transfer_pipeline_parameters(encoded, encoded)

    concrete_call = replace(call, shared_workspace_buffers=workspaces)
    builder = _PipelineSourceBuilder(
        module,
        {},
        {module.entry: [encoded]},
        lambda *_: {},
        lambda *_: {},
        lambda *_: (),
    )
    plan = builder._leaf_stage(
        _Context(module.entry, {}, {}, {}, ()),
        concrete_call,
        {"call": call.call_id},
    )

    assert [value["name"] for value in plan["channels"]] == [
        "key", "projection",
    ]
    assert [
        field["name"]
        for field in plan["channels"][1]["fields"]
    ] == ["left", "right"]
    assert [value["name"] for value in plan["consumer_workspaces"]] == [
        "scratch",
    ]
    assert [value["offset_bytes"] for value in plan["workspaces"]] == [
        0, 8192, 16384, 24576,
    ]

    rendered = deepcopy(package)
    schedule = rendered["pipeline_schedule"]
    schedule["stages"] = [plan]
    consumer_endpoints = [
        *(channel["reader_name"] for channel in plan["channels"]),
        *(value["variable"] for value in plan["consumer_workspaces"]),
    ]
    producer_endpoints = [
        channel["writer_name"] for channel in plan["channels"]
    ]
    schedule["consumer_endpoints"] = [
        *consumer_endpoints,
        *(value["writer_name"] for value in schedule["handoffs"]),
    ]
    schedule["producer_endpoints"] = [
        *producer_endpoints,
        *(value["reader_name"] for value in schedule["handoffs"]),
    ]
    schedule["shared_arena_nbytes"] = 32768
    _replace_pipeline_event_arguments(
        schedule["consumer_events"], "consumer", consumer_endpoints
    )
    _replace_pipeline_event_arguments(
        schedule["producer_events"], "producer", producer_endpoints
    )
    wrapper = next(
        value for value in rendered["render_calls"]
        if value["call"] == call.call_id
    )
    for key in (
        "pipeline_channels",
        "pipeline_consumer_workspaces",
        "pipeline_consumer_parameters",
        "pipeline_producer_parameters",
    ):
        wrapper[key] = encoded[key]

    source = render_tir_package(rendered, "unit")
    compile(source, "pipeline.py", "exec")
    assert source.count("capacity=2,\n        scope=\"cta\"") == 2
    assert (
        f"weight={plan['channels'][0]['fields'][0]['workspace']['variable']}"
        in source
    )
    assert (
        f"left={plan['channels'][1]['fields'][0]['workspace']['variable']}"
        in source
    )
    assert (
        f"right={plan['channels'][1]['fields'][1]['workspace']['variable']}"
        in source
    )
    assert plan["consumer_workspaces"][0]["variable"] in source
