# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Bind a formal pipeline resource interface to one physical function call.

Mirrors nncase's PipelineDeviceFunctionInterface: function bodies are shared,
but pipe identities, descriptors, and caller-owned MemSpans are not. No source
body substitution or model-specific schedule reconstruction is performed here.
"""

from copy import deepcopy
from collections.abc import Mapping

from triton.flagmega.errors import CodegenError
from .descriptor_resources import DescriptorResources


def descriptor_requests(specs):
    """A callee's local descriptor instances become requests at its caller."""
    if isinstance(specs, DescriptorResources):
        return specs.requests()
    return tuple({
        **{key: value for key, value in spec.items() if key != "name"},
        "parameter": spec["name"],
    } for spec in specs)


def instantiate_resources(schedule: Mapping, shared_by_name: Mapping, stem: str):
    """Clone only the resource interface, never the executable function body."""
    names = {}
    for stage in schedule["stages"]:
        names[stage["id"]] = f"{stem}__{stage['id']}"
        for workspace in stage["workspaces"]:
            names[workspace["variable"]] = f"{stem}__{workspace['variable']}"
        for channel in stage["channels"]:
            for field in ("pipe_name", "reader_name", "writer_name"):
                names[channel[field]] = f"{stem}__{channel[field]}"
    for handoff in schedule["handoffs"]:
        for field in ("id", "pipe_name", "reader_name", "writer_name"):
            names[handoff[field]] = f"{stem}__{handoff[field]}"

    # One deepcopy preserves shared workspace records referenced by channels
    # and consumer workspaces. Rebinding them independently risks stale spans.
    resources = deepcopy({
        "stages": schedule["stages"], "handoffs": schedule["handoffs"],
    })
    for stage in resources["stages"]:
        for workspace in stage["workspaces"]:
            formal = workspace["buffer_name"]
            try:
                actual = shared_by_name[formal]
            except KeyError as error:
                raise CodegenError(f"Pipeline call has no Shared binding for {formal!r}.") from error
            shape = tuple(dimension.fixed_value for dimension in actual.dimensions)
            if shape != tuple(workspace["shape"]):
                raise CodegenError(f"Pipeline Shared binding {formal!r} changes its formal shape.")
            if (
                actual.elem_type.value != workspace["element_type"]
                or tuple(value.fixed_value for value in actual.strides) != tuple(workspace["strides"])
                or actual.mem_span.buffer.memory_space != "shared"
            ):
                raise CodegenError(f"Pipeline Shared binding {formal!r} changes its formal payload ABI.")
            workspace.update({
                "buffer_name": actual.name,
                "offset_bytes": actual.mem_span.absolute_start.fixed_value,
                "nbytes": actual.mem_span.size.fixed_value,
                "physical_buffer": actual.mem_span.buffer.id,
                "allocation_alignment_bytes": actual.mem_span.buffer.alignment,
            })

    def rename(value):
        if isinstance(value, str):
            return names.get(value, value)
        if isinstance(value, dict):
            return {key: rename(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return type(value)(rename(item) for item in value)
        return value

    return rename(resources), names


__all__ = ["descriptor_requests", "instantiate_resources"]
