# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

from copy import deepcopy
from dataclasses import replace

import pytest

from triton.flagmega.codegen.triton.pipeline_function_abi import instantiate_resources
from triton.flagmega.codegen.triton.tir_package import describe_tir_package
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import KernelInvoke
from triton.flagmega.ir.tir import iter_tir_children


@pytest.fixture
def formal_interface(compile_pipeline_module):
    module = compile_pipeline_module(reusable=True)
    definition, = describe_tir_package(module)["device_functions"]
    pending = [module.execution_function_map["worker"].body]
    while pending:
        value = pending.pop()
        if isinstance(value, KernelInvoke) and value.shared_workspace_buffers:
            buffer, = value.shared_workspace_buffers
            return definition["schedule"], buffer
        pending.extend(iter_tir_children(value))
    raise AssertionError("worker has no Shared workspace")


def test_resource_instantiation_preserves_formal_ir_and_rebases_every_view(formal_interface):
    schedule, formal = formal_interface
    original = deepcopy(schedule)
    actual = replace(formal, name="caller_scratch", mem_span=replace(
        formal.mem_span, buffer=replace(formal.mem_span.buffer, id="caller_shared", start=4096),
    ))
    first, first_names = instantiate_resources(schedule, {formal.name: actual}, "first")
    second, second_names = instantiate_resources(schedule, {formal.name: actual}, "second")
    assert schedule == original
    assert first_names.keys() == second_names.keys()
    assert set(first_names.values()).isdisjoint(second_names.values())
    assert set(first) == set(second) == {"stages", "handoffs"}
    for resources in (first, second):
        stage, = resources["stages"]
        workspace, = stage["workspaces"]
        assert workspace["buffer_name"] == actual.name
        assert workspace["offset_bytes"] == actual.mem_span.absolute_start.fixed_value
        assert workspace["physical_buffer"] == "caller_shared"
        assert stage["channels"][0]["fields"][0]["workspace"] == workspace


def test_missing_shared_binding_is_diagnosed(formal_interface):
    schedule, _ = formal_interface
    with pytest.raises(CodegenError, match="no Shared binding"):
        instantiate_resources(schedule, {}, "broken")


def test_binding_preserves_physical_alignment_separately_from_view_alignment(formal_interface):
    schedule, formal = formal_interface
    actual = replace(formal, mem_span=replace(formal.mem_span,
        buffer=replace(formal.mem_span.buffer, alignment=4096, start=4096)))
    resources, _ = instantiate_resources(schedule, {formal.name: actual}, "aligned")
    workspace, = resources["stages"][0]["workspaces"]
    assert workspace["alignment_bytes"] == 1024
    assert workspace["allocation_alignment_bytes"] == 4096


@pytest.mark.parametrize("change", ["shape", "element_type", "strides", "memory_space"])
def test_rebinding_cannot_change_shared_payload_abi(formal_interface, change):
    schedule, formal = formal_interface
    if change == "shape":
        actual = replace(formal, dimensions=(*formal.dimensions[:-1], 64))
    elif change == "element_type":
        actual = replace(formal, elem_type="bool")
    elif change == "strides":
        actual = replace(formal, strides=(*formal.strides[:-1], 0))
    else:
        actual = replace(formal, mem_span=replace(
            formal.mem_span, buffer=replace(formal.mem_span.buffer, memory_space="workspace"),
        ))
    with pytest.raises(CodegenError, match="formal.*ABI|formal shape"):
        instantiate_resources(schedule, {formal.name: actual}, "broken")
