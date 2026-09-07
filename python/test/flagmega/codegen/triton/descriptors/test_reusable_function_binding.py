# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega.codegen.triton.tir_package import (
    describe_tir_package,
    render_tir_package,
)
from triton.flagmega.runtime import TensorDescriptorCache

def test_each_reusable_call_binds_its_own_root_storage_descriptor(
    two_call_descriptor_module,
):
    module = two_call_descriptor_module()
    package = describe_tir_package(module)
    specs = package["host_tensor_descriptor_specs"]

    assert len(specs) == 2
    assert len({value["name"] for value in specs}) == 2
    assert {value["source"] for value in specs} == {"workspace"}
    assert len({value["offset_bytes"] for value in specs}) == 2
    assert TensorDescriptorCache.validate_specs(
        specs, source_names=package["runtime_binding"]["signature"]
    ) == tuple(value["name"] for value in specs)

    projection_wrapper = next(
        value for value in package["render_calls"]
        if value["call"] == "projection"
    )
    assert projection_wrapper["descriptor_parameters"] == (
        "weight_descriptor",
    )
    function_calls = tuple(
        value for value in package["entry_events"]
        if value["kind"] == "function_call"
    )
    assert len(function_calls) == 2
    assert {
        value["arguments"].rsplit(", ", 1)[-1]
        for value in function_calls
    } == {value["name"] for value in specs}

    source = render_tir_package(package, "unit")
    definition, = package["device_functions"]
    assert definition["function"] == "worker"
    assert len(definition["host_tensor_descriptor_specs"]) == 1
    assert len(definition["schedule"]["consumer_events"]) == 1
    assert source.count("def _flagmega_function_worker__consumer(") == 1
    assert source.count("def _flagmega_worker_call_0_projection(") == 1
    assert "_flagmega_dense_matmul_tensor_descriptor_gemv_accumulate" in source
    assert "qwen" not in source.lower()
