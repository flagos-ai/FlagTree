# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega.codegen.triton.tir_package import describe_tir_package


@pytest.mark.parametrize("wrapper_depth", [1, 2])
@pytest.mark.parametrize("auxiliary", [False, True])
def test_nested_function_abi_rebinds_resources_without_cloning_bodies(
    compile_pipeline_module, wrapper_depth, auxiliary,
):
    implementation = "tir.dense_matmul.tensor_descriptor_smem_pipeline_" + (
        "aux_gemv" if auxiliary else "gemv")
    package = describe_tir_package(compile_pipeline_module(
        reusable=True, worker_depth=2, wrapper_depth=wrapper_depth,
        implementation=implementation,
    ))
    definitions = package["device_functions"]
    assert [item["function"] for item in definitions] == [
        "worker", *(f"wrapper_{index}" for index in range(wrapper_depth)),
    ]
    for index, definition in enumerate(definitions):
        assert len(definition["schedule"]["stages"]) == 2
        assert len(definition["host_tensor_descriptor_specs"]) == 2
        if index:
            for role in ("consumer", "producer", "auxiliary" if auxiliary else "producer"):
                events = definition["schedule"][f"{role}_events"]
                calls = [item for item in events if item["kind"] == "function_call"]
                assert len(calls) == 1
                assert calls[0]["callee"] == definitions[index - 1]["function"]
    assert len(package["pipeline_schedule"]["stages"]) == 4
    descriptors = package["host_tensor_descriptor_specs"]
    assert len({item["name"] for item in descriptors}) == 4
    assert len({item["offset_bytes"] for item in descriptors}) == 4
