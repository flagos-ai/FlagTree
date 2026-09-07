# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega.codegen.triton import describe_tir_package, render_tir_package


@pytest.mark.parametrize("auxiliary", [False, True])
def test_pipeline_descriptor_is_a_device_handle_without_coordinate_rebasing(
    compile_pipeline_module, auxiliary,
):
    implementation = (
        "tir.dense_matmul.tensor_descriptor_smem_pipeline_aux_gemv" if auxiliary
        else "tir.dense_matmul.tensor_descriptor_smem_pipeline_gemv"
    )
    package = describe_tir_package(compile_pipeline_module(
        reusable=True, implementation=implementation,
    ))
    specs = package["host_tensor_descriptor_specs"]
    assert len(specs) == 2
    for spec in specs:
        assert spec.get("storage") == "device"
        assert spec["kind"] == "single"
        assert spec["box_shape"][-1] == 64
        assert spec["swizzle_mode"] == 3
    assert specs[0]["offset_bytes"] != specs[1]["offset_bytes"]
    source = render_tir_package(package, "unit")
    assert "tle.gpu.reinterpret_tensor_map(" in source
    assert "tle.gpu.tensor_map_fenceproxy_acquire(" in source
    assert " + shard_index * 128" not in source
