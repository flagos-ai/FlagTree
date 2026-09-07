# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Package-level coverage for the generic packed dense TMA pipeline."""

from triton.flagmega.codegen.triton import (
    describe_tir_package,
    render_tir_package,
)


def test_packed_descriptor_pipeline_survives_bufferization_and_package_render(
    packed_descriptor_pipeline_module,
):
    module = packed_descriptor_pipeline_module
    package = describe_tir_package(module)
    call = next(
        value
        for value in package["render_calls"]
        if value["call"] == "projection.vectorized.compute"
    )

    assert call["semantic_op"] == "ntt.packed_matmul"
    assert call["family"] == "dense_matmul"
    assert call["variant"] == (
        "packed_tensor_descriptor_smem_pipeline_gemv"
    )
    assert call["host_tensor_descriptor_requests"] == ({
        "parameter": "weight_descriptor",
        "rebase_axis": -1,
        "source": "rdata",
        "kind": "single",
        "storage": "device",
        "box_shape": (64, 2, 2, 64),
        "swizzle_mode": 3,
        "offset_bytes": 0,
        "dtype": "bfloat16",
        "shape": (128, 256, 2, 64),
        "strides": (32768, 128, 64, 1),
        "block_shape": (64, 2, 2, 64),
        "source_shape_axes": ((), (), (), ()),
        "padding": "zero",
    },)
    assert call["shared_workspaces"] == ({
        "name": "weight_stage",
        "dtype": "bfloat16",
        "shape": (4, 64, 2, 2, 64),
        "strides": (16384, 256, 128, 64, 1),
        "offset_bytes": 0,
        "nbytes": 131072,
        "alignment_bytes": 1024,
        "matrix_compatible": True,
        "physical_buffer": call["shared_workspaces"][0]["physical_buffer"],
    },)
    assert call["transfer_pipeline"] == {
        "capacity": 4,
        "channels": [{
            "name": "weight",
            "source_argument_indices": [1],
            "shared_workspace_indices": [0],
            "source_alignment_bytes": 16,
        }],
        "consumer_shared_workspace_indices": [],
        "auxiliary_consumer": None,
    }

    source = render_tir_package(package, "unit")
    compile(source, "packed_dense_matmul_package.py", "exec")
    assert "tle.gpu.copy(" in source
    assert "weight_descriptor" in source
    assert "dense_local_k_start" in source
    assert "dense_local_n_start" in source
    assert "// 16" in source
    assert "// 8" in source
    assert "dense_stage_k // 16" in source
    assert "dense_local_n // 8" in source
    assert "tle.gpu.BlockEncoding(" in source
    assert "qwen" not in source.lower()


def test_packed_dense_consumer_stage_has_only_value_dependencies(
    packed_descriptor_pipeline_module,
):
    package = describe_tir_package(packed_descriptor_pipeline_module)
    call = next(
        value
        for value in package["render_calls"]
        if value["call"] == "projection.vectorized.compute"
    )
    source = render_tir_package(package, "unit")

    symbol = call["symbol"]
    start = source.index(f"def {symbol}__consumer_stage(")
    end = source.index("\n\n@triton.jit", start)
    stage = source[start:end]
    header_end = stage.index("):") + 2
    assert stage[:header_end] == (
        f"def {symbol}__consumer_stage(\n"
        "    weight_stage,\n"
        "    dense_local_k_start,\n"
        "    dense_source,\n"
        "    dense_source_owner_active,\n"
        "    dense_source_active_extent,\n"
        "):"
    )
    assert "shard_coord" not in stage
    assert "shard_index" not in stage
    assert "_descriptor" not in stage[:header_end]


def test_packed_descriptor_table_has_one_local_tensor_map_per_mesh_owner(
    packed_descriptor_table_pipeline_module,
):
    package = describe_tir_package(packed_descriptor_table_pipeline_module)
    call = next(
        value
        for value in package["render_calls"]
        if value["call"] == "projection.vectorized.compute"
    )
    request = call["host_tensor_descriptor_requests"][0]
    spec = next(
        value
        for value in package["host_tensor_descriptor_specs"]
        if value["name"].endswith("weight_descriptor_descriptor")
    )

    assert request["kind"] == "table"
    assert spec["kind"] == "table"
    assert len(spec["entries"]) == 128
    assert spec["block_shape"] == [64, 2, 2, 64]
    assert spec["entries"][0] == {
        "offset_bytes": 0,
        "shape": [128, 2, 2, 64],
        "strides": [32768, 128, 64, 1],
        "source_shape_axes": [[], [], [], []],
    }
    assert spec["entries"][1]["offset_bytes"] == 512
    assert spec["entries"][-1]["offset_bytes"] == 127 * 512

    source = render_tir_package(package, "unit")
    compile(source, "packed_dense_matmul_table_package.py", "exec")
    assert f"dense_weight_descriptor_entry = {call['weight_descriptor']} + shard_index * 128" in source
    assert "tle.gpu.tensor_map_fenceproxy_acquire(" in source
    assert "tle.gpu.reinterpret_tensor_map(" in source
    producer_start = source.index(f"def {call['symbol']}__producer(")
    producer_end = source.index("\n\n@triton.jit", producer_start)
    producer = source[producer_start:producer_end]
    assert "shard_index" in producer
    assert all("shard_coord" not in value for value in call["descriptor_offsets"])
    assert "dense_local_k_start" in call["descriptor_offsets"][0]
    assert "dense_local_n_start" in call["descriptor_offsets"][1]
