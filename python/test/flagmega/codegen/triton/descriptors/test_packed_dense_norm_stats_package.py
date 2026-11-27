# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Package coverage for packed projection/residual/RMS-statistics fusion."""

from copy import deepcopy

from triton.flagmega.codegen.triton import (
    describe_tir_package,
    render_tir_package,
)
from triton.flagmega.codegen.triton.kernel_call_renderers import (
    _dense_matmul_norm_stats_call,
)
from triton.flagmega.codegen.triton.runtime_binding import (
    describe_function_runtime_binding,
)


def test_packed_norm_stats_pipeline_has_explicit_results_and_no_private_partials(
    packed_norm_stats_descriptor_pipeline_module,
):
    package = describe_tir_package(packed_norm_stats_descriptor_pipeline_module)
    call = next(
        value
        for value in package["render_calls"]
        if value["semantic_op"] == "ntt.matmul_norm_stats"
    )

    assert call["family"] == "dense_matmul"
    assert call["variant"] == (
        "packed_tensor_descriptor_smem_pipeline_gemv_norm_stats"
    )
    assert tuple(output["formal"] for output in call["outputs"]) == (
        "result_0",
        "result_1",
    )
    assert tuple(call["workspaces"]) == ()
    assert call["owner_count"] == 128
    assert call["tiles_per_owner"] == 1
    assert call["local_n_capacity"] == 16
    assert call["internal_grid_barriers"] == 0
    assert len(call["host_tensor_descriptor_requests"]) == 1
    descriptor = call["host_tensor_descriptor_requests"][0]
    assert descriptor["shape"] == (128, 256, 2, 64)
    assert descriptor["block_shape"] == (64, 2, 2, 64)
    assert tuple(value["name"] for value in call["shared_workspaces"]) == (
        "weight_stage",
    )
    assert not any(
        value["family"] in {"add_norm_stats", "gather_reduce_add_norm_stats"}
        for value in package["render_calls"]
    )
    fused = next(
        value
        for value in package["render_calls"]
        if value["semantic_op"] == "ntt.gather_reduce_norm_apply"
    )
    assert fused["family"] == "gather_reduce_norm_apply"
    assert fused["variant"] == "sum"
    assert fused["partial_owner_count"] == 128
    assert fused["internal_grid_barriers"] == 0
    assert not any(
        value["family"] == "distributed_boxing"
        and value["variant"] == "gather_reduce_scatter"
        for value in package["render_calls"]
    )


def test_packed_norm_stats_template_publishes_one_owner_local_bf16_partial(
    packed_norm_stats_descriptor_pipeline_module,
):
    source = render_tir_package(
        describe_tir_package(packed_norm_stats_descriptor_pipeline_module),
        "unit",
    )
    package = describe_tir_package(packed_norm_stats_descriptor_pipeline_module)
    call = next(
        value
        for value in package["render_calls"]
        if value["semantic_op"] == "ntt.matmul_norm_stats"
    )
    consumer_start = source.index(f"def {call['symbol']}__consumer(")
    consumer_end = source.find("\n\n# flagmega-kernel:", consumer_start)
    consumer_source = source[
        consumer_start:None if consumer_end < 0 else consumer_end
    ]

    compile(source, "packed_dense_norm_stats_package.py", "exec")
    assert "dense_owner_n_iteration" not in consumer_source
    assert "tle.distributed_barrier(FLAGMEGA_GRID_MESH)" not in consumer_source
    assert ")[:, None]," in consumer_source
    assert "dense_stats_value * dense_stats_value" in consumer_source
    assert "dense_stats_square_sum" in consumer_source
    assert "norm_stats_partials" not in consumer_source
    assert "qwen" not in source.lower()
    assert "# flagmega-kernel: gather_reduce_norm_apply/sum platform=generic" in source
    assert "norm_partial_owner" in source


def test_packed_norm_stats_table_uses_owner_entry_and_local_tma_offsets(
    packed_norm_stats_descriptor_table_pipeline_module,
):
    package = describe_tir_package(
        packed_norm_stats_descriptor_table_pipeline_module
    )
    call = next(
        value
        for value in package["render_calls"]
        if value["semantic_op"] == "ntt.matmul_norm_stats"
    )
    source = render_tir_package(package, "unit")

    assert call["descriptor_kind"] == "table"
    assert call["host_tensor_descriptor_requests"][0]["kind"] == "table"
    assert call["descriptor_offsets"][:2] == (
        "tl.full((), ((dense_local_k_start) // 16), tl.int32)",
        "tl.full((), ((dense_local_n_start) // 8), tl.int32)",
    )
    assert f"dense_weight_descriptor_entry = {call['weight_descriptor']} + shard_index * 128" in source
    assert "tensor_map_fenceproxy_acquire(" in source
    assert "reinterpret_tensor_map(" in source
    compile(source, "packed_dense_norm_stats_table_package.py", "exec")


def test_descriptor_norm_stats_accepts_block_local_logical_coordinate_result(
    packed_norm_stats_descriptor_pipeline_module,
):
    binding = describe_function_runtime_binding(
        packed_norm_stats_descriptor_pipeline_module
    )
    raw = deepcopy(next(
        value
        for value in binding["call_abi"]["kernel_calls"]
        if value["semantic_op"] == "ntt.matmul_norm_stats"
    ))
    value_abi = raw["outputs"][0]["buffers"][0]["abi"]
    assert value_abi["coordinate_space"] == "canonical_global"
    value_abi["storage_kind"] = "replicated_local"

    encoded = _dense_matmul_norm_stats_call(raw)

    assert encoded["result_writer_active"] == "True"


def test_descriptor_norm_stats_accepts_a_parent_shard_local_result(
    packed_norm_stats_descriptor_pipeline_module,
):
    binding = describe_function_runtime_binding(
        packed_norm_stats_descriptor_pipeline_module
    )
    raw = deepcopy(next(
        value
        for value in binding["call_abi"]["kernel_calls"]
        if value["semantic_op"] == "ntt.matmul_norm_stats"
    ))
    value_abi = raw["outputs"][0]["buffers"][0]["abi"]
    value_abi["storage_kind"] = "compact_local"
    value_abi["coordinate_space"] = "parent_shard_local"
    value_abi["storage_coordinate_expressions"] = tuple(
        value_abi["logical_coordinate_expressions"]
    )

    encoded = _dense_matmul_norm_stats_call(raw)

    assert "shard_" in encoded["result_offset"]
    assert encoded["result_writer_active"] == "True"


def test_portable_packed_norm_stats_template_defines_the_helper_it_calls(
    packed_norm_stats_portable_pipeline_module,
):
    source = render_tir_package(
        describe_tir_package(packed_norm_stats_portable_pipeline_module),
        "unit",
    )

    compile(source, "portable_packed_dense_norm_stats_package.py", "exec")
    helper = "_flagmega_dense_matmul_packed_k_major_gemv_norm_stats_accumulate"
    assert f"def {helper}(" in source
    assert f"dense_accumulator += {helper}(" in source
