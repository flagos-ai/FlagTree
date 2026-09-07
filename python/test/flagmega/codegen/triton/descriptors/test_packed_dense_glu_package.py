# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Package-level coverage for the generic packed SwiGLU TMA pipeline."""

from copy import deepcopy

from triton.flagmega.codegen.triton import describe_tir_package, render_tir_package


def test_packed_glu_pipeline_materializes_dual_descriptors_and_one_channel(
    packed_glu_descriptor_pipeline_module,
):
    package = describe_tir_package(packed_glu_descriptor_pipeline_module)
    call = next(
        value
        for value in package["render_calls"]
        if value["family"] == "dense_matmul_glu"
    )

    assert call["semantic_op"] == "nn.packed_dense_matmul_glu"
    assert call["family"] == "dense_matmul_glu"
    assert call["variant"] == (
        "packed_tensor_descriptor_smem_pipeline_gemv"
    )
    assert tuple(
        request["parameter"]
        for request in call["host_tensor_descriptor_requests"]
    ) == ("gate_weight_descriptor", "up_weight_descriptor")
    assert all(
        request["block_shape"] == (64, 2, 2, 64)
        for request in call["host_tensor_descriptor_requests"]
    )
    assert len(call["shared_workspaces"]) == 2
    assert call["shared_workspaces"][0]["shape"] == (4, 64, 2, 2, 64)
    assert call["shared_workspaces"][1]["name"] == "lhs_stage"
    assert call["shared_workspaces"][1]["shape"] == (1, 2048)
    assert call["pipeline_consumer_workspaces"] == ({
        "name": "lhs_stage",
        "parameter": "pipeline_consumer_lhs_stage",
    },)
    assert call["transfer_pipeline"]["channels"] == [{
        "name": "weight",
        "source_argument_indices": [1, 2],
        "shared_workspace_indices": [0],
        "source_alignment_bytes": 16,
    }]


def test_packed_glu_pipeline_source_has_interleaved_tiles_and_dynamic_offsets(
    packed_glu_descriptor_pipeline_module,
):
    package = describe_tir_package(packed_glu_descriptor_pipeline_module)
    source = render_tir_package(package, "unit")

    compile(source, "packed_dense_glu_package.py", "exec")
    assert "dense_glu_gate_sequence = dense_glu_tile_sequence * 2" in source
    assert "dense_glu_up_sequence = dense_glu_gate_sequence + 1" in source
    assert "dense_glu_local_k_start" in source
    assert "dense_glu_local_n_start" in source
    assert "// 16" in source
    assert "// 8" in source
    assert "dense_glu_gate_ready.slot.weight" in source
    assert "dense_glu_up_ready.slot.weight" in source
    assert source.count("dense_glu_input_copy_k = tle.gpu.set_layout(") == 1
    assert source.count("tle.gpu.async_commit_group()") >= 1
    assert "pipeline_consumer_lhs_stage" in source
    assert "dense_glu_source_pointer = tle.gpu.local_ptr(" in source
    assert "tl.load(dense_glu_source_pointer)" in source
    copy_commit = source.index("dense_glu_input_copy_k = tle.gpu.set_layout(")
    first_weight_wait = source.index(
        "dense_glu_gate_ready = pipeline_weight_reader.wait("
    )
    copy_wait = source.index("tle.gpu.async_wait_group(0)", copy_commit)
    assert copy_commit < first_weight_wait < copy_wait
    assert "qwen" not in source.lower()


def test_complete_lhs_consumer_stage_has_only_value_dependencies(
    packed_glu_descriptor_pipeline_module,
):
    package = describe_tir_package(packed_glu_descriptor_pipeline_module)
    call = next(
        value
        for value in package["render_calls"]
        if value["family"] == "dense_matmul_glu"
    )
    source = render_tir_package(package, "unit")

    symbol = call["symbol"]
    start = source.index(f"def {symbol}__consumer_stage(")
    end = source.index("\n\n@triton.jit", start)
    consumer_stage = source[start:end]
    header_end = consumer_stage.index("):") + 2
    header = consumer_stage[:header_end]

    assert header == (
        f"def {symbol}__consumer_stage(\n"
        "    gate_weight_stage,\n"
        "    up_weight_stage,\n"
        "    pipeline_consumer_lhs_stage,\n"
        "    dense_glu_local_k_start,\n"
        "):"
    )
    assert "shard_coord" not in consumer_stage
    assert "shard_index" not in consumer_stage


def test_direct_lhs_consumer_stage_has_only_value_dependencies(
    packed_glu_direct_lhs_descriptor_pipeline_module,
):
    package = describe_tir_package(
        packed_glu_direct_lhs_descriptor_pipeline_module
    )
    call = next(
        value
        for value in package["render_calls"]
        if value["family"] == "dense_matmul_glu"
    )
    source = render_tir_package(package, "unit")

    symbol = call["symbol"]
    start = source.index(f"def {symbol}__consumer_stage(")
    end = source.index("\n\n@triton.jit", start)
    consumer_stage = source[start:end]
    header_end = consumer_stage.index("):") + 2
    header = consumer_stage[:header_end]

    assert header == (
        f"def {symbol}__consumer_stage(\n"
        "    gate_weight_stage,\n"
        "    up_weight_stage,\n"
        "    dense_glu_local_k_start,\n"
        "    dense_glu_source,\n"
        "    dense_glu_source_owner_active,\n"
        "    dense_glu_source_active_extent,\n"
        "):"
    )
    assert "shard_coord" not in consumer_stage
    assert "shard_index" not in consumer_stage
    assert "_descriptor" not in header


def test_inline_consumer_candidate_removes_noinline_boundary(
    packed_glu_inline_descriptor_pipeline_module,
):
    package = describe_tir_package(packed_glu_inline_descriptor_pipeline_module)
    call = next(
        value
        for value in package["render_calls"]
        if value["family"] == "dense_matmul_glu"
    )
    source = render_tir_package(package, "unit")

    symbol = call["symbol"]
    start = source.index(f"def {symbol}__consumer_stage(")
    annotation = source[source.rfind("@triton.jit", 0, start):start]
    assert annotation.strip() == "@triton.jit"


def test_paired_weight_pipeline_uses_one_sequence_for_gate_and_up(
    packed_glu_paired_inline_descriptor_pipeline_module,
):
    package = describe_tir_package(
        packed_glu_paired_inline_descriptor_pipeline_module
    )
    call = next(
        value
        for value in package["render_calls"]
        if value["family"] == "dense_matmul_glu"
    )
    source = render_tir_package(package, "unit")

    assert call["paired_weight_fields"] is True
    assert tuple(
        value["name"] for value in call["shared_workspaces"]
    ) == ("gate_stage", "up_stage")
    assert call["pipeline_channels"][0]["fields"] == (
        {"name": "gate", "workspace_name": "gate_stage"},
        {"name": "up", "workspace_name": "up_stage"},
    )
    assert "dense_glu_ready.slot.gate" in source
    assert "dense_glu_ready.slot.up" in source
    assert "dense_glu_gate_sequence" not in source
    assert "dense_glu_up_sequence" not in source
    assert source.count("pipeline_weight_writer.acquire(") == 1
    assert source.count("pipeline_weight_writer.commit(") == 1
    assert source.count("pipeline_weight_reader.wait(") == 1
    assert source.count("pipeline_weight_reader.release(") == 1
    compile(source, "packed_dense_glu_paired_package.py", "exec")


def test_paired_owner_descriptor_tables_rebase_both_weights_to_local_offsets(
    packed_glu_paired_table_inline_descriptor_pipeline_module,
):
    package = describe_tir_package(
        packed_glu_paired_table_inline_descriptor_pipeline_module
    )
    call = next(
        value
        for value in package["render_calls"]
        if value["family"] == "dense_matmul_glu"
    )
    requests = call["host_tensor_descriptor_requests"]
    source = render_tir_package(package, "unit")

    assert call["descriptor_kind"] == "table"
    assert tuple(value["kind"] for value in requests) == ("table", "table")
    assert all(len(value["entries"]) == 128 for value in requests)
    expected_local_shape = (
        call["local_k_capacity"] // 16,
        call["local_n_capacity"] // 8,
        2,
        64,
    )
    assert all(
        value["entries"][0]["shape"] == expected_local_shape
        for value in requests
    )
    assert f"dense_glu_gate_descriptor_entry = {call['gate_weight_descriptor']} + shard_index * 128" in source
    assert f"dense_glu_up_descriptor_entry = {call['up_weight_descriptor']} + shard_index * 128" in source
    assert source.count("tle.gpu.reinterpret_tensor_map(") >= 2
    assert "((dense_glu_local_n_start) // 8)" in source
    assert "((dense_glu_local_k_start) // 16)" in source
    compile(source, "packed_dense_glu_paired_table_package.py", "exec")


def test_paired_full_lhs_pipeline_has_shared_source_and_inline_stage(
    packed_glu_paired_full_lhs_descriptor_pipeline_module,
):
    package = describe_tir_package(
        packed_glu_paired_full_lhs_descriptor_pipeline_module
    )
    call = next(
        value
        for value in package["render_calls"]
        if value["family"] == "dense_matmul_glu"
    )
    source = render_tir_package(package, "unit")

    assert call["paired_weight_fields"] is True
    assert call["lhs_stage_extent"] == 2048
    assert call["pipeline_consumer_workspaces"] == ({
        "name": "lhs_stage",
        "parameter": "pipeline_consumer_lhs_stage",
    },)
    symbol = call["symbol"]
    start = source.index(f"def {symbol}__consumer_stage(")
    annotation = source[source.rfind("@triton.jit", 0, start):start]
    assert annotation.strip() == "@triton.jit"
    stage_end = source.index("\n\n@triton.jit", start)
    stage = source[start:stage_end]
    assert "tl.load(dense_glu_source_pointer)" in stage
    assert "dense_glu_source_owner_active" not in stage
    assert "dense_glu_ready.slot.gate" in source
    assert "dense_glu_ready.slot.up" in source
    compile(source, "packed_dense_glu_paired_full_lhs.py", "exec")


def test_full_lhs_inline_candidate_combines_shared_source_and_inline_stage(
    packed_glu_full_lhs_inline_descriptor_pipeline_module,
):
    package = describe_tir_package(
        packed_glu_full_lhs_inline_descriptor_pipeline_module
    )
    call = next(
        value
        for value in package["render_calls"]
        if value["family"] == "dense_matmul_glu"
    )
    source = render_tir_package(package, "unit")

    assert "inline_consumer_stage" not in call
    assert call["pipeline_consumer_workspaces"] == ({
        "name": "lhs_stage",
        "parameter": "pipeline_consumer_lhs_stage",
    },)
    symbol = call["symbol"]
    start = source.index(f"def {symbol}__consumer_stage(")
    annotation = source[source.rfind("@triton.jit", 0, start):start]
    assert annotation.strip() == "@triton.jit"
    stage_end = source.index("\n\n@triton.jit", start)
    stage = source[start:stage_end]
    assert "tl.load(dense_glu_source_pointer)" in stage
    assert "dense_glu_source_owner_active" not in stage


def test_pipeline_renderer_emits_a_valid_empty_role(
    packed_glu_direct_lhs_descriptor_pipeline_module,
):
    package = deepcopy(describe_tir_package(
        packed_glu_direct_lhs_descriptor_pipeline_module
    ))
    package["pipeline_schedule"]["producer_events"] = []

    source = render_tir_package(package, "unit")

    compile(source, "pipeline_empty_producer.py", "exec")
    producer = source.index("def flagmega_main__producer(")
    consumer = source.index("def flagmega_main(", producer)
    assert "    pass" in source[producer:consumer]
