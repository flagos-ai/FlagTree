# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Package/source ABI coverage for the packed BF16 QKV MMA pipeline."""

from triton.flagmega.codegen.triton.tir_package import (
    describe_tir_package,
    render_tir_package,
)


def _qkv_call(
    package,
    implementation="tir.qkv_parallel_linear.packed_partial_mma_smem_pipeline",
):
    return next(
        call for call in package["render_calls"]
        if call["implementation"] == implementation
    )


def test_packed_qkv_mma_package_owns_exact_typed_resources(
    packed_qkv_mma_pipeline_module,
):
    package = describe_tir_package(packed_qkv_mma_pipeline_module)
    call = _qkv_call(package)
    schedule = package["pipeline_schedule"]

    assert call["block_n"] == 256
    assert call["block_k"] == 64
    assert "mma_logical_row_xor" not in call
    assert all(value["alignment_bytes"] == 1024 for value in call["shared_workspaces"])
    assert all(value["offset_bytes"] % 1024 == 0 for value in call["shared_workspaces"])
    assert tuple(value["shape"] for value in call["shared_workspaces"]) == (
        (2, 1, 32, 4, 2, 64),
        (1, 256),
    )
    descriptor = call["host_tensor_descriptor_requests"][0]
    assert descriptor["shape"] == (128, 32, 16, 2, 64)
    assert descriptor["strides"] == (65536, 128, 4096, 64, 1)
    assert descriptor["block_shape"] == (1, 32, 4, 2, 64)
    assert schedule["shared_arena_required_nbytes"] == 66048
    assert schedule["shared_arena_nbytes"] == 66048


def test_packed_qkv_mma_source_indexes_each_projection_output_row(
    packed_qkv_mma_pipeline_module,
):
    package = describe_tir_package(packed_qkv_mma_pipeline_module)
    source = render_tir_package(package, "unit")

    compile(source, "packed_qkv_mma.py", "exec")
    assert "qkv_local_n_offsets" in source
    assert "tl.dot(" in source
    assert "MmaEncoding(" in source
    assert "qkv_logical_m = qkv_c_m\n" in source
    assert "qkv_c_m ^" not in source
    assert "qwen" not in source.lower()


def test_packed_qkv_owner_table_matches_the_physical_owner_prefix(
    packed_qkv_mma_descriptor_table_pipeline_module,
):
    implementation = (
        "tir.qkv_parallel_linear."
        "packed_partial_mma_descriptor_table_smem_pipeline"
    )
    package = describe_tir_package(
        packed_qkv_mma_descriptor_table_pipeline_module
    )
    call = _qkv_call(package, implementation)
    descriptor = call["host_tensor_descriptor_requests"][0]
    source = render_tir_package(package, "unit")

    assert call["descriptor_kind"] == "table"
    assert tuple(
        value["shape"] for value in call["shared_workspaces"]
    ) == ((2, 32, 4, 2, 64), (1, 256))
    assert descriptor["kind"] == "table"
    assert descriptor["block_shape"] == (32, 4, 2, 64)
    assert len(call["descriptor_offsets"]) == len(descriptor["block_shape"])
    assert len(descriptor["entries"]) == 128
    assert descriptor["entries"][0]["shape"] == (32, 16, 2, 64)
    assert (
        descriptor["entries"][1]["offset_bytes"]
        - descriptor["entries"][0]["offset_bytes"]
        == 131072
    )
    assert f"qkv_weight_descriptor_entry = {call['weight_descriptor']} + shard_index * 128" in source
    assert "qkv_weight_descriptor = tle.gpu.reinterpret_tensor_map(" in source
    compile(source, "packed_qkv_mma_owner_table.py", "exec")


def test_pipeline_source_uses_first_class_barriers_exactly_once(
    packed_qkv_mma_pipeline_module,
):
    package = describe_tir_package(packed_qkv_mma_pipeline_module)
    events = package["pipeline_schedule"]["consumer_events"]

    assert events[0]["kind"] == "tir.kernel_call"
    assert events[0]["family"] == "distributed_boxing"
    assert events[0]["barrier_before"] is False
    for previous, current in zip(events, events[1:]):
        assert not (
            previous["kind"] == "barrier"
            and current.get("barrier_before", False)
        )
