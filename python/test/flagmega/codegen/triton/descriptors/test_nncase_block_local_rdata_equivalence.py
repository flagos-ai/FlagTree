# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Packed-QKV readonly owner table matches nncase BlockLocalRdata."""

from triton.flagmega.codegen.triton.tir_package import describe_tir_package


def test_owner_major_rdata_uses_one_fixed_capacity_slice_per_mesh_owner(
    packed_qkv_mma_pipeline_module,
):
    package = describe_tir_package(packed_qkv_mma_pipeline_module)
    call = next(
        value
        for value in package["render_calls"]
        if value["family"] == "qkv_parallel_linear"
    )
    descriptor = call["host_tensor_descriptor_requests"][0]

    assert descriptor["source"] == "rdata"
    assert descriptor["shape"][0] == 128
    assert descriptor["block_shape"][0] == 1
    assert call["descriptor_offsets"][0] == (
        "tl.full((), shard_index, tl.int32)"
    )
    assert descriptor["strides"][0] == call["weight_owner_stride"]
    assert descriptor["strides"][0] * 2 == 131072


def test_descriptor_table_entries_use_the_same_owner_stride(
    packed_qkv_mma_descriptor_table_pipeline_module,
):
    package = describe_tir_package(
        packed_qkv_mma_descriptor_table_pipeline_module
    )
    call = next(
        value
        for value in package["render_calls"]
        if value["family"] == "qkv_parallel_linear"
    )
    descriptor = call["host_tensor_descriptor_requests"][0]
    entries = descriptor["entries"]

    assert descriptor["kind"] == "table"
    assert len(entries) == 128
    assert all(
        right["offset_bytes"] - left["offset_bytes"] == 131072
        for left, right in zip(entries, entries[1:])
    )
