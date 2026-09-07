# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega.ir import DType, IRBuilder, make_buffer_plan, tensor_type


def test_equivalent_weight_refs_share_one_readonly_physical_allocation():
    builder = IRBuilder(dialect="tir", stage="selected_tir")
    weight_type = tensor_type(DType.BFLOAT16, [16, 16])
    first = builder.weight(
        "weight", weight_type, source="model.safetensors", key="layer.weight", id="first"
    )
    second = builder.weight(
        "weight", weight_type, source="model.safetensors", key="layer.weight", id="second"
    )
    builder.function("main", (), (first, second))

    plan = make_buffer_plan(builder.build(entry="main"))
    assert plan.buffer_map["first"].physical_id == plan.buffer_map["second"].physical_id
    assert plan.buffer_map["first"].offset == plan.buffer_map["second"].offset
    assert len([value for value in plan.allocations if value.memory_space == "rdata"]) == 1
    assert plan.rdata_bytes == 512


def test_same_key_from_different_sources_does_not_alias():
    builder = IRBuilder(dialect="tir", stage="selected_tir")
    weight_type = tensor_type(DType.BFLOAT16, [16, 16])
    first = builder.weight(
        "weight", weight_type, source="model-00001.safetensors", key="weight", id="first"
    )
    second = builder.weight(
        "weight", weight_type, source="model-00002.safetensors", key="weight", id="second"
    )
    builder.function("main", (), (first, second))

    plan = make_buffer_plan(builder.build(entry="main"))
    assert plan.buffer_map["first"].physical_id != plan.buffer_map["second"].physical_id
    assert plan.rdata_bytes == 1024
