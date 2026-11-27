# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler


def _packed_matmul_module() -> fm.IRModule:
    builder = fm.IRBuilder(dialect="high_level", stage="packed")
    value_type = fm.tensor_type("bfloat16", [1, 1024])
    weight_type = fm.tensor_type(fm.vector_type("float8_e4m3fn", (2, 16)), [1024, 32])
    scale_type = fm.tensor_type("float32", [8, 8])
    value = builder.var("value", value_type, id="value")
    weight = builder.weight(
        "weight", weight_type, source="unit.safetensors", key="weight", id="weight"
    )
    scale = builder.weight(
        "scale", scale_type, source="unit.safetensors", key="scale", id="scale"
    )
    output = builder.call(
        "math.packed_block_scaled_matmul",
        (value, weight, scale),
        value_type,
        id="output",
        attrs={
            "weight_block_n": 128,
            "weight_block_k": 128,
            "k_pack": 2,
            "k_vector": 16,
            "packed_layout": "n_major_k_packed",
        },
    )
    builder.function("main", (value,), (output,))
    return fm.verify_module(builder.build(entry="main"))


def test_lower_tir_preserves_mesh_kernel_types_and_reshard_program():
    result = Compiler().compile(_packed_matmul_module()).module
    launch = result.metadata["launch_contract"]
    distribution = result.metadata["tir_distribution"]
    kernel = result.node_map["output"]
    dispatch = fm.kernel_dispatch_for_call(result, kernel)
    kernel_distribution = dispatch.parameters["distribution"]

    assert tuple(launch["grid_mesh"]["hierarchy"]) == (8, 16)
    assert launch["grid_mesh"]["name"] == "yx"
    assert launch["grid_mesh"]["hierarchy_levels"] == "bb"
    assert distribution["schema"] == "flagmega.tir-distribution/v1"
    assert distribution["adapters"]
    assert {value["op"] for value in distribution["adapters"]} == {
        "distributed.sharded_view",
        "distributed.boxing",
    }
    assert kernel_distribution["placement"] == launch["grid_mesh"]
    assert kernel_distribution["input_types"]
    assert kernel_distribution["output_type"] is not None
    assert any("DistributedType" in type(value).__name__ for value in kernel_distribution["input_types"])


def test_distribution_lowering_round_trips_as_editable_python_ir(tmp_path):
    result = Compiler().compile(_packed_matmul_module()).module
    checkpoint = fm.emit_module(result, tmp_path / "bufferized.py")

    loaded = fm.load_module(checkpoint)
    assert loaded.semantic_hash == result.semantic_hash
    assert fm.kernel_dispatch_for_call(
        loaded, loaded.node_map["output"]
    ).parameters["distribution"] == (
        fm.kernel_dispatch_for_call(result, result.node_map["output"]).parameters["distribution"]
    )
