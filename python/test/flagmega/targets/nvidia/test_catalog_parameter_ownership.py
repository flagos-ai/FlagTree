# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega.targets.nvidia.implementations import (
    sm90_triton_implementation_model,
)


def test_catalog_exposes_physical_knobs_but_not_semantic_weight_geometry():
    model = sm90_triton_implementation_model()

    for name in (
        "tir.block_fp8.simt",
        "tir.block_fp8.mma",
        "tir.matmul_glu.simt",
        "tir.matmul_glu.mma",
    ):
        implementation = model.implementation(name)
        assert implementation is not None
        assert tuple(implementation.parameters) == ("tile_n",)

    qkv = model.implementation("tir.qkv_parallel_linear.packed_fused_gemv")
    assert qkv is not None
    assert dict(qkv.parameters) == {"block_k": 64, "tile_n": 16}


def test_previously_hidden_tiles_are_owned_by_the_implementation_catalog():
    model = sm90_triton_implementation_model()

    embedding = model.implementation("tir.embedding.decode")
    recurrent = model.implementation("tir.gdn_recurrent.persistent")
    tensor_load = model.implementation("tir.distributed_boxing.tensor_load")
    tensor_store = model.implementation("tir.distributed_boxing.tensor_store")
    assert embedding is not None
    assert recurrent is not None
    assert tensor_load is not None
    assert tensor_store is not None
    assert embedding.parameters["elements_per_program"] == 16
    assert recurrent.parameters["projection_tile"] == 128
    assert tensor_load.parameters["tile"] == 1024
    assert tensor_store.parameters["tile"] == 1024
