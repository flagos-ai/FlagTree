# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega.codegen.triton.templates import (
    KernelTemplateSpec,
    TritonTemplateRegistry,
)
from triton.flagmega.targets.nvidia.implementations import (
    sm90_triton_implementation_model,
)


@pytest.mark.parametrize(
    "implementation_id,helper",
    (
        ("tir.dense_matmul.gemv", "_flagmega_dense_matmul_gemv_accumulate"),
        (
            "tir.dense_matmul.packed_k_major_gemv",
            "_flagmega_dense_matmul_packed_k_major_gemv_accumulate",
        ),
        (
            "tir.dense_matmul.split_k_gemv",
            "_flagmega_dense_matmul_split_k_gemv_accumulate",
        ),
        (
            "tir.dense_matmul.split_k_packed_k_major_gemv",
            "_flagmega_dense_matmul_split_k_packed_k_major_gemv_accumulate",
        ),
        (
            "tir.dense_matmul.split_k_n_packed_k_major_gemv",
            "_flagmega_dense_matmul_split_k_n_packed_k_major_gemv_accumulate",
        ),
    ),
)
def test_registered_dense_variant_defines_its_local_accumulate_abi(
    implementation_id, helper,
):
    implementation = sm90_triton_implementation_model().implementation(
        implementation_id
    )
    assert implementation is not None
    source = TritonTemplateRegistry().render_kernel(
        KernelTemplateSpec(
            implementation.family,
            implementation.variant,
            "nvidia",
            "sm90",
        ),
        {},
    ).source

    compile(source, f"{implementation.variant}.py", "exec")
    assert f"def {helper}(" in source
    assert "source_offsets" in source
    assert "weight_offsets" in source
    assert "shard_index" not in source
    assert "mesh_hierarchy" not in source
