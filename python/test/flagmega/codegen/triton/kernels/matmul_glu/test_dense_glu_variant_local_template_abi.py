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
        (
            "tir.dense_matmul_glu.gemv_tn16",
            "_flagmega_dense_matmul_glu_gemv_accumulate",
        ),
        (
            "tir.dense_matmul_glu.packed_k_major_gemv_tn16",
            "_flagmega_dense_matmul_glu_packed_k_major_gemv_accumulate",
        ),
    ),
)
def test_registered_dense_glu_variant_defines_its_local_accumulate_abi(
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
    assert "gate_weight_offsets" in source
    assert "up_weight_offsets" in source
    assert "shard_index" not in source
    assert "mesh_hierarchy" not in source
