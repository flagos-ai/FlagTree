# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega.codegen.triton.templates import (
    KernelTemplateSpec,
    TritonTemplateRegistry,
)
from triton.flagmega.targets.nvidia.implementations import (
    sm90_triton_implementation_model,
)


def test_hybrid_split_k_n_uses_a_mesh_parameterized_generic_template():
    implementation = sm90_triton_implementation_model().implementation(
        "tir.dense_matmul.split_k_n_packed_k_major_gemv"
    )

    assert implementation is not None
    assert implementation.contract["distribution_kind"] == (
        "output_reduction_split"
    )
    assert implementation.facts["portable_triton"] is True
    registry = TritonTemplateRegistry()
    spec = KernelTemplateSpec(
        implementation.family,
        implementation.variant,
        "nvidia",
        "sm90",
    )
    assert registry.resolve(spec) == (
        "kernels/dense_matmul/split_k_n_packed_k_major_gemv.py.jinja"
    )
    source = registry.render_kernel(spec, {}).source
    compile(source, "split_k_n_packed_k_major_gemv.py", "exec")

    assert (
        "def _flagmega_dense_matmul_split_k_n_packed_k_major_gemv_accumulate("
        in source
    )
    assert "source_offsets" in source
    assert "weight_offsets" in source
    assert "mesh_size" not in source
    assert "mesh_hierarchy" not in source
    assert "sm90" not in source.lower()
