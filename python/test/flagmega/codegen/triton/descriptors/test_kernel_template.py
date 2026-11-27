# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega.codegen.triton.templates import (
    KernelTemplateSpec,
    TritonTemplateRegistry,
)


def test_descriptor_gemv_template_uses_local_tile_coordinates_only():
    source = TritonTemplateRegistry().render_kernel(
        KernelTemplateSpec(
            "dense_matmul", "tensor_descriptor_gemv", "nvidia", "sm90"
        ),
        {},
    ).source

    compile(source, "tensor_descriptor_gemv.py", "exec")
    assert "weight_descriptor.load(" in source
    assert "descriptor_n_offset" in source
    assert "descriptor_k_offset" in source
    assert "shard_index" not in source
    assert "mesh_hierarchy" not in source
    assert "qwen" not in source.lower()
