# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega.codegen.triton.templates import (
    KernelTemplateSpec,
    TritonTemplateRegistry,
)
from triton.flagmega.targets.nvidia.implementations import (
    sm90_triton_implementation_model,
)


def test_simt_fallback_has_a_complete_parameter_and_template_contract():
    implementation = sm90_triton_implementation_model().implementation(
        "tir.block_fp8.simt"
    )

    assert implementation is not None
    assert implementation.parameters == {"tile_n": 16}
    assert implementation.requires == ("fp8",)
    registry = TritonTemplateRegistry()
    spec = KernelTemplateSpec("block_fp8", "simt", "nvidia", "sm90")
    assert registry.resolve(spec) == "kernels/block_fp8/simt.py.jinja"
    source = registry.render_kernel(spec, {"tn": 16}).source
    compile(source, "block_fp8_simt.py", "exec")
    assert "tl.dot" not in source
    assert "weight_row_offsets" in source
    assert "destination_offsets" in source
