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
        "tir.matmul_glu.simt"
    )

    assert implementation is not None
    assert implementation.parameters == {"tile_n": 16}
    assert implementation.requires == ("fp8",)
    registry = TritonTemplateRegistry()
    spec = KernelTemplateSpec("matmul_glu", "simt", "nvidia", "sm90")
    assert registry.resolve(spec) == "kernels/matmul_glu/simt.py.jinja"
    source = registry.render_kernel(spec, {"tn": 16}).source
    compile(source, "matmul_glu_simt.py", "exec")
    assert "tl.dot" not in source
    assert "gate_weight" in source
    assert "up_weight" in source
    assert "destination_offsets" in source
