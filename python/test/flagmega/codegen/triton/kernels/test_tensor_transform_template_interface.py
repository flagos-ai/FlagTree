# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega.codegen.triton.templates import KernelTemplateSpec, TritonTemplateRegistry


@pytest.mark.parametrize("family",
                         ["pad", "slice", "pack", "unpack", "concat", "broadcast_to", "softmax", "reduce_sum", "top_k"])
def test_tensor_transform_template_can_render_without_a_package_call_list(family):
    source = TritonTemplateRegistry().render_kernel(KernelTemplateSpec(family, "local", "nvidia", "sm90"), {}).source
    compile(source, f"{family}.py", "exec")
    assert f"# flagmega-kernel: {family}/local platform=generic" in source
