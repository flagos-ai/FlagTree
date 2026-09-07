# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega.codegen.triton.templates import (
    KernelTemplateSpec,
    TritonTemplateRegistry,
)


def test_persistent_template_exposes_only_the_local_shard_microkernel():
    source = TritonTemplateRegistry().render_kernel(
        KernelTemplateSpec("gdn_convolution", "persistent", "nvidia", "sm90"),
        {},
    ).source

    compile(source, "gdn_convolution_persistent.py", "exec")
    assert "def _flagmega_gdn_convolution_local_tile(" in source
    assert "def _flagmega_gdn_convolution(" not in source
    assert "shard_index" not in source
