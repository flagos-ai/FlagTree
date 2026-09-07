# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega.codegen.triton.templates import (
    KernelTemplateSpec,
    TritonTemplateRegistry,
)


def test_mma_template_contains_only_the_local_shard_microkernel():
    source = TritonTemplateRegistry().render_kernel(
        KernelTemplateSpec("matmul_glu", "mma", "nvidia", "sm90"),
        {"tn": 16},
    ).source

    compile(source, "matmul_glu_mma.py", "exec")
    assert "def _flagmega_block_glu_local_tile(" in source
    assert "def _flagmega_block_glu(" not in source
    assert "shard_index" not in source
