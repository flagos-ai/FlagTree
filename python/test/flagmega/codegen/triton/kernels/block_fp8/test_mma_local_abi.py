# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega.codegen.triton.templates import (
    KernelTemplateSpec,
    TritonTemplateRegistry,
)


def test_mma_template_contains_only_the_local_shard_microkernel():
    registry = TritonTemplateRegistry()
    spec = KernelTemplateSpec("block_fp8", "mma", "nvidia", "sm90")

    source = registry.render_kernel(spec, {"tn": 16}).source

    compile(source, "block_fp8_mma.py", "exec")
    assert "def _flagmega_block_linear_local_tile(" in source
    assert "def _flagmega_block_linear(" not in source
    assert "def _flagmega_block_linear_split_k(" not in source
    assert "shard_y" not in source
    assert "shard_x" not in source
    assert "shard_index" not in source
