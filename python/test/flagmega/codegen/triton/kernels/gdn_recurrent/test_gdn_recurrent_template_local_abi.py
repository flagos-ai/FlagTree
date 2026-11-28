# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega.codegen.triton.templates import (
    KernelTemplateSpec,
    TritonTemplateRegistry,
)


def test_persistent_template_exposes_only_local_shard_microkernels():
    source = TritonTemplateRegistry().render_kernel(
        KernelTemplateSpec("gdn_recurrent", "persistent", "nvidia", "sm90"),
        {
            "recurrent_value_tile": 128,
            "head_block": 128,
            "query_scale_repr": "0.125",
        },
    ).source

    compile(source, "gdn_recurrent_persistent.py", "exec")
    assert "def _flagmega_dense_rows(" in source
    assert "def _flagmega_gdn_recurrent_local_core_tile(" in source
    assert "def _flagmega_gdn_recurrent_local_norm_tile(" in source
    assert "def _flagmega_dense_scalar(" not in source
    assert "def _flagmega_gdn_recurrent(" not in source
    assert "shard_index" not in source
