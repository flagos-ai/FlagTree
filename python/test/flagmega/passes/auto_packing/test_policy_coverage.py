# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega.rules.ntt.packing import NttPackingPolicy


def test_pyntt_packing_policy_catalog_covers_every_represented_rule_family():
    """Drift gate for nncase PyNTT rules representable in current FlagMega IR.

    ScaledMatMul and NVFP4 families are intentionally absent until their IR
    operations exist; silently naming them here would create fake passes.
    """

    assert NttPackingPolicy.op_names == {
        "math.block_scaled_matmul",
        "math.matmul",
        "math.vectorized_matmul",
        "nn.dense_matmul_glu",
        "nn.matmul_glu",
        "nn.qkv_parallel_linear",
    }


def test_packing_policy_identity_is_machine_independent():
    identity = NttPackingPolicy(vector_bytes=16, k_pack=2).identity.lower()

    assert identity.startswith("pyntt-auto-packing/")
    assert "sm90" not in identity
    assert "qwen" not in identity
