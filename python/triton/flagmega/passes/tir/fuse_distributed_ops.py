# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Post-distribution boundary rewriting and explicit multi-input fusion."""

from triton.flagmega.passes.rewriter import DataflowPass
from triton.flagmega.passes.tir.fuse_attention_gate import fuse_attention_gate
from triton.flagmega.passes.tir.lower_sparse_experts import lower_sparse_experts
from triton.flagmega.rules.ntt.fuse_gather_reduce_add_norm_apply import fuse_gather_reduce_add_norm_apply_rule
from triton.flagmega.rules.ntt.fuse_gather_reduce_norm_apply import fuse_gather_reduce_norm_apply_rule


def fuse_distributed_ops(module, *, fusion_rules=()):
    rewritten = DataflowPass(
        "FuseDistributedOps",
        (fuse_gather_reduce_add_norm_apply_rule(), fuse_gather_reduce_norm_apply_rule(),
         *fusion_rules),
        rewrite_constants=False,
    ).run(module)
    return lower_sparse_experts(fuse_attention_gate(rewritten))
