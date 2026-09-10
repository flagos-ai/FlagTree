# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""One dataflow fixed point for post-distribution local fusion strategies."""

from triton.flagmega.passes.rewriter import DataflowPass
from triton.flagmega.rules.ntt.fuse_gather_reduce_add_norm_apply import fuse_gather_reduce_add_norm_apply_rule
from triton.flagmega.rules.ntt.fuse_gather_reduce_norm_apply import fuse_gather_reduce_norm_apply_rule
from triton.flagmega.rules.ntt.fuse_gather_reduce_qkv_rope_with_cache import fuse_gather_reduce_qkv_rope_with_cache_rule


def fuse_distributed_ops(module, *, fusion_rules=()):
    return DataflowPass(
        "FuseDistributedOps",
        (fuse_gather_reduce_add_norm_apply_rule(), fuse_gather_reduce_norm_apply_rule(),
         fuse_gather_reduce_qkv_rope_with_cache_rule(), *fusion_rules),
        rewrite_constants=False,
    ).run(module)
