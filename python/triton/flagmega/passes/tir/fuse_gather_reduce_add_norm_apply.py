# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Compatibility entry points for the grouped dataflow fusion rules."""

from dataclasses import dataclass

from triton.flagmega.ir import IRModule
from triton.flagmega.passes.rewriter import DataflowPass
from triton.flagmega.rules.ntt.fuse_gather_reduce_add_norm_apply import fuse_gather_reduce_add_norm_apply_rule


def fuse_gather_reduce_add_norm_apply(module: IRModule) -> IRModule:
    return DataflowPass("fuse_gather_reduce_add_norm_apply", (fuse_gather_reduce_add_norm_apply_rule(),),
                        rewrite_constants=False, remove_unused=False).run(module)


@dataclass(frozen=True)
class FuseGatherReduceAddNormApplyPass:
    name: str = "FuseGatherReduceAddNormApply"
    preserves: frozenset[str] = frozenset()

    def run(self, module: IRModule) -> IRModule:
        return fuse_gather_reduce_add_norm_apply(module)


__all__ = [
    "FuseGatherReduceAddNormApplyPass",
    "fuse_gather_reduce_add_norm_apply",
]
