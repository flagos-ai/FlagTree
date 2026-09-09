# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Compatibility entry points for the grouped dataflow fusion rules."""

from dataclasses import dataclass

from triton.flagmega.ir import IRModule
from triton.flagmega.passes.rewriter import DataflowPass
from triton.flagmega.rules.ntt.fuse_gather_reduce_qkv_rope_with_cache import fuse_gather_reduce_qkv_rope_with_cache_rule


def fuse_gather_reduce_qkv_rope_with_cache(module: IRModule) -> IRModule:
    return DataflowPass("fuse_gather_reduce_qkv_rope_with_cache", (fuse_gather_reduce_qkv_rope_with_cache_rule(),),
                        rewrite_constants=False, remove_unused=False).run(module)


@dataclass(frozen=True)
class FuseGatherReduceQKVRoPEWithCachePass:
    name: str = "FuseGatherReduceQKVRoPEWithCache"
    preserves: frozenset[str] = frozenset()

    def run(self, module: IRModule) -> IRModule:
        return fuse_gather_reduce_qkv_rope_with_cache(module)


__all__ = [
    "FuseGatherReduceQKVRoPEWithCachePass",
    "fuse_gather_reduce_qkv_rope_with_cache",
]
