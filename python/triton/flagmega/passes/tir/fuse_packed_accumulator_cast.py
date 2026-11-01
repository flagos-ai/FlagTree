# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Opt into the existing typed PostOps rules for packed accumulator casts."""

from triton.flagmega.codegen.triton.fusion import can_fuse
from triton.flagmega.passes.rewriter import DataflowPass
from triton.flagmega.rules.neutral.pre_post_ops import pre_post_ops_rules
from triton.flagmega.rules.neutral.post_ops_view import post_ops_through_view_rules


def fuse_packed_accumulator_casts(module, *, predicate=None):
    """Keep FP32 semantics and move only the proven final BF16 cast.

    The rules preserve exported/shared values and retire consumed selections.
    A private ShardedView may separate the producer and cast: lift the exact
    conversion onto the producer's ownership and retain the result view. This
    is one composite rewrite, so ordinary cast/view commutation cannot undo it.
    The backend capability check
    admits only a BF16 cast of a complete FP32 packed projection, never a
    partial reduction or an arbitrary MatMul expression.
    """

    def supported(candidate, source):
        return (predicate is None or predicate(candidate)) and can_fuse(candidate, source)

    return DataflowPass(
        "FusePackedAccumulatorCast",
        (
            *pre_post_ops_rules(("ntt.packed_matmul",), supported),
            *post_ops_through_view_rules(("ntt.packed_matmul",), supported, views=("distributed.sharded_view",)),
        ),
        rewrite_constants=False,
    ).run(module)


__all__ = ["fuse_packed_accumulator_casts"]
