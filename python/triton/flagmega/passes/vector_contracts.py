# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Shared lifetime facts for typed-vector rewrite expressions."""

from __future__ import annotations

from triton.flagmega.ir import IRModule, Node


# These operations remain physical typed-vector compute after
# LowerVectorizationContracts.  Other selected vector expressions are equality
# witnesses used to choose a schedule and are reconstructed as their semantic
# operation before TIR selection.
NATIVE_VECTOR_COMPUTE_OPS = frozenset({
    "math.vectorized_binary",
    "math.vectorized_unary",
    "nn.norm_apply",
    "nn.norm_stats",
    "nn.qkv_rope_with_cache",
    "ntt.matmul_norm_stats_combine",
    "ntt.matmul_norm_stats",
    "ntt.packed_matmul",
    "ntt.vectorized_cast",
    "ntt.vectorized_rope",
})


def vectorization_root(node: Node) -> str | None:
    value = node.metadata.get("vectorization_semantic_id")
    if value is None:
        value = node.metadata.get("vectorization_root")
    if value is None and node.metadata.get("vectorized_from") is not None:
        value = node.id
    return None if value is None else str(value)


def retained_vectorization_roots(module: IRModule) -> frozenset[str]:
    """Return expression roots whose typed-vector graph survives lowering."""

    result = {
        root
        for node in module.nodes
        if (root := vectorization_root(node)) is not None
        if node.op in NATIVE_VECTOR_COMPUTE_OPS
    }
    # Pad/Slice do not yet have native vector TIR implementations.  Their
    # entire expression must therefore return to the semantic scalar graph.
    result.difference_update(
        root
        for node in module.nodes
        if (root := vectorization_root(node)) is not None
        if node.op in {"tensors.pad", "tensors.slice_to_shape"}
    )
    return frozenset(result)


def is_transient_vectorization_boundary(
    module: IRModule,
    node: Node,
    *,
    retained_roots: frozenset[str] | None = None,
) -> bool:
    """Whether a representation view disappears before TIR selection."""

    root = vectorization_root(node)
    if root is None:
        return False
    retained = (
        retained_vectorization_roots(module)
        if retained_roots is None
        else retained_roots
    )
    return root not in retained


__all__ = [
    "NATIVE_VECTOR_COMPUTE_OPS",
    "is_transient_vectorization_boundary",
    "retained_vectorization_roots",
    "vectorization_root",
]
