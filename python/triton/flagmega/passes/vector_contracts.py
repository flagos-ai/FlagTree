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
    "nn.sparse_experts_gate_up",
    "nn.sparse_experts_down",
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

    result: set[str] = set()
    for node in module.nodes:
        if node.op not in NATIVE_VECTOR_COMPUTE_OPS:
            continue
        # Propagation gives helpers the generated compute ID, while the
        # compute also carries its scalar semantic ID and original schedule
        # root. These are aliases of one retained expression, not independent
        # lifetimes. Taking only vectorization_root()'s first match would
        # misclassify the other aliases as transient at reusable boundaries.
        result.add(node.id)
        result.update(str(node.metadata[key]) for key in ("vectorization_semantic_id", "vectorization_root")
                      if node.metadata.get(key) is not None)
    # Padding/cropping are executable tensor transforms. In particular a
    # packed RHS may already be a function parameter: scalar reconstruction
    # cannot recover its logical MatMul operands from that physical ABI.
    # Keep the exact Pad/Pack/Compute/Unpack/Slice expression selected by the
    # vector rule; constant padding still materializes with its weight island.
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
