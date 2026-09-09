# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""nncase VectorizeSparseExpertsPropagation: absorb hidden-axis output Pack."""

from triton.flagmega.ir import TensorType, VectorType
from triton.flagmega.ir.ops.nn.sparse_experts_down import SparseExpertsDown
from triton.flagmega.ir.ops.tensors.pack import normalize_axes
from triton.flagmega.pattern_match import F
from triton.flagmega.rules import RewriteResult, RewriteRule
from triton.flagmega.rules.neutral._utility import make_node
from triton.flagmega.rules.ntt.vectorize.utility import propagation_result_metadata


def sparse_experts_propagation_rules() -> tuple[RewriteRule, ...]:
    pattern = F.tensors.is_pack(F.nn.is_sparse_experts_down(call_name="down"), call_name="pack")

    def rewrite(result, module):
        pack, down = result["pack"], result["down"]
        if (not isinstance(down.type, TensorType) or isinstance(down.type.dtype, VectorType)
                or not isinstance(pack.type, TensorType) or not isinstance(pack.type.dtype, VectorType)):
            return None
        lanes = tuple(pack.attrs["lanes"])
        axes = normalize_axes(
            tuple(pack.attrs["axes"]) if "axes" in pack.attrs else (pack.attrs["axis"], ) * len(lanes), 2)
        if not axes or any(axis != 1 for axis in axes):
            return None
        replacement = make_node(
            SparseExpertsDown.op_name,
            pack.id,
            tuple(module.node_map[name] for name in down.inputs),
            {**dict(down.attrs), "output_dtype": pack.type.dtype},
            propagation_result_metadata(down, pack, axes=axes, lanes=lanes, rule="VectorizeSparseExpertsPropagation",
                                        internal_role="propagated-compute"),
        )
        return RewriteResult(replacement)

    return (RewriteRule("VectorizeSparseExpertsPropagation", pattern, rewrite), )


__all__ = ["sparse_experts_propagation_rules"]
