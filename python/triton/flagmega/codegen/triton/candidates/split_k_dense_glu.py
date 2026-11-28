# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Scoped SM90 candidates for explicit two-program GU semantics."""
from triton.flagmega.ir.ops.ntt.split_k_dense_glu import SplitKDenseGluProjection, SplitKDenseGluEpilogue
from .core import TritonCandidateProposal


class SplitKDenseGluCandidateProvider:
    op_names = frozenset((SplitKDenseGluProjection.op_name, SplitKDenseGluEpilogue.op_name))

    def propose(self, node, context):
        if (getattr(context.target, 'codegen_platform', None) != 'nvidia'
                or getattr(context.target, 'codegen_architecture', None) != 'sm90'):
            return None
        family = 'split_k_dense_glu'
        variant = 'projection' if node.op == SplitKDenseGluProjection.op_name else 'epilogue'
        variants = tuple(
            context.configure_implementation(value)
            for value in context.implementations(family)
            if value.variant == variant)
        if not variants:
            return None
        return TritonCandidateProposal(variants, context.choose_default(family, variants))


__all__ = ['SplitKDenseGluCandidateProvider']
