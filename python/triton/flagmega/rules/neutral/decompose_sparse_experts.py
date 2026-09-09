# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Expose the selected-expert GateUp/Down handoff, as in nncase."""

from triton.flagmega.ir.ops.nn.sparse_experts import SparseExperts
from triton.flagmega.pattern_match import F, wildcard
from triton.flagmega.rules import RewriteResult, RewriteRule
from triton.flagmega.rules.neutral._utility import decomposition_metadata


def decompose_sparse_experts_rule() -> RewriteRule:
    pattern = F.nn.is_sparse_experts(
        **{
            parameter.name: wildcard(parameter.name, type_pattern=parameter.type_pattern)
            for parameter in SparseExperts.input_parameters
        },
        call_name="call",
    )

    def rewrite(result, module):
        source = result["call"]
        operands = tuple(result[parameter.name] for parameter in SparseExperts.input_parameters)
        gate, down = SparseExperts.stage_calls(
            operands,
            source.attrs,
            name=source.id,
            metadata=decomposition_metadata(source, "DecomposeSparseExperts"),
        )
        return RewriteResult(down, (gate, ))

    return RewriteRule("DecomposeSparseExperts", pattern, rewrite)


__all__ = ["decompose_sparse_experts_rule"]
