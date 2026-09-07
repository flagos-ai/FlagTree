# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Typed equality saturation for pure FlagMega IR islands."""

from triton.flagmega.egraph.graph import EClassView, EGraph, ENode
from triton.flagmega.egraph.matcher import find_egraph_matches
from triton.flagmega.egraph.extractor import EGraphExtractor, ExtractionResult, ForcedChoice
from triton.flagmega.egraph.rewriter import (
    AlternativeSelector,
    EGraphAlternative,
    EGraphRewriter,
    EGraphSession,
    NodeCost,
    RewriteIteration,
)

__all__ = [
    "AlternativeSelector",
    "EClassView",
    "EGraph",
    "EGraphAlternative",
    "EGraphExtractor",
    "EGraphRewriter",
    "EGraphSession",
    "find_egraph_matches",
    "ENode",
    "ExtractionResult",
    "ForcedChoice",
    "NodeCost",
    "RewriteIteration",
]
