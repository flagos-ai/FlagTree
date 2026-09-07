# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""nncase-style structural pattern matching for FlagMega expression IR."""

from triton.flagmega.pattern_match.call_pattern import CallPattern, is_call
from triton.flagmega.pattern_match.const_pattern import ConstPattern, is_const
from triton.flagmega.pattern_match.functional import F
from triton.flagmega.pattern_match.matcher import find_matches, try_match, try_match_root
from triton.flagmega.pattern_match.op_pattern import OpPattern, is_op
from triton.flagmega.pattern_match.or_pattern import OrPattern, is_alt
from triton.flagmega.pattern_match.options import MatchOptions
from triton.flagmega.pattern_match.pattern import ExprPattern, Pattern, wildcard
from triton.flagmega.pattern_match.result import MatchResult
from triton.flagmega.pattern_match.vargs_pattern import VArgsPattern, is_vargs, is_vargs_repeat
from triton.flagmega.pattern_match.var_pattern import VarPattern, is_var

__all__ = [
    "CallPattern",
    "ConstPattern",
    "ExprPattern",
    "F",
    "MatchResult",
    "MatchOptions",
    "OpPattern",
    "OrPattern",
    "Pattern",
    "VArgsPattern",
    "VarPattern",
    "find_matches",
    "is_alt",
    "is_call",
    "is_const",
    "is_op",
    "is_var",
    "is_vargs",
    "is_vargs_repeat",
    "try_match",
    "try_match_root",
    "wildcard",
]
