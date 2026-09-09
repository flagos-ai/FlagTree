# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Neutral rule groups used by the two nncase-shaped AutoPacking phases."""

from triton.flagmega.rules import RewriteRule
from triton.flagmega.rules.neutral.fold_bitcast_bitcast import (
    fold_bitcast_bitcast_rule,
)
from triton.flagmega.rules.neutral.fold_get_item_tuple import (
    fold_get_item_tuple_rule,
)
from triton.flagmega.rules.neutral.fold_pack_bitcast import fold_pack_bitcast_rule
from triton.flagmega.rules.neutral.fold_pack_reshape import fold_pack_reshape_rule
from triton.flagmega.rules.neutral.fold_pack_transpose import fold_pack_transpose_rule
from triton.flagmega.rules.neutral.unpack_to_bitcast import unpack_to_bitcast_rule
from triton.flagmega.rules.neutral.fold_pointwise_padded_slice import fold_pointwise_padded_slice_rule


def auto_packing_neutral_rules() -> tuple[RewriteRule, ...]:
    """Rules following target packing in nncase's AutoPacking dataflow pass.

    ``FoldConstCall`` is intentionally represented by FlagMega's later,
    explicit ``FreezeConstantIslands`` boundary.  Folding it here would erase
    the editable Python recipe required by edit-and-resume compilation.
    """

    return (
        fold_get_item_tuple_rule(),
        fold_pack_transpose_rule(),
        fold_pack_reshape_rule(),
        fold_pack_bitcast_rule(),
        fold_bitcast_bitcast_rule(),
    )


def post_boundary_auto_packing_neutral_rules() -> tuple[RewriteRule, ...]:
    """Neutral equalities following target post-boundary propagation."""

    return (
        unpack_to_bitcast_rule(),
        fold_pointwise_padded_slice_rule(),
        *auto_packing_neutral_rules(),
    )


__all__ = [
    "auto_packing_neutral_rules",
    "post_boundary_auto_packing_neutral_rules",
]
