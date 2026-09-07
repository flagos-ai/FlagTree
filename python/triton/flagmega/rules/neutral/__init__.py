# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Target-independent semantic rewrite rules."""

from triton.flagmega.rules.neutral.auto_packing import (
    auto_packing_neutral_rules,
    post_boundary_auto_packing_neutral_rules,
)
from triton.flagmega.rules.neutral.decompose_layer_norm import decompose_layer_norm_rule
from triton.flagmega.rules.neutral.decompose_rms_norm import decompose_rms_norm_rule
from triton.flagmega.rules.neutral.fuse_norm_stats_apply import fuse_norm_stats_apply_rule
from triton.flagmega.rules.neutral.fuse_wide_glu import fuse_wide_glu_rule
from triton.flagmega.rules.neutral.fuse_norm_apply_cast import fuse_norm_apply_cast_rule
from triton.flagmega.rules.neutral.fold_bind_norm_stats import fold_bind_norm_stats_rule
from triton.flagmega.rules.neutral.fold_bitcast_bitcast import fold_bitcast_bitcast_rule
from triton.flagmega.rules.neutral.fold_get_item_tuple import fold_get_item_tuple_rule
from triton.flagmega.rules.neutral.fold_pack_bitcast import fold_pack_bitcast_rule
from triton.flagmega.rules.neutral.fold_pack_reshape import fold_pack_reshape_rule
from triton.flagmega.rules.neutral.fold_pack_transpose import fold_pack_transpose_rule
from triton.flagmega.rules.neutral.unpack_to_bitcast import unpack_to_bitcast_rule

__all__ = [
    "auto_packing_neutral_rules",
    "decompose_layer_norm_rule",
    "decompose_rms_norm_rule",
    "fuse_norm_stats_apply_rule",
    "fuse_wide_glu_rule",
    "fuse_norm_apply_cast_rule",
    "fold_bind_norm_stats_rule",
    "fold_bitcast_bitcast_rule",
    "fold_get_item_tuple_rule",
    "fold_pack_bitcast_rule",
    "fold_pack_reshape_rule",
    "fold_pack_transpose_rule",
    "post_boundary_auto_packing_neutral_rules",
    "unpack_to_bitcast_rule",
]
