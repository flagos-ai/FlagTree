# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Pack/de-vectorize propagation rules."""

from triton.flagmega.rules.ntt.vectorize.propagation.binary import binary_propagation_rules
from triton.flagmega.rules.ntt.vectorize.propagation.cast import cast_propagation_rules
from triton.flagmega.rules.ntt.vectorize.propagation.fold import fold_boundary_rules
from triton.flagmega.rules.ntt.vectorize.propagation.unary import unary_propagation_rules
from triton.flagmega.rules.ntt.vectorize.propagation.layout import layout_propagation_rules
from triton.flagmega.rules.ntt.vectorize.propagation.concat import concat_propagation_rules
from triton.flagmega.rules.ntt.vectorize.propagation.reshape import reshape_propagation_rules
from triton.flagmega.rules.ntt.vectorize.propagation.rope import rope_propagation_rules
from triton.flagmega.rules.ntt.vectorize.propagation.sparse_experts import sparse_experts_propagation_rules
from triton.flagmega.rules.ntt.vectorize.propagation.slice import slice_propagation_rules
from triton.flagmega.rules.ntt.vectorize.propagation.broadcast import broadcast_propagation_rules


def propagation_rules():
    return (
        *binary_propagation_rules(),
        *cast_propagation_rules(),
        *unary_propagation_rules(),
        *concat_propagation_rules(),
        *layout_propagation_rules(),
        *slice_propagation_rules(),
        *broadcast_propagation_rules(),
        *reshape_propagation_rules(),
        *rope_propagation_rules(),
        *sparse_experts_propagation_rules(),
        *fold_boundary_rules(),
    )


__all__ = [
    "binary_propagation_rules",
    "cast_propagation_rules",
    "concat_propagation_rules",
    "fold_boundary_rules",
    "layout_propagation_rules",
    "propagation_rules",
    "reshape_propagation_rules",
    "rope_propagation_rules",
    "sparse_experts_propagation_rules",
    "slice_propagation_rules",
    "broadcast_propagation_rules",
    "unary_propagation_rules",
]
