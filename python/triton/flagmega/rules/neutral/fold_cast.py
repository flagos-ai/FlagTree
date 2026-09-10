# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Target-independent cast identities and relaxed floating round trips.

The round-trip rule deliberately permits loss of the intermediate floating
rounding. It is not an equivalence for integer truncation or booleanization.
"""

from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.types import DType
from triton.flagmega.pattern_match import F, wildcard
from triton.flagmega.rules import RewriteRedirect, RewriteRule


def fold_cast_rule():
    pattern = F.tensors.is_cast(wildcard("input"), call_name="root")

    def rewrite(match, module):
        root, source = match["root"], match["input"]
        if root.type == source.type:
            return RewriteRedirect(source.id)
        if source.op != "tensors.cast":
            return root
        original = module.node_map[source.inputs[0]]
        floating = {DType.FLOAT32, DType.BFLOAT16, DType.FLOAT8_E4M3FN}
        if (root.type == original.type and tensor_of(root.type).dtype in floating
                and tensor_of(source.type).dtype in floating):
            return RewriteRedirect(original.id)
        return root

    return RewriteRule("FoldCast", pattern, rewrite)
