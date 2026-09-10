# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Select an FP32 MatMul result at a private widening boundary.

This is a relaxed numerical rewrite: the F16/BF16 result rounding is removed,
not hidden inside an epilogue. Run before vectorization/distribution so the
new output dtype participates in layout and collective planning.
"""

from dataclasses import replace

from triton.flagmega.ir import DType, TensorType
from triton.flagmega.ir.ops.math.matmul import MatMul
from triton.flagmega.pattern_match import F
from triton.flagmega.rules import RewriteResult, RewriteRule


def fold_matmul_cast_rule():
    producer = F.math.is_matmul(call_name="matmul").with_user_count(1)
    pattern = F.tensors.is_cast(producer, dtype="float32", call_name="root")

    def rewrite(match, module):
        root, matmul = match["root"], match["matmul"]
        if (not isinstance(matmul.type, TensorType) or matmul.type.dtype not in {DType.FLOAT16, DType.BFLOAT16}
                or any(matmul.id in function.outputs for function in module.functions)):
            return root
        prepared = MatMul.prepare(tuple(module.node_map[value] for value in matmul.inputs),
                                  {**matmul.attrs, "output_data_type": DType.FLOAT32})
        if prepared.result_type != root.type:
            return root
        return RewriteResult(
            replace(matmul, id=root.id, type=prepared.result_type, attrs=prepared.attrs,
                    metadata={**matmul.metadata, **root.metadata, "formed_by": "FoldMatMulCast"}))

    return RewriteRule("FoldMatMulCast", pattern, rewrite)


__all__ = ["fold_matmul_cast_rule"]
