# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Lossless projection promotion with exact packed-coordinate preservation."""

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.model import Node
from triton.flagmega.ir.types import DType, VectorType
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.ops.ntt.vectorized_cast import VectorizedCast
from triton.flagmega.ir.ops.tensors.cast import Cast


def promoted_projection_type(source_type, result_type):
    """Return the promoted type, preserving split units and any partial axes.

    This is a type/layout proof, not permission to commute a cast through a
    collective. A local epilogue must additionally prove full type equality.
    Projection BF16 rounding always precedes residual FP32 addition.
    """
    if source_type == result_type:
        return source_type
    source, result = tensor_of(source_type), tensor_of(result_type)
    if source == result:
        return source_type
    source_dtype = source.dtype.elem_type if isinstance(source.dtype, VectorType) else source.dtype
    result_dtype = result.dtype.elem_type if isinstance(result.dtype, VectorType) else result.dtype
    if (source_dtype != DType.BFLOAT16 or result_dtype != DType.FLOAT32
            or source.rank != result.rank):
        return None
    operand = Node("<projection>", "builtin.var", (), source_type, attrs={"name": "<projection>"})
    try:
        if isinstance(source.dtype, VectorType) and isinstance(result.dtype, VectorType):
            if len(source.dtype.lanes) != 1 or len(result.dtype.lanes) != 1:
                return None
            attrs = VectorizedCast.normalize_attrs({"new_type": result.dtype, "vectorize_axes": (-1,)})
            promoted = VectorizedCast.infer_type((operand,), attrs)
        elif not isinstance(source.dtype, VectorType) and not isinstance(result.dtype, VectorType):
            promoted = Cast.infer_type((operand,), Cast.normalize_attrs({"dtype": result.dtype}))
        else:
            return None
    except IRSchemaError:
        return None
    return promoted if tensor_of(promoted) == result else None


def is_projection_promotion(node, nodes):
    return (node.op in {"tensors.cast", "ntt.vectorized_cast"}
            and promoted_projection_type(nodes[node.inputs[0]].type, node.type) == node.type)
