# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""NormStats owns exact BF16 widening before FP32 square/reduction."""

from math import prod
from triton.flagmega.ir import DType, VectorType, TensorLayout
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.ops.nn._norm import normalize_axis, scalar_dtype
from triton.flagmega.ir.ops.ntt.vectorized_cast import cast_vector_axes
from triton.flagmega.rules import RewriteResult, RewriteRule
from triton.flagmega.rules.neutral._utility import make_node


def fuse_norm_stats_input_cast_rule():
    def rewrite(node, module):
        original = module.node_map[node.inputs[0]]
        converted = original
        # A packet Bitcast can factor lanes but must preserve dense scalar
        # order. Only reinterpret FP32 packets, never integer bit patterns.
        while converted.op == "tensors.bitcast":
            before = module.node_map[converted.inputs[0]]
            a, b = tensor_of(before.type), tensor_of(converted.type)
            if (scalar_dtype(a.dtype) != DType.FLOAT32 or scalar_dtype(b.dtype) != DType.FLOAT32
                    or a.layout != TensorLayout() or b.layout != TensorLayout()):
                return None
            converted = before
        if converted.op not in {"tensors.cast", "ntt.vectorized_cast"}:
            return None
        value = module.node_map[converted.inputs[0]]
        source, target, observed = map(lambda n: tensor_of(n.type), (value, converted, original))
        if (scalar_dtype(source.dtype) != DType.BFLOAT16 or scalar_dtype(target.dtype) != DType.FLOAT32
                or scalar_dtype(observed.dtype) != DType.FLOAT32
                or source.layout != TensorLayout() or target.layout != TensorLayout()
                or source.rank != observed.rank or source.rank == 0):
            return None
        if converted.op == "ntt.vectorized_cast":
            ia, oa = cast_vector_axes(source.dtype.lanes, target.dtype.lanes,
                                     converted.attrs["vectorize_axes"], source.rank)
            if any(axis != source.rank - 1 for axis in (*ia, *oa)):
                return None
        elif source.shape != target.shape or getattr(source.dtype, "lanes", ()) != getattr(target.dtype, "lanes", ()):
            return None
        # NormStats unpacks every lane on the final axis. Prefix/head
        # coordinates are untouched and the entire suffix cardinality agrees.
        axis = normalize_axis(node.attrs["axis"], observed.rank)
        if source.shape[:-1] != observed.shape[:-1]:
            return None
        sl, tl = prod(getattr(source.dtype, "lanes", ())), prod(getattr(observed.dtype, "lanes", ()))
        if source.shape[-1] * sl != observed.shape[-1] * tl:
            return None
        if source.shape[:axis] != observed.shape[:axis]:
            return None
        helpers = ()
        # Preserve the selected statistics packet geometry/reduction ordering.
        # BF16 packet regrouping is a dense storage view, never a conversion.
        lanes = getattr(observed.dtype, "lanes", ())
        wanted = VectorType(DType.BFLOAT16, lanes) if lanes else DType.BFLOAT16
        if source.dtype != wanted:
            value = make_node("tensors.bitcast", f"{node.id}.input_packet", (value,), {"dtype": wanted},
                              {"formed_by": "FuseNormStatsInputCast"})
            helpers = (value,)
        replacement = make_node(node.op, node.id, (value,), node.attrs,
                                {**node.metadata, "formed_by": "FuseNormStatsInputCast"})
        if replacement.type != node.type:
            return None
        return RewriteResult(replacement, helpers)
    return RewriteRule("FuseNormStatsInputCast", lambda node, _: node.op == "nn.norm_stats", rewrite,
                       root_ops=("nn.norm_stats",))


__all__ = ["fuse_norm_stats_input_cast_rule"]
