# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Vectorize additive normalization statistics on the reduction suffix."""

from __future__ import annotations

from triton.flagmega.ir import DType, IRModule, Node, TensorType, VectorType
from triton.flagmega.ir.ops.nn._norm import normalize_axis
from triton.flagmega.ir.ops.nn.norm_stats import NormStats
from triton.flagmega.rules import RewriteResult
from triton.flagmega.rules.ntt.vectorize.base import VectorizeCandidate
from triton.flagmega.rules.ntt.vectorize.utility import (
    padding_for,
    prepare_packed_input,
    root_metadata,
)


class VectorizeNormStats:
    name = "VectorizeNormStats"
    op_names = frozenset({"nn.norm_stats"})

    def __init__(self, *, lane_bytes: int = 16) -> None:
        self.lane_bytes = lane_bytes

    def candidates(self, node: Node, module: IRModule) -> tuple[VectorizeCandidate, ...]:
        if node.op != "nn.norm_stats" or not node.effect.is_pure:
            return ()
        value = module.node_map[node.inputs[0]].type
        if (
            not isinstance(value, TensorType)
            or value.rank == 0
            or isinstance(value.dtype, VectorType)
            or not isinstance(value.dtype, DType)
            or value.dtype not in {DType.BFLOAT16, DType.FLOAT32}
        ):
            return ()
        axis = normalize_axis(int(node.attrs["axis"]), value.rank)
        vector_axis = value.rank - 1
        if vector_axis < axis:
            return ()
        lane = self.lane_bytes // value.dtype.itemsize
        if lane <= 1:
            return ()
        pads = padding_for(value, (vector_axis,), (lane,))
        # Padding a reduction suffix changes its normalization cardinality.
        # nncase accepts this rule only when the vector lane divides exactly.
        if pads is None or any(pads[index] for index in range(axis, value.rank)):
            return ()
        return (VectorizeCandidate(
            "vectorization.norm_stats.reduction_axis",
            self.name,
            (vector_axis,),
            (lane,),
            {"axis": vector_axis, "lane": lane, "vector_bytes": self.lane_bytes},
            {"padding": list(pads), "egraph_equivalent": True},
        ),)

    def rewrite(self, node: Node, module: IRModule, candidate: VectorizeCandidate) -> RewriteResult:
        axes = candidate.axes
        lanes = candidate.lanes
        helpers, packed = prepare_packed_input(
            module.node_map[node.inputs[0]],
            axes=axes,
            lanes=lanes,
            root_id=node.id,
            input_index=0,
            forced_pad=tuple(candidate.facts["padding"]),
        )
        attrs = dict(node.attrs)
        replacement = Node(
            node.id,
            "nn.norm_stats",
            (packed.id,),
            NormStats.infer_type((packed,), attrs),
            node.effect,
            attrs,
            root_metadata(node, candidate),
        )
        return RewriteResult(replacement, helpers)


__all__ = ["VectorizeNormStats"]
