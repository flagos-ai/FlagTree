# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Vectorize RMSNorm on its reduction axis."""

from __future__ import annotations

from triton.flagmega.ir import DType, IRModule, Node, TensorType, VectorType
from triton.flagmega.rules import RewriteResult
from triton.flagmega.rules.ntt.vectorize.base import VectorizeCandidate
from triton.flagmega.rules.ntt.vectorize.utility import finish_vector_result, padding_for, prepare_packed_input, internal_metadata


class VectorizeRMSNorm:
    name = "VectorizeRMSNorm"
    op_names = frozenset({"nn.rms_norm"})
    layout_input_indices = (0,)

    def __init__(self, *, lane_bytes: int = 16) -> None:
        self.lane_bytes = lane_bytes

    def candidates(self, node: Node, module: IRModule) -> tuple[VectorizeCandidate, ...]:
        if node.op != "nn.rms_norm" or not node.effect.is_pure or not isinstance(node.type, TensorType):
            return ()
        value = module.node_map[node.inputs[0]].type
        weight = module.node_map[node.inputs[1]].type
        if (
            not isinstance(value, TensorType) or not isinstance(weight, TensorType)
            or value.rank < 1 or weight.rank != 1
            or isinstance(value.dtype, VectorType) or not isinstance(value.dtype, DType)
            or value.dtype != weight.dtype or not value.shape[-1].is_fixed
            or value.dtype not in {DType.BFLOAT16, DType.FLOAT32}
        ):
            return ()
        lane = self.lane_bytes // value.dtype.itemsize
        if lane <= 1:
            return ()
        value_axes = (value.rank - 1,)
        weight_axes = (0,)
        value_pads = padding_for(value, value_axes, (lane,))
        weight_pads = padding_for(weight, weight_axes, (lane,))
        assert value_pads is not None and weight_pads is not None
        return (VectorizeCandidate(
            "vectorization.norm.reduction_axis",
            self.name,
            value_axes,
            (lane,),
            {
                "value_axes": list(value_axes), "weight_axes": list(weight_axes),
                "lanes": [lane], "logical_extent": value.shape[-1].fixed_value,
                "vector_bytes": self.lane_bytes,
            },
            {
                "value_padding": list(value_pads), "weight_padding": list(weight_pads),
                "egraph_equivalent": True,
            },
        ),)

    def rewrite(self, node: Node, module: IRModule, candidate: VectorizeCandidate) -> RewriteResult:
        value_axes = tuple(candidate.parameters["value_axes"])
        weight_axes = tuple(candidate.parameters["weight_axes"])
        lanes = tuple(candidate.parameters["lanes"])
        value_helpers, value = prepare_packed_input(
            module.node_map[node.inputs[0]], axes=value_axes, lanes=lanes,
            root_id=node.id, input_index=0, forced_pad=tuple(candidate.facts["value_padding"]),
        )
        weight_helpers, weight = prepare_packed_input(
            module.node_map[node.inputs[1]], axes=weight_axes, lanes=lanes,
            root_id=node.id, input_index=1, forced_pad=tuple(candidate.facts["weight_padding"]),
        )
        compute = Node(
            f"{node.id}.vectorized.compute",
            "nn.vectorized_rms_norm",
            (value.id, weight.id),
            value.type,
            attrs={
                "value_axes": value_axes,
                "weight_axes": weight_axes,
                "logical_extent": int(candidate.parameters["logical_extent"]),
                "epsilon": float(node.attrs["epsilon"]),
                "weight_bias": float(node.attrs["weight_bias"]),
            },
            metadata=internal_metadata(node.id, "compute"),
        )
        return finish_vector_result(
            node,
            compute,
            (*value_helpers, *weight_helpers),
            axes=value_axes,
            pads=tuple(candidate.facts["value_padding"]),
            candidate=candidate,
        )


__all__ = ["VectorizeRMSNorm"]
