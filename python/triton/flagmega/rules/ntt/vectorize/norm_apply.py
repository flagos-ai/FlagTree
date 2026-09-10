# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Vectorize normalization apply while statistics remain scalar FP32."""

from __future__ import annotations

from triton.flagmega.ir import DType, IRModule, Node, TensorType, VectorType
from triton.flagmega.ir.ops.nn._norm import normalize_axis
from triton.flagmega.ir.ops.nn.norm_apply import NormApply
from triton.flagmega.rules import RewriteResult
from triton.flagmega.rules.ntt.vectorize.base import VectorizeCandidate
from triton.flagmega.rules.ntt.vectorize.utility import (
    finish_vector_result,
    padding_for,
    prepare_packed_input,
    internal_metadata,
)


class VectorizeNormApply:
    name = "VectorizeNormApply"
    op_names = frozenset({"nn.norm_apply"})
    # Value/output coordinates agree; scale/bias follow this choice during
    # rewriting, while scalar statistics do not belong to the layout region.
    layout_input_indices = (0,)

    def __init__(self, *, lane_bytes: int = 16) -> None:
        self.lane_bytes = lane_bytes

    def candidates(self, node: Node, module: IRModule) -> tuple[VectorizeCandidate, ...]:
        if node.op != "nn.norm_apply" or not node.effect.is_pure:
            return ()
        value = module.node_map[node.inputs[0]].type
        scale = module.node_map[node.inputs[2]].type
        bias = module.node_map[node.inputs[3]].type
        if (
            not isinstance(value, TensorType)
            or not isinstance(scale, TensorType)
            or not isinstance(bias, TensorType)
            or value.rank == 0
            or isinstance(value.dtype, VectorType)
            or not isinstance(value.dtype, DType)
            or value.dtype not in {DType.BFLOAT16, DType.FLOAT32}
            or scale.dtype not in {DType.BFLOAT16, DType.FLOAT32}
            or bias.dtype not in {DType.BFLOAT16, DType.FLOAT32}
        ):
            return ()
        axis = normalize_axis(int(node.attrs["axis"]), value.rank)
        vector_axis = value.rank - 1
        parameter_axis = vector_axis - axis
        if parameter_axis < 0 or scale.rank <= parameter_axis or bias.rank <= parameter_axis:
            return ()
        lane = self.lane_bytes // value.dtype.itemsize
        # Lanes describe the same logical suffix coordinates, not equal byte
        # widths. NormApply already defines FP32 arithmetic with independent
        # value/scale/bias dtypes, so pack each operand in its own element type.
        if lane <= 1:
            return ()
        value_pads = padding_for(value, (vector_axis,), (lane,))
        scale_pads = padding_for(scale, (parameter_axis,), (lane,))
        bias_pads = padding_for(bias, (parameter_axis,), (lane,))
        if value_pads is None or scale_pads is None or bias_pads is None:
            return ()
        if any(value_pads[index] for index in range(axis, value.rank)):
            return ()
        if any(scale_pads) or any(bias_pads):
            return ()
        return (VectorizeCandidate(
            "vectorization.norm_apply.reduction_axis",
            self.name,
            (vector_axis,),
            (lane,),
            {
                "value_axis": vector_axis,
                "parameter_axis": parameter_axis,
                "lane": lane,
                "vector_bytes": self.lane_bytes,
            },
            {
                "value_padding": list(value_pads),
                "scale_padding": list(scale_pads),
                "bias_padding": list(bias_pads),
                "egraph_equivalent": True,
            },
        ),)

    def rewrite(self, node: Node, module: IRModule, candidate: VectorizeCandidate) -> RewriteResult:
        value_axes = candidate.axes
        lanes = candidate.lanes
        parameter_axes = (int(candidate.parameters["parameter_axis"]),) * len(lanes)
        prefix: list[Node] = []
        value_helpers, value = prepare_packed_input(
            module.node_map[node.inputs[0]],
            axes=value_axes,
            lanes=lanes,
            root_id=node.id,
            input_index=0,
            forced_pad=tuple(candidate.facts["value_padding"]),
        )
        prefix.extend(value_helpers)
        scale_helpers, scale = prepare_packed_input(
            module.node_map[node.inputs[2]],
            axes=parameter_axes,
            lanes=lanes,
            root_id=node.id,
            input_index=2,
            forced_pad=tuple(candidate.facts["scale_padding"]),
        )
        prefix.extend(scale_helpers)
        bias_helpers, bias = prepare_packed_input(
            module.node_map[node.inputs[3]],
            axes=parameter_axes,
            lanes=lanes,
            root_id=node.id,
            input_index=3,
            forced_pad=tuple(candidate.facts["bias_padding"]),
        )
        prefix.extend(bias_helpers)
        stats = module.node_map[node.inputs[1]]
        attrs = dict(node.attrs)
        compute_type = NormApply.infer_type((value, stats, scale, bias), attrs)
        compute = Node(
            f"{node.id}.vectorized.compute",
            "nn.norm_apply",
            (value.id, stats.id, scale.id, bias.id),
            compute_type,
            attrs=attrs,
            metadata=internal_metadata(node.id, "compute"),
        )
        return finish_vector_result(
            node,
            compute,
            prefix,
            axes=value_axes,
            pads=tuple(candidate.facts["value_padding"]),
            candidate=candidate,
        )


__all__ = ["VectorizeNormApply"]
