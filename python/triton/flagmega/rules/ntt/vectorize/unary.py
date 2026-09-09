# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Vectorize scalar unary expressions into VectorizedUnary candidates."""

from __future__ import annotations

from triton.flagmega.ir import DType, IRModule, Node, TensorType
from triton.flagmega.ir.ops.math.vectorized_unary import VectorizedUnary
from triton.flagmega.rules import RewriteResult
from triton.flagmega.rules.ntt.vectorize.base import VectorizeCandidate
from triton.flagmega.rules.ntt.vectorize.utility import finish_vector_result, generate_axis_candidates, padding_for, prepare_packed_input, internal_metadata


class VectorizeUnary:
    name = "VectorizeUnary"
    op_names = frozenset(definition.op_name for definition in VectorizedUnary.scalar_definitions.values())

    def __init__(
        self,
        *,
        lane_bytes: int = 16,
        max_axes: int = 2,
    ) -> None:
        self.lane_bytes = lane_bytes
        self.max_axes = max_axes

    def candidates(self, node: Node, module: IRModule) -> tuple[VectorizeCandidate, ...]:
        if (
            node.op not in self.op_names or not node.effect.is_pure or not isinstance(node.type, TensorType)
            or node.type.dtype not in {DType.BFLOAT16, DType.FLOAT32}
        ):
            return ()
        candidates = generate_axis_candidates(
            node.type,
            rule=self.name,
            lane_bytes=self.lane_bytes,
            max_axes=self.max_axes,
        )
        return candidates

    def rewrite(self, node: Node, module: IRModule, candidate: VectorizeCandidate) -> RewriteResult:
        pads = padding_for(node.type, candidate.axes, candidate.lanes)
        assert pads is not None
        helpers, packed = prepare_packed_input(
            module.node_map[node.inputs[0]],
            axes=candidate.axes,
            lanes=candidate.lanes,
            root_id=node.id,
            input_index=0,
            forced_pad=pads,
        )
        compute = Node(
            f"{node.id}.vectorized.compute",
            "math.vectorized_unary",
            (packed.id,),
            packed.type,
            attrs={"unary_op": node.op.removeprefix("math.")},
            metadata=internal_metadata(node.id, "compute"),
        )
        return finish_vector_result(
            node,
            compute,
            helpers,
            axes=candidate.axes,
            pads=pads,
            candidate=candidate,
        )


__all__ = ["VectorizeUnary"]
