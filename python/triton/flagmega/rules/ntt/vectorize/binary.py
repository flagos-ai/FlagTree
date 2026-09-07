# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Vectorize scalar binary expressions into VectorizedBinary candidates."""

from __future__ import annotations

from triton.flagmega.ir import DType, IRModule, Node, TensorType
from triton.flagmega.rules import RewriteResult
from triton.flagmega.rules.ntt.vectorize.base import VectorizeCandidate
from triton.flagmega.rules.ntt.vectorize.utility import finish_vector_result, generate_axis_candidates, padding_for, prepare_packed_input, internal_metadata


class VectorizeBinary:
    name = "VectorizeBinary"
    op_names = frozenset({"math.add", "math.mul"})

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
            node.op not in {"math.add", "math.mul"} or not node.effect.is_pure
            or not isinstance(node.type, TensorType)
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
        prefix: list[Node] = []
        packed: list[Node] = []
        for index, input_id in enumerate(node.inputs):
            helpers, value = prepare_packed_input(
                module.node_map[input_id],
                axes=candidate.axes,
                lanes=candidate.lanes,
                root_id=node.id,
                input_index=index,
                forced_pad=pads,
            )
            prefix.extend(helpers)
            packed.append(value)
        compute = Node(
            f"{node.id}.vectorized.compute",
            "math.vectorized_binary",
            tuple(value.id for value in packed),
            packed[0].type,
            attrs={"binary_op": node.op.removeprefix("math.")},
            metadata=internal_metadata(node.id, "compute"),
        )
        return finish_vector_result(
            node,
            compute,
            prefix,
            axes=candidate.axes,
            pads=pads,
            candidate=candidate,
        )


__all__ = ["VectorizeBinary"]
