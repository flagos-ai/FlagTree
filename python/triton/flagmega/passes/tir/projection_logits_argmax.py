# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Legality analysis for projection/cast/argmax epilogue fusion."""

from __future__ import annotations

from dataclasses import dataclass

from triton.flagmega.ir import DType, IRModule, Node, TensorType, logical_type


@dataclass(frozen=True)
class ProjectionLogitsArgmaxMatch:
    producer: str
    adapters: tuple[str, ...]
    logits: str
    sampler: str


_TRANSPARENT_ADAPTERS = frozenset({
    "distributed.boxing",
    "distributed.sharded_view",
})


def find_projection_logits_argmax_matches(
    module: IRModule,
) -> dict[str, ProjectionLogitsArgmaxMatch]:
    """Find exact ``bf16 matmul -> f32 cast -> greedy_sample`` boundaries.

    The FP32 cast must be a public function result and have exactly one graph
    consumer, the sampler.  The projection itself must be used only by the
    cast.  These conditions let codegen publish the same logits while folding
    both the cast and local argmax into the projection owner's epilogue.
    """

    users: dict[str, list[Node]] = {node.id: [] for node in module.nodes}
    for node in module.nodes:
        for input_id in node.inputs:
            users.setdefault(input_id, []).append(node)
    public_outputs = {
        output_id
        for function in module.functions
        for output_id in function.outputs
    }
    matches: dict[str, ProjectionLogitsArgmaxMatch] = {}
    for producer in module.nodes:
        producer_type = logical_type(producer.type)
        if (
            producer.op not in {"math.matmul", "math.packed_dense_matmul"}
            or not isinstance(producer_type, TensorType)
            or producer_type.dtype != DType.BFLOAT16
        ):
            continue
        current = producer
        adapters: list[str] = []
        while True:
            projection_users = users.get(current.id, ())
            if len(projection_users) != 1:
                break
            candidate = projection_users[0]
            if candidate.op not in _TRANSPARENT_ADAPTERS:
                break
            if (
                len(candidate.inputs) != 1
                or logical_type(candidate.type) != logical_type(current.type)
            ):
                break
            adapters.append(candidate.id)
            current = candidate
        else:  # pragma: no cover - the loop always exits through a graph edge
            projection_users = ()
        if len(projection_users) != 1:
            continue
        cast = projection_users[0]
        cast_type = logical_type(cast.type)
        if (
            cast.op != "tensors.cast"
            or not isinstance(cast_type, TensorType)
            or cast_type.dtype != DType.FLOAT32
            or cast_type.shape != producer_type.shape
            or cast.id not in public_outputs
        ):
            continue
        cast_users = users.get(cast.id, ())
        if len(cast_users) != 1:
            continue
        sampler = cast_users[0]
        if sampler.op != "nn.greedy_sample" or sampler.inputs != (cast.id,):
            continue
        matches[producer.id] = ProjectionLogitsArgmaxMatch(
            producer=producer.id,
            adapters=tuple(adapters),
            logits=cast.id,
            sampler=sampler.id,
        )
    return matches


__all__ = [
    "ProjectionLogitsArgmaxMatch",
    "find_projection_logits_argmax_matches",
]
