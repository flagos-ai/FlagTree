# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Pad/pack/slice construction shared by vectorization rules."""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations
from typing import Mapping, Sequence

from triton.flagmega.ir import DType, Node, TensorType, VectorType
from triton.flagmega.ir.dim_expr import try_div_exactly
from triton.flagmega.ir.ops.tensors.pack import Pack, axis_lane_products, normalize_axes
from triton.flagmega.ir.ops.tensors.pad import Pad
from triton.flagmega.ir.ops.tensors.slice_to_shape import SliceToShape
from triton.flagmega.ir.ops.tensors.unpack import Unpack
from triton.flagmega.rules import RewriteResult
from triton.flagmega.rules.ntt.vectorize.base import VectorizeCandidate


def generate_axis_candidates(
    value_type: TensorType,
    *,
    rule: str,
    lane_bytes: int,
    max_axes: int,
) -> tuple[VectorizeCandidate, ...]:
    if isinstance(value_type.dtype, VectorType) or not isinstance(value_type.dtype, DType) or value_type.rank == 0:
        return ()
    lane = lane_bytes // value_type.dtype.itemsize
    if lane <= 1:
        return ()
    candidates: list[VectorizeCandidate] = []
    for count in range(1, min(max_axes, value_type.rank) + 1):
        for axes in combinations(range(value_type.rank), count):
            lanes = (lane,) * len(axes)
            pads = padding_for(value_type, axes, lanes)
            if pads is None:
                continue
            if any(pads) and any(not dimension.is_fixed for dimension in value_type.shape):
                # SliceToShape deliberately stores an editable static shape;
                # dynamic padding needs a future shape-expression Slice op.
                continue
            candidate_id = "vectorization.last_axis" if axes == (value_type.rank - 1,) else (
                "vectorization.axes_" + "_".join(str(axis) for axis in axes)
            )
            candidates.append(VectorizeCandidate(
                candidate_id,
                rule,
                axes,
                lanes,
                {"axes": list(axes), "lanes": list(lanes), "vector_bytes": lane_bytes},
                {"padding": list(pads), "egraph_equivalent": True},
            ))
    return tuple(candidates)


def padding_for(value_type: TensorType, axes: Sequence[int], lanes: Sequence[int]) -> tuple[int, ...] | None:
    normalized = normalize_axes(axes, value_type.rank)
    products = axis_lane_products(lanes, normalized)
    pads = [0] * value_type.rank
    for axis, lane_product in products.items():
        dimension = value_type.shape[axis]
        if dimension.is_fixed:
            pads[axis] = (-dimension.fixed_value) % lane_product
        elif try_div_exactly(dimension, lane_product) is None:
            return None
    return tuple(pads)


def prepare_packed_input(
    value: Node,
    *,
    axes: tuple[int, ...],
    lanes: tuple[int, ...],
    root_id: str,
    input_index: int,
    pad_value: float = 0.0,
    forced_pad: tuple[int, ...] | None = None,
) -> tuple[tuple[Node, ...], Node]:
    assert isinstance(value.type, TensorType)
    pads = forced_pad if forced_pad is not None else padding_for(value.type, axes, lanes)
    if pads is None:
        raise ValueError(f"Input {value.id!r} cannot be vectorized on axes {axes}.")
    helpers: list[Node] = []
    current = value
    if any(pads):
        attrs = {"pad_end": tuple(pads), "pad_value": pad_value}
        pad_type = Pad.infer_type((current,), attrs)
        current = Node(
            f"{root_id}.vectorized.pad{input_index}",
            "tensors.pad",
            (current.id,),
            pad_type,
            attrs=attrs,
            metadata=internal_metadata(root_id, "pad"),
        )
        helpers.append(current)
    pack_attrs = {"lanes": lanes, "axes": axes}
    pack_type = Pack.infer_type((current,), pack_attrs)
    current = Node(
        f"{root_id}.vectorized.pack{input_index}",
        "tensors.pack",
        (current.id,),
        pack_type,
        attrs=pack_attrs,
        metadata=internal_metadata(root_id, "pack"),
    )
    helpers.append(current)
    return tuple(helpers), current


def finish_vector_result(
    original: Node,
    compute: Node,
    prefix: Sequence[Node],
    *,
    axes: tuple[int, ...],
    pads: tuple[int, ...],
    candidate: VectorizeCandidate,
) -> RewriteResult:
    metadata = root_metadata(original, candidate)
    compute = replace(compute, metadata={**dict(compute.metadata), **metadata})
    helpers = list(prefix)
    if not isinstance(compute.type, TensorType) or not isinstance(compute.type.dtype, VectorType):
        return RewriteResult(
            Node(original.id, compute.op, compute.inputs, original.type, original.effect, compute.attrs, metadata),
            tuple(helpers),
        )
    unpack_attrs = {"axes": axes}
    unpack_type = Unpack.infer_type((compute,), unpack_attrs)
    if any(pads):
        unpack = Node(
            f"{original.id}.vectorized.unpack",
            "tensors.unpack",
            (compute.id,),
            unpack_type,
            attrs=unpack_attrs,
            metadata=internal_metadata(original.id, "unpack"),
        )
        helpers.extend((compute, unpack))
        shape = tuple(dimension.fixed_value for dimension in original.type.shape)
        replacement = Node(
            original.id,
            "tensors.slice_to_shape",
            (unpack.id,),
            original.type,
            original.effect,
            {"shape": shape},
            metadata,
        )
    else:
        helpers.append(compute)
        replacement = Node(
            original.id,
            "tensors.unpack",
            (compute.id,),
            original.type,
            original.effect,
            unpack_attrs,
            metadata,
        )
    return RewriteResult(replacement, tuple(helpers))


def internal_metadata(root_id: str, role: str) -> Mapping[str, object]:
    return {"vectorization_internal": True, "vectorization_role": role, "vectorization_root": root_id}


def is_generated_boundary(node: Node) -> bool:
    """Whether a Pack/Unpack boundary was introduced by AutoVectorize."""

    return node.metadata.get("vectorization_internal") is True or "vectorized_from" in node.metadata


def propagation_helper_metadata(boundary: Node, root_id: str, role: str) -> Mapping[str, object]:
    """Mark helpers only when propagation is inside compiler-owned vector IR."""

    if is_generated_boundary(boundary):
        return internal_metadata(root_id, role)
    return {"vectorization_propagated": role}


def propagation_result_metadata(
    semantic: Node,
    boundary: Node,
    *,
    axes: Sequence[int],
    lanes: Sequence[int],
    rule: str,
    internal_role: str | None = None,
) -> Mapping[str, object]:
    """Describe how a propagated vector node recovers its scalar semantic op.

    Explicit user-authored vector IR stays vector IR.  Compiler-generated
    boundaries additionally carry a complete recovery recipe consumed before
    AutoDistribution and AutoPacking.
    """

    if not is_generated_boundary(boundary):
        return {**dict(boundary.metadata), "vectorization_propagated": rule}
    metadata: dict[str, object] = {
        **dict(semantic.metadata),
        "vectorized_from": semantic.op,
        "vectorization_inputs": list(semantic.inputs),
        "vectorization_attrs": dict(semantic.attrs),
        "vectorization_semantic_id": semantic.id,
        "vectorization_candidate": str(
            boundary.metadata.get("vectorization_candidate", "vectorization.propagated")
        ),
        "vector_lanes": [int(value) for value in lanes],
        "vector_axes": [int(value) for value in axes],
        "vectorization_rule": rule,
    }
    if internal_role is not None:
        metadata.update(
            internal_metadata(
                str(boundary.metadata.get("vectorization_root", boundary.id)),
                internal_role,
            )
        )
    return metadata


def root_metadata(original: Node, candidate: VectorizeCandidate) -> Mapping[str, object]:
    return {
        **dict(original.metadata),
        "vectorized_from": original.op,
        "vectorization_inputs": list(original.inputs),
        "vectorization_attrs": dict(original.attrs),
        "vectorization_candidate": candidate.id,
        "vector_lanes": list(candidate.lanes),
        "vector_axes": list(candidate.axes),
        "vectorization_rule": candidate.rule,
    }


__all__ = [
    "finish_vector_result", "generate_axis_candidates", "internal_metadata", "padding_for",
    "is_generated_boundary", "prepare_packed_input", "propagation_helper_metadata",
    "propagation_result_metadata", "root_metadata",
]
