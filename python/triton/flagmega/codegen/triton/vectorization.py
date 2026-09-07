# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Typed AutoVectorize decisions lowered to executable Triton schedules.

Most semantic vector expressions are lowered to schedule contracts after
AutoDistribution. Operations such as ``ntt.packed_matmul`` consume their typed
Pack/Unpack boundaries directly and retain both the physical ``VectorType`` and
this schedule contract. Unlike the old metadata hint, this object is part of
the selected TIR ABI and must determine physical indexing/tile parameters in
codegen.
"""

from __future__ import annotations

from math import prod
from typing import Mapping

from triton.flagmega.errors import CodegenError, IRVerificationError
from triton.flagmega.ir import Node, TensorType, VectorType, logical_type
from triton.flagmega.ir.ops.tensors.pack import normalize_axes


SCALAR_VECTORIZATION = {"kind": "scalar", "axes": (), "lanes": (), "lane_count": 1}


def vectorization_contract(node: Node) -> dict[str, object]:
    """Return the normalized, semantic vector-layout contract for ``node``."""

    candidate = node.metadata.get("selected_vectorization")
    # Explicit/propagated VectorizedCast is physical typed IR, not a scalar
    # equality witness. It can be authored without an AutoVectorize choice.
    # Its operation attributes and result type own the lane mapping.
    if node.op == "ntt.vectorized_cast":
        value_type = logical_type(node.type)
        if not isinstance(value_type, TensorType) or not isinstance(value_type.dtype, VectorType):
            raise IRVerificationError("VectorizedCast needs a vector result type.", node_id=node.id)
        axes = normalize_axes(tuple(int(axis) for axis in node.attrs["vectorize_axes"]), value_type.rank)
        lanes = tuple(value_type.dtype.lanes)
        return {"kind": "axes", "axes": axes, "lanes": lanes, "lane_count": prod(lanes),
                "source_candidate": str(candidate) if candidate is not None else "typed-ir"}
    if candidate is None:
        return dict(SCALAR_VECTORIZATION)
    axes = tuple(int(value) for value in node.metadata.get("selected_vector_axes", ()))
    lanes = tuple(int(value) for value in node.metadata.get("selected_vector_lanes", ()))
    if not axes or len(axes) != len(lanes) or any(value <= 1 for value in lanes):
        raise IRVerificationError(
            f"Node {node.id!r} has an incomplete selected vectorization contract: "
            f"axes={axes}, lanes={lanes}.",
            node_id=node.id,
        )
    value_type = logical_type(node.type)
    if not isinstance(value_type, TensorType):
        raise IRVerificationError(
            f"Node {node.id!r} attaches tensor vectorization to {value_type!r}.",
            node_id=node.id,
        )
    normalized_axes = tuple(axis + value_type.rank if axis < 0 else axis for axis in axes)
    if any(axis < 0 or axis >= value_type.rank for axis in normalized_axes):
        raise IRVerificationError(
            f"Node {node.id!r} vectorization axes {axes} exceed rank {value_type.rank}.",
            node_id=node.id,
        )
    if node.op in {"math.matmul", "math.packed_dense_matmul", "ntt.packed_matmul"}:
        kind = "output_axis"
        if normalized_axes != (value_type.rank - 1,):
            raise IRVerificationError(
                f"Triton dense MatMul currently lowers only its output axis, got "
                f"{normalized_axes} on {node.id!r}.",
                node_id=node.id,
            )
    elif node.op in {
        "nn.norm_apply",
        "nn.rms_norm",
        "ntt.gather_reduce_norm_apply",
    }:
        kind = "reduction_axis"
        if normalized_axes != (value_type.rank - 1,):
            raise IRVerificationError(
                f"Triton RMSNorm currently lowers only its last reduction axis, got "
                f"{normalized_axes} on {node.id!r}.",
                node_id=node.id,
            )
    else:
        # Elementwise execution is layout-polymorphic. The physical schedule
        # traverses the selected packed axes (including padding masks) instead
        # of constraining the semantic rule to a last-axis shortcut.
        kind = "axes"
    return {
        "kind": kind,
        "axes": normalized_axes,
        "lanes": lanes,
        "lane_count": prod(lanes),
        "source_candidate": str(candidate),
    }


def configured_vector_schedule(
    contract: Mapping[str, object],
    *,
    lowering: str,
    **physical: object,
) -> dict[str, object]:
    """Build the serializable proof that a target candidate consumes a layout."""

    return {
        "contract": dict(contract),
        "lowering": str(lowering),
        "physical": dict(physical),
    }


def consumed_vector_contracts(
    module,
    requirements: Mapping[str, tuple[str, str]],
) -> dict[str, object] | None:
    """Return contracts only when every fused semantic boundary is vector-legal.

    ``requirements`` maps a fused role to ``(node_id, required_kind)``.  A
    scalar agent choice therefore disables the fused implementation instead
    of being silently ignored by it.
    """

    values: dict[str, object] = {}
    for role, (node_id, required_kind) in requirements.items():
        contract = vectorization_contract(module.node_map[node_id])
        if contract["kind"] != required_kind:
            return None
        values[str(role)] = contract
    return values


def require_vector_schedule(parameters: Mapping[str, object], node_id: str) -> Mapping[str, object]:
    """Reject the historical metadata-only vectorization failure mode."""

    value = parameters.get("vector_schedule")
    if not isinstance(value, Mapping):
        raise CodegenError(
            f"Selected TIR node {node_id!r} has no executable vector_schedule; "
            "AutoVectorize decisions may not survive as metadata-only hints."
        )
    contract = value.get("contract")
    physical = value.get("physical")
    if not isinstance(contract, Mapping) or not isinstance(physical, Mapping):
        raise CodegenError(
            f"Selected TIR node {node_id!r} has a malformed vector_schedule."
        )
    return value


__all__ = [
    "SCALAR_VECTORIZATION",
    "configured_vector_schedule",
    "consumed_vector_contracts",
    "require_vector_schedule",
    "vectorization_contract",
]
