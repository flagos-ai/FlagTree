# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Target-neutral executable distribution contracts for Triton providers."""

from __future__ import annotations

from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import (
    DistributedType,
    IRModule,
    IRType,
    Node,
    ReduceOp,
    SBPBroadCast,
    SBPSplit,
    TupleType,
)
from triton.flagmega.ir.ops.tensors._k_major import parse_k_major_layout


def dense_matmul_distribution_contract(
    node: Node, module,
) -> dict[str, object]:
    """Describe the exact executable distribution schedule of a dense MatMul."""

    output = node.type
    if not isinstance(output, DistributedType) or output.partial is None:
        return {"kind": "canonical"}
    if node.op not in {
        "math.matmul", "math.packed_dense_matmul", "ntt.packed_matmul",
    } or len(node.inputs) < 2:
        raise CodegenError(
            f"Partial dense MatMul contract is unsupported for {node.op!r} on {node.id!r}."
        )

    partial = output.partial
    if partial.reduce_op is not ReduceOp.SUM:
        raise CodegenError(
            f"Dense MatMul {node.id!r} requires Sum partial output, got "
            f"{partial.reduce_op.value}."
        )
    lhs = module.node_map[node.inputs[0]].type
    rhs = module.node_map[node.inputs[1]].type
    if not isinstance(lhs, DistributedType) or not isinstance(rhs, DistributedType):
        raise CodegenError(
            f"Dense MatMul {node.id!r} partial output requires distributed operands."
        )
    if lhs.placement != output.placement or rhs.placement != output.placement:
        raise CodegenError(
            f"Dense MatMul {node.id!r} operands and output require one placement."
        )

    if node.op == "math.matmul":
        transpose_a = bool(node.attrs.get("transpose_a", False))
        transpose_b = bool(node.attrs.get("transpose_b", False))
        lhs_m_axis = lhs.tensor.rank - 1 if transpose_a else lhs.tensor.rank - 2
        lhs_k_axis = lhs.tensor.rank - 2 if transpose_a else lhs.tensor.rank - 1
        rhs_k_axis = rhs.tensor.rank - 1 if transpose_b else rhs.tensor.rank - 2
        rhs_n_axis = rhs.tensor.rank - 2 if transpose_b else rhs.tensor.rank - 1
    elif node.op == "math.packed_dense_matmul":
        packed_layout = node.attrs.get("packed_layout")
        if not isinstance(packed_layout, str):
            raise CodegenError(
                f"Partial dense MatMul contract is unsupported for packed node "
                f"{node.id!r} without an explicit packed_layout."
            )
        _, _, mesh_interleaved = parse_k_major_layout(packed_layout)
        lhs_m_axis = 0
        lhs_k_axis = lhs.tensor.rank - 1
        rhs_k_axis = 0
        rhs_n_axis = 2 if mesh_interleaved else 1
    elif node.op == "ntt.packed_matmul":
        lhs_m_axis = 0
        lhs_k_axis = lhs.tensor.rank - 1
        rhs_k_axis = 0
        rhs_n_axis = 1
    else:
        raise CodegenError(
            f"Partial dense MatMul contract is unsupported for {node.op!r} on {node.id!r}."
        )
    axes = tuple(partial.axes)
    _require_reduction_split(lhs, lhs_k_axis, axes, node.id, "lhs")
    _require_reduction_split(rhs, rhs_k_axis, axes, node.id, "rhs")
    output_m_axes = _split_axes(output.axis_policies[0])
    lhs_m_axes = _split_axes(lhs.axis_policies[lhs_m_axis])
    if output_m_axes != lhs_m_axes:
        raise CodegenError(
            f"Dense MatMul {node.id!r} output-M and lhs-M splits must use the "
            "same placement axes."
        )
    output_n_axes = _split_axes(output.axis_policies[-1])
    rhs_output_axes = _split_axes(rhs.axis_policies[rhs_n_axis])
    if output_n_axes != rhs_output_axes:
        raise CodegenError(
            f"Dense MatMul {node.id!r} output-N and rhs-N splits must use the "
            "same placement axes."
        )
    output_split_axes = tuple(dict.fromkeys((*output_m_axes, *output_n_axes)))
    if set(output_split_axes) & set(axes):
        raise CodegenError(
            f"Dense MatMul {node.id!r} output and reduction placement axes must be disjoint."
        )
    owner_count = 1
    for axis in axes:
        owner_count *= output.placement.hierarchy[axis]
    return {
        "kind": (
            "output_reduction_split" if output_split_axes else "reduction_split"
        ),
        "partial_axes": axes,
        "output_axes": output_split_axes,
        "owner_count": owner_count,
        "lhs_k_axis": lhs_k_axis,
        "rhs_k_axis": rhs_k_axis,
    }


def _require_reduction_split(
    value: DistributedType,
    reduction_axis: int,
    partial_axes: tuple[int, ...],
    node_id: str,
    role: str,
) -> None:
    """Verify only the semantic K relation needed to produce a partial.

    Contiguous versus block-cyclic mapping is intentionally absent here.  A
    normal MatMul microkernel receives the divided dense local tensors; the
    buffer access renderer owns the mapping from those local coordinates to
    physical storage.
    """

    if value.partial is not None:
        raise CodegenError(
            f"Dense MatMul {node_id!r} {role} cannot itself be partial."
        )
    policy = value.axis_policies[reduction_axis]
    if (
        not isinstance(policy, SBPSplit)
        or tuple(policy.hierarchy_axes) != partial_axes
    ):
        raise CodegenError(
            f"Dense MatMul {node_id!r} {role} reduction axis must be split over "
            f"placement axes {partial_axes}."
        )


def _split_axes(policy) -> tuple[int, ...]:
    if isinstance(policy, SBPBroadCast):
        return ()
    if isinstance(policy, SBPSplit):
        return tuple(policy.hierarchy_axes)
    raise CodegenError("Dense MatMul output policies must be broadcast or split.")


def is_non_replicated_type(value: IRType) -> bool:
    """Return whether an IR type carries split or partial runtime storage.

    Candidate identifiers are deliberately not part of this decision.  Auto
    Distribution providers are free to use descriptive ids such as
    ``layout_0001``; the materialized type is the semantic source of truth.
    """

    if isinstance(value, DistributedType):
        return value.partial is not None or any(
            not isinstance(policy, SBPBroadCast)
            for policy in value.axis_policies
        )
    if isinstance(value, TupleType):
        return any(is_non_replicated_type(field) for field in value.fields)
    return False


def non_replicated_value_ids(module: IRModule) -> tuple[str, ...]:
    return tuple(node.id for node in module.nodes if is_non_replicated_type(node.type))


def requires_distributed_grid(module: IRModule) -> bool:
    return bool(non_replicated_value_ids(module))


__all__ = [
    "dense_matmul_distribution_contract",
    "is_non_replicated_type",
    "non_replicated_value_ids",
    "requires_distributed_grid",
]
