# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Dense MatMul+GLU candidates from a target-owned implementation catalog."""

from __future__ import annotations

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir import (
    DistributedType,
    Node,
    TensorType,
    VectorType,
    logical_type,
)
from triton.flagmega.ir.distributed_type import SBPBroadCast, SBPSplit, local_shape
from triton.flagmega.ir.ops.tensors._k_major import parse_k_major_layout
from triton.flagmega.codegen.triton.vectorization import consumed_vector_contracts

from .core import TritonCandidateContext, TritonCandidateProposal


class DenseMatmulGluCandidateProvider:
    op_names = frozenset({"nn.dense_matmul_glu", "nn.packed_dense_matmul_glu"})

    def propose(
        self,
        node: Node,
        context: TritonCandidateContext,
    ) -> TritonCandidateProposal | None:
        packed = node.op == "nn.packed_dense_matmul_glu"
        packed_layout = str(node.attrs.get("packed_layout", ""))
        contract = {
            "input_kind": "packed" if packed else "logical",
            "fusion": "none",
        }
        if packed:
            contract["packed_layout"] = packed_layout
        source_type = context.module.node_map[node.inputs[0]].type
        reduction_extent = _local_scalar_last_axis_extent(source_type)

        def applicable(implementation) -> bool:
            if implementation.parameters.get("descriptor_kind") == "table" and any(
                not isinstance(context.module.node_map[value].type, DistributedType)
                for value in node.inputs[1:3]
            ):
                return False
            block_k = implementation.parameters.get("block_k")
            tile_n = implementation.parameters.get("tile_n")
            required_reduction_extent = implementation.contract.get(
                "required_local_reduction_extent"
            )
            if (
                required_reduction_extent is not None
                and reduction_extent != required_reduction_extent
            ):
                return False
            lhs_stage_extent = implementation.parameters.get("lhs_stage_extent")
            if lhs_stage_extent is not None and (
                reduction_extent is None
                or not isinstance(lhs_stage_extent, int)
                or isinstance(lhs_stage_extent, bool)
                or lhs_stage_extent < reduction_extent
                or lhs_stage_extent & (lhs_stage_extent - 1)
            ):
                return False
            if implementation.contract.get("requires_full_reduction_tiles", False):
                if (
                    reduction_extent is None
                    or not isinstance(block_k, int)
                    or reduction_extent % block_k
                ):
                    return False
            return (
                not implementation.contract.get(
                    "requires_contiguous_local_tiles", False
                )
                or _supports_packed_contiguous_descriptor_tiles(
                    node,
                    context.module,
                    block_k=block_k,
                    tile_n=tile_n,
                    allow_output_tail=implementation.contract.get("supports_masked_output_tiles", False),
                )
            )

        variants = tuple(
            context.configure_implementation(
                implementation,
                semantic_parameters=(
                    {"packed_layout": packed_layout} if packed else None
                ),
            )
            for implementation in context.implementations(
                "dense_matmul_glu", **contract
            )
            if applicable(implementation)
        )
        norm_match = context.norm_consumer_matches.get(node.id)
        norm_vectors = (
            None
            if norm_match is None
            else consumed_vector_contracts(
                context.module,
                {"input_norm": (norm_match.producer, "reduction_axis")},
            )
        )
        if (
            packed
            and norm_match is not None
            and norm_vectors is not None
            and context.is_reusable(node)
        ):
            variants += tuple(
                context.configure_implementation(
                    implementation,
                    semantic_parameters={
                        "packed_layout": packed_layout,
                        "input_norm_fusion": "consumer_local_staging",
                        "input_norm": norm_match.producer,
                        "input_norm_weight": norm_match.weight,
                        "input_norm_epsilon": norm_match.epsilon,
                        "input_norm_weight_bias": norm_match.weight_bias,
                        **(
                            {"input_norm_stats": norm_match.stats}
                            if norm_match.stats is not None else {}
                        ),
                        "consumed_vector_contracts": norm_vectors,
                    },
                    facts={
                        "requires_reusable_function": True,
                        "norm_result_single_use": True,
                        "materializes_input_norm": False,
                    },
                )
                for implementation in context.implementations(
                    "dense_matmul_glu",
                    input_kind="packed",
                    fusion="input_norm",
                    packed_layout=packed_layout,
                )
                if applicable(implementation)
            )
        if not variants:
            return None
        return TritonCandidateProposal(
            variants,
            context.choose_default("dense_matmul_glu", variants),
        )


__all__ = ["DenseMatmulGluCandidateProvider"]


def _supports_packed_contiguous_descriptor_tiles(
    node: Node,
    module,
    *,
    block_k,
    tile_n,
    allow_output_tail=False,
) -> bool:
    """Prove dual packed weights expose the same affine TMA tile contract."""

    if (
        node.op != "nn.packed_dense_matmul_glu"
        or not isinstance(block_k, int)
        or isinstance(block_k, bool)
        or not isinstance(tile_n, int)
        or isinstance(tile_n, bool)
        or block_k <= 0
        or tile_n <= 0
        or len(node.inputs) < 3
    ):
        return False
    values = tuple(module.node_map[value].type for value in node.inputs[:3])
    source, gate, up = values
    result = node.type
    try:
        n_lane, k_lane, mesh_interleaved = parse_k_major_layout(
            str(node.attrs.get("packed_layout", ""))
        )
    except IRSchemaError:
        return False
    source_type = logical_type(source)
    result_type = logical_type(result)
    weight_types = tuple(logical_type(value) for value in (gate, up))
    if (
        mesh_interleaved
        or not isinstance(source_type, TensorType)
        or not isinstance(result_type, TensorType)
        or any(
            not _is_k_major_weight(value, n_lane=n_lane, k_lane=k_lane)
            for value in weight_types
        )
    ):
        return False
    for value, value_type in zip((gate, up), weight_types, strict=True):
        assert isinstance(value_type, TensorType)
        if not all(
            _axis_has_contiguous_local_order(value, axis)
            for axis in range(value_type.rank)
        ):
            return False
    if not (
        _axis_has_contiguous_local_order(source, source_type.rank - 1)
        and _axis_has_contiguous_local_order(result, result_type.rank - 1)
    ):
        return False
    local_k = _local_scalar_last_axis_extent(source)
    local_n = _local_scalar_last_axis_extent(result)
    return (
        local_k is not None
        and local_n is not None
        and local_k >= block_k
        and local_n > 0
        and local_k % block_k == 0
        and (allow_output_tail or (local_n >= tile_n and local_n % tile_n == 0))
    )


def _is_k_major_weight(value, *, n_lane: int, k_lane: int) -> bool:
    if not isinstance(value, TensorType):
        return False
    if isinstance(value.dtype, VectorType):
        return (
            value.rank == 2
            and bool(value.dtype.lanes)
            and value.dtype.lanes[0] == n_lane
            and value.dtype.lane_count == n_lane * k_lane
        )
    return (
        value.rank == 4
        and value.shape[-2].is_fixed
        and value.shape[-1].is_fixed
        and value.shape[-2].fixed_value * value.shape[-1].fixed_value
        == n_lane * k_lane
    )


def _axis_has_contiguous_local_order(value, axis: int) -> bool:
    if not isinstance(value, DistributedType):
        return True
    policy = value.axis_policies[axis]
    return isinstance(policy, SBPBroadCast) or (
        isinstance(policy, SBPSplit) and policy.is_contiguous
    )


def _local_scalar_last_axis_extent(value) -> int | None:
    value_type = logical_type(value)
    if not isinstance(value_type, TensorType):
        return None
    shape = local_shape(value) if isinstance(value, DistributedType) else value_type.shape
    if not shape or not shape[-1].is_fixed:
        return None
    lane_count = (
        value_type.dtype.lane_count
        if isinstance(value_type.dtype, VectorType)
        else 1
    )
    return shape[-1].fixed_value * lane_count
