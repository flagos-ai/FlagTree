# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Dense MatMul candidates selected from a target-owned implementation catalog."""

from __future__ import annotations

from math import prod
from functools import lru_cache
from itertools import product

from triton.flagmega.ir import (
    Candidate,
    DType,
    DistributedType,
    Node,
    NoneType,
    TensorType,
    TupleType,
    VectorType,
    logical_type,
)
from triton.flagmega.ir.distributed_type import (
    ReduceOp,
    SBPBroadCast,
    SBPSplit,
    local_shape,
)
from triton.flagmega.ir.ops.math.matmul import MatMul
from triton.flagmega.ir.local_shard import local_shard_descriptor
from triton.flagmega.ir.ops.ntt.packed_matmul import PackedMatMul
from triton.flagmega.ir.ops.tensors._k_major import k_major_layout_name
from triton.flagmega.codegen.triton.vectorization import (
    configured_vector_schedule,
    consumed_vector_contracts,
    vectorization_contract,
)
from triton.flagmega.codegen.triton.distribution import (
    dense_matmul_distribution_contract,
)

from .core import TritonCandidateContext, TritonCandidateProposal


class DenseMatmulCandidateProvider:
    op_names = frozenset({
        "math.matmul", "math.packed_dense_matmul", "ntt.packed_matmul",
    })

    def propose(
        self,
        node: Node,
        context: TritonCandidateContext,
    ) -> TritonCandidateProposal | None:
        output_type = logical_type(node.type)
        if (
            not isinstance(output_type, TensorType)
            or _scalar_dtype(output_type) != DType.BFLOAT16
        ):
            return None
        vector_contract = vectorization_contract(node)
        distribution_contract = dense_matmul_distribution_contract(
            node, context.module
        )
        if node.op in {"math.packed_dense_matmul", "ntt.packed_matmul"}:
            return self._propose_packed(
                node, context, vector_contract, distribution_contract
            )
        return self._propose_logical(
            node, context, vector_contract, distribution_contract
        )

    def _propose_packed(
        self,
        node: Node,
        context: TritonCandidateContext,
        vector_contract,
        distribution_contract,
    ) -> TritonCandidateProposal | None:
        output_type = logical_type(node.type)
        assert isinstance(output_type, TensorType)
        input_type = logical_type(context.module.node_map[node.inputs[0]].type)
        if not isinstance(input_type, TensorType):
            return None
        output_dimension = output_type.shape[-1]
        reduction_dimension = input_type.shape[-1]
        output_extent = (
            output_dimension.fixed_value * _last_axis_lane_count(output_type)
            if output_dimension.is_fixed else None
        )
        reduction_extent = (
            reduction_dimension.fixed_value if reduction_dimension.is_fixed else None
        )
        if node.op == "ntt.packed_matmul":
            weight_type = logical_type(context.module.node_map[node.inputs[1]].type)
            if (
                not isinstance(weight_type, TensorType)
                or not isinstance(weight_type.dtype, VectorType)
                or len(weight_type.dtype.lanes) != 3
            ):
                return None
            n_vector, k_pack, k_vector = weight_type.dtype.lanes
            packed_layout = k_major_layout_name(
                n_vector, k_pack * k_vector
            )
        else:
            packed_layout = str(node.attrs["packed_layout"])

        def applicable(epilogue: str):
            return tuple(
                implementation
                for implementation in context.implementations(
                    "dense_matmul",
                    input_kind="packed",
                    epilogue=epilogue,
                    packed_layout=packed_layout,
                    vectorization_kind=vector_contract["kind"],
                    distribution_kind=distribution_contract["kind"],
                )
                if output_extent is not None
                and reduction_extent is not None
                and (
                    implementation.parameters.get("descriptor_kind") != "table"
                    or isinstance(context.module.node_map[node.inputs[1]].type, DistributedType)
                )
                and _supports_lhs_staging(
                    implementation, context.module.node_map[node.inputs[0]].type
                )
                and (
                    implementation.contract.get("supports_masked_tiles", False)
                    or (
                        (implementation.contract.get("supports_masked_output_tiles", False)
                         or output_extent % int(implementation.parameters["tile_n"]) == 0)
                        and reduction_extent % int(implementation.parameters["block_k"]) == 0
                    )
                )
                and (
                    not (
                        implementation.contract.get("requires_contiguous_local_tiles", False)
                        or implementation.contract.get("requires_affine_local_tiles", False)
                    )
                    or _supports_packed_descriptor_tiles(
                        node,
                        context.module,
                        block_k=implementation.parameters.get("block_k"),
                        tile_n=implementation.parameters.get("tile_n"),
                        affine_rhs=implementation.contract.get("requires_affine_local_tiles", False),
                        allow_output_tail=implementation.contract.get("supports_masked_output_tiles", False),
                    )
                )
            )

        variants: tuple[Candidate, ...] = tuple(
            _configure_dense_implementation(
                context,
                implementation,
                vector_contract,
                distribution_contract,
                semantic_parameters={"packed_layout": packed_layout},
            )
            for implementation in applicable("none")
        )
        match = (
            context.projection_norm_matches.get(node.id)
            if context.is_reusable(node)
            else None
        )
        if match is not None:
            fused_vectors = consumed_vector_contracts(
                context.module,
                {
                    "residual_add": (match.residual_add, "axes"),
                    "norm": (match.norm_consumer, "reduction_axis"),
                },
            )
        else:
            fused_vectors = None
        if match is not None and fused_vectors is not None:
            variants += tuple(
                _configure_dense_implementation(
                    context,
                    implementation,
                    vector_contract,
                    distribution_contract,
                    semantic_parameters={
                        "packed_layout": packed_layout,
                        "epilogue": "residual_norm_stats",
                        "residual_add": match.residual_add,
                        "residual_input": match.residual_input,
                        "norm_consumer": match.norm_consumer,
                        "norm_stats_partials": "cta",
                        "projection_adapters": match.adapters,
                        "consumed_vector_contracts": fused_vectors,
                    },
                    facts={
                        "requires_reusable_function": True,
                        "projection_result_single_use": True,
                        "materializes_residual": True,
                    },
                )
                for implementation in applicable("residual_norm_stats")
            )
        logits_match = context.projection_logits_matches.get(node.id)
        logits_vectors = (
            None
            if logits_match is None
            else consumed_vector_contracts(
                context.module,
                {"logits_cast": (logits_match.logits, "axes")},
            )
        )
        if logits_match is not None and logits_vectors is not None:
            variants += tuple(
                _configure_dense_implementation(
                    context,
                    implementation,
                    vector_contract,
                    distribution_contract,
                    semantic_parameters={
                        "packed_layout": packed_layout,
                        "epilogue": "cast_logits_argmax",
                        "logits_output": logits_match.logits,
                        "sampler": logits_match.sampler,
                        "consumed_vector_contracts": logits_vectors,
                    },
                    facts={
                        "projection_result_single_use": True,
                        "materializes_projection": False,
                        "materializes_fp32_logits": True,
                    },
                )
                for implementation in applicable("cast_logits_argmax")
            )
        if not variants:
            return None
        return TritonCandidateProposal(
            variants, context.choose_default("dense_matmul", variants)
        )
    def _propose_logical(
        self,
        node: Node,
        context: TritonCandidateContext,
        vector_contract,
        distribution_contract,
    ) -> TritonCandidateProposal | None:
        input_type = logical_type(context.module.node_map[node.inputs[0]].type)
        reduction_extent = (
            input_type.shape[-1].fixed_value
            if isinstance(input_type, TensorType) and input_type.shape[-1].is_fixed
            else None
        )
        local_output_extent = _local_scalar_last_axis_extent(node.type)

        def applicable(implementation) -> bool:
            required_transpose_b = implementation.contract.get("transpose_b")
            if (
                required_transpose_b is not None
                and bool(node.attrs.get("transpose_b", False))
                is not bool(required_transpose_b)
            ):
                return False
            maximum_local_output = implementation.contract.get(
                "maximum_local_output_extent"
            )
            if (
                maximum_local_output is not None
                and (
                    local_output_extent is None
                    or local_output_extent > int(maximum_local_output)
                )
            ):
                return False
            if not implementation.contract.get("requires_full_reduction_tiles", False):
                full_reduction_tiles = True
            else:
                block_k = implementation.parameters.get("block_k")
                full_reduction_tiles = (
                    reduction_extent is not None
                    and isinstance(block_k, int)
                    and reduction_extent % block_k == 0
                )
            return full_reduction_tiles and (
                not implementation.contract.get(
                    "requires_contiguous_local_tiles", False
                )
                or _supports_contiguous_descriptor_tiles(
                    node,
                    context.module,
                    block_k=implementation.parameters.get("block_k"),
                    tile_n=implementation.parameters.get("tile_n"),
                )
            )

        variants = tuple(
            _configure_dense_implementation(
                context, implementation, vector_contract, distribution_contract
            )
            for implementation in context.implementations(
                "dense_matmul",
                input_kind="logical",
                epilogue="none",
                vectorization_kind=vector_contract["kind"],
                distribution_kind=distribution_contract["kind"],
            )
            if applicable(implementation)
        )
        match = (
            context.projection_norm_matches.get(node.id)
            if context.is_reusable(node)
            else None
        )
        if match is not None:
            fused_vectors = consumed_vector_contracts(
                context.module,
                {
                    "residual_add": (match.residual_add, "axes"),
                    "norm": (match.norm_consumer, "reduction_axis"),
                },
            )
        else:
            fused_vectors = None
        if match is not None and fused_vectors is not None:
            variants += tuple(
                _configure_dense_implementation(
                    context,
                    implementation,
                    vector_contract,
                    distribution_contract,
                    semantic_parameters={
                        "epilogue": "residual_norm_stats",
                        "residual_add": match.residual_add,
                        "residual_input": match.residual_input,
                        "norm_consumer": match.norm_consumer,
                        "norm_stats_partials": "cta",
                        "projection_adapters": match.adapters,
                        "consumed_vector_contracts": fused_vectors,
                    },
                    facts={
                        "requires_reusable_function": True,
                        "projection_result_single_use": True,
                        "materializes_residual": True,
                    },
                )
                for implementation in context.implementations(
                    "dense_matmul",
                    input_kind="logical",
                    epilogue="residual_norm_stats",
                    vectorization_kind=vector_contract["kind"],
                    distribution_kind=distribution_contract["kind"],
                )
                if applicable(implementation)
            )
        output_type = logical_type(node.type)
        assert isinstance(output_type, TensorType)
        output_dimension = output_type.shape[-1]
        output_extent = (
            output_dimension.fixed_value if output_dimension.is_fixed else None
        )
        logits_match = context.projection_logits_matches.get(node.id)
        logits_vectors = (
            None
            if logits_match is None
            else consumed_vector_contracts(
                context.module,
                {"logits_cast": (logits_match.logits, "axes")},
            )
        )
        if (
            logits_match is not None
            and logits_vectors is not None
            and output_extent is not None
        ):
            variants += tuple(
                _configure_dense_implementation(
                    context,
                    implementation,
                    vector_contract,
                    distribution_contract,
                    semantic_parameters={
                        "epilogue": "cast_logits_argmax",
                        "logits_output": logits_match.logits,
                        "sampler": logits_match.sampler,
                        "consumed_vector_contracts": logits_vectors,
                    },
                    facts={
                        "projection_result_single_use": True,
                        "materializes_projection": False,
                        "materializes_fp32_logits": True,
                    },
                )
                for implementation in context.implementations(
                    "dense_matmul",
                    input_kind="logical",
                    epilogue="cast_logits_argmax",
                    vectorization_kind=vector_contract["kind"],
                    distribution_kind=distribution_contract["kind"],
                )
                if applicable(implementation)
                and output_extent % int(implementation.parameters["tile_n"]) == 0
            )
        if not variants:
            return None
        return TritonCandidateProposal(
            variants, context.choose_default("dense_matmul", variants)
        )


def _supports_contiguous_descriptor_tiles(
    node: Node,
    module,
    *,
    block_k,
    tile_n,
) -> bool:
    """Check the local affine rectangles required by one descriptor load."""

    if (
        node.op != "math.matmul"
        or bool(node.attrs.get("transpose_a", False))
        or not isinstance(block_k, int)
        or isinstance(block_k, bool)
        or not isinstance(tile_n, int)
        or isinstance(tile_n, bool)
        or block_k <= 0
        or tile_n <= 0
    ):
        return False
    lhs = module.node_map[node.inputs[0]].type
    output = node.type
    lhs_type = logical_type(lhs)
    output_type = logical_type(output)
    if not isinstance(lhs_type, TensorType) or not isinstance(output_type, TensorType):
        return False
    lhs_k_axis = lhs_type.rank - 1
    output_n_axis = output_type.rank - 1
    if not _axis_has_contiguous_local_order(lhs, lhs_k_axis):
        return False
    if not _axis_has_contiguous_local_order(output, output_n_axis):
        return False
    lhs_dimensions = local_shape(lhs) if isinstance(lhs, DistributedType) else lhs_type.shape
    output_dimensions = (
        local_shape(output) if isinstance(output, DistributedType) else output_type.shape
    )
    lhs_extent = lhs_dimensions[lhs_k_axis].fixed_value
    output_extent = output_dimensions[output_n_axis].fixed_value
    lhs_extent = (
        None
        if lhs_extent is None
        else lhs_extent * _last_axis_lane_count(lhs_type)
    )
    output_extent = (
        None
        if output_extent is None
        else output_extent * _last_axis_lane_count(output_type)
    )
    return (
        lhs_extent is not None
        and output_extent is not None
        and lhs_extent >= block_k
        and output_extent >= tile_n
        and lhs_extent % block_k == 0
        and output_extent % tile_n == 0
    )


def _supports_packed_descriptor_tiles(
    node: Node,
    module,
    *,
    block_k,
    tile_n,
    affine_rhs: bool = False,
    allow_output_tail: bool = False,
) -> bool:
    """Prove that each packed descriptor issue names one affine rectangle."""

    if (
        node.op != "ntt.packed_matmul"
        or not isinstance(block_k, int)
        or isinstance(block_k, bool)
        or not isinstance(tile_n, int)
        or isinstance(tile_n, bool)
        or block_k <= 0
        or tile_n <= 0
    ):
        return False
    lhs = module.node_map[node.inputs[0]].type
    rhs = module.node_map[node.inputs[1]].type
    output = node.type
    lhs_type = logical_type(lhs)
    rhs_type = logical_type(rhs)
    output_type = logical_type(output)
    if (
        not isinstance(lhs_type, TensorType)
        or not isinstance(output_type, TensorType)
        or not isinstance(rhs_type, TensorType)
        or not isinstance(rhs_type.dtype, VectorType)
        or rhs_type.dtype.lanes != (8, 2, 8)
    ):
        return False
    if affine_rhs:
        # An owner table encodes RHS global strides. LHS and result use the
        # ordinary scalar local-shard address mapper, not TMA descriptors.
        if not all(_axis_has_affine_local_order(rhs, axis) for axis in range(rhs_type.rank)):
            return False
    elif not all((
        _axis_has_contiguous_local_order(lhs, lhs_type.rank - 1),
        _axis_has_contiguous_local_order(rhs, rhs_type.rank - 2),
        _axis_has_contiguous_local_order(rhs, rhs_type.rank - 1),
        _axis_has_contiguous_local_order(output, output_type.rank - 1),
    )):
        return False
    local_k = _local_scalar_last_axis_extent(lhs)
    local_n = _local_scalar_last_axis_extent(output)
    return (
        local_k is not None
        and local_n is not None
        and local_k >= block_k
        and local_n >= tile_n
        and local_k % block_k == 0
        and (allow_output_tail or local_n % tile_n == 0)
    )


@lru_cache(maxsize=1024)
def _axis_has_affine_local_order(value, axis: int) -> bool:
    if not isinstance(value, DistributedType):
        return True
    return all(
        local_shard_descriptor(value, owner).axes[axis].affine_stride is not None
        for owner in product(*(range(extent) for extent in value.placement.hierarchy))
    )


def _axis_has_contiguous_local_order(value, axis: int) -> bool:
    if not isinstance(value, DistributedType):
        return True
    policy = value.axis_policies[axis]
    return isinstance(policy, SBPBroadCast) or (
        isinstance(policy, SBPSplit) and policy.is_contiguous
    )


def _local_scalar_last_axis_extent(value) -> int | None:
    logical = logical_type(value)
    if not isinstance(logical, TensorType):
        return None
    shape = local_shape(value) if isinstance(value, DistributedType) else logical.shape
    if not shape or not shape[-1].is_fixed:
        return None
    return shape[-1].fixed_value * _last_axis_lane_count(logical)


class MatMulNormStatsCandidateProvider:
    """Select a direct multi-result projection/residual/statistics kernel."""

    op_names = frozenset({"ntt.matmul_norm_stats"})

    def propose(
        self,
        node: Node,
        context: TritonCandidateContext,
    ) -> TritonCandidateProposal | None:
        result_type = node.type
        output_type = logical_type(result_type)
        if (
            not isinstance(output_type, TupleType)
            or len(output_type.fields) != 2
            or not isinstance(output_type.fields[0], TensorType)
            or _scalar_dtype(output_type.fields[0]) != DType.BFLOAT16
            or bool(node.attrs["use_mean"])
        ):
            return None
        value_type = output_type.fields[0]
        if (
            any(not dimension.is_fixed for dimension in value_type.shape[:-1])
            or prod(
                dimension.fixed_value for dimension in value_type.shape[:-1]
            ) != 1
        ):
            return None
        axis = int(node.attrs["axis"])
        axis = axis + value_type.rank if axis < 0 else axis
        if axis != value_type.rank - 1:
            return None
        lhs = context.module.node_map[node.inputs[0]]
        rhs = context.module.node_map[node.inputs[1]]
        rhs_layout = node.attrs.get("rhs_layout")
        if rhs_layout is None:
            matmul_attrs = {
                "transpose_a": bool(node.attrs["transpose_a"]),
                "transpose_b": bool(node.attrs["transpose_b"]),
            }
            matmul_type = MatMul.infer_type((lhs, rhs), matmul_attrs)
            synthetic_op = MatMul.op_name
            input_kind = "logical"
            packed_layout = None
        elif rhs_layout == "k_major":
            none = Node("<none>", "builtin.none", (), NoneType())
            matmul_attrs = {
                "fused_reduce": False,
                "output_data_type": DType.BFLOAT16,
                "rhs_layout": "k_major",
            }
            matmul_type = PackedMatMul.infer_type(
                (lhs, rhs, none, none), matmul_attrs
            )
            rhs_type = logical_type(rhs.type)
            if (
                not isinstance(rhs_type, TensorType)
                or not isinstance(rhs_type.dtype, VectorType)
                or len(rhs_type.dtype.lanes) != 3
            ):
                return None
            n_vector, k_pack, k_vector = rhs_type.dtype.lanes
            packed_layout = k_major_layout_name(
                n_vector, k_pack * k_vector
            )
            synthetic_op = PackedMatMul.op_name
            input_kind = "packed"
        else:
            return None
        vector_metadata = {
            **dict(node.metadata),
            **dict(node.metadata.get("matmul_vectorization", {})),
        }
        synthetic = Node(
            node.id,
            synthetic_op,
            node.inputs[:2],
            matmul_type,
            node.effect,
            matmul_attrs,
            vector_metadata,
        )
        vector_contract = vectorization_contract(synthetic)
        distribution_contract = dense_matmul_distribution_contract(
            synthetic, context.module
        )
        residual_add = node.metadata.get("residual_add")
        residual_input = node.metadata.get("residual_input")
        norm_consumer = node.metadata.get("norm_consumer")
        if not all(
            isinstance(value, str)
            for value in (residual_add, residual_input, norm_consumer)
        ):
            return None
        fused_vectors = consumed_vector_contracts(
            context.module,
            {
                "residual_add": (residual_add, "axes"),
                "norm": (norm_consumer, "reduction_axis"),
            },
        )
        if fused_vectors is None:
            return None
        input_type = logical_type(lhs.type)
        reduction_extent = (
            input_type.shape[-1].fixed_value
            if isinstance(input_type, TensorType) and input_type.shape[-1].is_fixed
            else None
        )

        output_extent = _local_scalar_last_axis_extent(matmul_type)
        owner_count = _output_partition_owner_count(matmul_type)
        if (
            not isinstance(result_type, TupleType)
            or result_type.fields[0] != matmul_type
        ):
            return None
        stats_type = result_type.fields[1]
        output_partition_axes = _local_norm_stats_partition_axes(
            matmul_type, stats_type
        )
        if output_partition_axes is None:
            return None
        owner_partial_fast_path = (
            isinstance(matmul_type, DistributedType)
            and owner_count == prod(matmul_type.placement.hierarchy)
            and output_partition_axes
            == tuple(range(matmul_type.placement.rank))
        )

        def applicable(implementation) -> bool:
            if not _supports_lhs_staging(implementation, lhs.type):
                return False
            statistics_kind = implementation.contract.get("statistics_kind")
            if statistics_kind == "owner_partial":
                if not owner_partial_fast_path:
                    return False
            elif statistics_kind != "local_shard":
                return False
            block_k = implementation.parameters.get("block_k")
            tile_n = implementation.parameters.get("tile_n")
            full_tiles = (
                reduction_extent is not None
                and isinstance(block_k, int)
                and reduction_extent % block_k == 0
                and output_extent is not None
                and isinstance(tile_n, int)
                and output_extent % tile_n == 0
            )
            if not implementation.contract.get(
                "requires_full_reduction_tiles", False
            ):
                full_tiles = True
            if not full_tiles:
                return False
            affine_rhs = implementation.contract.get("requires_affine_local_tiles", False)
            if not (implementation.contract.get("requires_contiguous_local_tiles", False) or affine_rhs):
                return True
            descriptor_proof = (
                _supports_packed_descriptor_tiles
                if input_kind == "packed"
                else _supports_contiguous_descriptor_tiles
            )
            return descriptor_proof(
                synthetic,
                context.module,
                block_k=block_k,
                tile_n=tile_n,
                **({"affine_rhs": affine_rhs} if input_kind == "packed" else {}),
            )

        candidates = tuple(
            _configure_dense_implementation(
                context,
                implementation,
                vector_contract,
                distribution_contract,
                semantic_parameters={
                    "epilogue": "residual_norm_stats",
                    **(
                        {"packed_layout": packed_layout}
                        if packed_layout is not None else {}
                    ),
                    "residual_add": residual_add,
                    "residual_input": residual_input,
                    "norm_consumer": norm_consumer,
                    "projection_adapters": tuple(
                        node.metadata.get("projection_adapters", ())
                    ),
                    "consumed_vector_contracts": fused_vectors,
                    "explicit_results": ("value", "norm_stats"),
                    "owner_count": owner_count,
                    "statistics_kind": str(
                        implementation.contract["statistics_kind"]
                    ),
                    "output_partition_axes": output_partition_axes,
                },
                facts={
                    "projection_result_single_use": True,
                    "materializes_residual": True,
                    "explicit_norm_stats_result": True,
                },
            )
            for implementation in context.implementations(
                "dense_matmul",
                input_kind=input_kind,
                epilogue="residual_norm_stats",
                vectorization_kind=vector_contract["kind"],
                distribution_kind=distribution_contract["kind"],
            )
            if applicable(implementation)
        )
        if not candidates:
            return None
        return TritonCandidateProposal(
            candidates,
            context.choose_default("dense_matmul", candidates),
        )


def _supports_lhs_staging(implementation, value_type) -> bool:
    """Check a bounded Shared LHS resource against scalar local capacity."""

    capacity = implementation.parameters.get("lhs_stage_extent")
    if capacity is None:
        return True
    extent = _local_scalar_last_axis_extent(value_type)
    if implementation.parameters.get("lhs_copy_kind", "sync") == "async":
        from triton.flagmega.codegen.triton.row_transfer import has_full_static_rows

        tile = implementation.parameters.get("lhs_copy_tile")
        if (not isinstance(tile, int) or isinstance(tile, bool) or tile <= 0
                or extent is None or extent % tile or not has_full_static_rows(value_type)):
            return False
    return (
        isinstance(capacity, int)
        and not isinstance(capacity, bool)
        and capacity > 0
        and extent is not None
        and 0 < extent <= capacity
    )


def _output_partition_owner_count(value_type) -> int:
    if not isinstance(value_type, DistributedType):
        return 1
    policy = value_type.axis_policies[-1]
    if not isinstance(policy, SBPSplit):
        return 1
    result = 1
    for axis in policy.hierarchy_axes:
        result *= value_type.placement.hierarchy[axis]
    return result


def _local_norm_stats_partition_axes(value_type, stats_type) -> tuple[int, ...] | None:
    """Validate the local-shard statistics contract of a direct GEMV.

    A normal kernel computes only the local output shard.  Consequently its
    square sum is materialized when N is broadcast, or is Sum-partial over
    exactly the placement axes that partition N.  This is the same generic
    local-buffer contract used by nncase's PackedMatMulNormStats TIR and does
    not depend on a model shape or target architecture.
    """

    if not isinstance(value_type, DistributedType):
        return () if not isinstance(stats_type, DistributedType) else None
    if (
        not isinstance(stats_type, DistributedType)
        or stats_type.placement != value_type.placement
    ):
        return None
    output_policy = value_type.axis_policies[-1]
    if isinstance(output_policy, SBPBroadCast):
        axes: tuple[int, ...] = ()
    elif isinstance(output_policy, SBPSplit):
        axes = tuple(output_policy.hierarchy_axes)
    else:
        return None
    partial = stats_type.partial
    if not axes:
        return () if partial is None else None
    if (
        partial is None
        or partial.reduce_op is not ReduceOp.SUM
        or tuple(partial.axes) != axes
    ):
        return None
    return axes


def _configure_dense_implementation(
    context,
    implementation,
    vector_contract,
    distribution_contract,
    *,
    semantic_parameters=None,
    facts=None,
):
    tile_n = int(implementation.parameters["tile_n"])
    lane_count = int(vector_contract["lane_count"])
    if vector_contract["kind"] != "scalar" and tile_n % lane_count:
        raise ValueError(
            f"Dense implementation {implementation.id!r} tile_n={tile_n} does not "
            f"preserve vector lane group {lane_count}."
        )
    parameters = dict(semantic_parameters or {})
    parameters["distribution_schedule"] = dict(distribution_contract)
    parameters["vector_schedule"] = configured_vector_schedule(
        vector_contract,
        lowering="output_tile",
        tile_n=tile_n,
    )
    return context.configure_implementation(
        implementation,
        semantic_parameters=parameters,
        facts=facts,
    )


def _scalar_dtype(value: TensorType):
    return value.dtype.elem_type if isinstance(value.dtype, VectorType) else value.dtype


def _last_axis_lane_count(value: TensorType) -> int:
    if not isinstance(value.dtype, VectorType):
        return 1
    # PackedMatMul carries its one output-N lane in the last physical axis.
    return value.dtype.lane_count


__all__ = ["DenseMatmulCandidateProvider", "MatMulNormStatsCandidateProvider"]
