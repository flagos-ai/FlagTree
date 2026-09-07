# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Per-family NTT distributed candidate providers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from itertools import combinations, product as cartesian_product
from math import prod

from triton.flagmega.ir import (
    BlockCyclicSplit,
    ContiguousSplit,
    DistributedType,
    IRType,
    Node,
    NoneType,
    Placement,
    RefType,
    SBP,
    SBPPartial,
    SBPSplit,
    TensorType,
    TupleType,
    VectorType,
    is_distributable,
    local_shape,
    local_tensor_type,
    scale_split_units,
)
from triton.flagmega.ir.distributed_inference import (
    broadcast_ir_type,
    broadcast_type,
    split_type,
    tensor_of,
)
from triton.flagmega.ir.ops.tensors._k_major import parse_k_major_layout
from triton.flagmega.ir.ops.ntt.packed_qkv_parallel_linear_combine import (
    can_materialize_packed_qkv,
)
from triton.flagmega.passes.auto_distributed.candidates import (
    DistributedCandidate,
    DistributedCandidateContext,
    DistributedCandidateProviderBase,
)
from triton.flagmega.passes.auto_distributed.candidate_identity import (
    distributed_candidate_id,
)


def _replicated(context: DistributedCandidateContext, cost: int) -> DistributedCandidate:
    node = context.source_call
    return DistributedCandidate(
        f"distribution.{node.id}.replicated",
        broadcast_ir_type(node.type, context.placement),
        tuple(
            broadcast_ir_type(context.module.node_map[value].type, context.placement)
            for value in node.inputs
        ),
        cost,
        "broadcast-replicated",
    )


class BroadcastCandidateProvider(DistributedCandidateProviderBase):
    """Explicit generic inference for ops that are legal only when replicated.

    This is a reviewed provider, not an unknown-op fallback.  The op's normal
    distributed type inference still validates the materialized candidate.
    """

    allows_partial_inputs = False
    is_exhaustive = True

    def __init__(self, op_names: frozenset[str]) -> None:
        if not op_names:
            raise ValueError("BroadcastCandidateProvider requires at least one op name.")
        self.op_names = frozenset(op_names)

    def _enumerate_candidates(
        self,
        context: DistributedCandidateContext,
    ) -> tuple[DistributedCandidate, ...]:
        work = max(_ir_type_bytes(context.source_call.type), 1)
        candidate = _replicated(context, min(work, 2_000_000_000))
        return (
            DistributedCandidate(
                candidate.id,
                candidate.return_type,
                candidate.input_types,
                candidate.operation_cost,
                "explicit-broadcast-inference",
                candidate.target_op,
                candidate.objective_kind,
                "flagmega.explicit-broadcast-work/v1",
                ("registered-broadcast-provider",),
            ),
        )


class EmbeddingCandidateProvider(DistributedCandidateProviderBase):
    """Project a decode embedding's feature layout through the table lookup.

    This is the rank-general equivalent of nncase's ``GatherCandidateProvider``
    for ``gather(weight, indices, axis=0)``.  The vocabulary axis must remain
    broadcast because every owner may observe any token id.  Output axes that
    originate in ``indices`` also remain broadcast; the trailing embedding
    feature axis may be split and is projected exactly to weight axis 1.
    """

    op_names = frozenset({"nn.embedding"})
    allows_partial_inputs = False
    is_exhaustive = True

    def _enumerate_candidates(
        self,
        context: DistributedCandidateContext,
    ) -> tuple[DistributedCandidate, ...]:
        node = context.source_call
        if len(node.inputs) != 2:
            return ()
        module = context.module
        indices = tensor_of(module.node_map[node.inputs[0]].type)
        weight = tensor_of(module.node_map[node.inputs[1]].type)
        output = tensor_of(node.type)
        if (
            weight.rank != 2
            or output.rank != indices.rank + 1
            or output.shape[:-1] != indices.shape
            or output.shape[-1] != weight.shape[1]
        ):
            return ()

        work = max(_tensor_bytes(output), 1)
        replicated = _replicated(context, work)
        values = [replace(replicated, reason="embedding-replicated")]
        placement = context.placement
        for hierarchy_axes in _mesh_axis_combinations(placement):
            for output_index, feature_policy in enumerate(
                context.split_candidates(
                    output,
                    output.rank - 1,
                    hierarchy_axes,
                    purpose="output",
                )
            ):
                weight_policies = [SBP.broadcast(), feature_policy]
                output_policies = [
                    SBP.broadcast() for _ in range(output.rank - 1)
                ]
                output_policies.append(feature_policy)
                weight_type = DistributedType(
                    weight,
                    tuple(weight_policies),
                    placement,
                )
                output_type = DistributedType(
                    output,
                    tuple(output_policies),
                    placement,
                )
                if not is_distributable(
                    weight, tuple(weight_policies), placement
                ):
                    continue
                values.append(
                    DistributedCandidate(
                        _candidate_id(
                            node.id,
                            f"variant_{output_index}.feature_split",
                            hierarchy_axes,
                            placement,
                        ),
                        output_type,
                        (
                            broadcast_type(indices, placement),
                            weight_type,
                        ),
                        max(
                            work // _shard_count(placement, hierarchy_axes),
                            1,
                        ),
                        "embedding-exact-output-sbp",
                        objective_kind="analytic",
                        objective_model="flagmega.embedding-gather-bytes/v1",
                        objective_evidence=(
                            "nncase-gather-output-layout-projection",
                            "feature-local-bytes",
                            "vocabulary-axis-broadcast",
                        ),
                    )
                )
        return tuple(values)


def _tensor_bytes(value: IRType) -> int:
    tensor = tensor_of(value)
    if any(not dimension.is_fixed for dimension in tensor.shape):
        return 1 << 20
    return prod(dimension.fixed_value for dimension in tensor.shape) * tensor.dtype.itemsize


def _ir_type_bytes(value: IRType) -> int:
    if isinstance(value, RefType):
        return 0
    if isinstance(value, TupleType):
        return sum(_ir_type_bytes(field) for field in value.fields)
    return _tensor_bytes(value)


def _matmul_work(output: TensorType, value: TensorType) -> int:
    if any(not dimension.is_fixed for dimension in (*output.shape, *value.shape)):
        return 10_000_000
    output_lanes = getattr(output.dtype, "lane_count", 1)
    value_lanes = getattr(value.dtype, "lane_count", 1)
    elements = prod(dimension.fixed_value for dimension in output.shape) * output_lanes
    reduction = value.shape[-1].fixed_value * value_lanes
    return min(elements * reduction, 2_000_000_000)


def _packed_qkv_local_shape_cost(
    input_type: IRType,
    output_type: TupleType,
    local_mac_work: int,
) -> int:
    """Rank equal-work packed QKV plans by their regular local K/N shape.

    A scalar-MAC count alone cannot distinguish split-N from orthogonal
    split-K/split-N plans: both divide the same global work by the same owner
    count.  Each of the three projections must nevertheless traverse its
    local K and N domains.  ``abs(K - N)`` is a target-independent lower-order
    imbalance term which prefers useful two-dimensional local work without
    encoding an accelerator tile, model shape, or calibrated latency.

    Partial materialization remains a separate analytic edge cost in the
    combine provider, so the global solver selects the hybrid only when this
    shape advantage pays for its actual collective boundary.
    """

    local_input = (
        local_tensor_type(input_type)
        if isinstance(input_type, DistributedType)
        else tensor_of(input_type)
    )
    local_outputs = tuple(
        local_tensor_type(field)
        if isinstance(field, DistributedType)
        else tensor_of(field)
        for field in output_type.fields
    )
    if (
        not local_input.shape
        or not local_input.shape[-1].is_fixed
        or any(not output.shape or not output.shape[-1].is_fixed for output in local_outputs)
    ):
        return min(local_mac_work, 2_000_000_000)
    local_k = (
        local_input.shape[-1].fixed_value
        * getattr(local_input.dtype, "lane_count", 1)
    )
    local_ns = tuple(
        output.shape[-1].fixed_value * getattr(output.dtype, "lane_count", 1)
        for output in local_outputs
    )
    imbalance = sum(abs(local_k - local_n) for local_n in local_ns)
    return min(local_mac_work + imbalance, 2_000_000_000)


_PACKED_QKV_OBJECTIVE = {
    "objective_kind": "heuristic",
    "objective_model": "flagmega.packed-qkv-local-shape-balance/v1",
    "objective_evidence": (
        "local-scalar-macs",
        "local-k-n-imbalance",
        "combine-communication-accounted-separately",
        "target-and-model-independent",
    ),
}


def _mesh_axis_combinations(placement: Placement) -> tuple[tuple[int, ...], ...]:
    """Match nncase's non-empty placement-axis combination enumeration."""

    return tuple(
        axes
        for count in range(1, placement.rank + 1)
        for axes in combinations(range(placement.rank), count)
        if _shard_count(placement, axes) > 1
    )


def _shard_count(placement: Placement, hierarchy_axes: tuple[int, ...]) -> int:
    return prod(placement.hierarchy[axis] for axis in hierarchy_axes)


def _can_split(
    hierarchy_axes: tuple[int, ...],
    placement: Placement,
    *tensor_axes: tuple[TensorType, int],
) -> bool:
    divisor = _shard_count(placement, hierarchy_axes)
    return all(_axis_divides(tensor, tensor_axis, divisor) for tensor, tensor_axis in tensor_axes)


def _candidate_id(
    node_id: str,
    kind: str,
    hierarchy_axes: tuple[int, ...],
    placement: Placement,
) -> str:
    if placement.rank == 1:
        return f"distribution.{node_id}.{kind}"
    axes = "_".join(str(axis) for axis in hierarchy_axes)
    return f"distribution.{node_id}.axes_{axes}.{kind}"


class MatMulCandidateProvider(DistributedCandidateProviderBase):
    op_names = frozenset({
        "math.block_scaled_matmul",
        "math.packed_block_scaled_matmul",
        "math.packed_dense_matmul",
        "math.matmul",
    })
    allows_partial_inputs = True
    is_exhaustive = True

    def _enumerate_candidates(self, context: DistributedCandidateContext) -> tuple[DistributedCandidate, ...]:
        node = context.source_call
        module = context.module
        value = tensor_of(module.node_map[node.inputs[0]].type)
        weight = tensor_of(module.node_map[node.inputs[1]].type)
        output = tensor_of(node.type)
        work = _matmul_work(output, value)
        placement = context.placement
        original_inputs = tuple(module.node_map[value_id].type for value_id in node.inputs)

        transpose_a = bool(node.attrs.get("transpose_a", False)) if node.op == "math.matmul" else False
        transpose_b = bool(node.attrs.get("transpose_b", True)) if node.op == "math.matmul" else True
        value_k_axis = 0 if transpose_a else 1
        scalar_packed = node.op == "math.packed_dense_matmul"
        packed_layout = (
            str(node.attrs.get("packed_layout"))
            if scalar_packed else None
        )
        packed_n_lane = 1
        packed_k_lane = 1
        mesh_interleaved = False
        if packed_layout is not None:
            packed_n_lane, packed_k_lane, mesh_interleaved = parse_k_major_layout(
                packed_layout
            )
        elif (
            node.op == "math.packed_block_scaled_matmul"
            and isinstance(weight.dtype, VectorType)
        ):
            packed_k_lane = weight.dtype.lane_count
        if scalar_packed:
            # [K/KLane,N/NLane,payload...]: only the physical N-group axis maps to an
            # independently materializable output split.
            weight_n_axis = 2 if mesh_interleaved else 1
            weight_k_axis = 0
        else:
            weight_n_axis = 0 if transpose_b else 1
            weight_k_axis = 1 if transpose_b else 0
        values = [_replicated(context, work)]

        output_splits: list[tuple[tuple[int, ...], SBPSplit, SBPSplit]] = []
        if not mesh_interleaved:
            for hierarchy_axes in _mesh_axis_combinations(placement):
                for output_policy in context.split_candidates(
                    output,
                    output.rank - 1,
                    hierarchy_axes,
                    purpose="output",
                ):
                    weight_n_policy = (
                        scale_split_units(output_policy, 1, packed_n_lane)
                        if scalar_packed
                        else output_policy
                    )
                    if weight_n_policy is None:
                        continue
                    weight_policies = [SBP.broadcast() for _ in weight.shape]
                    weight_policies[weight_n_axis] = weight_n_policy
                    if not is_distributable(
                        weight, tuple(weight_policies), placement
                    ):
                        continue
                    output_splits.append(
                        (hierarchy_axes, output_policy, weight_n_policy)
                    )

            for output_index, (
                hierarchy_axes,
                output_policy,
                weight_n_policy,
            ) in enumerate(output_splits):
                output_inputs = list(original_inputs)
                output_inputs[0] = broadcast_type(value, placement)
                weight_policies = [SBP.broadcast() for _ in weight.shape]
                weight_policies[weight_n_axis] = weight_n_policy
                output_inputs[1] = DistributedType(
                    weight, tuple(weight_policies), placement
                )
                for index in range(2, len(output_inputs)):
                    output_inputs[index] = broadcast_ir_type(
                        output_inputs[index], placement
                    )
                output_type = DistributedType(
                    output,
                    tuple(
                        output_policy if axis == output.rank - 1 else SBP.broadcast()
                        for axis in range(output.rank)
                    ),
                    placement,
                )
                values.append(DistributedCandidate(
                    _candidate_id(
                        node.id,
                        f"variant_{output_index}.output_split",
                        hierarchy_axes,
                        placement,
                    ),
                    output_type,
                    tuple(output_inputs),
                    max(work // _shard_count(placement, hierarchy_axes), 1),
                    "matmul-output-sbp",
                ))

        if mesh_interleaved:
            for hierarchy_axes in _mesh_axis_combinations(placement):
                if (
                    not weight.shape[weight_n_axis].is_fixed
                    or weight.shape[weight_n_axis].fixed_value
                    != _shard_count(placement, hierarchy_axes)
                ):
                    continue
                output_inputs = list(original_inputs)
                output_inputs[0] = broadcast_type(value, placement)
                output_inputs[1] = split_type(
                    weight, weight_n_axis, placement, hierarchy_axes)
                output_type = split_type(
                    output,
                    output.rank - 1,
                    placement,
                    hierarchy_axes,
                    block_size=packed_n_lane,
                )
                values.append(DistributedCandidate(
                    _candidate_id(
                        node.id,
                        f"mesh_interleaved_n{packed_n_lane}_output_split",
                        hierarchy_axes,
                        placement,
                    ),
                    output_type,
                    tuple(output_inputs),
                    max(work // _shard_count(placement, hierarchy_axes), 1),
                    "matmul-mesh-interleaved-output-sbp",
                ))

        if (
            node.op in {"math.matmul", "math.packed_dense_matmul"}
            and not mesh_interleaved
            and value.rank == 2
            and output.rank == 2
        ):
            # nncase permits a packed/logical MatMul to consume disjoint mesh
            # axes for output-N and reduction-K at the same time.  This is the
            # regular 2-D tensor-parallel form: the output remains split on one
            # hierarchy subset and partial-Sum on another.
            reduction_splits: list[tuple[tuple[int, ...], SBPSplit, SBPSplit]] = []
            for reduction_axes in _mesh_axis_combinations(placement):
                for input_k_policy in context.split_candidates(
                    value,
                    value_k_axis,
                    reduction_axes,
                    purpose="reduction",
                ):
                    weight_k_policy = (
                        scale_split_units(input_k_policy, 1, packed_k_lane)
                        if node.op == "math.packed_dense_matmul"
                        else input_k_policy
                    )
                    if weight_k_policy is None:
                        continue
                    reduction_splits.append(
                        (reduction_axes, input_k_policy, weight_k_policy)
                    )
            for reduction_index, (
                reduction_axes,
                input_k_policy,
                weight_k_policy,
            ) in enumerate(reduction_splits):
                for output_index, (
                    output_axes,
                    output_policy,
                    weight_n_policy,
                ) in enumerate(output_splits):
                    if set(reduction_axes) & set(output_axes):
                        continue
                    hybrid_inputs = list(original_inputs)
                    input_policies = [SBP.broadcast() for _ in value.shape]
                    input_policies[value_k_axis] = input_k_policy
                    hybrid_inputs[0] = DistributedType(
                        value, tuple(input_policies), placement
                    )
                    weight_policies = [SBP.broadcast() for _ in weight.shape]
                    weight_policies[weight_k_axis] = weight_k_policy
                    weight_policies[weight_n_axis] = weight_n_policy
                    hybrid_inputs[1] = DistributedType(
                        weight, tuple(weight_policies), placement)
                    for index in range(2, len(hybrid_inputs)):
                        hybrid_inputs[index] = broadcast_ir_type(
                            hybrid_inputs[index], placement)
                    output_policies = [SBP.broadcast() for _ in output.shape]
                    output_policies[output.rank - 1] = output_policy
                    hybrid_output = DistributedType(
                        output,
                        tuple(output_policies),
                        placement,
                        partial=SBPPartial(reduction_axes),
                    )
                    fallback = max(
                        work // (
                            _shard_count(placement, reduction_axes)
                            * _shard_count(placement, output_axes)
                        ),
                        1,
                    )
                    values.append(DistributedCandidate(
                        _candidate_id(
                            node.id,
                            (
                                "output_"
                                + str(output_index)
                                + "_reduction_"
                                + str(reduction_index)
                            ),
                            tuple((*reduction_axes, *output_axes)),
                            placement,
                        ),
                        hybrid_output,
                        tuple(hybrid_inputs),
                        fallback,
                        "matmul-output-K-sbp-partial",
                    ))
        if (
            not mesh_interleaved
            and value.rank == 2
            and (weight.rank == 2 or scalar_packed)
        ):
            reduction_splits = []
            for hierarchy_axes in _mesh_axis_combinations(placement):
                for input_k_policy in context.split_candidates(
                    value,
                    value_k_axis,
                    hierarchy_axes,
                    purpose="reduction",
                ):
                    if (
                        node.op in {
                            "math.block_scaled_matmul",
                            "math.packed_block_scaled_matmul",
                        }
                        and not _is_scale_group_aligned_reduction(
                            node,
                            value,
                            value_k_axis,
                            input_k_policy,
                            placement,
                        )
                    ):
                        continue
                    weight_k_policy = scale_split_units(
                        input_k_policy, 1, packed_k_lane
                    )
                    if weight_k_policy is None:
                        continue
                    weight_policies = [SBP.broadcast() for _ in weight.shape]
                    weight_policies[weight_k_axis] = weight_k_policy
                    if not is_distributable(
                        weight, tuple(weight_policies), placement
                    ):
                        continue
                    reduction_splits.append(
                        (hierarchy_axes, input_k_policy, weight_k_policy)
                    )
            for reduction_index, (
                hierarchy_axes,
                input_k_policy,
                weight_k_policy,
            ) in enumerate(reduction_splits):
                reduction_inputs = list(original_inputs)
                input_policies = [SBP.broadcast() for _ in value.shape]
                input_policies[value_k_axis] = input_k_policy
                reduction_inputs[0] = DistributedType(
                    value, tuple(input_policies), placement
                )
                weight_policies = [SBP.broadcast() for _ in weight.shape]
                weight_policies[weight_k_axis] = weight_k_policy
                reduction_inputs[1] = DistributedType(
                    weight, tuple(weight_policies), placement
                )
                for index in range(2, len(reduction_inputs)):
                    reduction_inputs[index] = broadcast_ir_type(
                        reduction_inputs[index], placement)
                partial = DistributedType(
                    output,
                    tuple(SBP.broadcast() for _ in output.shape),
                    placement,
                    partial=SBPPartial(hierarchy_axes),
                )
                fallback = max(work // _shard_count(placement, hierarchy_axes), 1)
                values.append(DistributedCandidate(
                    _candidate_id(
                        node.id,
                        f"variant_{reduction_index}.reduction_split",
                        hierarchy_axes,
                        placement,
                    ),
                    partial,
                    tuple(reduction_inputs),
                    fallback,
                    "matmul-K-sbp-partial",
                ))
        return tuple(values)


def _is_scale_group_aligned_reduction(
    node,
    value: TensorType,
    value_k_axis: int,
    split: SBPSplit,
    placement: Placement,
) -> bool:
    """Apply nncase's legality rule for dynamic block-FP8 K sharding."""

    if not split.is_contiguous:
        return False
    policies = [SBP.broadcast() for _ in value.shape]
    policies[value_k_axis] = split
    distributed = DistributedType(value, tuple(policies), placement)
    local_k = local_shape(distributed)[value_k_axis]
    if not local_k.is_fixed:
        return False
    scalar_k = local_k.fixed_value * getattr(value.dtype, "lane_count", 1)
    block_k = int(node.attrs.get("weight_block_k", 0))
    return block_k > 0 and scalar_k % block_k == 0


class PackedQKVParallelLinearCandidateProvider(DistributedCandidateProviderBase):
    """Couple Q/K/V K/N policies like nncase's packed-QKV provider."""

    op_names = frozenset({"ntt.packed_qkv_parallel_linear"})
    allows_partial_inputs = False
    is_exhaustive = True

    def _enumerate_candidates(
        self,
        context: DistributedCandidateContext,
    ) -> tuple[DistributedCandidate, ...]:
        node = context.source_call
        module = context.module
        if not isinstance(node.type, TupleType) or len(node.type.fields) != 3:
            return ()
        output_fields = tuple(tensor_of(field) for field in node.type.fields)
        if any(not isinstance(field.dtype, VectorType) for field in output_fields):
            return ()
        n_vectors = tuple(field.dtype.lanes[0] for field in output_fields)
        if len(set(n_vectors)) != 1:
            return ()
        n_vector = n_vectors[0]
        input_tensor = tensor_of(module.node_map[node.inputs[0]].type)
        weights = tuple(
            tensor_of(module.node_map[input_id].type)
            for input_id in node.inputs[1:4]
        )
        if any(
            not isinstance(weight.dtype, VectorType)
            or weight.dtype.lanes[0] != n_vector
            for weight in weights
        ):
            return ()
        k_lane_products = tuple(
            weight.dtype.lanes[1] * weight.dtype.lanes[2]
            for weight in weights
            if isinstance(weight.dtype, VectorType)
        )
        if len(k_lane_products) != 3:
            return ()
        tails = tuple(module.node_map[input_id].type for input_id in node.inputs[4:13])
        work = min(
            sum(_matmul_work(output, input_tensor) for output in output_fields),
            2_000_000_000,
        )
        placement = context.placement
        replicated = _replicated(context, work)
        candidates = [replace(
            replicated,
            operation_cost=_packed_qkv_local_shape_cost(
                replicated.input_types[0],
                replicated.return_type,
                work,
            ),
            **_PACKED_QKV_OBJECTIVE,
        )]

        output_splits: list[
            tuple[tuple[int, ...], tuple[SBPSplit, SBPSplit, SBPSplit]]
        ] = []
        for output_axes in _mesh_axis_combinations(placement):
            per_output_policies = tuple(
                context.split_candidates(
                    output, 1, output_axes, purpose="output"
                )
                for output in output_fields
            )
            for output_policies in cartesian_product(*per_output_policies):
                if (
                    not _have_coupled_qkv_output_policies(output_policies)
                    or not all(
                        is_distributable(
                            output,
                            (SBP.broadcast(), output_policy),
                            placement,
                        )
                        and is_distributable(
                            weight,
                            (SBP.broadcast(), output_policy),
                            placement,
                        )
                        for output, weight, output_policy in zip(
                            output_fields, weights, output_policies
                        )
                    )
                ):
                    continue
                output_splits.append((output_axes, output_policies))

        for output_index, (output_axes, output_policies) in enumerate(output_splits):
            output_types = TupleType(tuple(
                DistributedType(
                    output,
                    (SBP.broadcast(), output_policy),
                    placement,
                )
                for output, output_policy in zip(output_fields, output_policies)
            ))
            inputs = (
                broadcast_type(input_tensor, placement),
                *(
                    DistributedType(
                        weight,
                        (SBP.broadcast(), output_policy),
                        placement,
                    )
                    for weight, output_policy in zip(weights, output_policies)
                ),
                *_qkv_tail_types(
                    tails,
                    placement,
                    output_policies=output_policies,
                    partial=False,
                ),
            )
            local_work = max(work // _shard_count(placement, output_axes), 1)
            candidates.append(DistributedCandidate(
                _candidate_id(
                    node.id,
                    f"variant_{output_index}.output_split",
                    output_axes,
                    placement,
                ),
                output_types,
                tuple(inputs),
                _packed_qkv_local_shape_cost(inputs[0], output_types, local_work),
                "packed-qkv-output-sbp",
                **_PACKED_QKV_OBJECTIVE,
            ))

        if all(isinstance(value, NoneType) for value in tails[:3]):
            reduction_splits: list[
                tuple[tuple[int, ...], SBPSplit, tuple[SBPSplit, ...]]
            ] = []
            for reduction_axes in _mesh_axis_combinations(placement):
                for input_k_policy in context.split_candidates(
                    input_tensor, 1, reduction_axes, purpose="reduction"
                ):
                    weight_k_policies = tuple(
                        scale_split_units(input_k_policy, 1, lane_product)
                        for lane_product in k_lane_products
                    )
                    if any(policy is None for policy in weight_k_policies):
                        continue
                    typed_weight_k_policies = tuple(
                        policy
                        for policy in weight_k_policies
                        if isinstance(policy, SBPSplit)
                    )
                    if len(typed_weight_k_policies) != 3 or not all(
                        is_distributable(
                            weight,
                            (weight_k_policy, SBP.broadcast()),
                            placement,
                        )
                        for weight, weight_k_policy in zip(
                            weights, typed_weight_k_policies
                        )
                    ):
                        continue
                    reduction_splits.append(
                        (reduction_axes, input_k_policy, typed_weight_k_policies)
                    )

            for reduction_index, (
                reduction_axes,
                input_k_policy,
                weight_k_policies,
            ) in enumerate(reduction_splits):
                output_types = TupleType(tuple(
                    DistributedType(
                        output,
                        (SBP.broadcast(), SBP.broadcast()),
                        placement,
                        partial=SBPPartial(reduction_axes),
                    )
                    for output in output_fields
                ))
                inputs = (
                    DistributedType(
                        input_tensor,
                        (SBP.broadcast(), input_k_policy),
                        placement,
                    ),
                    *(
                        DistributedType(
                            weight,
                            (weight_k_policy, SBP.broadcast()),
                            placement,
                        )
                        for weight, weight_k_policy in zip(
                            weights, weight_k_policies
                        )
                    ),
                    *_qkv_tail_types(
                        tails,
                        placement,
                        output_policies=(
                            SBP.broadcast(),
                            SBP.broadcast(),
                            SBP.broadcast(),
                        ),
                        partial=True,
                    ),
                )
                local_work = max(
                    work // _shard_count(placement, reduction_axes), 1
                )
                candidates.append(DistributedCandidate(
                    _candidate_id(
                        node.id,
                        f"variant_{reduction_index}.reduction_split",
                        reduction_axes,
                        placement,
                    ),
                    output_types,
                    tuple(inputs),
                    _packed_qkv_local_shape_cost(
                        inputs[0], output_types, local_work
                    ),
                    "packed-qkv-K-sbp-partial",
                    **_PACKED_QKV_OBJECTIVE,
                ))

                for output_index, (output_axes, output_policies) in enumerate(
                    output_splits
                ):
                    if set(reduction_axes).intersection(output_axes):
                        continue
                    hybrid_outputs = TupleType(tuple(
                        DistributedType(
                            output,
                            (SBP.broadcast(), output_policy),
                            placement,
                            partial=SBPPartial(reduction_axes),
                        )
                        for output, output_policy in zip(
                            output_fields, output_policies
                        )
                    ))
                    hybrid_inputs = (
                        DistributedType(
                            input_tensor,
                            (SBP.broadcast(), input_k_policy),
                            placement,
                        ),
                        *(
                            DistributedType(
                                weight,
                                (weight_k_policy, output_policy),
                                placement,
                            )
                            for weight, weight_k_policy, output_policy in zip(
                                weights, weight_k_policies, output_policies
                            )
                        ),
                        *_qkv_tail_types(
                            tails,
                            placement,
                            output_policies=output_policies,
                            partial=True,
                        ),
                    )
                    local_work = max(
                        work // (
                            _shard_count(placement, reduction_axes)
                            * _shard_count(placement, output_axes)
                        ),
                        1,
                    )
                    candidates.append(DistributedCandidate(
                        _candidate_id(
                            node.id,
                            (
                                "output_"
                                + str(output_index)
                                + "_reduction_"
                                + str(reduction_index)
                            ),
                            tuple((*reduction_axes, *output_axes)),
                            placement,
                        ),
                        hybrid_outputs,
                        tuple(hybrid_inputs),
                        _packed_qkv_local_shape_cost(
                            hybrid_inputs[0], hybrid_outputs, local_work
                        ),
                        "packed-qkv-output-K-sbp-partial",
                        **_PACKED_QKV_OBJECTIVE,
                    ))
        return tuple(candidates)


class PackedQKVParallelLinearCombineCandidateProvider(
    DistributedCandidateProviderBase
):
    """Materialize all three Q/K/V partials through one collective boundary."""

    op_names = frozenset({"ntt.packed_qkv_parallel_linear_combine"})
    allows_partial_inputs = True
    is_exhaustive = True

    def _enumerate_candidates(
        self,
        context: DistributedCandidateContext,
    ) -> tuple[DistributedCandidate, ...]:
        node = context.source_call
        if len(context.available_input_types) != 1:
            return ()
        expected = node.attrs.get("output_type")
        if not isinstance(expected, TupleType) or len(expected.fields) != 3:
            return ()
        results: list[DistributedCandidate] = []
        seen: set[tuple[IRType, IRType]] = set()
        for input_type in context.available_input_types[0]:
            output_type = _packed_qkv_materialized_type(input_type)
            if (
                output_type is None
                or not _same_tuple_tensors(output_type, expected)
                or not can_materialize_packed_qkv(input_type, output_type)
                or (input_type, output_type) in seen
            ):
                continue
            seen.add((input_type, output_type))
            communication = _packed_qkv_combine_cost(
                input_type,
                output_type,
                context.reshard_cost_model.grid_synchronization_cost,
            )
            results.append(DistributedCandidate(
                distributed_candidate_id(
                    node.id,
                    "packed_qkv_combine",
                    output_type,
                    (input_type,),
                ),
                output_type,
                (input_type,),
                min(communication, 2_000_000_000),
                "packed-qkv-combine-sbp",
                target_op="ntt.packed_qkv_parallel_linear_combine",
                objective_kind="analytic",
                objective_model="flagmega.packed-qkv-combine-distribution/v2",
                objective_evidence=(
                    "coupled-three-field-sum-partial-materialization",
                    "local-output-by-partial-fan-in",
                ),
                target_attrs={"output_type": output_type},
            ))
        return tuple(results)

    def create_candidate_attrs(
        self,
        context: DistributedCandidateContext,
        return_type: IRType,
    ) -> Mapping[str, object]:
        del context
        return {"output_type": return_type}


def _packed_qkv_materialized_type(value: IRType) -> TupleType | None:
    if not isinstance(value, TupleType) or len(value.fields) != 3:
        return None
    if all(not isinstance(field, DistributedType) for field in value.fields):
        return value
    if not all(isinstance(field, DistributedType) for field in value.fields):
        return None
    return TupleType(tuple(replace(field, partial=None) for field in value.fields))


def _packed_qkv_combine_cost(
    input_type: TupleType,
    output_type: TupleType,
    grid_synchronization_cost: int,
) -> int:
    """Port nncase's local-state read/write cost for QKV partial reduction."""

    if input_type == output_type:
        return 0
    total = 0
    for input_field, output_field in zip(input_type.fields, output_type.fields):
        if (
            not isinstance(input_field, DistributedType)
            or not isinstance(output_field, DistributedType)
            or input_field.partial is None
        ):
            return 2_000_000_000
        local_bytes = _tensor_bytes(local_tensor_type(output_field))
        fan_in = _shard_count(input_field.placement, input_field.partial.axes)
        total = min(
            total + local_bytes * fan_in + local_bytes,
            2_000_000_000,
        )
    return min(total + grid_synchronization_cost, 2_000_000_000)


def _same_tuple_tensors(lhs: IRType, rhs: IRType) -> bool:
    if not isinstance(lhs, TupleType) or not isinstance(rhs, TupleType):
        return False
    if len(lhs.fields) != len(rhs.fields):
        return False
    return all(
        tensor_of(left) == tensor_of(right)
        for left, right in zip(lhs.fields, rhs.fields)
    )


def _qkv_tail_types(
    tails: tuple[IRType, ...],
    placement: Placement,
    *,
    output_policies: tuple[SBP, SBP, SBP],
    partial: bool,
) -> tuple[IRType, ...]:
    values: list[IRType] = []
    for index, value in enumerate(tails):
        if isinstance(value, NoneType):
            values.append(value)
            continue
        tensor = tensor_of(value)
        if index < 3:
            if partial:
                raise ValueError("Partial packed-QKV candidates cannot carry biases.")
            values.append(DistributedType(
                tensor, (output_policies[index],), placement
            ))
        else:
            values.append(broadcast_type(tensor, placement))
    return tuple(values)


def _have_coupled_qkv_output_policies(
    policies: tuple[SBPSplit, SBPSplit, SBPSplit],
) -> bool:
    """Match nncase's coupling rule without forcing equal block sizes.

    Q, K, and V may have different logical extents.  Their block-cyclic
    policies are coupled when they traverse the same placement stages; each
    field may choose its own block size.  Contiguous policies have no such
    indirection and therefore must be identical.
    """

    reference = policies[0]
    return all(
        len(policy.stages) == len(reference.stages)
        and all(
            left.hierarchy_axes == right.hierarchy_axes
            and (
                isinstance(left.distribution, BlockCyclicSplit)
                and isinstance(right.distribution, BlockCyclicSplit)
                or isinstance(left.distribution, ContiguousSplit)
                and isinstance(right.distribution, ContiguousSplit)
                and left.distribution == right.distribution
            )
            for left, right in zip(policy.stages, reference.stages)
        )
        for policy in policies[1:]
    )


class MatMulGluCandidateProvider(DistributedCandidateProviderBase):
    op_names = frozenset({
        "nn.dense_matmul_glu",
        "nn.packed_dense_matmul_glu",
        "nn.matmul_glu",
        "nn.packed_matmul_glu",
    })
    allows_partial_inputs = False
    is_exhaustive = True

    def _enumerate_candidates(self, context: DistributedCandidateContext) -> tuple[DistributedCandidate, ...]:
        node = context.source_call
        module = context.module
        value = tensor_of(module.node_map[node.inputs[0]].type)
        output = tensor_of(node.type)
        work = min(_matmul_work(output, value) * 2, 2_000_000_000)
        placement = context.placement
        original_inputs = [module.node_map[value_id].type for value_id in node.inputs]
        gate = tensor_of(original_inputs[1])
        up = tensor_of(original_inputs[2])
        weight_n_axis = 1 if node.op == "nn.packed_dense_matmul_glu" else 0
        n_lane = (
            parse_k_major_layout(str(node.attrs["packed_layout"]))[0]
            if node.op == "nn.packed_dense_matmul_glu"
            else 1
        )
        values = [_replicated(context, work)]
        split_index = 0
        for hierarchy_axes in _mesh_axis_combinations(placement):
            for output_policy in context.split_candidates(
                output,
                output.rank - 1,
                hierarchy_axes,
                purpose="output",
            ):
                weight_n_policy = scale_split_units(output_policy, 1, n_lane)
                if weight_n_policy is None:
                    continue
                weight_policies = [SBP.broadcast() for _ in gate.shape]
                weight_policies[weight_n_axis] = weight_n_policy
                if not is_distributable(gate, tuple(weight_policies), placement):
                    continue
                inputs = list(original_inputs)
                inputs[0] = broadcast_type(value, placement)
                inputs[1] = DistributedType(
                    gate, tuple(weight_policies), placement
                )
                inputs[2] = DistributedType(
                    up, tuple(weight_policies), placement
                )
                for index in range(3, len(inputs)):
                    inputs[index] = broadcast_type(
                        tensor_of(inputs[index]), placement
                    )
                output_policies = [SBP.broadcast() for _ in output.shape]
                output_policies[output.rank - 1] = output_policy
                values.append(DistributedCandidate(
                    _candidate_id(
                        node.id,
                        f"variant_{split_index}.output_split",
                        hierarchy_axes,
                        placement,
                    ),
                    DistributedType(
                        output, tuple(output_policies), placement
                    ),
                    tuple(inputs),
                    max(work // _shard_count(placement, hierarchy_axes), 1),
                    "matmul-glu-output-sbp",
                ))
                split_index += 1
        return tuple(values)


def _axis_divides(tensor: TensorType, axis: int, divisor: int) -> bool:
    dimension = tensor.shape[axis]
    return not dimension.is_fixed or dimension.fixed_value % divisor == 0


class BinaryCandidateProvider(DistributedCandidateProviderBase):
    op_names = frozenset({
        "math.add",
        "math.mul",
        "math.vectorized_binary",
    })
    allows_partial_inputs = False
    is_exhaustive = True

    def _enumerate_candidates(self, context: DistributedCandidateContext) -> tuple[DistributedCandidate, ...]:
        node = context.source_call
        module = context.module
        output = tensor_of(node.type)
        work = max(_tensor_bytes(output) // max(output.dtype.itemsize, 1), 1)
        values = [_replicated(context, work)]
        for hierarchy_axes in _mesh_axis_combinations(context.placement):
            if not _can_split(
                hierarchy_axes,
                context.placement,
                (output, output.rank - 1),
            ):
                continue
            split = split_type(
                output, output.rank - 1, context.placement, hierarchy_axes)
            values.append(DistributedCandidate(
                _candidate_id(node.id, "exact_split", hierarchy_axes, context.placement),
                split,
                (split, split),
                max(work // _shard_count(context.placement, hierarchy_axes), 1),
                "binary-exact-output-sbp",
            ))
        return tuple(values)


class GdnConvolutionCandidateProvider(DistributedCandidateProviderBase):
    op_names = frozenset({"nn.gdn_convolution"})
    allows_partial_inputs = True
    is_exhaustive = True

    def _enumerate_candidates(self, context: DistributedCandidateContext) -> tuple[DistributedCandidate, ...]:
        node = context.source_call
        module = context.module
        placement = context.placement
        logical_inputs = [module.node_map[value].type for value in node.inputs]
        replicated = _replicated(context, 900_000_000)
        assert isinstance(node.type, TupleType) and len(node.type.fields) == 2
        qkv = tensor_of(logical_inputs[0])
        conv_weight = tensor_of(logical_inputs[2])
        output = tensor_of(node.type.fields[0])
        work = 900_000_000
        values = [replicated]
        hierarchy_axes = tuple(range(placement.rank))
        divisor = _shard_count(placement, hierarchy_axes)
        if _can_split(
            hierarchy_axes,
            placement,
            (qkv, 1),
            (conv_weight, 0),
            (output, 1),
        ):
            values.append(DistributedCandidate(
                _candidate_id(node.id, "channel_split", hierarchy_axes, placement),
                TupleType((
                    split_type(output, 1, placement, hierarchy_axes),
                    node.type.fields[1],
                )),
                (
                    split_type(qkv, 1, placement, hierarchy_axes),
                    logical_inputs[1],
                    split_type(conv_weight, 0, placement, hierarchy_axes),
                ),
                max(work // divisor, 1),
                "gdn-convolution-channel-sbp",
            ))
            values.extend(_gdn_direct_partial_candidates(
                context,
                tensor_axis=1,
                materialized_return=TupleType((
                    split_type(output, 1, placement, hierarchy_axes),
                    node.type.fields[1],
                )),
                base_inputs=(
                    split_type(qkv, 1, placement, hierarchy_axes),
                    logical_inputs[1],
                    split_type(conv_weight, 0, placement, hierarchy_axes),
                ),
                source_input_index=0,
                work=work,
                reason="gdn-convolution-direct-sum-partial",
            ))
        return tuple(values)


class GdnRecurrentCandidateProvider(DistributedCandidateProviderBase):
    op_names = frozenset({"nn.gdn_recurrent_core"})
    allows_partial_inputs = True
    is_exhaustive = True

    def _enumerate_candidates(self, context: DistributedCandidateContext) -> tuple[DistributedCandidate, ...]:
        node = context.source_call
        module = context.module
        placement = context.placement
        logical_inputs = [module.node_map[value].type for value in node.inputs]
        replicated = _replicated(context, 1_200_000_000)
        assert isinstance(node.type, TupleType) and isinstance(node.type.fields[0], TensorType)
        z = tensor_of(logical_inputs[2])
        output = tensor_of(node.type.fields[0])
        work = 1_200_000_000
        values = [replicated]
        hierarchy_axes = tuple(range(placement.rank))
        divisor = _shard_count(placement, hierarchy_axes)
        if _can_split(hierarchy_axes, placement, (z, 1), (output, 1)):
            inputs: list[IRType] = [logical_inputs[0]]
            for index, value in enumerate(logical_inputs[1:], start=1):
                tensor = tensor_of(value)
                inputs.append(
                    split_type(tensor, 1, placement, hierarchy_axes)
                    if index == 2
                    else broadcast_type(tensor, placement)
                )
            return_type = TupleType((
                split_type(output, 1, placement, hierarchy_axes),
                node.type.fields[1],
            ))
            values.append(DistributedCandidate(
                _candidate_id(node.id, "head_split", hierarchy_axes, placement),
                return_type,
                tuple(inputs),
                max(work // divisor, 1),
                "gdn-recurrent-head-sbp",
            ))
            values.extend(_gdn_direct_partial_candidates(
                context,
                tensor_axis=1,
                materialized_return=return_type,
                base_inputs=tuple(inputs),
                source_input_index=2,
                work=work,
                reason="gdn-recurrent-direct-sum-partial",
            ))
        return tuple(values)


def _gdn_direct_partial_candidates(
    context: DistributedCandidateContext,
    *,
    tensor_axis: int,
    materialized_return: IRType,
    base_inputs: tuple[IRType, ...],
    source_input_index: int,
    work: int,
    reason: str,
) -> tuple[DistributedCandidate, ...]:
    """Reuse legal upstream Sum-partial values, matching nncase's providers."""

    if len(context.available_input_types) != len(base_inputs):
        return ()
    result: list[DistributedCandidate] = []
    placement = context.placement
    all_axes = set(range(placement.rank))
    source_tensor = tensor_of(context.module.node_map[context.source_call.inputs[source_input_index]].type)
    for available in context.available_input_types[source_input_index]:
        if not isinstance(available, DistributedType):
            continue
        if available.tensor != source_tensor or available.placement != placement or available.partial is None:
            continue
        policy = available.axis_policies[tensor_axis]
        split_axes = set(policy.hierarchy_axes) if hasattr(policy, "hierarchy_axes") else set()
        partial_axes = set(available.partial.axes)
        if (
            any(
                not isinstance(item, type(SBP.broadcast()))
                for index, item in enumerate(available.axis_policies)
                if index != tensor_axis
            )
            or split_axes & partial_axes
            or split_axes | partial_axes != all_axes
        ):
            continue
        inputs = list(base_inputs)
        inputs[source_input_index] = available
        axes = "_".join(str(axis) for axis in available.partial.axes)
        result.append(DistributedCandidate(
            f"distribution.{context.source_call.id}.partial_{axes}.direct",
            materialized_return,
            tuple(inputs),
            max(work // placement.size, 1),
            reason,
        ))
    return tuple(result)


__all__ = [
    "BinaryCandidateProvider",
    "BroadcastCandidateProvider",
    "EmbeddingCandidateProvider",
    "GdnConvolutionCandidateProvider",
    "GdnRecurrentCandidateProvider",
    "MatMulCandidateProvider",
    "MatMulGluCandidateProvider",
]
