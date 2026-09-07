# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""nncase-shaped distribution candidates for explicit normalization dataflow."""

from __future__ import annotations

from dataclasses import replace
from math import prod

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir import (
    DistributedType,
    IRType,
    Node,
    TupleType,
    local_tensor_type,
)
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.ops.nn.norm_apply import NormApply
from triton.flagmega.ir.ops.nn.norm_stats import NormStats
from triton.flagmega.ir.ops.ntt.matmul_norm_stats_combine import (
    MatMulNormStatsCombine,
    can_materialize_matmul_partial,
)
from triton.flagmega.passes.auto_distributed.candidates import (
    DistributedCandidate,
    DistributedCandidateContext,
    DistributedCandidateProviderBase,
)
from triton.flagmega.passes.auto_distributed.candidate_identity import (
    distributed_candidate_id,
)


class NormStatsCandidateProvider(DistributedCandidateProviderBase):
    op_names = frozenset({"nn.norm_stats"})
    allows_partial_inputs = False
    is_exhaustive = True

    def _enumerate_candidates(
        self, context: DistributedCandidateContext,
    ) -> tuple[DistributedCandidate, ...]:
        node = context.source_call
        value = tensor_of(context.module.node_map[node.inputs[0]].type)
        results: list[DistributedCandidate] = []
        for input_type in _candidate_distributed_inputs(context, 0, value):
            input_node = _typed_node("<norm_stats_input>", input_type)
            try:
                output_type = NormStats.infer_type((input_node,), node.attrs)
            except IRSchemaError:
                continue
            assert isinstance(output_type, DistributedType)
            factors = NormStats.cost_factors((input_node,), node.attrs, output_type)
            results.append(DistributedCandidate(
                distributed_candidate_id(
                    node.id, "norm_stats", output_type, (input_type,)
                ),
                output_type,
                (input_type,),
                (
                    _local_elements(input_type)
                    if factors is None
                    else context.operation_cost_model.get_latency(
                        factors, output_type
                    )
                ),
                "norm-stats-additive-sbp",
                objective_model=(
                    "flagmega.norm-stats-local-work/v1"
                    if factors is None
                    else context.operation_cost_model.identity
                ),
                objective_evidence=(
                    "additive-statistics",
                    "partial-on-reduced-split",
                    "op-definition-cost-factors",
                    "hierarchical-target-latency",
                ),
            ))
        return tuple(results)


class NormApplyCandidateProvider(DistributedCandidateProviderBase):
    op_names = frozenset({"nn.norm_apply"})
    allows_partial_inputs = False
    is_exhaustive = True

    def _enumerate_candidates(
        self, context: DistributedCandidateContext,
    ) -> tuple[DistributedCandidate, ...]:
        node = context.source_call
        module = context.module
        value = tensor_of(module.node_map[node.inputs[0]].type)
        scale = tensor_of(module.node_map[node.inputs[2]].type)
        bias = tensor_of(module.node_map[node.inputs[3]].type)
        axis = _normalize_axis(int(node.attrs["axis"]), value.rank)
        results: list[DistributedCandidate] = []
        for input_type in _candidate_distributed_inputs(context, 0, value):
            policies = input_type.axis_policies
            partial_stats = NormStats.infer_type(
                (_typed_node("<norm_apply_input>", input_type),),
                {"axis": axis, "use_mean": bool(node.attrs["use_mean"])},
            )
            assert isinstance(partial_stats, DistributedType)
            stats_type = replace(partial_stats, partial=None)
            suffix_policies = tuple(policies[axis:])
            try:
                scale_type = DistributedType(scale, suffix_policies, context.placement)
                bias_type = DistributedType(bias, suffix_policies, context.placement)
                inputs = (
                    _typed_node("<norm_apply_input>", input_type),
                    _typed_node("<norm_apply_stats>", stats_type),
                    _typed_node("<norm_apply_scale>", scale_type),
                    _typed_node("<norm_apply_bias>", bias_type),
                )
                output_type = NormApply.infer_type(inputs, node.attrs)
            except IRSchemaError:
                continue
            factors = NormApply.cost_factors(inputs, node.attrs, output_type)
            results.append(DistributedCandidate(
                distributed_candidate_id(
                    node.id,
                    "norm_apply",
                    output_type,
                    tuple(item.type for item in inputs),
                ),
                output_type,
                tuple(item.type for item in inputs),
                (
                    _local_elements(input_type)
                    if factors is None
                    else context.operation_cost_model.get_latency(
                        factors, output_type
                    )
                ),
                "norm-apply-compatible-sbp",
                objective_model=(
                    "flagmega.norm-apply-local-work/v1"
                    if factors is None
                    else context.operation_cost_model.identity
                ),
                objective_evidence=(
                    "materialized-statistics",
                    "suffix-parameter-policy",
                    "op-definition-cost-factors",
                    "hierarchical-target-latency",
                ),
            ))
        return tuple(results)


class BindNormStatsCandidateProvider(DistributedCandidateProviderBase):
    op_names = frozenset({"nn.bind_norm_stats"})
    allows_partial_inputs = False
    is_exhaustive = True

    def _enumerate_candidates(
        self, context: DistributedCandidateContext,
    ) -> tuple[DistributedCandidate, ...]:
        node = context.source_call
        value = tensor_of(context.module.node_map[node.inputs[0]].type)
        results: list[DistributedCandidate] = []
        for input_type in _candidate_distributed_inputs(context, 0, value):
            expected = NormStats.infer_type(
                (_typed_node("<bind_norm_stats_input>", input_type),), node.attrs)
            assert isinstance(expected, DistributedType)
            materialized = replace(expected, partial=None)
            results.append(DistributedCandidate(
                distributed_candidate_id(
                    node.id,
                    "bind_norm_stats",
                    materialized,
                    (input_type, materialized),
                ),
                materialized,
                (input_type, materialized),
                0,
                "bind-materialized-norm-stats-sbp",
                objective_kind="analytic",
                objective_model="flagmega.semantic-zero/v1",
                objective_evidence=("constraint-only-op",),
            ))
        return tuple(results)


class MatMulNormStatsCombineCandidateProvider(DistributedCandidateProviderBase):
    """Search partial materialization and both normalization outputs together."""

    op_names = frozenset({"ntt.matmul_norm_stats_combine"})
    allows_partial_inputs = True
    is_exhaustive = True

    def _enumerate_candidates(
        self, context: DistributedCandidateContext,
    ) -> tuple[DistributedCandidate, ...]:
        node = context.source_call
        if len(context.available_input_types) != 2 or not isinstance(node.type, TupleType):
            return ()
        expected_tensors = tuple(tensor_of(field) for field in node.type.fields)
        results: list[DistributedCandidate] = []
        seen: set[tuple[IRType, IRType]] = set()
        addend_types = list(context.available_input_types[1])
        # A requested value layout may originate at the projection, not just
        # the addend. Preserve its exact staged N policy when materializing
        # partials; the selected edge still proves the addend reshard.
        for projection in context.available_input_types[0]:
            if isinstance(projection, DistributedType) and projection.placement == context.placement:
                materialized = replace(projection, partial=None)
                if materialized not in addend_types:
                    addend_types.append(materialized)
        # A materialized addend may always be narrowed from Broadcast to a
        # storage-only Split view.  This is essential for a projection whose
        # output-N is split: the fused kernel must preserve that split and
        # publish owner-local normalization statistics instead of gathering
        # the whole value only to split it again at NormApply.  The reshard
        # realization policy remains responsible for proving that the chosen
        # view is physically legal.
        for candidate in context.leaf_candidate_types(expected_tensors[0]):
            if candidate not in addend_types:
                addend_types.append(candidate)
        for addend_type in addend_types:
            if tensor_of(addend_type) != expected_tensors[0]:
                continue
            try:
                stats_type = NormStats.infer_type(
                    (_typed_node("<combine_addend>", addend_type),),
                    {
                        "axis": int(node.attrs["axis"]),
                        "use_mean": bool(node.attrs["use_mean"]),
                    },
                )
            except IRSchemaError:
                continue
            if tensor_of(stats_type) != expected_tensors[1]:
                continue
            for input_type in context.available_input_types[0]:
                relation = (input_type, addend_type)
                if relation in seen or not can_materialize_matmul_partial(
                    input_type, addend_type
                ):
                    continue
                seen.add(relation)
                output_type = TupleType((addend_type, stats_type))
                typed_inputs = (
                    _typed_node("<combine_input>", input_type),
                    _typed_node("<combine_addend>", addend_type),
                )
                factors = MatMulNormStatsCombine.cost_factors(
                    typed_inputs, node.attrs, output_type
                )
                if factors is None:
                    local_work = _type_local_elements(addend_type)
                    communication = (
                        _type_local_elements(input_type)
                        if isinstance(input_type, DistributedType)
                        and input_type.partial is not None
                        else 0
                    )
                    operation_cost = min(
                        local_work * (4 if node.attrs["use_mean"] else 3)
                        + communication
                        + (
                            context.reshard_cost_model.grid_synchronization_cost
                            if input_type != addend_type
                            else 0
                        ),
                        2_000_000_000,
                    )
                    objective_model = (
                        "flagmega.matmul-norm-stats-combine-distribution/v1"
                    )
                else:
                    operation_cost = context.operation_cost_model.get_latency(
                        factors, output_type
                    )
                    objective_model = context.operation_cost_model.identity
                results.append(DistributedCandidate(
                    distributed_candidate_id(
                        node.id,
                        "matmul_norm_stats_combine",
                        output_type,
                        relation,
                    ),
                    output_type,
                    relation,
                    operation_cost,
                    "matmul-partial-materialize-add-norm-stats-sbp",
                    objective_kind="analytic",
                    objective_model=objective_model,
                    objective_evidence=(
                        "sum-partial-materialization",
                        "collective-grid-synchronization",
                        "addend-defines-materialized-layout",
                        "additive-statistics-output",
                        "op-definition-cost-factors",
                        "hierarchical-target-latency",
                    ),
                ))
        return tuple(results)


def _typed_node(node_id: str, value_type) -> Node:
    return Node(node_id, "builtin.var", (), value_type, attrs={"name": node_id})


def _candidate_distributed_inputs(
    context: DistributedCandidateContext,
    input_index: int,
    tensor,
) -> tuple[DistributedType, ...]:
    """Keep producer-specific split units before adding leaf layouts.

    PyNTT contiguous candidates may deliberately reserve a larger local
    capacity than ``extent / owners``.  Re-enumerating only canonical leaf
    policies silently changes that contract at NormStats/NormApply and forces
    an otherwise unnecessary boxing edge.
    """

    values: list[DistributedType] = []
    if input_index < len(context.available_input_types):
        for value in context.available_input_types[input_index]:
            if (
                isinstance(value, DistributedType)
                and value.tensor == tensor
                and value.placement == context.placement
                and value.partial is None
                and value not in values
            ):
                values.append(value)
    for value in context.leaf_candidate_types(tensor):
        if value not in values:
            values.append(value)
    return tuple(values)


def _normalize_axis(axis: int, rank: int) -> int:
    value = axis + rank if axis < 0 else axis
    if value < 0 or value >= rank:
        raise IRSchemaError(f"Normalization axis {axis} is out of range for rank {rank}.")
    return value


def _local_elements(value: DistributedType) -> int:
    tensor = local_tensor_type(value)
    if any(not dimension.is_fixed for dimension in tensor.shape):
        return 1 << 20
    lanes = getattr(tensor.dtype, "lane_count", 1)
    return min(prod(dimension.fixed_value for dimension in tensor.shape) * lanes, 2_000_000_000)


def _type_local_elements(value: IRType) -> int:
    tensor = local_tensor_type(value) if isinstance(value, DistributedType) else tensor_of(value)
    if any(not dimension.is_fixed for dimension in tensor.shape):
        return 1 << 20
    lanes = getattr(tensor.dtype, "lane_count", 1)
    return min(
        prod(dimension.fixed_value for dimension in tensor.shape) * lanes,
        2_000_000_000,
    )


__all__ = [
    "BindNormStatsCandidateProvider",
    "MatMulNormStatsCombineCandidateProvider",
    "NormApplyCandidateProvider",
    "NormStatsCandidateProvider",
]
