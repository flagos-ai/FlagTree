# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Distributed candidate provider for semantic QKV/RoPE/cache fusion."""

from __future__ import annotations

from itertools import product
from math import prod

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir import (
    DistributedType,
    IRType,
    Node,
    RefType,
    TensorType,
    TupleType,
    get_definition,
    local_tensor_type,
    ParameterKind,
    SBP,
)
from triton.flagmega.ir.distributed_inference import broadcast_type
from triton.flagmega.passes.auto_distributed.candidates import (
    DistributedCandidate,
    DistributedCandidateContext,
    DistributedCandidateProviderBase,
)
from triton.flagmega.passes.auto_distributed.candidate_identity import (
    distributed_candidate_id,
)


class QKVRoPEWithCacheCandidateProvider(DistributedCandidateProviderBase):
    """Port nncase's operation-inference candidate relation exactly."""

    op_names = frozenset({"nn.qkv_rope_with_cache"})
    allows_partial_inputs = False
    is_exhaustive = True

    def _enumerate_candidates(
        self,
        context: DistributedCandidateContext,
    ) -> tuple[DistributedCandidate, ...]:
        node = context.source_call
        if node.op not in self.op_names or len(context.available_input_types) != 10:
            return ()
        definition = get_definition(node.op)
        try:
            head_axis = tuple(node.attrs["qkv_layout"]).index("head")
        except (KeyError, TypeError, ValueError):
            return ()
        results: list[DistributedCandidate] = []
        seen: set[tuple[IRType, tuple[IRType, ...]]] = set()
        choices = []
        for index, (values, input_id, parameter) in enumerate(zip(
            context.available_input_types,
            node.inputs,
            definition.input_parameters,
        )):
            logical = context.module.node_map[input_id].type
            if index == 0:
                choices.append(
                    _qkv_candidate_input_types(
                        values, logical, context, head_axis=head_axis
                    )
                )
            else:
                choices.append(_candidate_input_types(
                    values,
                    logical,
                    context,
                    distribute=parameter.parameter_kind == ParameterKind.INPUT,
                ))
        choices = tuple(choices)
        for input_types in product(*choices):
            if not isinstance(input_types[0], TupleType) or len(input_types[0].fields) != 3:
                continue
            typed_inputs = tuple(
                Node(
                    f"<{node.id}.input.{index}>",
                    "builtin.var",
                    (),
                    value_type,
                    attrs={"name": f"<{node.id}.input.{index}>"},
                )
                for index, value_type in enumerate(input_types)
            )
            try:
                return_type = definition.infer_type(typed_inputs, node.attrs)
            except (IRSchemaError, AssertionError, TypeError, ValueError):
                continue
            if not isinstance(return_type, TupleType) or len(return_type.fields) != 2:
                continue
            relation = (return_type, tuple(input_types))
            if relation in seen:
                continue
            seen.add(relation)
            results.append(
                DistributedCandidate(
                    distributed_candidate_id(
                        node.id,
                        "qkv_rope_cache",
                        return_type,
                        tuple(input_types),
                    ),
                    return_type,
                    tuple(input_types),
                    min(
                        sum(_local_bytes(value) for value in input_types),
                        2_000_000_000,
                    ),
                    "qkv-rope-cache-output-sbp",
                    target_op=node.op,
                    objective_kind="analytic",
                    objective_model=(
                        "flagmega.qkv-rope-with-cache-distribution/v1"
                    ),
                    objective_evidence=(
                        "norm-stats-and-apply-type-relations",
                        "rope-type-relation",
                        "cache-update-type-relation",
                    ),
                )
            )
        return tuple(results)


def _local_bytes(value: IRType) -> int:
    if isinstance(value, RefType):
        return 0
    if isinstance(value, TupleType):
        return sum(_local_bytes(field) for field in value.fields)
    tensor = local_tensor_type(value) if isinstance(value, DistributedType) else value
    if not isinstance(tensor, TensorType):
        return 0
    if any(not dimension.is_fixed for dimension in tensor.shape):
        return 1 << 20
    elements = prod(dimension.fixed_value for dimension in tensor.shape)
    return elements * tensor.dtype.itemsize


def _candidate_input_types(
    values: tuple[IRType, ...],
    logical: IRType,
    context: DistributedCandidateContext,
    *,
    distribute: bool,
) -> tuple[IRType, ...]:
    if not distribute:
        return (logical,)
    result: list[IRType] = []
    for value in values:
        candidate = _lift_logical_tensors(value, context)
        if candidate not in result:
            result.append(candidate)
    return tuple(result)


def _qkv_candidate_input_types(
    values: tuple[IRType, ...],
    logical: IRType,
    context: DistributedCandidateContext,
    *,
    head_axis: int,
) -> tuple[IRType, ...]:
    """Enumerate the head-local layouts consumed by fused Q/K/V work.

    A reshape may be materialized from canonical storage and therefore expose
    only a broadcast inferred type.  nncase still offers downstream
    ``ShardedView`` candidates at the Q/K/V head boundary: query heads can use
    one placement axis while the usually smaller KV-head domain can use a
    different axis.  Enumerating those semantic layouts here lets the global
    AutoDistribution solve insert the views and account for the downstream
    local work; it does not prescribe which mesh axis a model must use.
    """

    if not isinstance(logical, TupleType) or len(logical.fields) != 3:
        return _candidate_input_types(
            values, logical, context, distribute=True
        )
    options: list[tuple[IRType, ...]] = []
    for field in logical.fields:
        tensor = field.tensor if isinstance(field, DistributedType) else field
        if not isinstance(tensor, TensorType) or not 0 <= head_axis < tensor.rank:
            return ()
        policies = [SBP.broadcast() for _ in tensor.shape]
        candidates: list[IRType] = [
            DistributedType(tensor, tuple(policies), context.placement)
        ]
        for placement_axis, owner_count in enumerate(
            context.placement.hierarchy
        ):
            if owner_count <= 1:
                continue
            for split in context.split_candidates(
                tensor,
                head_axis,
                (placement_axis,),
                purpose="output",
            ):
                split_policies = list(policies)
                split_policies[head_axis] = split
                candidate = DistributedType(
                    tensor, tuple(split_policies), context.placement
                )
                if candidate not in candidates:
                    candidates.append(candidate)
        options.append(tuple(candidates))

    result: list[IRType] = []
    for value in values:
        candidate = _lift_logical_tensors(value, context)
        if candidate not in result:
            result.append(candidate)
    for fields in product(*options):
        candidate = TupleType(tuple(fields), logical.is_variadic)
        if candidate not in result:
            result.append(candidate)
    return tuple(result)


def _lift_logical_tensors(
    value: IRType,
    context: DistributedCandidateContext,
) -> IRType:
    if isinstance(value, DistributedType):
        return value
    if isinstance(value, TensorType):
        return broadcast_type(value, context.placement)
    if isinstance(value, TupleType):
        return TupleType(
            tuple(_lift_logical_tensors(field, context) for field in value.fields),
            value.is_variadic,
        )
    return value


__all__ = ["QKVRoPEWithCacheCandidateProvider"]
