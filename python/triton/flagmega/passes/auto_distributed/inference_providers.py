# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Candidates derived from an operation's own distributed type contract."""

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
)
from triton.flagmega.ir.distributed_inference import broadcast_ir_type
from triton.flagmega.passes.auto_distributed.candidates import (
    DistributedCandidate,
    DistributedCandidateContext,
    DistributedCandidateProviderBase,
)
from triton.flagmega.passes.auto_distributed.candidate_identity import (
    distributed_candidate_id,
)


class TypeInferenceCandidateProvider(DistributedCandidateProviderBase):
    """Mirror nncase's generic cartesian-candidate type inference path.

    The provider owns no layout policy.  It feeds the producer layouts already
    available in the search graph into the operation's handwritten
    ``infer_type`` implementation.  A replicated form is included explicitly
    so originators can enter the distributed domain through a reviewed edge.
    Invalid combinations are rejected by the operation definition itself.
    """

    allows_partial_inputs = False
    is_exhaustive = True

    def __init__(self, op_names: frozenset[str]) -> None:
        if not op_names:
            raise ValueError(
                "TypeInferenceCandidateProvider requires at least one op name.")
        self.op_names = frozenset(op_names)

    def _enumerate_candidates(
        self,
        context: DistributedCandidateContext,
    ) -> tuple[DistributedCandidate, ...]:
        node = context.source_call
        if node.op not in self.op_names:
            return ()
        logical_inputs = tuple(
            context.module.node_map[input_id].type for input_id in node.inputs
        )
        if len(context.available_input_types) != len(logical_inputs):
            return ()
        definition = get_definition(node.op)
        choices = tuple(
            _candidate_input_types(
                available,
                logical,
                context,
                distribute=parameter.parameter_kind == ParameterKind.INPUT,
            )
            for available, logical, parameter in zip(
                context.available_input_types,
                logical_inputs,
                definition.input_parameters,
            )
        )
        if any(not values for values in choices):
            return ()

        results: list[DistributedCandidate] = []
        seen: set[tuple[IRType, tuple[IRType, ...]]] = set()
        for input_types in product(*choices):
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
            except (IRSchemaError, AssertionError, ValueError):
                continue
            relation = (return_type, tuple(input_types))
            if relation in seen:
                continue
            seen.add(relation)
            factors = definition.cost_factors(
                tuple(typed_inputs), node.attrs, return_type
            )
            results.append(DistributedCandidate(
                distributed_candidate_id(
                    node.id, "inferred", return_type, tuple(input_types)
                ),
                return_type,
                tuple(input_types),
                (
                    _operation_local_work_bytes(return_type, tuple(input_types))
                    if factors is None
                    else context.operation_cost_model.get_latency(
                        factors, return_type
                    )
                ),
                "operation-type-inference-sbp",
                objective_kind="heuristic" if factors is None else "analytic",
                objective_model=(
                    "flagmega.operation-type-inference-local-work/v1"
                    if factors is None
                    else context.operation_cost_model.identity
                ),
                objective_evidence=(
                    "producer-candidate-layout",
                    "operation-owned-distributed-type-inference",
                    "op-definition-cost-factors",
                    "hierarchical-target-latency",
                ),
            ))
        return tuple(results)


def _candidate_input_types(
    available: tuple[IRType, ...],
    logical: IRType,
    context: DistributedCandidateContext,
    *,
    distribute: bool,
) -> tuple[IRType, ...]:
    if not distribute:
        # nncase's VisitLeafArgument terminates ParameterKind.Attribute
        # operands.  The operand remains a graph value (and editable Python
        # expression), but it cannot acquire an SBP contract.
        return (logical,)
    values: list[IRType] = [broadcast_ir_type(logical, context.placement)]
    for value in available:
        # A logical tensor originates outside the distributed domain. Mirror
        # nncase TryAddOriginator: expose target-owned leaf layouts at this use
        # edge, not just B. Otherwise a Cast directly on a function parameter
        # can never be shard-local, even if its consumer and refined ABI are S.
        # Already-distributed producers retain only their available contracts;
        # arbitrary new distributions must still be explicit reshard edges.
        if isinstance(value, TensorType):
            candidates = context.leaf_candidate_types(value)
        else:
            candidates = (broadcast_ir_type(value, context.placement) if _is_logical_tensor(value) else value,)
        for candidate in candidates:
            if candidate not in values:
                values.append(candidate)
    return tuple(values)


def _is_logical_tensor(value: IRType) -> bool:
    if isinstance(value, TensorType):
        return True
    if isinstance(value, TupleType):
        return any(_is_logical_tensor(field) for field in value.fields)
    return False


def _local_work_bytes(value: IRType) -> int:
    if isinstance(value, RefType):
        return 0
    if isinstance(value, TupleType):
        return min(sum(_local_work_bytes(field) for field in value.fields), 2_000_000_000)
    tensor = local_tensor_type(value) if isinstance(value, DistributedType) else value
    if not isinstance(tensor, TensorType):
        return 0
    if any(not dimension.is_fixed for dimension in tensor.shape):
        return 1 << 20
    # ``VectorType.itemsize`` already includes its payload lanes.
    elements = prod(dimension.fixed_value for dimension in tensor.shape)
    return min(elements * tensor.dtype.itemsize, 2_000_000_000)


def _operation_local_work_bytes(
    return_type: IRType,
    input_types: tuple[IRType, ...],
) -> int:
    output_bytes = _local_work_bytes(return_type)
    if output_bytes:
        return output_bytes
    # Stateful operations return an identity/reference handle.  The handle has
    # no payload bytes, but reading or writing its tensor operand is still
    # local work and must distinguish replicated from sharded candidates.
    return min(sum(_local_work_bytes(value) for value in input_types), 2_000_000_000)


__all__ = ["TypeInferenceCandidateProvider"]
