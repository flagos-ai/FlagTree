# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Candidates derived from an operation's own distributed type contract."""

from __future__ import annotations

from dataclasses import replace
from math import prod

from triton.flagmega.ir import (
    DistributedType,
    IRType,
    ParameterKind,
    RefType,
    TensorType,
    TupleType,
    get_definition,
    local_tensor_type,
)
from triton.flagmega.ir.distributed_inference import broadcast_ir_type
from triton.flagmega.passes.auto_distributed.candidate_identity import (
    distributed_candidate_id,
)
from triton.flagmega.passes.auto_distributed.candidates import (
    DistributedCandidate,
    DistributedCandidateContext,
    DistributedCandidateProviderBase,
    DistributedCandidateTuple,
)
from triton.flagmega.passes.auto_distributed.type_analysis import infer_candidate


class TypeInferenceCandidateProvider(DistributedCandidateProviderBase):
    """Combine forward inference with operation-owned inverse type relations.

    The provider owns no layout policy.  It feeds the producer layouts already
    available in the search graph into the operation's handwritten
    ``infer_type`` implementation. Ops with a declared inverse may also use
    target-owned output seeds and consumer demands, even when their requested
    input types need an explicit reshard. Every inverse relation is checked by
    forward inference; unknown inverse semantics are never invented here.
    """

    allows_partial_inputs = False
    is_exhaustive = True

    def __init__(self, op_names: frozenset[str]) -> None:
        if not op_names:
            raise ValueError(
                "TypeInferenceCandidateProvider requires at least one op name.")
        self.op_names = frozenset(op_names)

    def get_return_candidate_types(self, context, default_return_types):
        values = dict.fromkeys(c.return_type for c in self.get_candidates(context))
        for output_type in default_return_types:
            if self._inverse_candidates(context, output_type):
                values[output_type] = None
        return tuple(values)

    def _candidates_for_return(self, context, return_type):
        values = {c.id: c for c in self.get_candidates(context) if c.return_type == return_type}
        values.update((c.id, c) for c in self._inverse_candidates(context, return_type))
        return tuple(values.values())

    def try_get_input_type_tuples(self, context, return_type):
        return tuple(DistributedCandidateTuple(c.input_types, c.reason)
                     for c in self._candidates_for_return(context, return_type))

    def create_candidate(self, context, return_type, inputs):
        candidate = next(c for c in self._candidates_for_return(context, return_type)
                         if c.input_types == inputs.input_types and c.reason == inputs.reason)
        return replace(candidate, target_op=context.source_call.op)

    def _inverse_candidates(self, context, output_type):
        if context.source_call.op not in self.op_names:
            return ()
        if isinstance(output_type, DistributedType) and output_type.placement != context.placement:
            return ()
        definition = get_definition(context.source_call.op)
        logical_inputs = tuple(context.module.node_map[value].type for value in context.source_call.inputs)
        tuples = definition.infer_distributed_input_types(output_type, logical_inputs, context.source_call.attrs)
        return tuple(candidate for inputs in tuples or ()
                     if (candidate := self._candidate(context, tuple(inputs))) is not None
                     and candidate.return_type == output_type)

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
        # A variadic ParameterInfo describes every remaining operand, not
        # one tuple-valued edge. Preserve each actual edge in the candidate.
        parameters = tuple(
            parameter
            for parameter in definition.input_parameters
            for _ in (range(parameter.input_index, len(logical_inputs)) if parameter.variadic else (parameter.input_index,))
        )
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
                parameters,
                strict=True,
            )
        )
        if any(not values for values in choices):
            return ()

        results = {}
        for input_types in definition.distributed_input_type_tuples(choices, node.attrs):
            candidate = self._candidate(context, tuple(input_types))
            if candidate is not None:
                results[candidate.id] = candidate
        output_types = definition.distributed_output_type_candidates(choices, node.type, node.attrs)
        # Target-owned output seeds are useful only when the operation has an
        # inverse relation. A provider cannot synthesize one by copying SBPs.
        broad = broadcast_ir_type(node.type, context.placement)
        if definition.infer_distributed_input_types(broad, logical_inputs, node.attrs) is not None:
            output_types = (*output_types, *context.leaf_candidate_types(broad.tensor)) if isinstance(
                broad, DistributedType) else output_types
        for output_type in dict.fromkeys(output_types):
            for candidate in self._inverse_candidates(context, output_type):
                results[candidate.id] = candidate
        return tuple(results.values())

    def _candidate(self, context, input_types):
        if any(_has_partial(value) for value in input_types):
            return None
        node = context.source_call
        analysis = infer_candidate(context, get_definition(node.op), input_types)
        if analysis is None:
            return None
        return_type, factors = analysis
        return DistributedCandidate(
            distributed_candidate_id(node.id, "operation-type-inference-sbp", return_type, input_types),
            return_type,
            tuple(input_types),
            (_operation_local_work_bytes(return_type, tuple(input_types)) if factors is None
             else context.operation_cost_model.get_latency(factors, return_type)),
            "operation-type-inference-sbp",
            objective_kind="heuristic" if factors is None else "analytic",
            objective_model=("flagmega.operation-type-inference-local-work/v1" if factors is None
                             else context.operation_cost_model.identity),
            objective_evidence=("producer-candidate-layout", "operation-owned-distributed-type-inference",
                                "op-definition-cost-factors", "hierarchical-target-latency"),
        )


def _has_partial(value):
    return (isinstance(value, DistributedType) and value.partial is not None
            or isinstance(value, TupleType) and any(_has_partial(field) for field in value.fields))


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
