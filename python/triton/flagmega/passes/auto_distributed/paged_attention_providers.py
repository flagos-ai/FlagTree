# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Distributed providers for explicit paged-attention split states."""

from __future__ import annotations

from collections.abc import Mapping
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
)
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.passes.auto_distributed.candidates import (
    DistributedCandidate,
    DistributedCandidateContext,
    DistributedCandidateProviderBase,
)
from triton.flagmega.passes.auto_distributed.candidate_identity import (
    distributed_candidate_id,
)
from triton.flagmega.passes.auto_distributed.inference_providers import (
    TypeInferenceCandidateProvider,
)


class PagedAttentionPartialCandidateProvider(TypeInferenceCandidateProvider):
    """Infer partial-state candidates from available query layouts."""

    def __init__(self) -> None:
        super().__init__(frozenset({"ntt.paged_attention_partial"}))


class PagedAttentionCombineCandidateProvider(DistributedCandidateProviderBase):
    """Enumerate legal materialization/reduce-scatter output contracts."""

    op_names = frozenset({"ntt.paged_attention_combine"})
    allows_partial_inputs = True
    is_exhaustive = True

    def _enumerate_candidates(
        self,
        context: DistributedCandidateContext,
    ) -> tuple[DistributedCandidate, ...]:
        node = context.source_call
        if len(context.available_input_types) != 3:
            return ()
        expected = node.attrs.get("output_type")
        if not isinstance(expected, IRType):
            return ()
        expected_tensor = tensor_of(expected)
        definition = get_definition(node.op)
        outputs = context.leaf_candidate_types(expected_tensor)
        candidates: list[DistributedCandidate] = []
        seen: set[tuple[IRType, tuple[IRType, ...]]] = set()
        for input_types in product(*context.available_input_types):
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
            for output_type in outputs:
                normalized = definition.normalize_attrs({
                    **dict(node.attrs),
                    "output_type": output_type,
                })
                try:
                    inferred = definition.infer_type(typed_inputs, normalized)
                except (IRSchemaError, AssertionError, ValueError):
                    continue
                if inferred != output_type:
                    continue
                relation = (output_type, tuple(input_types))
                if relation in seen:
                    continue
                seen.add(relation)
                attrs = definition.ir_attrs(normalized)
                candidates.append(DistributedCandidate(
                    distributed_candidate_id(
                        node.id,
                        "paged_attention_combine",
                        output_type,
                        tuple(input_types),
                    ),
                    output_type,
                    tuple(input_types),
                    min(sum(_local_bytes(value) for value in input_types), 2_000_000_000),
                    "paged-attention-combine-sbp",
                    target_op=node.op,
                    objective_kind="analytic",
                    objective_model="flagmega.paged-attention-combine-distribution/v1",
                    objective_evidence=(
                        "matching-p-max-p-sum-p-sum-states",
                        "output-type-rebuild",
                    ),
                    target_attrs=attrs,
                ))
        return tuple(candidates)

    def create_candidate_attrs(
        self,
        context: DistributedCandidateContext,
        return_type: IRType,
    ) -> Mapping[str, object]:
        definition = get_definition(context.source_call.op)
        return definition.ir_attrs(definition.normalize_attrs({
            **dict(context.source_call.attrs),
            "output_type": return_type,
        }))


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


__all__ = [
    "PagedAttentionCombineCandidateProvider",
    "PagedAttentionPartialCandidateProvider",
]
