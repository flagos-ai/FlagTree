# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Candidate contracts mirroring nncase AutoDistributed providers."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping, Protocol, Sequence

from triton.flagmega.ir import (
    IRModule,
    IRType,
    Node,
    DistributedType,
    Placement,
    SBP,
    SBPSplit,
    TensorType,
    is_distributable,
    leaf_candidate_policies,
)
from triton.flagmega.passes.auto_distributed.split_candidates import (
    ContiguousDistributedSplitCandidateProvider,
    DistributedSplitCandidateContext,
    DistributedSplitCandidateProvider,
)
from triton.flagmega.passes.auto_distributed.reshard_cost import (
    DistributedReshardCostModel,
)
from triton.flagmega.passes.auto_distributed.operation_cost import (
    DistributedOperationCostModel,
)
from triton.flagmega.passes.auto_distributed.candidate_identity import (
    distributed_candidate_id,
)


@dataclass(frozen=True)
class DistributedCandidateContext:
    """One immutable source/layout/policy snapshot for provider enumeration.

    Return-type, input-tuple and target queries share this snapshot. A caller
    changing IR or provider policy creates a new context (``replace`` resets
    candidate snapshots). Pure type queries may share a fully op/type/attribute
    keyed memo within a search; a new search owns a fresh memo.
    """

    module: IRModule
    source_call: Node
    placement: Placement
    available_input_types: tuple[tuple[IRType, ...], ...]
    split_candidate_provider: DistributedSplitCandidateProvider = field(
        default_factory=ContiguousDistributedSplitCandidateProvider
    )
    reshard_cost_model: DistributedReshardCostModel = field(
        default_factory=DistributedReshardCostModel
    )
    operation_cost_model: DistributedOperationCostModel = field(
        default_factory=DistributedOperationCostModel
    )
    type_inference_memo: dict = field(default_factory=dict, compare=False, repr=False)
    _candidate_snapshots: dict[int, tuple[object, tuple[DistributedCandidate, ...]]] = field(
        default_factory=dict, init=False, repr=False, compare=False,
    )

    def split_candidates(
        self,
        tensor: TensorType,
        tensor_axis: int,
        hierarchy_axes: tuple[int, ...],
        *,
        purpose: str = "generic",
    ) -> tuple["SBPSplit", ...]:
        dimension = tensor.shape[tensor_axis]
        divisor = 1
        for axis in hierarchy_axes:
            divisor *= self.placement.hierarchy[axis]
        if divisor <= 1:
            return ()
        granularity = (
            None
            if (
                not dimension.is_fixed
                or dimension.fixed_value % divisor
            )
            else dimension.fixed_value // divisor
        )
        candidates = self.split_candidate_provider.get_candidates(
            DistributedSplitCandidateContext(
                tensor,
                tensor_axis,
                self.placement,
                hierarchy_axes,
                granularity,
                dimension.fixed_value if dimension.is_fixed else None,
                purpose,
            )
        )
        # A target owns the split distribution.  In particular, PyNTT's
        # block-cyclic policy permits ragged logical extents whereas a
        # contiguous split still requires exact divisibility.  Validate each
        # returned policy instead of rejecting the tensor before the target
        # has had a chance to describe its legal representation.
        result: list[SBPSplit] = []
        for candidate in candidates:
            policies = [SBP.broadcast() for _ in tensor.shape]
            policies[tensor_axis] = candidate
            if (
                candidate not in result
                and is_distributable(tensor, tuple(policies), self.placement)
            ):
                result.append(candidate)
        return tuple(result)

    def leaf_candidate_types(self, tensor: TensorType) -> tuple[DistributedType, ...]:
        """Enumerate target-owned leaf SBPs through this policy snapshot."""

        return tuple(
            DistributedType(tensor, policies, self.placement)
            for policies in leaf_candidate_policies(
                tensor, self.placement, split_candidates=self.split_candidates
            )
        )


@dataclass(frozen=True)
class DistributedCandidateTuple:
    input_types: tuple[IRType, ...]
    reason: str | None = None


@dataclass(frozen=True)
class DistributedCandidate:
    id: str
    return_type: IRType
    input_types: tuple[IRType, ...]
    operation_cost: int
    reason: str
    target_op: str | None = None
    objective_kind: str = "heuristic"
    objective_model: str = "flagmega.distributed-work/v1"
    objective_evidence: tuple[str, ...] = ()
    target_attrs: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if not self.id or not self.reason:
            raise ValueError("DistributedCandidate requires a non-empty id and reason.")
        if (
            isinstance(self.operation_cost, bool)
            or not isinstance(self.operation_cost, int)
            or self.operation_cost < 0
            or self.operation_cost > 2_000_000_000
        ):
            raise ValueError(
                "DistributedCandidate operation_cost must be an integer in [0, 2_000_000_000]."
            )
        if self.objective_kind not in {"analytic", "heuristic", "measured", "agent"}:
            raise ValueError(f"Unknown distributed objective kind {self.objective_kind!r}.")
        if not self.objective_model:
            raise ValueError("DistributedCandidate requires an objective model identifier.")
        evidence = tuple(str(value) for value in self.objective_evidence)
        if not evidence:
            evidence = (f"candidate-reason:{self.reason}",)
        object.__setattr__(self, "objective_evidence", evidence)


class DistributedCandidateProvider(Protocol):
    op_names: frozenset[str]
    allows_partial_inputs: bool
    is_exhaustive: bool

    def get_return_candidate_types(
        self,
        context: DistributedCandidateContext,
        default_return_types: Sequence[IRType],
    ) -> tuple[IRType, ...]: ...

    def try_get_input_type_tuples(
        self,
        context: DistributedCandidateContext,
        return_type: IRType,
    ) -> tuple[DistributedCandidateTuple, ...] | None: ...

    def create_candidate_target(
        self,
        context: DistributedCandidateContext,
        return_type: IRType,
    ) -> str: ...

    def create_candidate_attrs(
        self,
        context: DistributedCandidateContext,
        return_type: IRType,
    ) -> Mapping[str, object] | None: ...

    def create_candidate(
        self,
        context: DistributedCandidateContext,
        return_type: IRType,
        inputs: DistributedCandidateTuple,
    ) -> DistributedCandidate: ...


class DistributedCandidateProviderBase:
    """Handwritten provider base with nncase's three-part public contract.

    Concrete providers enumerate their small, target-specific candidate set in
    one local method. Search still consumes return types and input tuples
    separately, exactly like nncase. Public candidates are renamed from their
    implementation-local labels to the complete type relation, so inserting or
    reordering a legal candidate cannot retarget an editable selection.
    """

    def _enumerate_candidates(
        self,
        context: DistributedCandidateContext,
    ) -> tuple[DistributedCandidate, ...]:
        raise NotImplementedError

    def get_return_candidate_types(
        self,
        context: DistributedCandidateContext,
        default_return_types: Sequence[IRType],
    ) -> tuple[IRType, ...]:
        del default_return_types
        values: list[IRType] = []
        for candidate in self.get_candidates(context):
            if candidate.return_type not in values:
                values.append(candidate.return_type)
        return tuple(values)

    def try_get_input_type_tuples(
        self,
        context: DistributedCandidateContext,
        return_type: IRType,
    ) -> tuple[DistributedCandidateTuple, ...] | None:
        values = tuple(
            DistributedCandidateTuple(candidate.input_types, candidate.reason)
            for candidate in self.get_candidates(context)
            if candidate.return_type == return_type
        )
        return values

    def create_candidate_target(
        self,
        context: DistributedCandidateContext,
        return_type: IRType,
    ) -> str:
        del return_type
        return context.source_call.op

    def create_candidate_attrs(
        self,
        context: DistributedCandidateContext,
        return_type: IRType,
    ) -> Mapping[str, object] | None:
        del context, return_type
        return None

    def create_candidate(
        self,
        context: DistributedCandidateContext,
        return_type: IRType,
        inputs: DistributedCandidateTuple,
    ) -> DistributedCandidate:
        candidate = next(
            candidate
            for candidate in self.get_candidates(context)
            if candidate.return_type == return_type
            and candidate.input_types == inputs.input_types
            and candidate.reason == inputs.reason
        )
        return replace(
            candidate,
            target_op=self.create_candidate_target(context, return_type),
            target_attrs=self.create_candidate_attrs(context, return_type),
        )

    def get_candidates(self, context: DistributedCandidateContext) -> tuple[DistributedCandidate, ...]:
        """Enumerate once for all queries in the nncase provider protocol."""

        cached = context._candidate_snapshots.get(id(self))
        if cached is not None and cached[0] is self:
            return cached[1]
        values = tuple(
            replace(
                candidate,
                id=distributed_candidate_id(
                    context.source_call.id,
                    candidate.reason,
                    candidate.return_type,
                    candidate.input_types,
                ),
            )
            for candidate in self._enumerate_candidates(context)
        )
        ids = tuple(candidate.id for candidate in values)
        if len(set(ids)) != len(ids):
            duplicates = sorted({value for value in ids if ids.count(value) > 1})
            raise ValueError(
                "Distributed candidates must have unique semantic identities: "
                + ", ".join(duplicates)
            )
        context._candidate_snapshots[id(self)] = (self, values)
        return values


class DistributedCandidateProviderRegistry:
    """Explicit target-owned provider registry; no IoC or assembly scanning."""

    def __init__(self) -> None:
        self._providers: dict[str, DistributedCandidateProvider] = {}
        self._split_candidate_provider: DistributedSplitCandidateProvider = (
            ContiguousDistributedSplitCandidateProvider()
        )

    @property
    def split_candidate_provider(self) -> DistributedSplitCandidateProvider:
        return self._split_candidate_provider

    def set_split_candidate_provider(
        self,
        provider: DistributedSplitCandidateProvider,
    ) -> None:
        self._split_candidate_provider = provider

    def add(self, provider: DistributedCandidateProvider) -> None:
        for op_name in provider.op_names:
            if op_name in self._providers:
                raise ValueError(f"A distributed provider is already registered for {op_name!r}.")
            self._providers[op_name] = provider

    def try_get(self, op_name: str) -> DistributedCandidateProvider | None:
        return self._providers.get(op_name)

    @property
    def op_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._providers))


__all__ = [
    "DistributedCandidate",
    "DistributedCandidateContext",
    "DistributedCandidateProvider",
    "DistributedCandidateProviderRegistry",
    "DistributedCandidateTuple",
    "DistributedCandidateProviderBase",
]
