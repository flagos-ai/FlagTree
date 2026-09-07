# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Target-owned construction of legal tensor-axis split policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from triton.flagmega.ir import Placement, SBP, SBPSplit, TensorType


@dataclass(frozen=True)
class DistributedSplitCandidateContext:
    tensor_type: TensorType
    tensor_axis: int
    placement: Placement
    hierarchy_axes: tuple[int, ...]
    contiguous_granularity: int | None
    maximum_extent: int | None
    purpose: str = "generic"

    def __post_init__(self) -> None:
        if self.purpose not in {"generic", "output", "reduction"}:
            raise ValueError(
                f"Unknown distributed split candidate purpose {self.purpose!r}."
            )


class DistributedSplitCandidateProvider(Protocol):
    @property
    def identity(self) -> str: ...

    def get_candidates(
        self,
        context: DistributedSplitCandidateContext,
    ) -> tuple[SBPSplit, ...]: ...


class ContiguousDistributedSplitCandidateProvider:
    """Portable fallback used by targets without a physical block policy."""

    identity = "contiguous-split/v1"

    def get_candidates(
        self,
        context: DistributedSplitCandidateContext,
    ) -> tuple[SBPSplit, ...]:
        return (
            SBP.split_contiguous(
                context.hierarchy_axes,
                context.contiguous_granularity,
            ),
        )


__all__ = [
    "ContiguousDistributedSplitCandidateProvider",
    "DistributedSplitCandidateContext",
    "DistributedSplitCandidateProvider",
]
