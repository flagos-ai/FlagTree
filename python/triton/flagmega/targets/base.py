# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Target protocol and explicit registry."""

from __future__ import annotations

from typing import Protocol

from triton.flagmega.ir import IRModule, Placement, SelectionPoint
from triton.flagmega.rules.ntt.vectorize import VectorizeRuleRegistry
from triton.flagmega.passes.auto_distributed import (
    DistributedCandidateProviderRegistry,
    DistributedReshardRealizationPolicy,
)


class Target(Protocol):
    name: str
    policy_version: str
    codegen_platform: str
    codegen_architecture: str
    triton_implementation_model: object

    def propose_vectorization(self, module: IRModule) -> IRModule: ...

    def apply_vectorization(self, module: IRModule) -> IRModule: ...

    def register_auto_vectorize_rules(self, registry: VectorizeRuleRegistry) -> None: ...

    def register_pack_propagation_rules(self, registry: VectorizeRuleRegistry) -> None: ...

    def propose_distribution(self, module: IRModule) -> IRModule: ...

    def apply_distribution(self, module: IRModule) -> IRModule: ...

    def auto_distribute(self, module: IRModule) -> IRModule: ...

    def distributed_placements(self, module: IRModule) -> tuple[Placement, ...]: ...

    def register_auto_distributed_candidate_providers(
        self, registry: DistributedCandidateProviderRegistry,
    ) -> None: ...

    def distributed_reshard_realization_policy(self) -> DistributedReshardRealizationPolicy: ...

    def distributed_reshard_cost_model(self): ...

    def distributed_operation_cost_model(self): ...

    def propose_packing(self, module: IRModule) -> IRModule: ...

    def apply_packing(self, module: IRModule) -> IRModule: ...

    def register_post_auto_packing_passes(self, registry) -> None: ...

    def propose_tir(self, module: IRModule) -> IRModule: ...

    def lower_to_tir(self, module: IRModule) -> IRModule: ...

    def propose_microkernels(self, module: IRModule) -> IRModule: ...

    def select_microkernels(self, module: IRModule) -> IRModule: ...

    def plan_launch(self, module: IRModule, kernel_nodes) -> dict[str, object]: ...

    def plan_codegen_package(self, module: IRModule, kernel_nodes) -> dict[str, object]: ...

    def bufferize(self, module: IRModule) -> IRModule: ...

    def with_bufferize_opt_level(self, level: str) -> Target: ...

    def plan_function_memory(self, module: IRModule) -> IRModule: ...

    def plan_memory_synchronization(self, module: IRModule) -> IRModule: ...

    def add_default_selections(
        self,
        module: IRModule,
        points: tuple[SelectionPoint, ...],
        rationale: str,
        *,
        policy_version: str | None = None,
    ) -> IRModule: ...

    def verify(self, module: IRModule) -> None: ...


_TARGETS: dict[str, Target] = {}


def register_target(target: Target) -> None:
    if target.name in _TARGETS:
        raise ValueError(f"Target {target.name!r} is already registered.")
    _TARGETS[target.name] = target


def get_target(name: str) -> Target:
    try:
        return _TARGETS[name]
    except KeyError as error:
        raise ValueError(f"Unknown FlagMega target {name!r}; available: {sorted(_TARGETS)}") from error


def target_names() -> tuple[str, ...]:
    return tuple(sorted(_TARGETS))
