# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Backend-independent NTT target composition.

Like nncase's ``NTTTarget``, this class owns the NTT compilation stages and
their rule registration. Concrete machines provide capabilities, memory
spaces, workspace annotation, and legality verification.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from triton.flagmega.ir import IRModule, Placement, SelectionPoint
from triton.flagmega.passes.auto_distributed import DistributedCandidateProviderRegistry
from triton.flagmega.passes.auto_distributed.auto_distributed import AutoDistributedPass
from triton.flagmega.passes.auto_distributed.realization import DistributedReshardRealizationPolicy
from triton.flagmega.passes.auto_vectorize import AutoVectorizePass
from triton.flagmega.rules.ntt.vectorize import VectorizeRuleRegistry
from triton.flagmega.targets.ntt_options import NttTargetOptions


class NttTarget(ABC):
    """Shared NTT policy facade parameterized by a concrete target machine."""

    name: str
    policy_version: str
    codegen_platform: str
    codegen_architecture: str

    def __init__(
        self,
        capability,
        options: NttTargetOptions,
        *,
        launch_planner,
        package_planner,
        packing_policy,
        vectorization_policy,
        distribution_policy,
        reshard_cost_model,
        operation_cost_model,
        selection_policy,
        bufferization_policy,
        tir_selection_policy,
        tir_lowering_policy,
        microkernel_selection_policy,
        triton_implementation_model=None,
    ) -> None:
        self.capability = capability
        self.options = options
        if not callable(launch_planner):
            raise ValueError("NttTarget requires a concrete launch planner.")
        if not callable(package_planner):
            raise ValueError("NttTarget requires a concrete package planner.")
        self.launch_planner = launch_planner
        self.package_planner = package_planner
        for name, value in (
            ("vectorization policy", vectorization_policy),
            ("distribution policy", distribution_policy),
            ("distributed reshard cost model", reshard_cost_model),
            ("distributed operation cost model", operation_cost_model),
            ("selection policy", selection_policy),
            ("bufferization policy", bufferization_policy),
            ("packing policy", packing_policy),
            ("TIR selection policy", tir_selection_policy),
            ("TIR lowering policy", tir_lowering_policy),
            ("TIR microkernel selection policy", microkernel_selection_policy),
        ):
            if value is None:
                raise ValueError(f"NttTarget requires a concrete {name}.")
        self.vectorization_policy = vectorization_policy
        self.distribution_policy = distribution_policy
        self.reshard_cost_model = reshard_cost_model
        self.operation_cost_model = operation_cost_model
        self.selection_policy = selection_policy
        self.bufferization_policy = bufferization_policy
        self.packing_policy = packing_policy
        self.tir_selection_policy = tir_selection_policy
        self.tir_lowering_policy = tir_lowering_policy
        self.microkernel_selection_policy = microkernel_selection_policy
        if triton_implementation_model is None:
            raise ValueError("NttTarget requires a concrete Triton implementation model.")
        self.triton_implementation_model = triton_implementation_model

    def propose_vectorization(self, module: IRModule) -> IRModule:
        return AutoVectorizePass.propose(module, self)

    def apply_vectorization(self, module: IRModule) -> IRModule:
        return AutoVectorizePass.run(module, self)

    def register_auto_vectorize_rules(self, registry: VectorizeRuleRegistry) -> None:
        self.vectorization_policy.register_rules(registry)

    def register_pack_propagation_rules(self, registry: VectorizeRuleRegistry) -> None:
        self.vectorization_policy.register_propagation_rules(registry)

    def distributed_placements(self, module: IRModule) -> tuple[Placement, ...]:
        return self.distribution_policy.placements(module)

    def register_auto_distributed_candidate_providers(
        self,
        registry: DistributedCandidateProviderRegistry,
    ) -> None:
        self.distribution_policy.register_candidate_providers(registry)

    def distributed_reshard_realization_policy(
        self,
    ) -> DistributedReshardRealizationPolicy:
        return self.distribution_policy.reshard_realization_policy()

    def distributed_reshard_cost_model(self):
        return self.reshard_cost_model

    def distributed_operation_cost_model(self):
        return self.operation_cost_model

    def auto_distribute(self, module: IRModule) -> IRModule:
        # Distributed type inference owns VectorType just like every other
        # element type.  Lowering a selected vector expression to a scalar
        # schedule contract here used to erase its physical type before the
        # distribution search and made the distributed dump misleading.
        return AutoDistributedPass.run(module, self)

    def propose_distribution(self, module: IRModule) -> IRModule:
        return AutoDistributedPass.propose(module, self)

    def apply_distribution(self, module: IRModule) -> IRModule:
        return AutoDistributedPass.apply(module, self)

    def propose_packing(self, module: IRModule) -> IRModule:
        return self.packing_policy.propose(module, self)

    def apply_packing(self, module: IRModule) -> IRModule:
        return self.packing_policy.apply(module, self)

    def register_post_auto_packing_passes(self, registry) -> None:
        """Register backend-level graph passes that run after ABI threading.

        Concrete physical machines do not participate in this hook. Backends
        such as PyNTT may extend it with semantic NTT rewrites.
        """

        del registry

    def propose_tir(self, module: IRModule) -> IRModule:
        return self.tir_selection_policy.propose(module, self)

    def lower_to_tir(self, module: IRModule) -> IRModule:
        return self.tir_lowering_policy.lower(module, self)

    def propose_microkernels(self, module: IRModule) -> IRModule:
        return self.microkernel_selection_policy.propose(module, self)

    def select_microkernels(self, module: IRModule) -> IRModule:
        return self.microkernel_selection_policy.apply(module, self)

    def plan_launch(self, module: IRModule, kernel_nodes) -> dict[str, object]:
        return dict(self.launch_planner(module, tuple(kernel_nodes)))

    def plan_codegen_package(
        self, module: IRModule, kernel_nodes,
    ) -> dict[str, object]:
        return dict(self.package_planner(
            module,
            tuple(kernel_nodes),
            self.capability,
            self.options,
        ))

    def bufferize(self, module: IRModule) -> IRModule:
        return self.bufferization_policy.bufferize(module)

    def plan_function_memory(self, module: IRModule) -> IRModule:
        return self.bufferization_policy.plan_function_memory(module)

    def plan_memory_synchronization(self, module: IRModule) -> IRModule:
        return self.bufferization_policy.plan_memory_synchronization(module)

    def add_default_selections(
        self,
        module: IRModule,
        points: tuple[SelectionPoint, ...],
        rationale: str,
        *,
        policy_version: str | None = None,
    ) -> IRModule:
        return self.selection_policy.add_defaults(
            module,
            points,
            rationale,
            capability=self.capability,
            target_name=self.name,
            policy_version=policy_version or self.policy_version,
        )

    @abstractmethod
    def verify(self, module: IRModule) -> None:
        """Verify concrete-machine legality and serialized target identity."""


__all__ = ["NttTarget"]
