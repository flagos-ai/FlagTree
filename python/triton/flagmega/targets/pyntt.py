# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""PyNTT/Triton backend composition over a replaceable target machine."""

from __future__ import annotations

from triton.flagmega.codegen.triton.lowering import TritonTirLoweringPolicy
from triton.flagmega.codegen.triton.selection import TritonTirSelectionPolicy
from triton.flagmega.codegen.triton.microkernels import (
    TritonMicroKernelSelectionPolicy,
    default_triton_microkernel_registry,
)
from triton.flagmega.passes.auto_distributed.policy import NttDistributionPolicy
from triton.flagmega.passes.auto_distributed.realization import (
    PyNttDistributedReshardRealizationPolicy,
)
from triton.flagmega.passes.tir.bufferize import NttBufferizationPolicy
from triton.flagmega.rules.ntt.packing import NttPackingPolicy
from triton.flagmega.rules.ntt.vectorize.policy import NttVectorizationPolicy
from triton.flagmega.targets.machine import NttTargetMachine
from triton.flagmega.targets.ntt import NttTarget
from triton.flagmega.targets.ntt_options import NttTargetOptions, PyNttTargetOptions
from triton.flagmega.targets.pyntt_split import PyNttDistributedSplitCandidateProvider
from triton.flagmega.rules.ntt.decompose_paged_attention import (
    paged_attention_split_plan,
)


class PyNttTarget(NttTarget):
    """NTT graph stages plus Triton codegen, parameterized by a machine.

    This is the counterpart of nncase's ``PyNTTTarget``.  Semantic rule-set
    registration is constructed here; the injected machine contributes only
    physical implementation and legality services.
    """

    backend_name = "pyntt"

    def __init__(
        self,
        machine: NttTargetMachine,
        *,
        target_name: str | None = None,
        policy_version: str | None = None,
        options: NttTargetOptions | None = None,
        vectorization_policy=None,
        distribution_policy=None,
        selection_policy=None,
        bufferization_policy=None,
        packing_policy=None,
        tir_selection_policy=None,
        tir_lowering_policy=None,
        microkernel_selection_policy=None,
        triton_implementation_model=None,
    ) -> None:
        self.machine = machine
        self.machine_name = machine.name
        self.machine_policy_version = machine.policy_version
        self.name = target_name or f"{self.backend_name}:{machine.name}"
        self.policy_version = policy_version or (
            f"{self.backend_name}/v1+{machine.policy_version}"
        )
        self.codegen_platform = machine.codegen_platform
        self.codegen_architecture = machine.codegen_architecture
        options = PyNttTargetOptions.from_ntt_options(
            options or machine.default_ntt_options()
        )
        packing_policy = packing_policy or NttPackingPolicy(
            vector_bytes=options.packing_vector_bytes,
            k_pack=options.packing_k_pack,
        )
        implementation_model = (
            triton_implementation_model or machine.triton_implementation_model()
        )
        split_candidate_provider = PyNttDistributedSplitCandidateProvider(
            options.block_cyclic_block_bytes
        )
        split_provider_factory = getattr(
            machine, "distributed_split_candidate_provider", None
        )
        if split_provider_factory is not None:
            split_candidate_provider = split_provider_factory(
                options, split_candidate_provider
            )
        bufferization_options = machine.bufferization_options()
        super().__init__(
            machine.capability,
            options,
            launch_planner=machine.plan_launch,
            package_planner=lambda module, kernels, _capability, active_options: (
                machine.plan_codegen_package(module, kernels, active_options)
            ),
            vectorization_policy=(
                vectorization_policy or NttVectorizationPolicy(
                    lane_bytes=options.vector_lane_bytes,
                    max_axes=options.vector_max_axes,
                )
            ),
            distribution_policy=(
                distribution_policy or NttDistributionPolicy(
                    options.placements,
                    split_candidate_provider,
                    PyNttDistributedReshardRealizationPolicy(),
                )
            ),
            reshard_cost_model=machine.distributed_reshard_cost_model(),
            operation_cost_model=machine.distributed_operation_cost_model(),
            selection_policy=selection_policy or machine.selection_policy(options),
            bufferization_policy=(
                bufferization_policy
                or NttBufferizationPolicy(bufferization_options)
            ),
            packing_policy=packing_policy,
            tir_selection_policy=(
                tir_selection_policy
                or TritonTirSelectionPolicy(machine.annotate_workspaces)
            ),
            tir_lowering_policy=tir_lowering_policy or TritonTirLoweringPolicy(),
            microkernel_selection_policy=(
                microkernel_selection_policy
                or TritonMicroKernelSelectionPolicy(
                    default_triton_microkernel_registry()
                )
            ),
            triton_implementation_model=implementation_model,
        )

    def verify(self, module) -> None:
        self.machine.verify(
            module,
            target_name=self.name,
            target_backend=self.backend_name,
            policy_version=self.policy_version,
            target_options=self.options,
            implementation_model=self.triton_implementation_model,
        )

    def register_post_auto_packing_passes(self, registry) -> None:
        super().register_post_auto_packing_passes(registry)
        registry.add("DecomposePagedAttention", "decompose-paged-attention")

__all__ = ["PyNttTarget", "paged_attention_split_plan"]
