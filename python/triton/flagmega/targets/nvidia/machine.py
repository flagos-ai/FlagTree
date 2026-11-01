# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""NVIDIA SM90 physical target-machine profile.

This module contains machine choices only.  It does not register graph rewrite,
packing, vectorization, or distribution rules.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

from triton.flagmega.ir import IRModule, Placement
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.targets.ntt_options import NttTargetOptions
from triton.flagmega.targets.nvidia.capability import Sm90Capability
from triton.flagmega.targets.portable_triton_implementations import (
    portable_triton_implementation_model,
)
from triton.flagmega.targets.nvidia.launch import sm90_launch_parameters
from triton.flagmega.targets.nvidia.memory import sm90_bufferization_options
from triton.flagmega.targets.nvidia.package import sm90_codegen_package_plan
from triton.flagmega.targets.selection import CapabilitySelectionPolicy
from triton.flagmega.passes.auto_distributed.operation_cost import (
    DistributedOperationCostModel,
)
from triton.flagmega.passes.auto_distributed.reshard_cost import (
    DistributedReshardCostModel,
)
from triton.flagmega.targets.nvidia.verifier import verify_sm90_module
from triton.flagmega.targets.nvidia.workspaces import attach_workspace_requirements
from triton.flagmega.targets.nvidia.shared_layout import verify_shared_workspaces


class NvidiaSm90Machine:
    """Physical services for an SM90 device, independent of NTT graph rules."""

    name = "nvidia-sm90"
    policy_version = "nvidia-sm90-machine/v4"
    codegen_platform = "nvidia"
    codegen_architecture = "sm90"

    def __init__(
        self,
        capability: Sm90Capability | None = None,
        *,
        implementation_model=None,
    ) -> None:
        self.capability = capability or Sm90Capability()
        if implementation_model is not None:
            verify_shared_workspaces(implementation_model)
        self._implementation_model = implementation_model

    def default_ntt_options(self) -> NttTargetOptions:
        # These are an editable backend configuration selected for this
        # machine profile, not constants embedded in any rule implementation.
        return NttTargetOptions(
            placements=(Placement((8, 16), "yx", "bb"),),
            vector_lane_bytes=16,
            vector_max_axes=1,
            packing_vector_bytes=16,
            packing_k_pack=2,
        )

    def triton_implementation_model(self):
        return self._implementation_model or _sm90_implementation_model()

    def bufferization_options(self):
        return sm90_bufferization_options(self.capability)

    def memory_synchronization_protocol(self, module):
        from collections.abc import Mapping
        from triton.flagmega.ir.bufferization.synchronization_protocol import MemorySynchronizationProtocol
        # Derived machines such as COREX own a different lowering protocol.
        if self.codegen_platform != "nvidia":
            return None
        package = module.metadata.get("codegen_package_plan")
        if not isinstance(package, Mapping) or not isinstance(package.get("launch"), Mapping):
            return None
        cooperative = bool(package["launch"].get("cooperative_grid", False))
        # Matches TLE lowerGridBarrier's CTA arrival/acquire pair; a local
        # entry uses _common.stage_barrier's one debug_barrier instead.
        return MemorySynchronizationProtocol(
            "nvidia-sm90/tle-cooperative-grid/v1" if cooperative else "nvidia-sm90/local-entry/v1",
            2 if cooperative else 1,
        )

    def distributed_reshard_cost_model(self):
        operation_cost = self.distributed_operation_cost_model()
        return DistributedReshardCostModel(
            grid_synchronization_cost=operation_cost.grid_synchronization_cycles,
            operation_cost_model=operation_cost,
        )

    def distributed_operation_cost_model(self):
        # H800 target-machine values mirror nncase's canonical catalog.  The
        # aggregation implementation remains backend independent.
        return DistributedOperationCostModel(
            block_local_read_bytes_per_cycle=1024,
            block_local_write_bytes_per_cycle=1024,
            block_local_latency_cycles=20,
            elementwise_elements_per_cycle=128,
            simt_fma_per_cycle=64,
            chip_global_read_bytes_per_cycle=1908,
            chip_global_write_bytes_per_cycle=1908,
            chip_global_latency_cycles=300,
            block_synchronization_cycles=25,
            grid_synchronization_cycles=2200,
            identity="nvidia.h800-target-op-cost/v2",
        )

    def selection_policy(self, options: NttTargetOptions):
        del options
        return CapabilitySelectionPolicy()

    def annotate_workspaces(self, node, candidates, module, *, mesh_hierarchy):
        return attach_workspace_requirements(
            node,
            candidates,
            module,
            mesh_hierarchy=mesh_hierarchy,
        )

    def plan_launch(self, module, kernel_nodes) -> dict[str, object]:
        return sm90_launch_parameters(module, kernel_nodes)

    def plan_codegen_package(
        self,
        module,
        kernel_nodes,
        options: NttTargetOptions,
    ) -> dict[str, object]:
        return sm90_codegen_package_plan(
            module,
            kernel_nodes,
            self.capability,
            options,
        )

    def verify(
        self,
        module: IRModule,
        *,
        target_name: str,
        target_backend: str,
        policy_version: str,
        target_options: NttTargetOptions,
        implementation_model,
    ) -> None:
        if module.stage in {
            "selected_tir",
            "canonicalized_tir",
            "microkernel_candidates",
            "selected_microkernels",
            "packaged_tir",
            "memory_placed_tir",
            "allocated_tir",
            "scheduled_tir",
            "synchronized_tir",
            "bufferized_tir",
        }:
            expected = {
                "target_backend": target_backend,
                "target_machine": self.name,
                "target_machine_policy": self.policy_version,
            }
            mismatches = {
                name: (value, module.metadata.get(name))
                for name, value in expected.items()
                if module.metadata.get(name) != value
            }
            if mismatches:
                raise IRVerificationError(
                    f"SM90 backend/machine identity differs from the active target: {mismatches}.",
                    stage=module.stage,
                )
        verify_sm90_module(
            module,
            self.capability,
            target_name=target_name,
            policy_version=policy_version,
            target_options=target_options,
            implementation_model=implementation_model,
        )


@lru_cache(maxsize=1)
def _sm90_implementation_model():
    from triton.flagmega.targets.nvidia.delta_rule import resident_delta_rule_implementations
    from triton.flagmega.targets.nvidia.sparse_experts_ordered import ordered_sparse_experts_implementations
    from triton.flagmega.targets.nvidia.dense_matmul import (
        packed_mma_accumulator_implementations, packed_mma_norm_stats_implementations, ordered_mma_glu_implementations,
    )
    base = portable_triton_implementation_model()
    # These GEMV tiles fit an eight-warp consumer. Requiring sixteen warps
    # enlarges every consumer in a persistent pipeline, where fixed device
    # calls prevent register redistribution to its state-heavy stages.
    eight_warp_gemv = {
        "tir.dense_matmul.split_k_n_packed_k_major_gemv_tn64",
        "tir.qkv_parallel_linear.packed_fused_gemv_tn256",
    }
    implementations = tuple(
        replace(value, facts={**value.facts, "ordered_mma_accumulation": True})
        if value.id == "tir.dense_matmul.packed_k_major_mma" else
        replace(value, parameters={**value.parameters, "bounded_accurate_trig": True})
        if value.family == "rotary_embedding" else
        replace(value, parameters={**value.parameters, "compute_num_warps": 8})
        if value.id in eight_warp_gemv else value
        for value in base.implementations
    )
    ordered = (*packed_mma_accumulator_implementations(), *packed_mma_norm_stats_implementations())
    glu = ordered_mma_glu_implementations()
    recurrent = resident_delta_rule_implementations()
    experts = ordered_sparse_experts_implementations()
    preferences = {
        **base.preferences,
        "dense_matmul": (*base.preferences["dense_matmul"],
                         "tir.dense_matmul.packed_k_major_mma", *(value.id for value in ordered)),
        "dense_matmul_glu": (*base.preferences["dense_matmul_glu"], *(value.id for value in glu)),
        **{value.family: (value.id, *base.preferences[value.family]) for value in recurrent},
        **{value.family: (*base.preferences.get(value.family, tuple(
            item.id for item in base.implementations if item.family == value.family)), value.id)
           for value in experts},
    }
    return replace(base, name="nvidia-sm90-ordered/v1", implementations=(
        *implementations, *ordered, *glu, *recurrent, *experts), preferences=preferences)


__all__ = ["NvidiaSm90Machine"]
