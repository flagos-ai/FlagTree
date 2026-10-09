# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""AMD RDNA physical target-machine profile (FlagMega-on-Radeon T0).

This registers a selectable amd-gfx1100 / amd-gfx1201 machine with real RDNA
capability + a single-GPU (degenerate 1-device) mesh. The *planner* services
(implementation model, bufferization, package, verifier, launch, workspaces)
are, for T0, reused from the NVIDIA machine as PLACEHOLDERS — they are only
invoked once a module is compiled past propose-tir, which is the T1-T5 scope.
They let the target construct and register; replacing each with an RDNA policy
(WMMA microkernels, LDS memory spaces, mbarrier sync, RDNA cost model) is the
remaining Target work.
"""

from __future__ import annotations

from triton.flagmega.ir import IRModule, Placement
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.targets.ntt_options import NttTargetOptions
from triton.flagmega.targets.amd.capability import (
    AmdCapability,
    Gfx1100Capability,
    Gfx1201Capability,
)
from triton.flagmega.targets.selection import CapabilitySelectionPolicy
from triton.flagmega.passes.auto_distributed.operation_cost import (
    DistributedOperationCostModel,
)
from triton.flagmega.passes.auto_distributed.reshard_cost import (
    DistributedReshardCostModel,
)

# T0 placeholders — reused from the NVIDIA machine until RDNA policies land.
from triton.flagmega.targets.nvidia.implementations import (
    sm90_triton_implementation_model as _placeholder_implementation_model,
)
from triton.flagmega.targets.nvidia.memory import (
    sm90_bufferization_options as _placeholder_bufferization_options,
)
from triton.flagmega.targets.nvidia.package import (
    sm90_codegen_package_plan as _placeholder_codegen_package_plan,
)
from triton.flagmega.targets.nvidia.launch import (
    sm90_launch_parameters as _placeholder_launch_parameters,
)
from triton.flagmega.targets.nvidia.verifier import verify_sm90_module as _placeholder_verify_module
from triton.flagmega.targets.nvidia.workspaces import attach_workspace_requirements
from triton.flagmega.targets.nvidia.shared_layout import verify_shared_workspaces


class _AmdRdnaMachine:
    """Physical services for an RDNA device, independent of NTT graph rules."""

    codegen_platform = "amd"

    def __init__(self, capability: AmdCapability, *, implementation_model=None) -> None:
        self.capability = capability
        self.codegen_architecture = capability.gfx_arch
        self.name = f"amd-{capability.gfx_arch}"
        self.policy_version = f"amd-{capability.gfx_arch}-machine/v0"
        if implementation_model is not None:
            verify_shared_workspaces(implementation_model)
        self._implementation_model = implementation_model

    def default_ntt_options(self) -> NttTargetOptions:
        # Single-GPU: a degenerate 1-device mesh (NVIDIA uses an 8x16 cluster).
        return NttTargetOptions(
            placements=(Placement((1, 1), "yx", "bb"),),
            vector_lane_bytes=16,
            vector_max_axes=1,
            packing_vector_bytes=16,
            packing_k_pack=2,
        )

    def triton_implementation_model(self):
        return self._implementation_model or _placeholder_implementation_model()

    def bufferization_options(self):
        return _placeholder_bufferization_options(self.capability)

    def distributed_reshard_cost_model(self):
        operation_cost = self.distributed_operation_cost_model()
        return DistributedReshardCostModel(
            grid_synchronization_cost=operation_cost.grid_synchronization_cycles,
            operation_cost_model=operation_cost,
        )

    def distributed_operation_cost_model(self):
        # Placeholder RDNA numbers; retune for gfx1100/gfx1201 in a later pass.
        return DistributedOperationCostModel(
            block_local_read_bytes_per_cycle=1024,
            block_local_write_bytes_per_cycle=1024,
            block_local_latency_cycles=20,
            elementwise_elements_per_cycle=64,
            simt_fma_per_cycle=32,
            chip_global_read_bytes_per_cycle=1024,
            chip_global_write_bytes_per_cycle=1024,
            chip_global_latency_cycles=300,
            block_synchronization_cycles=25,
            grid_synchronization_cycles=2200,
            identity=f"amd.{self.codegen_architecture}-target-op-cost/v0",
        )

    def selection_policy(self, options: NttTargetOptions):
        del options
        return CapabilitySelectionPolicy()

    def annotate_workspaces(self, node, candidates, module, *, mesh_hierarchy):
        return attach_workspace_requirements(
            node, candidates, module, mesh_hierarchy=mesh_hierarchy
        )

    def plan_launch(self, module, kernel_nodes) -> dict[str, object]:
        return _placeholder_launch_parameters(module, kernel_nodes)

    def plan_codegen_package(self, module, kernel_nodes, options: NttTargetOptions) -> dict[str, object]:
        return _placeholder_codegen_package_plan(module, kernel_nodes, self.capability, options)

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
                    f"AMD backend/machine identity differs from the active target: {mismatches}.",
                    stage=module.stage,
                )
        _placeholder_verify_module(
            module,
            self.capability,
            target_name=target_name,
            policy_version=policy_version,
            target_options=target_options,
            implementation_model=implementation_model,
        )


class AmdGfx1100Machine(_AmdRdnaMachine):
    def __init__(self, capability: AmdCapability | None = None, *, implementation_model=None) -> None:
        super().__init__(capability or Gfx1100Capability(), implementation_model=implementation_model)


class AmdGfx1201Machine(_AmdRdnaMachine):
    def __init__(self, capability: AmdCapability | None = None, *, implementation_model=None) -> None:
        super().__init__(capability or Gfx1201Capability(), implementation_model=implementation_model)


__all__ = ["AmdGfx1100Machine", "AmdGfx1201Machine"]
