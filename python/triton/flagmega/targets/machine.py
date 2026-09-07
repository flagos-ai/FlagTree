# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Physical-machine boundary for NTT backends.

An NTT backend owns graph rules and compilation stages.  A target machine owns
only physical capabilities, implementation variants, memory spaces, launch
geometry, package composition, and final legality checks.  Keeping this
boundary explicit mirrors nncase's ``NTTTarget`` + target-machine split and
prevents a machine generation from becoming the namespace of semantic rules.
"""

from __future__ import annotations

from typing import Protocol

from triton.flagmega.ir import IRModule
from triton.flagmega.targets.ntt_options import NttTargetOptions


class NttTargetMachine(Protocol):
    """Services supplied by one physical target-machine profile."""

    name: str
    policy_version: str
    codegen_platform: str
    codegen_architecture: str
    capability: object

    def default_ntt_options(self) -> NttTargetOptions: ...

    def triton_implementation_model(self): ...

    def bufferization_options(self): ...

    def distributed_reshard_cost_model(self): ...

    def distributed_operation_cost_model(self): ...

    def selection_policy(self, options: NttTargetOptions): ...

    def annotate_workspaces(self, node, candidates, module, *, mesh_hierarchy): ...

    def plan_launch(self, module, kernel_nodes) -> dict[str, object]: ...

    def plan_codegen_package(
        self,
        module,
        kernel_nodes,
        options: NttTargetOptions,
    ) -> dict[str, object]: ...

    def verify(
        self,
        module: IRModule,
        *,
        target_name: str,
        target_backend: str,
        policy_version: str,
        target_options: NttTargetOptions,
        implementation_model,
    ) -> None: ...


__all__ = ["NttTargetMachine"]
