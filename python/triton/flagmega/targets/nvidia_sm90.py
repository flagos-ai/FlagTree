# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Compatibility facade for PyNTT configured with an SM90 machine."""

from __future__ import annotations

from triton.flagmega.targets.pyntt import PyNttTarget
from triton.flagmega.targets.ntt_options import NttTargetOptions
from triton.flagmega.targets.nvidia import Sm90Capability
from triton.flagmega.targets.nvidia.machine import NvidiaSm90Machine


class NvidiaSm90Target(PyNttTarget):
    """Legacy target spelling; graph policies are owned by :class:`PyNttTarget`."""

    name = "nvidia-sm90"
    policy_version = "pyntt-nvidia-sm90/v31"
    codegen_platform = "nvidia"
    codegen_architecture = "sm90"

    def __init__(
        self,
        capability: Sm90Capability | None = None,
        *,
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
        machine = NvidiaSm90Machine(
            capability,
            implementation_model=triton_implementation_model,
        )
        active_options = options or machine.default_ntt_options()
        super().__init__(
            machine,
            target_name=self.name,
            policy_version=self.policy_version,
            options=active_options,
            vectorization_policy=vectorization_policy,
            distribution_policy=distribution_policy,
            selection_policy=selection_policy,
            bufferization_policy=bufferization_policy,
            packing_policy=packing_policy,
            tir_selection_policy=tir_selection_policy,
            tir_lowering_policy=tir_lowering_policy,
            microkernel_selection_policy=microkernel_selection_policy,
            triton_implementation_model=triton_implementation_model,
        )
