# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""PyNTT target facade for an AMD RDNA3 (gfx1100 / W7900) machine."""

from __future__ import annotations

from dataclasses import replace

from triton.flagmega.targets.pyntt import PyNttTarget
from triton.flagmega.targets.ntt_options import NttTargetOptions
from triton.flagmega.ir import Placement
from triton.flagmega.targets.amd.machine import AmdGfx1100Machine


class AmdGfx1100Target(PyNttTarget):
    name = "amd-gfx1100"
    policy_version = "pyntt-amd-gfx1100/v0"
    codegen_platform = "amd"
    codegen_architecture = "gfx1100"

    def __init__(self, capability=None, *, options: NttTargetOptions | None = None,
                 triton_implementation_model=None, **policies) -> None:
        machine = AmdGfx1100Machine(capability, implementation_model=triton_implementation_model)
        active_options = options or machine.default_ntt_options()
        super().__init__(
            machine,
            target_name=self.name,
            policy_version=self.policy_version,
            options=active_options,
            triton_implementation_model=triton_implementation_model,
            **policies,
        )

    def with_placement(self, placement: Placement) -> "AmdGfx1100Target":
        """Return the same RDNA policy with one explicit block placement.

        Used by the compiler's ``--placement`` option; the default placement is
        unchanged when no placement is requested.
        """
        if not isinstance(placement, Placement):
            raise TypeError("placement must be a Placement")
        return type(self)(
            self.machine.capability,
            options=replace(self.options, placements=(placement,)),
            triton_implementation_model=self.triton_implementation_model,
        )
