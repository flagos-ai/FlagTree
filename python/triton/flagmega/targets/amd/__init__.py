# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""AMD RDNA FlagMega target machine package (FlagMega-on-Radeon)."""

from triton.flagmega.targets.amd.capability import (
    AmdCapability,
    Gfx1100Capability,
    Gfx1201Capability,
)
from triton.flagmega.targets.amd.machine import AmdGfx1100Machine, AmdGfx1201Machine

__all__ = [
    "AmdCapability",
    "Gfx1100Capability",
    "Gfx1201Capability",
    "AmdGfx1100Machine",
    "AmdGfx1201Machine",
]
