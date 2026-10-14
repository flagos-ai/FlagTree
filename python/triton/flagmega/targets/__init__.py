# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""FlagMega target registry."""

from triton.flagmega.targets.base import Target, get_target, register_target, target_names
from triton.flagmega.targets.nvidia_sm90 import NvidiaSm90Target, Sm90Capability
from triton.flagmega.targets.ntt import NttTarget
from triton.flagmega.targets.ntt_options import NttTargetOptions, PyNttTargetOptions
from triton.flagmega.targets.pyntt import PyNttTarget
from triton.flagmega.targets.machine import NttTargetMachine
from triton.flagmega.targets.nvidia.machine import NvidiaSm90Machine
from triton.flagmega.targets.iluvatar_bi_v150 import (
    IluvatarBiV150Capability,
    IluvatarBiV150Machine,
    IluvatarBiV150Target,
)
from triton.flagmega.targets.nvidia_sm89 import NvidiaSm89Target

register_target(NvidiaSm90Target())
register_target(IluvatarBiV150Target())
register_target(NvidiaSm89Target())

__all__ = [
    "IluvatarBiV150Capability",
    "IluvatarBiV150Machine",
    "IluvatarBiV150Target",
    "NttTargetOptions",
    "PyNttTargetOptions",
    "NttTargetMachine",
    "NttTarget",
    "PyNttTarget",
    "NvidiaSm90Target",
    "NvidiaSm90Machine",
    "Sm90Capability",
    "Target",
    "get_target",
    "register_target",
    "target_names",
]
