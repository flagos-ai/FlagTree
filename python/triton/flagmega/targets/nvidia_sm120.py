# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Conservative SM120 profile for RTX 5060 Ti native quantized kernels."""

from dataclasses import dataclass
from typing import ClassVar

from triton.flagmega.ir import Placement
from triton.flagmega.targets.nvidia.capability import Sm90Capability
from triton.flagmega.targets.nvidia.machine import NvidiaSm90Machine
from triton.flagmega.targets.ntt_options import NttTargetOptions
from triton.flagmega.targets.pyntt import PyNttTarget


@dataclass(frozen=True)
class Sm120Capability(Sm90Capability):
    compute_major: ClassVar[int] = 12
    schema: ClassVar[str] = "flagmega.nvidia-sm120-capability/v1"
    compute_capability: tuple[int, int] = (12, 0)
    max_threads_per_sm: int = 1536
    max_shared_memory_bytes: int = 101376
    supports_tma: bool = False
    supports_warp_specialize: bool = False
    supports_mma_v3: bool = False


class NvidiaSm120Machine(NvidiaSm90Machine):
    """Reuse portable NVIDIA services with SM120 resource limits."""

    name = "nvidia-sm120"
    policy_version = "nvidia-sm120-machine/v1"
    codegen_architecture = "sm120"

    def __init__(self, capability=None, *, implementation_model=None):
        super().__init__(capability or Sm120Capability(), implementation_model=implementation_model)

    def default_ntt_options(self):
        # Keep the persistent grid within the 36-SM RTX 5060 Ti's capacity.
        return NttTargetOptions(
            placements=(Placement((4, 8), "yx", "bb"), ),
            vector_lane_bytes=16,
            vector_max_axes=1,
            packing_vector_bytes=16,
            packing_k_pack=2,
        )


class NvidiaSm120Target(PyNttTarget):
    name = "nvidia-sm120"
    policy_version = "pyntt-nvidia-sm120/v1"
    codegen_platform = "nvidia"
    codegen_architecture = "sm120"

    def __init__(self, capability=None, *, options=None, **policies):
        super().__init__(
            NvidiaSm120Machine(capability),
            target_name=self.name,
            policy_version=self.policy_version,
            options=options,
            **policies,
        )
