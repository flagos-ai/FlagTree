# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Translate the prepared residency contract into NVIDIA compilation bounds."""

from triton.flagmega.runtime.prepared import ResourceContract


def compilation_options(contract: ResourceContract) -> dict[str, object]:
    # The persistent grid has an explicit residency requirement. Without a
    # launch bound, ptxas guesses an occupancy target without knowing the
    # dynamically allocated Shared arena, and may spill to save registers for
    # CTAs that cannot be resident. Supply the same lower bound that resource
    # validation enforces; never relax the post-compilation spill check.
    return {
        "num_warps": contract.compute_num_warps,
        "ptx_options": f"--minnctapersm={contract.resident_blocks_per_sm}",
    }


__all__ = ["compilation_options"]
