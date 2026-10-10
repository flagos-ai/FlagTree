# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Translate the prepared residency contract into backend compilation bounds."""

import os

from triton.flagmega.runtime.prepared import ResourceContract


def compilation_options(contract: ResourceContract, target: str = "nvidia-sm90") -> dict[str, object]:
    # NVIDIA uses ptxas occupancy bounds; COREX has no equivalent ptx option.
    # Both backends still enforce the post-compilation resource contract.
    options = {"num_warps": contract.compute_num_warps}
    # Keep the normal compiler default unless an experiment explicitly asks
    # for a different software-pipeline depth.  This is intentionally an
    # opt-in launch-time knob: the persistent-kernel residency contract still
    # owns the warp count, while num_stages only changes loop pipelining.
    raw_stages = os.environ.get("FLAGMEGA_NUM_STAGES")
    if raw_stages:
        stages = int(raw_stages)
        if stages < 1 or stages > 8:
            raise ValueError(f"FLAGMEGA_NUM_STAGES must be in [1, 8], got {stages}")
        options["num_stages"] = stages
    if target != "iluvatar-bi-v150":
        options["ptx_options"] = f"--minnctapersm={contract.resident_blocks_per_sm}"
    # BI-V150 does not implement the CUDA cooperative-launch admission API.
    # Its grid barriers use the backend global-atomic protocol and ordinary
    # kernel launches; the driver also ignores a stale cooperative metadata bit.
    return options


__all__ = ["compilation_options"]
