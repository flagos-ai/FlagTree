# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Translate the prepared residency contract into backend compilation bounds."""

import os
from contextlib import contextmanager
from typing import Iterator

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
    # The ptxas occupancy flag (--minnctapersm) is NVIDIA-only; COREX has no
    # equivalent and the AMD/RDNA backend rejects it. RDNA compiles correctly
    # with just the warp count (occupancy tuning is a later performance pass).
    if target != "iluvatar-bi-v150" and not target.startswith("amd-"):
        options["ptx_options"] = f"--minnctapersm={contract.resident_blocks_per_sm}"
    # BI-V150 does not implement the CUDA cooperative-launch admission API.
    # Its grid barriers use the backend global-atomic protocol and ordinary
    # kernel launches; the driver also ignores a stale cooperative metadata bit.
    return options


@contextmanager
def compilation_knobs(target: str = "nvidia-sm90") -> Iterator[None]:
    """Process-wide compiler knobs to hold while compiling a megakernel.

    gfx1201: ConvertToBufferOps turns the megakernel's small-phase loads into
    buffer loads, but their base pointers arrive as noinline-call arguments in
    VGPRs, so each one becomes a serial readfirstlane waterfall loop. The
    GEMVs use int64 offsets and never take the buffer path, so disabling it
    only removes those waterfalls. Not on gfx1100: with FlagTree's LLVM 22,
    the rope function's gfx1100 code under that schedule corrupts one 4-lane
    quad of the query nondeterministically. LLVM 23 avoids it only by changing
    codegen (llvm #142716, #168169, #182446), so revisit after an LLVM bump
    with a multi-process reproducibility check. The knob mirrors itself into
    AMDGCN_USE_BUFFER_OPS, a cache-invalidating variable, so the cache key
    stays correct; the scope restores both afterwards.
    """
    # An explicit AMDGCN_USE_BUFFER_OPS from the user wins over this default.
    if target != "amd-gfx1201" or "AMDGCN_USE_BUFFER_OPS" in os.environ:
        yield
        return
    from triton import knobs

    with knobs.amd.scope():
        knobs.amd.use_buffer_ops = False
        yield


__all__ = ["compilation_knobs", "compilation_options"]
