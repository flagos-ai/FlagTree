# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""NVIDIA Shared storage contracts, separate from graph rules and SAT.

nncase's TritonTIRMicroKernelSelector uses 1024-byte NVMMA Shared alignment.
This is the complete 128B-swizzle period, not the 128-byte transfer width:
NVMMASharedEncodingAttr::getAlignment() is 128 * getMaxPhase(). A workspace
can be rebound at any legal SAT offset, so an index XOR cannot substitute
for this physical storage invariant.
"""

from dataclasses import replace

from triton.flagmega.errors import IRVerificationError


NVMMA_SHARED_ALIGNMENT_BYTES = 1024


def align_shared_workspaces(workspaces):
    """Attach the target's alignment floor while constructing its catalog."""
    return tuple(replace(value, alignment_bytes=max(value.alignment_bytes, NVMMA_SHARED_ALIGNMENT_BYTES))
                 if value.matrix_compatible else value for value in workspaces)


def verify_shared_workspaces(model):
    """Explicit injected/edited catalogs must already satisfy their ABI."""
    for implementation in model.implementations:
        for value in implementation.shared_workspaces:
            if value.matrix_compatible and value.alignment_bytes % NVMMA_SHARED_ALIGNMENT_BYTES:
                raise IRVerificationError(
                    f"NVIDIA Shared workspace {implementation.id!r}/{value.name!r} "
                    f"requires {NVMMA_SHARED_ALIGNMENT_BYTES}-byte NVMMA alignment."
                )


__all__ = ["NVMMA_SHARED_ALIGNMENT_BYTES", "align_shared_workspaces", "verify_shared_workspaces"]
