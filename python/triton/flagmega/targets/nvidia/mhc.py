# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""SM90-specific ordered hyper-connection projections."""
from triton.flagmega.codegen.triton.implementation import TritonImplementation


def mhc_contiguous_tf32_implementations():
    # Candidate selection enforces SM90; this is never a portable fallback.
    return (TritonImplementation(
        'tir.mhc_pre.contiguous_tf32', 'mhc_pre', 'contiguous_tf32',
        {'elements_per_program': 256, 'compute_num_warps': 4},
        {'indexing': 'local', 'projection_reduction': 'contiguous_tf32'},
        facts={'portable_triton': False}),)
