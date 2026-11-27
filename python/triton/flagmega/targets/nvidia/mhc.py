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


def mhc_cooperative_stream_fma_implementations():
    """Optional chip-cooperative schedules; targets opt in after measurement."""
    return (TritonImplementation(
        'tir.mhc_pre.cooperative_stream_fma', 'mhc_pre', 'cooperative_stream_fma',
        {'elements_per_program': 256, 'compute_num_warps': 8},
        {'indexing': 'local', 'projection_reduction': 'stream_fma_128'},
        requires=('cooperative_grid', 'grid_sync'),
        facts={'portable_triton': False, 'internal_grid_barriers': 2,
               'result_memory_scope': 'chip'}),)


def mhc_cooperative_contiguous_tf32_implementations():
    """Parallelize existing TF32 splits, retaining the ordered split merge."""
    return (TritonImplementation(
        'tir.mhc_pre.cooperative_contiguous_tf32', 'mhc_pre', 'cooperative_contiguous_tf32',
        {'elements_per_program': 256, 'compute_num_warps': 8},
        {'indexing': 'local', 'projection_reduction': 'contiguous_tf32'},
        requires=('cooperative_grid', 'grid_sync'),
        facts={'portable_triton': False, 'internal_grid_barriers': 2,
               'result_memory_scope': 'chip'}),)
