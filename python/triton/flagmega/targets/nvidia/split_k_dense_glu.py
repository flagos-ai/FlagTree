# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Optional normal SM90 two-program GU implementation catalog."""
from triton.flagmega.codegen.triton.implementation import TritonImplementation


def split_k_dense_glu_implementations():
    return (
        TritonImplementation(
            'tir.split_k_dense_glu_projection.participant_retire', 'split_k_dense_glu', 'projection', {
                'compute_num_warps': 8, 'block_k': 64, 'tile_n': 256, 'split_k': 2, 'num_stages': 3, 'program_grid':
                (48, 2, 1), 'program_owner_mapping': 'x-fast', 'maxnreg': 64
            }, requires=('tma', ), facts={'portable_triton': False, 'warp_specialized_entry': True}),
        TritonImplementation(
            'tir.split_k_dense_glu_epilogue.complete_sum', 'split_k_dense_glu', 'epilogue',
            {'compute_num_warps': 8, 'tile_n': 256, 'program_grid':
             (24, 1, 1), 'program_owner_mapping': 'x-fast'}, facts={'portable_triton': False}),
    )
