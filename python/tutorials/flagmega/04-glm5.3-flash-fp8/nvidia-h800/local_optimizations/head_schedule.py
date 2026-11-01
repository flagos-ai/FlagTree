# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Measured physical tiles for projections with many outputs per owner."""
from dataclasses import replace
from math import prod

from triton.flagmega import ir as fm


def rewrite(module):
    """Leave small projections unchanged; expose the edit in verified TIR."""
    definitions = []
    for definition in module.kernel_definitions:
        dispatch = definition.dispatch
        micro = dispatch.microkernel
        if micro.family == 'dense_matmul' and micro.variant == 'packed_k_major_mma':
            output, = definition.output_parameters
            buffer, = output.buffers
            kind = fm.local_tensor_type(buffer.distributed_type) if buffer.distributed_type else buffer.type
            lanes = prod(kind.dtype.lanes) if isinstance(kind.dtype, fm.VectorType) else 1
            extent = kind.shape[-1].fixed_value
            # The MMA layout distributes N fragments across every compute warp.
            # Select a tile covering those fragments whenever the owner has work
            # for them; total owner capacity is not a reason to retain N16.
            warp_columns = 8 * int(micro.parameters['compute_num_warps'])
            if extent is not None and extent * lanes >= warp_columns:
                micro = replace(micro, parameters={**micro.parameters, 'tile_n': warp_columns, 'block_k': 128})
                definition = replace(definition, dispatch=replace(dispatch, microkernel=micro))
        definitions.append(definition)
    return tuple(definitions)
