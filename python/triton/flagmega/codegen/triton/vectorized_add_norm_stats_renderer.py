# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Register composition of a vectorized residual and an owner-local statistic."""

from collections.abc import Mapping
from math import prod

from triton.flagmega.errors import CodegenError


def vectorized_add_norm_stats_call(raw):
    from triton.flagmega.codegen.triton.kernel_call_renderers import (
        _buffer, _canonical_writer_active, _elementwise_call, _pointer, _static_shape,
    )
    from triton.flagmega.codegen.triton.physical_access import emit_active_extent, emit_local_scalar_offset

    attrs = raw.get('semantic_attrs', {})
    parameters = raw['parameters']
    value = _buffer(raw, 'outputs', 'result_0')
    stats = _buffer(raw, 'outputs', 'result_1')
    value_abi, stats_abi = value['abi'], stats['abi']
    shape = _static_shape(value_abi, 'local_capacity_shape')
    axis = int(attrs.get('axis', -1))
    axis = axis + len(shape) if axis < 0 else axis
    if (raw.get('variant', parameters.get('variant')) != 'local16_xor_fma'
            or attrs.get('binary_op') != 'add' or attrs.get('use_mean', False)
            or attrs.get('local_reduction_order') != 'square_rn_xor8_fma_xor4_2_1_lane0'
            or axis != len(shape) - 1 or attrs.get('post_ops')
            or parameters.get('task_mapping') is not None
            or int(parameters.get('tile', 0)) != 16):
        raise CodegenError('VectorizedAddNormStats requires its declared local16 XOR/FMA contract.')
    if (not shape or prod(shape[:-1]) != 1
            or prod(shape) * int(value_abi.get('scalar_lane_count', 1)) != 16
            or value_abi.get('scalar_dtype') != 'float32'
            or stats_abi.get('scalar_dtype') != 'float32'
            or int(stats_abi.get('scalar_lane_count', 1)) != 1
            or _static_shape(stats_abi, 'local_capacity_shape') != (1, *shape[:-1], 1)):
        raise CodegenError('VectorizedAddNormStats needs sixteen F32 raw scalars and one F32 statistic.')
    distributed = value_abi.get('distributed_type')
    stats_distributed = stats_abi.get('distributed_type')
    if not isinstance(distributed, Mapping) or not isinstance(stats_distributed, Mapping):
        raise CodegenError('VectorizedAddNormStats requires explicit distributed raw/statistics ABIs.')
    split_axes = {
        int(a) for policy in distributed.get('axis_policies', ()) if policy.get('kind') == 'split'
        for stage in policy.get('stages', ()) for a in stage.get('hierarchy_axes', ())
    }
    partial = stats_distributed.get('partial')
    if (distributed.get('partial') is not None
            or any(p.get('kind') == 'partial' for p in distributed.get('axis_policies', ()))
            or not isinstance(partial, Mapping)
            or partial.get('reduce_op') != 'sum' or set(partial.get('axes', ())) != split_axes
            or not split_axes or distributed.get('placement') != stats_distributed.get('placement')
            or any(p.get('kind') != 'broadcast' for p in stats_distributed.get('axis_policies', ()))):
        raise CodegenError('VectorizedAddNormStats Psum ownership must match the raw reduction owners.')
    for operand in ('lhs', 'rhs'):
        kind = _buffer(raw, 'inputs', operand)['abi'].get('distributed_type', {})
        if (kind.get('partial') is not None
                or any(p.get('kind') == 'partial' for p in kind.get('axis_policies', ()))):
            raise CodegenError('VectorizedAddNormStats inputs must already be materialized.')

    # The shared mapper/emitter sees exactly the original arithmetic operation.
    # Only the explicit raw output is part of its elementwise contract.
    arithmetic = {
        **raw, 'family': 'elementwise', 'variant': 'add', 'semantic_op': 'math.vectorized_binary',
        'semantic_attrs': {k: v for k, v in attrs.items()
                           if k not in {'axis', 'use_mean', 'local_reduction_order'}},
        'outputs': tuple({**p, 'formal': 'result', 'buffers': tuple(
            {**b, 'formal': 'result'} for b in p['buffers'])}
            for p in raw['outputs'] if p['formal'] == 'result_0'),
    }
    call = _elementwise_call(arithmetic)
    if call['local_capacity'] != 16 or call['tile'] != 16:
        raise CodegenError('VectorizedAddNormStats vector schedule must use the direct sixteen-element domain.')
    stats_coordinates = ('0',) * len(_static_shape(stats_abi, 'local_capacity_shape'))
    stats_active = ' & '.join((
        _canonical_writer_active(stats_abi),
        *(f'(0 < ({emit_active_extent(stats_abi, i)}))' for i in range(len(stats_coordinates))),
        *(f'(0 < ({emit_active_extent(value_abi, i)}))' for i in range(len(shape) - 1)),
    ))
    return {**call, 'stats': _pointer(stats),
            'stats_offset': emit_local_scalar_offset(stats_abi, stats_coordinates),
            'stats_active': stats_active}
