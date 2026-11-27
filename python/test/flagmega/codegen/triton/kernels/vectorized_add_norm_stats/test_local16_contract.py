# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The composed leaf keeps the two output ABIs and a fixed local arithmetic tree."""

from copy import deepcopy

import pytest

from triton.flagmega.codegen.triton.vectorized_add_norm_stats_renderer import vectorized_add_norm_stats_call
from triton.flagmega.errors import CodegenError

ORDER = 'square_rn_xor8_fma_xor4_2_1_lane0'


def abi(shape, lanes=(), *, partial=False):
    from math import prod
    strides = []
    stride = prod(lanes)
    for extent in reversed(shape):
        strides.append(stride)
        stride *= extent
    policies = ({'kind': 'broadcast'}, ) * (len(shape) - 1) + (
        {'kind': 'split', 'stages': ({'hierarchy_axes': (0, 1), 'kind': 'contiguous', 'unit': 4}, )}, )
    return {
        'storage': 'workspace',
        'storage_kind': 'compact_per_owner',
        'pool_byte_offset': 0,
        'scalar_dtype': 'float32',
        'scalar_itemsize': 4,
        'logical_shape': tuple(shape),
        'local_capacity_shape': tuple(shape),
        'active_shape_expressions': tuple(map(str, shape)),
        'logical_coordinate_expressions': tuple(f'local_coord_{i}' for i in range(len(shape))),
        'scalar_storage_strides': tuple(reversed(strides)),
        'scalar_lane_shape': lanes,
        'scalar_lane_count': prod(lanes),
        'component_stride_scalar_elements': stride,
        'coordinate_space': 'local',
        'distributed_type': {
            'placement': {'hierarchy': (8, 16)},
            'axis_policies': ({'kind': 'broadcast'}, ) * len(shape) if partial else policies,
            'partial': {'axes': (0, 1), 'reduce_op': 'sum'} if partial else None,
        },
    }


def parameter(formal, kind):
    return {
        'formal': formal, 'buffers': ({'formal': formal, 'actual': formal, 'runtime_argument': formal, 'abi': kind}, )
    }


def raw_call():
    raw = abi((1, 4), (4, ))
    rhs = abi((1, 16))
    return {
        'family': 'vectorized_add_norm_stats',
        'variant': 'local16_xor_fma',
        'semantic_op': 'ntt.vectorized_add_norm_stats',
        'semantic_attrs': {
            'binary_op': 'add', 'axis': -1, 'use_mean': False, 'rhs_pack': {'axes': (1, ), 'lanes': (4, )},
            'local_reduction_order': ORDER
        },
        'parameters': {
            'variant': 'local16_xor_fma', 'tile': 16, 'vector_schedule':
            {'contract': {'kind': 'typed_elements', 'lanes': (4, )}, 'physical': {'elements_per_program': 16}}
        },
        'inputs': (parameter('lhs', deepcopy(raw)), parameter('rhs', rhs)),
        'outputs': (parameter('result_0', raw), parameter('result_1', abi((1, 1, 1), partial=True))),
        'workspaces': (),
    }


def test_raw_vector_and_scalar_rhs_use_original_pack_mapping():
    raw = raw_call()
    result = vectorized_add_norm_stats_call(raw)
    assert result['local_capacity'] == result['tile'] == 16
    assert result['output_type'] == 'tl.float32'
    # Evaluating the mapper must yield each rhs scalar exactly once in the raw lane order.
    for offset in range(16):
        assert eval(result['rhs_offset'], {}, {'local_offsets': offset}) == offset
        assert eval(result['result_offset'], {}, {'local_offsets': offset}) == offset
    # P is one independent owner slot, not a broadcast scalar address.
    assert '(shard_index) * 1' in result['stats']
    assert eval(result['stats_offset']) == 0
    assert result['fusion_lines'] == ()
    assert raw['semantic_op'] == 'ntt.vectorized_add_norm_stats'
    assert raw['outputs'][0]['formal'] == 'result_0'


@pytest.mark.parametrize('mutation', [
    'order', 'mean', 'axis', 'extent', 'dtype', 'partial', 'partial_axes', 'tile', 'post', 'task', 'lhs_axis_partial',
    'rhs_axis_partial', 'raw_axis_partial'
])
def test_direct_tir_rejects_unsupported_contracts(mutation):
    raw = raw_call()
    attrs = raw['semantic_attrs']
    out = raw['outputs'][0]['buffers'][0]['abi']
    stats = raw['outputs'][1]['buffers'][0]['abi']
    if mutation == 'order': attrs['local_reduction_order'] = 'group_fma'
    elif mutation == 'mean': attrs['use_mean'] = True
    elif mutation == 'axis': attrs['axis'] = 0
    elif mutation == 'extent': out['local_capacity_shape'] = (1, 8)
    elif mutation == 'dtype': out['scalar_dtype'] = 'bfloat16'
    elif mutation == 'partial': stats['distributed_type']['partial'] = None
    elif mutation == 'partial_axes': stats['distributed_type']['partial']['axes'] = (0, )
    elif mutation == 'tile': raw['parameters']['tile'] = 128
    elif mutation == 'post': attrs['post_ops'] = ({'invalid': 'unsupported'}, )
    elif mutation == 'task': raw['parameters']['task_mapping'] = {'program_kind': 'persistent'}
    elif mutation.endswith('_axis_partial'):
        name = mutation.removesuffix('_axis_partial')
        kind = out if name == 'raw' else raw['inputs'][0 if name == 'lhs' else 1]['buffers'][0]['abi']
        policies = list(kind['distributed_type']['axis_policies'])
        policies[0] = {'kind': 'partial', 'axes': (0, ), 'reduce_op': 'sum'}
        kind['distributed_type']['axis_policies'] = policies
    with pytest.raises(CodegenError):
        vectorized_add_norm_stats_call(raw)
