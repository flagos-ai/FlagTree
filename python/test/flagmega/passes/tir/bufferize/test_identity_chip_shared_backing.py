# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Collapsing broadcast CTA axes retains a complete local chip component."""
from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.call_abi import describe_local_buffer_abi
from triton.flagmega.codegen.triton.cooperative_access import require_chip_shared
from triton.flagmega.codegen.triton.distributed_abi import distributed_buffer_abi
from triton.flagmega.passes.tir.bufferize import BufferizationOptions


@pytest.mark.parametrize('dtype', ['float8_e4m3fn', 'bfloat16', 'float32'])
@pytest.mark.parametrize('identity_chain', [False, True])
def test_chip_mesh_view_has_no_redundant_identity_backing(dtype, identity_chain):
    tensor = fm.tensor_type(dtype, (16, 512))
    policies = (fm.SBP.broadcast(), fm.SBP.split_contiguous((0,), 128))
    source_type = fm.DistributedType(tensor, policies, fm.Placement((4, 8, 16), 'cyx', 'cbb'))
    target_type = replace(source_type, placement=fm.Placement((4, 1, 1), 'cyx', 'cbb'))
    builder = fm.IRBuilder(dialect='tir', stage='selected_tir')
    weight = builder.weight('weight', source_type, source='fixture.safetensors', key='weight', id='weight')
    view = builder.call('distributed.sharded_view', (weight,), target_type,
                        id='view', attrs={'new_type': target_type})
    if identity_chain:
        view = builder.call('distributed.sharded_view', (view,), target_type,
                            id='identity', attrs={'new_type': target_type})
    output = builder.call('tir.kernel', (view,), target_type, id='output')
    builder.function('main', (), (output,))
    module = builder.build(entry='main')
    options = replace(BufferizationOptions.generic(), rdata_chip_sharding=True)
    plan = fm.make_buffer_plan(module, options=options)
    source = plan.buffer_map['weight']
    expected_bytes = 16 * 128 * fm.DType(dtype).itemsize
    assert source.distributed_storage_kind is fm.DistributedBufferStorageKind.COMPACT_PER_CHIP
    assert source.mem_span.nbytes == expected_bytes
    for name in ('view', 'identity') if identity_chain else ('view',):
        descriptor = plan.buffer_map[name]
        assert descriptor.mem_span.must_alias(source.mem_span)
        assert descriptor.physical_access_span == source.physical_access_span
        assert descriptor.strides == source.strides == (128, 1)
        assert descriptor.distributed_backing_type is None
        abi = describe_local_buffer_abi(descriptor, plan)
        assert abi['coordinate_space'] == 'local'
        assert tuple(abi['local_capacity_shape']) == (16, 128)
        require_chip_shared({'abi': abi}, 'weight', owner='PartitionedBlockScaledLinear')
        for chip in range(4):
            current = distributed_buffer_abi(descriptor, chip)
            for block in (0, 127):
                original = distributed_buffer_abi(source, chip * 128 + block)
                assert current.component_offset_elements == original.component_offset_elements == 0
                for column in range(128):
                    assert current.storage_axis_coordinate(1, column) == original.storage_axis_coordinate(1, column) == column
                    assert current.logical_axis_coordinate(1, column) == original.logical_axis_coordinate(1, column) == chip * 128 + column
