# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Readonly cache identity describes stored owners, independently of views."""
from dataclasses import replace

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import rdata
from triton.flagmega.passes.tir.bufferize import BufferizationOptions


class MetadataCheckpoint:
    def cache_identity(self, keys):
        return {'schema': 'fixture', 'keys': sorted(keys)}


def cache_data(view_count):
    kind = fm.DistributedType(fm.tensor_type('float8_e4m3fn', (16, 512)),
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((0,), 128)),
        fm.Placement((4, 8, 16), 'cyx', 'cbb'))
    local_kind = replace(kind, placement=fm.Placement((4, 1, 1), 'cyx', 'cbb'))
    builder = fm.IRBuilder(dialect='tir', stage='selected_tir')
    value = builder.weight('weight', kind, source='fixture.safetensors', key='weight', id='weight')
    for index in range(view_count):
        value = builder.call('distributed.sharded_view', (value,), local_kind,
            id=f'view{index}', attrs={'new_type': local_kind})
    result = builder.call('tir.kernel', (value,), local_kind if view_count else kind, id='result')
    builder.function('main', (), (result,))
    module = builder.build(entry='main')
    plan = fm.make_buffer_plan(module, options=replace(BufferizationOptions.generic(), rdata_chip_sharding=True))
    module = replace(module, metadata={**module.metadata, 'buffer_plan': plan.to_data()})
    return rdata._rdata_cache_key_data(module, MetadataCheckpoint())


@pytest.mark.parametrize('view_count', [1, 2])
def test_alias_views_do_not_add_cache_payloads_or_change_owner_identity(view_count):
    baseline = cache_data(0)
    actual = cache_data(view_count)
    assert [buffer['id'] for buffer in actual['buffers']] == ['weight']
    assert actual == baseline
