# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Project both handoff endpoints only within one complete invocation."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.pipeline_programs import project_pipeline_schedule
from triton.flagmega.errors import CodegenError


def schedule():
    stages, handoffs, consumers, producers = [], [], [], []
    for index in (1, 2):
        stem = f'instance_{index}'
        stages.append(dict(id=stem, call='reusable_leaf',
            channels=[dict(reader_name=stem+'_reader', writer_name=stem+'_writer')],
            consumer_workspaces=[], auxiliary_consumer=None,
            workspaces=[dict(offset_bytes=(index-1)*16, nbytes=16,
                             alignment_bytes=16, allocation_alignment_bytes=16)]))
        handoffs.append(dict(id=stem+'_handoff', pipe_name=stem+'_handoff_pipe',
            reader_name=stem+'_handoff_reader', writer_name=stem+'_handoff_writer',
            semantic=True, repeated=False, wrap_around=False))
        consumers.append(dict(kind='function_call', call=f'call_{index}',
            endpoint_bindings={'leaf_reader':stem+'_reader', 'handoff_writer':stem+'_handoff_writer'}))
        producers.append(dict(kind='function_call', call=f'call_{index}',
            endpoint_bindings={'leaf_writer':stem+'_writer', 'handoff_reader':stem+'_handoff_reader'}))
    return dict(stages=stages, handoffs=handoffs, consumer_events=consumers,
                producer_events=producers, auxiliary_events=[],
                trip_scoped_closes=[], trip_scoped_drains=[])


@pytest.mark.parametrize('index', (0, 1))
def test_complete_invocation_retains_only_its_own_handoff(monkeypatch, index):
    # Physical allocation is validated independently; this regression targets
    # the endpoint ownership proof before projection chooses any allocation.
    shared = SimpleNamespace(name='shared', allocation_bytes=lambda n:n)
    monkeypatch.setattr(fm, 'verify_buffer_plan', lambda module:SimpleNamespace(memory_spaces=[shared]))
    original = schedule()
    before = deepcopy(original)
    segment = [original['consumer_events'][index]]
    projected = project_pipeline_schedule(None, original, segment)
    assert original == before
    assert projected['handoffs'] == [original['handoffs'][index]]
    assert projected['stages'] == [original['stages'][index]]
    assert projected['producer_events'] == [original['producer_events'][index]]
    handoff = projected['handoffs'][0]
    assert handoff['writer_name'] in projected['consumer_endpoints']
    assert handoff['reader_name'] in projected['producer_endpoints']
    assert set(projected['consumer_endpoints']) == set(segment[0]['endpoint_bindings'].values())
    assert set(projected['producer_endpoints']) == set(projected['producer_events'][0]['endpoint_bindings'].values())


@pytest.mark.parametrize('change', ('split_call', 'missing_reader', 'repeated', 'wrap_around'))
def test_handoff_crossing_a_program_boundary_remains_rejected(change):
    original = schedule()
    if change == 'split_call':
        original['producer_events'][0]['call'] = 'call_2'
    elif change == 'missing_reader':
        del original['producer_events'][0]['endpoint_bindings']['handoff_reader']
    else:
        original['handoffs'][0][change] = True
    with pytest.raises(CodegenError, match='handoff'):
        project_pipeline_schedule(None, original, [original['consumer_events'][0]])
