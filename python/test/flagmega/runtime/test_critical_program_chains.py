# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A minimum stream count must also retain the longer branch's critical chain."""

import pytest

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir.program_order import program_ancestors, program_stream_indices


def asymmetric_forks(count=2):
    programs = []
    for i in range(count):
        prefix = f'{i}.prefix'
        up, down, shared, join = (f'{i}.{name}' for name in ('up', 'down', 'shared', 'join'))
        programs.extend([
            dict(symbol=prefix, depends_on=() if not i else (f'{i-1}.join',)),
            dict(symbol=up, depends_on=(prefix,)),
            dict(symbol=down, depends_on=(up,)),
            dict(symbol=shared, depends_on=(prefix,)),
            dict(symbol=join, depends_on=(prefix, up, down, shared)),
        ])
    return programs


def test_repeated_longer_branch_keeps_its_complete_critical_path_on_one_stream():
    programs = asymmetric_forks()
    streams = program_stream_indices(programs)
    critical = [f'{i}.{name}' for i in range(2) for name in ('prefix', 'up', 'down', 'join')]
    assert len(set(streams.values())) == 2
    assert all(streams[left] == streams[right] for left, right in zip(critical, critical[1:])), streams
    for i in range(2):
        assert streams[f'{i}.shared'] != streams[f'{i}.up']


def test_critical_cover_preserves_every_original_dependency_and_parallel_branch():
    programs = asymmetric_forks(3)
    ancestors = program_ancestors(programs)
    streams = program_stream_indices(programs)
    assert len(set(streams.values())) == 2
    # Sharing a stream adds no ordering beyond the original transitive DAG.
    names = list(ancestors)
    for i, left in enumerate(names):
        for right in names[i+1:]:
            if streams[left] == streams[right]:
                assert left in ancestors[right]
    reverse = [{**p, 'depends_on': tuple(reversed(p['depends_on']))} for p in programs]
    assert program_stream_indices(reverse) == streams


def test_legacy_serialized_policy_keeps_its_original_stream_mapping():
    programs = asymmetric_forks()
    legacy = program_stream_indices(programs, policy='minimum_chains_v1')
    current = program_stream_indices(programs, policy='critical_path_v2')
    assert legacy != current
    assert legacy['1.prefix'] != legacy['1.up']
    assert len(set(legacy.values())) == len(set(current.values())) == 2


def test_unknown_policy_cannot_silently_select_another_schedule():
    with pytest.raises(IRVerificationError, match='stream policy'):
        program_stream_indices(asymmetric_forks(), policy='unknown')
    assert program_stream_indices(None, policy='minimum_chains_v1') is None
    assert program_stream_indices(None, policy='critical_path_v2') is None
