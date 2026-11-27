# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A mixed peer-reuse/local-RAW fence needs per-range completion proofs."""
from types import SimpleNamespace

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.program_dependencies import elide_completed_device_barriers


def access(offset, mode, *, peer=False):
    return SimpleNamespace(physical_id='workspace', offset=offset, nbytes=64,
        distributed_type=None, mode=mode,
        effect=fm.MemoryEffect(mode, fm.MemoryAccessScope.DEVICE if peer else fm.MemoryAccessScope.CHIP))


@pytest.mark.parametrize('case,removed', [('local', True), ('peer_reader', False),
                                       ('unordered_local', False), ('unknown_range', False)])
def test_mixed_epoch_fence_accepts_only_completed_peer_and_ordered_local_ranges(case, removed):
    programs = (dict(symbol='old', depends_on=()), dict(symbol='epoch', depends_on=('old',)),
        dict(symbol='local', depends_on=('epoch',)),
        dict(symbol='reader', depends_on=('epoch',) if case == 'unordered_local' else ('local',)))
    accesses = dict(old=(access(0, 'read', peer=True),), epoch=(),
        local=(access(64, 'write'),),
        reader=(access(0, 'write'), access(64, 'read', peer=case == 'peer_reader')))
    completion = dict(old=False, epoch=True, local=False, reader=False)
    ranges = [dict(physical_id='workspace', offset=0, nbytes=64),
              dict(physical_id='workspace', offset=64, nbytes=64)]
    if case == 'unknown_range':
        ranges.append(dict(physical_id='unknown', offset=0, nbytes=64))
    event = dict(kind='barrier', scope='device', before='reader',
                 hazards=('READ->WRITE', 'WRITE->READ'), ranges=ranges)
    result, elided = elide_completed_device_barriers([event, dict(kind='function_call')],
        'reader', programs, accesses, completion)
    assert bool(elided) is removed
    assert (result[0]['kind'] == 'function_call') is removed
