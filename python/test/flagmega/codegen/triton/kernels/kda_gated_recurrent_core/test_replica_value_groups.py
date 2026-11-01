# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Independent value tasks preserve every KDA owner and physical allocation."""

from copy import deepcopy
import importlib.util
from itertools import product
from pathlib import Path
import sys
from types import ModuleType

import pytest
from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry

from python.test.flagmega.codegen.triton.kernels.dense_matmul.test_replica_owner_groups import fixture


@pytest.fixture(scope='module')
def optimization():
    root = (Path(__file__).resolve().parents[6]
            / 'tutorials/flagmega/04-glm5.3-flash-fp8/nvidia-h800/local_optimizations')
    name = '_flagmega_kda_value_group_tests'
    package = ModuleType(name)
    package.__path__ = [str(root)]
    sys.modules[name] = package
    previous = TritonTemplateRegistry.render
    try:
        spec = importlib.util.spec_from_file_location(name + '.kda_work', root / 'kda_work.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        TritonTemplateRegistry.render = previous
    try:
        yield module
    finally:
        for key in tuple(sys.modules):
            if key == name or key.startswith(name + '.'):
                sys.modules.pop(key)


@pytest.mark.parametrize('owner_axes', [(0, 2), (2,)])
def test_every_head_value_chunk_has_one_writer_per_chip(optimization, owner_axes):
    call, context = fixture(owner_axes=owner_axes)
    call.update(query_head_offset='shard_coord2 * 128', state_head_offset='shard_coord2 * 16384')
    group = optimization.value_group(call, context)
    assert group['axes'] == [1]
    tasks = {}
    for coordinates in product(range(4), range(8), range(16)):
        # Even a logically broadcast chip axis has a separate physical pool.
        owner = (coordinates[0], *(coordinates[axis] for axis in owner_axes if axis != 0))
        replica = eval(group['index'], {'__builtins__': {}},
                       {f'shard_coord{axis}': value for axis, value in enumerate(coordinates)})
        tasks.setdefault(owner, []).extend(range(replica, 4, group['count']))
    assert all(sorted(chunks) == list(range(4)) for chunks in tasks.values())


@pytest.mark.parametrize('reason', ['replica_address', 'private_state', 'partial', 'exclusive', 'private_base'])
def test_unproved_state_ownership_cannot_share_value_tasks(optimization, reason):
    call, context = fixture(owner_axes=(0, 2))
    abi = call['outputs'][0]['buffers'][0]['abi']
    if reason == 'replica_address':
        call['state_head_offset'] = 'shard_coord1 * 16384'
    elif reason == 'private_state':
        abi['memory_sharing_scope'] = 'block'
    elif reason == 'partial':
        abi['distributed_type']['partial'] = {'axes': [1]}
    elif reason == 'exclusive':
        abi['distributed_type']['exclusive'] = {'axes': [1]}
    else:
        call['state'] = '_flagmega_block_scope_base(state, 65536)'
    assert optimization.value_group(call, context) is None


def test_value_writers_keep_other_owners_and_publish_each_chip_replica(optimization):
    abi = {'storage_kind': 'canonical_global', 'distributed_type': {
        'axis_policies': [{'kind': 'split', 'stages': [{'hierarchy_axes': [3]}]}],
        'placement': {'hierarchy': [2, 4, 8, 16], 'hierarchy_levels': 'cbbb'},
        'partial': None, 'exclusive': None,
    }}
    # Only axis 2 divides the value chunks. Axis 1 still elects its writer;
    # chip replicas must each publish into their own physical allocation.
    expression = optimization.writer_for_value_tasks(abi, {'axes': [2]})
    assert expression == '(shard_coord1 == 0)'
    compact = deepcopy(abi)
    compact['storage_kind'] = 'compact_per_chip'
    assert optimization.writer_for_value_tasks(compact, {'axes': [2]}) == 'True'
