# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A cooperative tile assignment must never merge distinct physical owners."""
import importlib.util
from copy import deepcopy
from itertools import product
from pathlib import Path

import pytest
from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry


@pytest.fixture(scope='module')
def replica_group():
    source = (Path(__file__).resolve().parents[6] / 'tutorials/flagmega/04-glm5.3-flash-fp8/nvidia-h800'
              / 'local_optimizations/replica_work.py')
    previous = TritonTemplateRegistry.render
    try:
        spec = importlib.util.spec_from_file_location('_glm53_replica_owner_groups', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        TritonTemplateRegistry.render = previous
    return module.replica_group


def fixture(hierarchy=(4, 8, 16), levels='cbb', owner_axes=(0, 1)):
    policy = {'kind':'split', 'stages':[{'hierarchy_axes':list(owner_axes)}]} if owner_axes else {'kind':'broadcast'}
    kind = {'axis_policies':[{'kind':'broadcast'},policy], 'partial':None, 'exclusive':None}
    abi = {'memory_sharing_scope':'chip', 'distributed_type':kind, 'distributed_backing_type':None}
    call = {'participation_active':'True', 'inputs':[{'buffers':[{'abi':deepcopy(abi)}]}],
            'outputs':[{'buffers':[{'abi':deepcopy(abi)}]}],
            'source':'x', 'weight':'w', 'result':'y', 'source_offset':'k', 'weight_offset':'n*4096+k',
            'result_offset':'n'}
    blocks = 1
    for size, level in zip(hierarchy, levels, strict=True):
        if level=='b':
            blocks *= size
    context = {'distributed_entry':True, 'mesh_hierarchy':hierarchy, 'mesh_levels':tuple(levels),
               'grid':(blocks,1,1)}
    return call, context


@pytest.mark.parametrize('hierarchy,levels,owner_axes', [
    ((8,16),'bb',(0,)),
    ((4,8,16),'cbb',(0,1)),
    ((4,8,16),'cbb',(0,)),
    ((2,4,8),'bbb',(1,)),
])
def test_each_owner_covers_every_output_tile_once(replica_group, hierarchy, levels, owner_axes):
    call, context = fixture(hierarchy,levels,owner_axes)
    group = replica_group(call,context)
    assert group is not None
    owners = {}
    tiles = 303  # The head has an eight-element final N tail.
    for coordinates in product(*(range(size) for size in hierarchy)):
        owner = tuple(coordinates[axis] for axis in owner_axes)
        if not owner_axes:
            owner = ()
        values = {f'shard_coord{axis}':value for axis,value in enumerate(coordinates)}
        replica = eval(group['index'], {'__builtins__':{}}, values)
        owners.setdefault(owner,[]).extend(range(replica,tiles,group['count']))
    assert all(sorted(actual)==list(range(tiles)) for actual in owners.values())


def test_backing_ownership_and_partial_components_are_preserved(replica_group):
    call, context = fixture(owner_axes=())
    backing = {'axis_policies':[{'kind':'split','stages':[{'hierarchy_axes':[1]}]}],
               'partial':{'axes':[0]}, 'exclusive':None}
    call['inputs'][0]['buffers'][0]['abi']['distributed_backing_type'] = backing
    assert replica_group(call,context)['axes']==[2]
    backing['partial']['axes'].append(2)
    assert replica_group(call,context) is None


def test_additional_operand_addresses_must_be_invariant_in_the_replica_group(replica_group):
    call, context = fixture(owner_axes=(0, 2))
    group = replica_group(call, context, expressions=("state + shard_coord2 * 16384",))
    assert group['axes'] == [1]
    assert replica_group(call, context, expressions=("state + shard_coord1 * 16384",)) is None
    assert replica_group(call, context, expressions=("_flagmega_block_scope_base(state, 65536)",)) is None


def test_two_dimensional_mesh_aliases_do_not_hide_replica_dependencies(replica_group):
    call, context = fixture(hierarchy=(8, 16), levels='bb', owner_axes=(1,))
    assert replica_group(call, context, expressions=("state + shard_x * 16384",))['axes'] == [0]
    assert replica_group(call, context, expressions=("state + shard_y * 16384",)) is None


@pytest.mark.parametrize('reason', ['private_storage','hidden_base','replica_address','exclusive',
                                    'partial_participation','truncated_grid','ordinary_launch'])
def test_unproven_shared_view_cannot_partition_work(replica_group,reason):
    call, context = fixture()
    abi = call['outputs'][0]['buffers'][0]['abi']
    if reason=='private_storage':
        abi['memory_sharing_scope']='block'
    elif reason=='hidden_base':
        call['result']='block_local_data + shard_index * 4864'
    elif reason=='replica_address':
        call['weight_offset']='n*4096 + k + shard_coord2*128'
    elif reason=='exclusive':
        abi['distributed_type']['exclusive']={'axes':[2], 'owner_coordinates':[0]}
    elif reason=='partial_participation':
        call['participation_active']='(shard_coord2 == 0)'
    elif reason=='truncated_grid':
        context['grid']=(64,1,1)
    elif reason=='ordinary_launch':
        context['distributed_entry']=False
    assert replica_group(call,context) is None


def private_read_fixture(*, rebased=False):
    from triton.flagmega.codegen.triton.physical_access import emit_buffer_pointer

    call, context = fixture(owner_axes=(0,))
    abi = call['inputs'][0]['buffers'][0]['abi']
    abi.update(memory_sharing_scope='block', coordinate_space='local', storage_kind='compact_local',
               storage='block_local_data', pooled=True, scalar_dtype='float32', scalar_itemsize=4,
               pool_byte_offset=0, pool_scope_count=1 if rebased else 128,
               pool_scope_stride_bytes=0 if rebased else 2048,
               pool_scope_index=None if rebased else 'block_index')
    abi['distributed_type']['tensor'] = {'kind': 'tensor'}
    abi['distributed_type']['placement'] = {'hierarchy': context['mesh_hierarchy'],
                                          'hierarchy_levels': ''.join(context['mesh_levels'])}
    call['inputs'][0]['buffers'][0]['runtime_argument'] = 'private_pool'
    call['inputs'][0]['role'] = 'input'
    call['source'] = emit_buffer_pointer(abi, 'private_pool')
    return call, context


@pytest.mark.parametrize('rebased', (False, True))
def test_complete_private_reads_can_feed_shared_replica_tasks(replica_group, rebased):
    call, context = private_read_fixture(rebased=rebased)
    group = replica_group(call, context, expressions=(call['source'],))
    assert group is not None and group['axes'] == [1, 2] and group['count'] == 128


@pytest.mark.parametrize('cause', ('view', 'backing', 'missing_type', 'partial', 'hidden_offset',
                                  'scope_index', 'inout'))
def test_private_read_replication_requires_an_exact_complete_component(replica_group, cause):
    call, context = private_read_fixture()
    abi = call['inputs'][0]['buffers'][0]['abi']
    if cause == 'view':
        abi['view_byte_offset'] = 8
    elif cause == 'backing':
        abi['distributed_backing_type'] = deepcopy(abi['distributed_type'])
    elif cause == 'missing_type':
        abi['distributed_type'] = None
    elif cause == 'partial':
        abi['distributed_type']['partial'] = {'axes': [0]}
    elif cause == 'hidden_offset':
        call['source_offset'] = 'k + block_index * 32'
    elif cause == 'scope_index':
        abi['pool_scope_index'] = 'shard_index'
    else:
        call['inputs'][0]['role'] = 'inout'
    assert replica_group(call, context) is None
