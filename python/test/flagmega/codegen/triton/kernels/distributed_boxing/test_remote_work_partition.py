# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Remote copies partition destination tiles only after physical ABI proofs."""
import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest
from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry
from triton.flagmega.errors import CodegenError


@pytest.fixture(scope='module')
def partition():
    path = (Path(__file__).resolve().parents[6] / 'tutorials/flagmega/04-glm5.3-flash-fp8/nvidia-h800'
            / 'local_optimizations/remote_work.py')
    previous = TritonTemplateRegistry.render
    try:
        spec = importlib.util.spec_from_file_location('_glm53_remote_partition', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        TritonTemplateRegistry.render = previous
    return module.gather_partition


def fixture():
    kind = {'partial':None, 'exclusive':None, 'axis_policies':[{'kind':'broadcast'},
            {'kind':'split','stages':[{'hierarchy_axes':[0,1,2]}]}]}
    source = {'storage':'workspace','memory_sharing_scope':'chip','coordinate_space':'local',
              'distributed_type':kind,'distributed_backing_type':None,'pool_byte_offset':0,
              'logical_shape':[1,8192],'local_capacity_shape':[1,16],
              'scalar_storage_strides':[16,1],'scalar_lane_count':1,'scalar_itemsize':4,
              'component_stride_scalar_elements':16,'placement_owner_count':512}
    result = {**deepcopy(source),'coordinate_space':'canonical_global','pool_byte_offset':65536,
              'local_capacity_shape':[1,8192],'scalar_storage_strides':[8192,1],
              'component_stride_scalar_elements':0,
              'distributed_type':{'partial':None,'exclusive':None,
                                  'axis_policies':[{'kind':'broadcast'},{'kind':'broadcast'}]}}
    leaf = {'reduction':False,'capacity':8192,'tile':1024,'source':'workspace + shard_index * 16',
            'source_offset':'unused_local_offset','result':'workspace + 16384','result_offset':'boxing_offsets',
            'remote_source':'peer_window','remote_source_offset':'boxing_offsets',
            'remote_owner':'boxing_offsets // 2048','remote_local_base':'workspace',
            'remote_window_pool':'workspace','active':'boxing_offsets < 8192',
            'writer_active':'(shard_coord0 == 1) & (shard_coord1 == 0) & (shard_coord2 == 0)'}
    return ({'participation_active':'True'},leaf,
            {'abi':source,'runtime_argument':'workspace'}, {'abi':result,'runtime_argument':'workspace'},
            {'distributed_entry':True,'mesh_levels':['c','b','b'],'mesh_hierarchy':[4,8,16],
             'grid':[128,1,1],'device_window_storage':'workspace'})


def test_unique_tiles_preserve_the_chip_writer_condition(partition):
    args = fixture()
    writer = partition(*args)
    assert writer is not None
    for chip in range(4):
        tiles = [tile for cta in range(128) for tile in range(cta*1024,8192,128*1024)
                 if eval(writer,{'__builtins__':{}},{'shard_coord0':chip})]
        assert tiles == (list(range(0,8192,1024)) if chip==1 else [])


@pytest.mark.parametrize('reason', ['private_source','wrong_window','private_result','split_result',
    'split_backing','partial','exclusive','remote_cta_base','remote_cta_offset','reduction',
    'partial_participation','truncated_grid','alias','padded_owner_alias','different_frame'])
def test_unproven_copy_keeps_its_original_ownership(partition,reason):
    call,leaf,source,result,context = fixture()
    if reason=='private_source': source['abi']['memory_sharing_scope']='block'
    elif reason=='wrong_window': leaf['remote_window_pool']='other_pool'
    elif reason=='private_result': result['abi']['coordinate_space']='local'
    elif reason=='split_result': result['abi']['distributed_type']['axis_policies'][1]={'kind':'split','stages':[{'hierarchy_axes':[1]}]}
    elif reason=='split_backing': result['abi']['distributed_backing_type']=deepcopy(source['abi']['distributed_type'])
    elif reason=='partial': source['abi']['distributed_type']['partial']={'axes':[0]}
    elif reason=='exclusive': result['abi']['distributed_type']['exclusive']={'axes':[2]}
    elif reason=='remote_cta_base': leaf['remote_local_base']='workspace + shard_index * 16'
    elif reason=='remote_cta_offset': leaf['remote_source_offset']='boxing_offsets + shard_coord1 * 32'
    elif reason=='reduction': leaf['reduction']=True
    elif reason=='partial_participation': call['participation_active']='(shard_coord2 == 0)'
    elif reason=='truncated_grid': context['grid']=[64,1,1]
    elif reason=='alias': result['abi']['pool_byte_offset']=16384
    elif reason=='padded_owner_alias': source['abi']['component_stride_scalar_elements']=64
    elif reason=='different_frame': result['runtime_argument']='other_frame'
    assert partition(call,leaf,source,result,context) is None


@pytest.mark.parametrize('cause', ('private_result', 'wrong_window', 'overlap'))
def test_declared_cooperative_gather_cannot_silently_drop_tile_assignment(partition, cause):
    call, leaf, source, result, context = fixture()
    call.update(family='distributed_boxing', inputs=[{'buffers': [source]}],
                outputs=[{'buffers': [result]}], leaves=[leaf],
                facts={'require_cooperative_remote_gather': True},
                parameters={'leaf_transitions': ('gather_reduce_scatter',)})
    context['render_calls'] = [call]
    if cause == 'private_result':
        result['abi']['memory_sharing_scope'] = 'block'
    elif cause == 'wrong_window':
        leaf['remote_window_pool'] = 'other'
    else:
        result['abi']['pool_byte_offset'] = 0
    with pytest.raises(CodegenError, match='cooperative peer gather'):
        partition.__globals__['override'](None, 'module.py.jinja', context)
