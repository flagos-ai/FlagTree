# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A completed device partial retains its component across local meshes."""

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.codegen.triton.tir_package import describe_tir_package
from triton.flagmega.errors import IRSchemaError, IRVerificationError
from triton.flagmega.ir.tir.buffer import Buffer
from triton.flagmega.passes.functions.graph import function_nodes
from triton.flagmega.passes.tir.bufferize.planner import BufferizationOptions


def partial_mesh_graph(dtype='float32', lanes=()):
    source_place = fm.Placement((4, 4, 16), 'cyx', 'cbb')
    target_place = fm.Placement((4, 8, 16), 'cyx', 'cbb')
    class Graph(fm.Module):
        def forward(self):
            x = self.input('x', fm.tensor_type(dtype, (4, 1, 4096)), id='x')
            local = fm.F.distributed.sharded_view(x, fm.DistributedType(x.type,
                (fm.SBP.split_contiguous((0,)), fm.SBP.broadcast(), fm.SBP.broadcast()), source_place))
            partial = fm.F.math.reduce_sum(local, axes=(0,), keep_dims=False, output_lanes=lanes,
                                           name='partial', metadata={'execution_mesh': (4, 16)})
            target_type = replace(partial.type, placement=target_place, axis_policies=(fm.SBP.broadcast(),
                fm.SBP.split_contiguous((1, 2))))
            view = fm.F.distributed.sharded_view(partial, target_type, name='repartitioned')
            result = fm.F.distributed.force_boxing(view, fm.logical_type(view.type), name='result',
                                                  metadata={'execution_mesh': (8, 16)})
            self.function('main', (x,), (result,))
    return Graph(dialect='distributed', stage='frozen_constants', entry='main', metadata={
        'auto_distribution': {'placement': source_place.to_data()},
    }).build()


@pytest.mark.parametrize('dtype,lanes', [('float32', ()), ('bfloat16', (2, 4))])
def test_normal_partial_mesh_handoff_preserves_one_chip_backing(tmp_path, dtype, lanes):
    module = Compiler().compile(partial_mesh_graph(dtype, lanes)).module
    plan = fm.verify_buffer_plan(module)
    source, view = (plan.buffer_map[name] for name in ('partial', 'repartitioned'))
    assert source.distributed_type.partial == view.distributed_type.partial == fm.SBP.partial((0,))
    assert source.physical_id == view.physical_id
    assert source.mem_span.must_alias(view.mem_span)
    assert source.distributed_storage_kind == view.distributed_storage_kind == fm.DistributedBufferStorageKind.COMPACT_PER_CHIP
    assert view.nbytes == source.nbytes == 4096 * fm.DType(dtype).itemsize
    assert view.storage_distributed_type.placement == view.distributed_type.placement
    assert fm.block_owner_count(view.storage_distributed_type) == 1
    assert source.mem_span.buffer.live_start <= source.live_start
    assert source.mem_span.buffer.live_end >= view.live_end
    collective_index = next(index for index, node in enumerate(function_nodes(module, module.function_map['main']))
                            if node.id == 'result')
    assert source.mem_span.buffer.live_end >= collective_index
    space = plan.memory_space_map[view.mem_span.buffer.memory_space]
    assert space.sharing_scope == fm.MemorySharingScope.CHIP and space.kind != 'shared'
    tir_buffer = Buffer('repartitioned', view.dtype, view.mem_span, view.distributed_type.tensor.shape,
        tuple(fm.dim(stride) for stride in view.strides), view.distributed_type,
        view.distributed_storage_kind, view.storage_distributed_type)
    assert tir_buffer.mem_span == view.mem_span
    description = describe_tir_package(module)
    assert [p['grid'] for p in description['kernels']] == [[64, 1, 1], [128, 1, 1]]
    assert [len(p['calls']) for p in description['kernels']] == [1, 1]
    restored = fm.load_module(fm.emit_module(module, tmp_path / 'partial.py'))
    assert restored.semantic_hash == module.semantic_hash
    assert describe_tir_package(restored)['kernels'] == description['kernels']


def test_each_consumer_owner_reads_its_coordinate_in_the_same_chip_component():
    from triton.flagmega.codegen.triton.distributed_abi import distributed_buffer_abi
    module = Compiler().compile(partial_mesh_graph()).module
    view = fm.verify_buffer_plan(module).buffer_map['repartitioned']
    for chip in range(4):
        for block in (0, 1, 17, 63, 127):
            abi = distributed_buffer_abi(view, chip * 128 + block)
            assert abi.partial_axes == (0,)
            assert abi.component_offset_elements == 0
            assert abi.storage_axis_coordinate(1, 0) == block * 32
            assert abi.storage_axis_coordinate(1, 31) == block * 32 + 31


@pytest.mark.parametrize('bad_partial', [None, fm.SBP.partial((0,), 'max'), fm.SBP.partial((1,))])
def test_parent_backing_cannot_forge_the_consumer_partial_policy(bad_partial):
    module = Compiler().compile(partial_mesh_graph()).module
    view = fm.verify_buffer_plan(module).buffer_map['repartitioned']
    forged = replace(view.storage_distributed_type, partial=bad_partial)
    with pytest.raises(ValueError, match='parent-shard backing'):
        replace(view, distributed_backing_type=forged)
    with pytest.raises(IRSchemaError, match='parent-shard backing'):
        Buffer('repartitioned', view.dtype, view.mem_span, view.distributed_type.tensor.shape,
               tuple(fm.dim(stride) for stride in view.strides), view.distributed_type,
               view.distributed_storage_kind, forged)


@pytest.mark.parametrize('source_kind', ['private_complete', 'chip_fragmented'])
def test_cross_mesh_partial_alias_requires_a_real_complete_chip_shared_component(source_kind):
    tensor = fm.tensor_type('float32', (1, 4096))
    source_place = fm.Placement((4, 4, 16), 'cyx', 'cbb')
    target_place = fm.Placement((4, 8, 16), 'cyx', 'cbb')
    policy = fm.SBP.split_contiguous((1, 2)) if source_kind == 'chip_fragmented' else fm.SBP.broadcast()
    source_type = fm.DistributedType(tensor, (fm.SBP.broadcast(), policy), source_place,
                                     partial=fm.SBP.partial((0,)))
    target_type = replace(source_type, placement=target_place,
                          axis_policies=(fm.SBP.broadcast(), fm.SBP.split_contiguous((1, 2))))
    builder = fm.IRBuilder(dialect='semantic_tir', stage='packaged_tir')
    lhs = builder.var('lhs', source_type)
    rhs = builder.var('rhs', source_type)
    produced = builder.call('math.add', (lhs, rhs), source_type, id='produced')
    view = builder.call('distributed.sharded_view', (produced,), target_type,
                        id='repartitioned', attrs={'new_type': target_type})
    result = builder.call('distributed.boxing', (view,), tensor, id='result', attrs={'new_type': tensor})
    builder.function('main', (lhs, rhs), (result,))
    options = BufferizationOptions.generic()
    if source_kind == 'chip_fragmented':
        options = replace(options, memory_spaces=tuple(
            replace(space, sharing_scope=fm.MemorySharingScope.CHIP) if space.name == 'workspace' else space
            for space in options.memory_spaces), block_local=None, rdata_chip_sharding=True)
    with pytest.raises(IRVerificationError, match='complete chip-shared component'):
        fm.make_buffer_plan(builder.build(entry='main'), options=options)
