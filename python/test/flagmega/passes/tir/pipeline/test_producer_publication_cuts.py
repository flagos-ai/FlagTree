# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Producer barrier membership follows actual source-publication scope proof."""
import importlib
from dataclasses import replace

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.ir.tir import iter_tir_children

from . import helpers
from . import test_interprocedural_pipeline_regions as interproc


def pass_module():
    return importlib.import_module('triton.flagmega.passes.tir.lower_transfer_pipeline_regions')


def mixed_module(*, publication_range_unrelated=False, nested=False):
    dispatch = helpers.pipeline_dispatch('pipeline_kernel', 0, shared_start=0)
    kernel = helpers.prim_function('pipeline_kernel', (dispatch,))
    write_kv = interproc._call('write_kv', 'ordinary', 'update', 'kv', (), transfer=False)
    kv_stage = interproc._call('kv_stage', kernel.name, 'kv', 'kv_result',
                              interproc._shared_for_call('main', 'kv_stage', dispatch))
    activation = interproc._call('activation', 'ordinary', 'kv_result', 'activation_result', (), transfer=False)
    weight_stage = interproc._call('weight_stage', kernel.name, 'weight', 'result',
                                  interproc._shared_for_call('main', 'weight_stage', dispatch))
    pub = fm.Barrier(fm.BarrierScope.CHIP, ('write_kv',), 'kv_stage', ranges=(
        (fm.SynchronizationRange('workspace', 'unrelated_norm_stat', 0, 4, 'read'),)
        if publication_range_unrelated else ()))
    activation_barrier = fm.Barrier(fm.BarrierScope.BLOCK, ('activation',), 'weight_stage')
    done = fm.Barrier(fm.BarrierScope.CHIP, ('weight_stage',), 'done')
    fields = (write_kv, pub, kv_stage, activation, activation_barrier, weight_stage, done)
    if nested:
        fields = (fm.Sequential(fields[:3]), fm.Sequential((fm.Sequential(fields[3:]),)))
    function = fm.T.execution_function('main', ('update', 'kv', 'weight'), ('result',), fm.Sequential(fields))
    graph = replace(helpers.module(kernel), execution_functions=(function,))
    return graph, (pub, activation_barrier, done)


def walk(statement):
    yield statement
    for child in iter_tir_children(statement):
        yield from walk(child)


@pytest.mark.parametrize('unrelated_range', [False, True])
@pytest.mark.parametrize('nested', [False, True])
def test_mixed_sources_keep_exact_publication_cut(unrelated_range, nested):
    graph, original_barriers = mixed_module(publication_range_unrelated=unrelated_range, nested=nested)
    lowered = pass_module().lower_transfer_pipeline_regions(graph)
    region = lowered.execution_function_map['main'].body.fields[0]
    producer = list(walk(region.produce_body))
    consumer = list(walk(region.consume_body))
    got = [x for x in producer if isinstance(x, fm.Barrier)]
    assert got == [original_barriers[0]], (
        'mixed mutable KV must retain only its selected publication cut; '
        f'consumer-only activation/completion cuts leaked into producer: {[x.before for x in got]}'
    )
    assert [x for x in consumer if isinstance(x, fm.Barrier)] == list(original_barriers)
    for kind, attr in ((fm.PipelineStage, 'stage_id'), (fm.PipelineDrain, 'stage_id'), (fm.PipelineHandoff, 'handoff_id')):
        assert [getattr(x, attr) for x in producer if isinstance(x, kind)] == [getattr(x, attr) for x in consumer if isinstance(x, kind)]
    assert len([x for x in producer if isinstance(x, fm.PipelineDrain)]) == 1
    assert len([x for x in producer if isinstance(x, fm.PipelineHandoff)]) == 1



def test_equal_barriers_at_distinct_tree_positions_are_not_same_occurrence():
    graph, barriers = mixed_module()
    original = graph.execution_function_map['main']
    duplicate = replace(barriers[0])
    assert duplicate == barriers[0] and duplicate is not barriers[0]
    graph = replace(graph, execution_functions=(replace(original, body=fm.Sequential(
        (*original.body.fields, fm.Sequential((duplicate,))))),))
    region = pass_module().lower_transfer_pipeline_regions(graph).execution_function_map['main'].body.fields[0]
    kept = [x for x in walk(region.produce_body) if isinstance(x, fm.Barrier)]
    assert len(kept) == 1 and kept[0] is barriers[0]
    assert sum(isinstance(x, fm.Barrier) for x in walk(region.consume_body)) == 4


def test_joint_axis_scope_keeps_both_witnesses_but_not_redundant_block():
    from collections import defaultdict
    placement = fm.Placement(hierarchy=(2, 4), name='yx', hierarchy_levels='bb')
    writer = interproc._call('writer', 'ordinary', 'update', 'source', (), transfer=False)
    reader = interproc._call('reader', 'pipeline', 'source', 'result', ())
    block = fm.Barrier(fm.BarrierScope.BLOCK, ('writer',), 'reader')
    axis0 = fm.Barrier(fm.BarrierScope.CHIP, ('writer',), 'reader', axis_group_axes=(0,))
    axis1 = fm.Barrier(fm.BarrierScope.CHIP, ('writer',), 'reader', axis_group_axes=(1,))
    order = (writer, block, axis0, axis1, reader)
    after, before = defaultdict(list), defaultdict(list)
    selected = pass_module()._add_execution_source_handoffs(
        order, {id(reader): fm.PipelineStage('reader_stage', reader)}, after, before, 'main', {},
        {'reader': (('writer', ('grid', (0, 1), placement)),)})
    assert selected == frozenset((2, 3))
    assert after[id(axis1)] == before[id(reader)] and not after[id(axis0)]


def test_full_grid_scope_replaces_redundant_earlier_axis_witness():
    placement = fm.Placement(hierarchy=(2, 4), name='yx', hierarchy_levels='bb')
    a = fm.Barrier(fm.BarrierScope.CHIP, ('writer',), 'reader', axis_group_axes=(0,))
    b = fm.Barrier(fm.BarrierScope.CHIP, ('writer',), 'reader')
    assert pass_module()._publication_barrier_indices((a, b), 0, 2, ('grid', (0, 1), placement)) == (1,)


def test_pipeline_callee_does_not_reintroduce_unrelated_caller_barriers():
    graph = interproc._interprocedural_module()
    main = graph.execution_function_map['main']
    writer = interproc._call('writer', 'ordinary', 'update', 'main_source', (), transfer=False)
    publication = fm.Barrier(fm.BarrierScope.CHIP, ('writer',), 'first')
    completion = fm.Barrier(fm.BarrierScope.CHIP, ('second',), 'done')
    main = replace(main, body=fm.Sequential((writer, publication, *main.body.fields, completion)))
    graph = replace(graph, execution_functions=(graph.execution_function_map['worker'], main))
    result = pass_module().lower_transfer_pipeline_regions(graph)
    region = result.execution_function_map['main'].body.fields[0]
    assert [x for x in walk(region.produce_body) if isinstance(x, fm.Barrier)] == [publication]
    assert [x for x in walk(region.consume_body) if isinstance(x, fm.Barrier)] == [publication, completion]
    for kind in (fm.PipelineStage, fm.PipelineDrain, fm.PipelineHandoff):
        assert [x for x in walk(region.produce_body) if isinstance(x, kind)] == [x for x in walk(region.consume_body) if isinstance(x, kind)]


def test_repeated_mutable_function_retains_existing_participation():
    graph, barriers = mixed_module()
    fn = graph.execution_function_map['main']
    graph = replace(graph, execution_functions=(replace(fn, attrs={'region_kind': 'for'}),))
    region = pass_module().lower_transfer_pipeline_regions(graph).execution_function_map['main'].body.fields[0]
    assert [x for x in walk(region.produce_body) if isinstance(x, fm.Barrier)] == list(barriers)
    assert [x for x in walk(region.consume_body) if isinstance(x, fm.Barrier)] == list(barriers)


def test_acyclic_tail_write_is_not_a_publication_predecessor():
    from . import test_producer_source_barriers as source_tests
    graph = source_tests._source_graph(fm.BarrierScope.CHIP, source_written=True, write_after=True)
    region = pass_module().lower_transfer_pipeline_regions(graph).execution_function_map['main'].body.fields[0]
    assert not any(isinstance(x, fm.Barrier) for x in walk(region.produce_body))
    assert sum(isinstance(x, fm.Barrier) for x in walk(region.consume_body)) == 2


def test_counted_mutable_loop_preserves_existing_barriers_and_tree_paths():
    graph, barriers = mixed_module(nested=True)
    fn = graph.execution_function_map['main']
    loop = fm.T.for_loop(fm.dim('i', minimum=0, maximum=1), fm.T.range(0, 2),
                         fm.T.LoopMode.SERIAL, fn.body)
    graph = replace(graph, execution_functions=(replace(fn, body=fm.Sequential((loop,))),))
    region = pass_module().lower_transfer_pipeline_regions(graph).execution_function_map['main'].body.fields[0]
    assert isinstance(region.produce_body.fields[0], fm.For)
    assert [x for x in walk(region.produce_body) if isinstance(x, fm.Barrier)] == list(barriers)
    assert [x for x in walk(region.consume_body) if isinstance(x, fm.Barrier)] == list(barriers)
