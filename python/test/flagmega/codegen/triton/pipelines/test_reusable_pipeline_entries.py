# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Complete pipeline entries share code, never their invocation bindings."""
from copy import deepcopy
import ast
from types import SimpleNamespace as NS

import pytest
from jinja2 import ChoiceLoader, DictLoader
from triton.flagmega.codegen.triton import function_programs
from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry


def pipeline(stem, offset=0):
    workspace = dict(variable=stem+'_tiles', shape=[3,64,512], dtype='tl.bfloat16',
        offset_bytes=offset, nbytes=196608, alignment_bytes=16, allocation_alignment_bytes=16,
        matrix_compatible=True, buffer_name=stem+'_backing', physical_buffer=stem+'_shared')
    channel = dict(name='weight', pipe_name=stem+'_pipe', reader_name=stem+'_reader',
        writer_name=stem+'_writer', capacity=3, fields=[dict(name='weight',workspace=workspace)])
    stage = dict(id=stem+'_stage',call='leaf',workspaces=[workspace],channels=[channel],
        consumer_workspaces=[],auxiliary_consumer=None,producer_warps=1,producer_registers=24)
    return dict(stages=[stage],handoffs=[],consumer_endpoints=[channel['reader_name']],
        producer_endpoints=[channel['writer_name']],auxiliary_endpoints=[],auxiliary_events=[],
        shared_arena_required_nbytes=offset+196608,shared_arena_nbytes=offset+196608,
        shared_arena_alignment_bytes=16,trip_scoped_closes=[],trip_scoped_drains=[])


def inputs(second_offset=0):
    formal = pipeline('formal')
    definition = dict(function='worker',symbol='worker_fn',signature_arguments=['value','weight'],
        typed_signature_arguments=['value','weight'],function_pool_abi=dict(pools=[],record_width=0),
        schedule=formal)
    programs, launches, physical = [], [], []
    for index in (0,1):
        stem = f'actual_{index}'
        scope = pipeline(stem, second_offset if index else 0)
        call = f'call_{index}'
        consumer = dict(kind='function_call',call=call,callee='worker',symbol='worker_fn__consumer',
            arguments=stem+'_reader, value, weight',argument_expressions=(stem+'_reader','value','weight'),
            endpoint_bindings={'formal_reader':stem+'_reader'},descriptor_argument_bindings={})
        producer = dict(kind='function_call',call=call,callee='worker',symbol='worker_fn__producer',
            arguments=stem+'_writer, value, weight',argument_expressions=(stem+'_writer','value','weight'),
            endpoint_bindings={'formal_writer':stem+'_writer'},descriptor_argument_bindings={})
        drain = dict(kind='drain',endpoints=[stem+'_reader'],stage_id=stem+'_stage')
        scope.update(consumer_events=[consumer,drain],producer_events=[producer,
            {**drain,'endpoints':[stem+'_writer']}])
        programs.append(dict(symbol=f'program_{index}',entry_events=scope['consumer_events'],
            device_functions=[definition],pipeline_schedule=scope))
        launches.append(dict(symbol=f'program_{index}',num_warps=4,grid=[128,1,1]))
        physical.append(dict(kind='function_call',call=call))
    descriptor = dict(signature_arguments=['input','weight_0','weight_1'],dynamic_argument_indices=[0],
        runtime_binding=dict(call_abi=dict(events=physical)),device_context_abi='runtime-handles/v2')
    return NS(entry='main'),descriptor,programs,launches


def project(monkeypatch, second_offset=0):
    monkeypatch.setattr(function_programs,'describe_function_runtime_binding',lambda *a,**k:{})
    def actual(module,root,definition,call,names,callee,physical):
        index=int(call['call'].removeprefix('call_'))
        return dict(value=dict(kind='argument',source='input'),
                    weight=dict(kind='argument',source=f'weight_{index}'))
    monkeypatch.setattr(function_programs,'_actual_bindings',actual)
    module,descriptor,programs,launches=inputs(second_offset)
    assert function_programs.project_function_programs(module,descriptor,programs,launches)
    return programs,launches


def test_complete_pipeline_entries_share_code_and_keep_both_roles(monkeypatch):
    programs,launches=project(monkeypatch)
    assert [k['entry_function'] for k in launches]==['worker','worker']
    assert launches[0]['entry_symbol']==launches[1]['entry_symbol']
    assert launches[0]['argument_bindings']!=launches[1]['argument_bindings']
    scope=programs[0]['pipeline_schedule']
    assert scope['consumer_endpoints']==['formal_reader']
    assert scope['producer_endpoints']==['formal_writer']
    assert scope['consumer_events'][0]['arguments']=='formal_reader, value, weight'
    assert scope['producer_events'][0]['arguments']=='formal_writer, value, weight'
    assert scope['consumer_events'][1]['endpoints']==['formal_reader']
    assert scope['producer_events'][1]['endpoints']==['formal_writer']
    assert scope['stages'][0]['channels'][0]['fields'][0]['workspace']['offset_bytes']==0


def test_complete_reusable_entry_inlines_only_its_root_invocation(monkeypatch):
    programs,_=project(monkeypatch)
    assert programs[0]['entry_inline_function']=='worker'
    context=deepcopy(programs[0])
    context.update(distributed_entry=False,mesh_hierarchy=[1])
    context['pipeline_schedule'].update(consumer_warps=4,producer_warps=1,producer_registers=24)
    nested=deepcopy(context['device_functions'][0])
    nested.update(function='nested',symbol='nested_fn')
    context['device_functions'].append(nested)
    for definition in context['device_functions']:
        for role in ('consumer','producer'):
            definition['schedule'][role+'_events']=[dict(kind='function_call',call='leaf',
                callee='leaf',symbol='leaf_'+role,arguments='value, weight')]
    source=TritonTemplateRegistry().render('entrypoints/call_graph.py.jinja',context)
    functions={node.name:node for node in ast.parse(source).body if isinstance(node,ast.FunctionDef)}
    def noinline(name):
        decorator=functions[name].decorator_list[0]
        return next(keyword.value.value for keyword in decorator.keywords if keyword.arg=='noinline')
    for role in ('consumer','producer'):
        assert noinline('worker_fn__'+role) is False
        assert noinline('nested_fn__'+role) is True


def test_unprojected_device_functions_keep_their_call_boundary():
    _,_,programs,_=inputs()
    context=deepcopy(programs[0])
    context.update(signature='value, weight',signature_arguments=['value','weight'],
        typed_signature_arguments=['value','weight'],distributed_entry=False,mesh_hierarchy=[1])
    context['pipeline_schedule'].update(consumer_warps=4,producer_warps=1,producer_registers=24)
    for role in ('consumer','producer'):
        context['device_functions'][0]['schedule'][role+'_events']=[dict(kind='function_call',
            call='leaf',callee='leaf',symbol='leaf_'+role,arguments='value, weight')]
    source=TritonTemplateRegistry().render('entrypoints/call_graph.py.jinja',context)
    functions={node.name:node for node in ast.parse(source).body if isinstance(node,ast.FunctionDef)}
    for role in ('consumer','producer'):
        decorator=functions['worker_fn__'+role].decorator_list[0]
        assert next(keyword.value.value for keyword in decorator.keywords if keyword.arg=='noinline') is True


def test_program_namespace_forwards_its_root_call_boundary(monkeypatch):
    programs,_=project(monkeypatch)
    program=deepcopy(programs[0])
    program.update(num_warps=4,render_calls=[])
    program['pipeline_schedule'].update(consumer_warps=4,producer_warps=1,producer_registers=24)
    for role in ('consumer','producer'):
        program['device_functions'][0]['schedule'][role+'_events']=[dict(kind='function_call',
            call='leaf',callee='leaf',symbol='leaf_'+role,arguments='value, weight')]
    registry=TritonTemplateRegistry()
    # The actual namespace wrapper must carry entry policy into its include;
    # this fixture isolates that boundary from unrelated algorithm templates.
    registry.environment.loader=ChoiceLoader([DictLoader({
        'module.py.jinja':'{% include "entrypoints/call_graph.py.jinja" %}'}),registry.environment.loader])
    source=registry.render('programs.py.jinja',dict(renderer_version='test',kernels=[],
        programs=[program],signature='value, weight',signature_arguments=['value','weight'],
        typed_signature_arguments=['value','weight'],scalar_arguments=[],
        distributed_entry=False,mesh_hierarchy=[1]))
    functions={node.name:node for node in ast.walk(ast.parse(source)) if isinstance(node,ast.FunctionDef)}
    for role in ('consumer','producer'):
        decorator=functions['worker_fn__'+role].decorator_list[0]
        assert next(keyword.value.value for keyword in decorator.keywords if keyword.arg=='noinline') is False


def test_different_shared_placement_requires_distinct_pipeline_entries(monkeypatch):
    programs,launches=project(monkeypatch,second_offset=16)
    assert launches[0]['entry_symbol']!=launches[1]['entry_symbol']
    assert programs[1]['pipeline_schedule']['stages'][0]['workspaces'][0]['offset_bytes']==16


def test_pipeline_entry_passes_each_cta_frame_to_both_roles():
    _,_,programs,_=inputs()
    scope=deepcopy(programs[0]['pipeline_schedule'])
    scope.update(consumer_warps=4,producer_warps=1,producer_registers=24)
    context=dict(symbol='scoped',signature='value, weight',signature_arguments=['value','weight'],
        typed_signature_arguments=['value','weight'],pipeline_schedule=scope,device_functions=[],
        distributed_entry=False,mesh_hierarchy=[1],scalar_arguments=[],
        scoped_entry_arguments=[dict(name='value',stride=8192,dtype='tl.bfloat16')])
    source=TritonTemplateRegistry().render('entrypoints/call_graph.py.jinja',context)
    functions={node.name:node for node in ast.parse(source).body if isinstance(node,ast.FunctionDef)}
    def frame_calls(node):
        return [call for call in ast.walk(node) if isinstance(call,ast.Call)
                and isinstance(call.func,ast.Name) and call.func.id=='_flagmega_block_scope_base']
    entry=functions['scoped']
    assert len(frame_calls(entry))==1
    assignment=next(node for node in entry.body if isinstance(node,ast.Assign)
        and any(isinstance(target,ast.Name) and target.id=='value' for target in node.targets))
    specialization=next(node for node in entry.body if isinstance(node,ast.Expr)
        and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute)
        and node.value.func.attr=='warp_specialize')
    assert entry.body.index(assignment)<entry.body.index(specialization)
    for task in specialization.value.args[0].elts:
        assert 'value' in [value.id for value in task.elts[1].elts if isinstance(value,ast.Name)]
    assert not frame_calls(functions['scoped__producer'])
    assert not frame_calls(functions['scoped__consumer'])
