# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Typed owner admission records only real execution and preserves outer sync."""
import ast
from copy import deepcopy
from math import prod
from types import SimpleNamespace

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.kernel_call_renderers import prepare_kernel_calls
from triton.flagmega.codegen.triton.tir_package import _kernel_source_event, _mesh_context
from triton.flagmega.codegen.triton.trace_config import TracePlan, digest
from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry
from triton.flagmega.errors import CodegenError
from python.test.flagmega.codegen.triton.distribution.test_norm_stats_execution_owners import _raw, _binding

def fixture(mode='ring', exclusive=False, hierarchy=(8, 16), levels='bb'):
    raw = _raw(hierarchy=hierarchy, levels=levels)
    if exclusive:
        for section in ('inputs', 'outputs'):
            _binding(raw, section)['distributed_type']['exclusive'] = {
                'axes': list(range(len(hierarchy))), 'owner_coordinates': [0] * len(hierarchy)}
    call, = prepare_kernel_calls([raw], function_name='main')
    event = _kernel_source_event(call, {}, {}, [], (), memory_dependencies=())
    module = SimpleNamespace(semantic_hash=digest(raw), entry='main',
        execution_function_map={'main': SimpleNamespace(body=fm.Sequential())})
    bindings = {'main': {'arguments': [], 'call_abi': {'kernel_calls': [raw]}}}
    grid = [prod(extent for extent, level in zip(hierarchy, levels) if level == 'b'), 1, 1]
    config = ({'schema': 'flagmega.triton-trace/v1', 'base_semantic_hash': module.semantic_hash,
        'layer_bindings': {'main': {'constant': -1}}, 'detailed_calls': [],
        'typed_ctas': list(range(grid[0])), 'capacity': 4096, 'coarse_calls': True} if mode == 'ring' else
        {'schema': 'flagmega.triton-census/v1', 'base_semantic_hash': module.semantic_hash,
        'layer_bindings': {'main': {'constant': -1}}, 'service_calls': [],
        'expected_occurrences': 2, 'ctas': grid[0]})
    barrier = {'kind': 'barrier', 'scope': 'grid', 'axis_group_axes': [],
               'after': ['stats'], 'before': 'reader', 'ranges': []}
    descriptor = {**_mesh_context(_binding(raw, 'outputs')['distributed_type']['placement']),
        'platform': 'nvidia', 'target_architecture': 'sm90', 'grid': grid,
        'distributed_entry': True, 'pipeline_schedule': {
        'consumer_events': [deepcopy(barrier), event, deepcopy(barrier)],
        'producer_events': [], 'auxiliary_events': [], 'stages': []},
        'device_functions': [], 'render_calls': [call]}
    return module, bindings, call, raw, event, descriptor, config

@pytest.mark.parametrize('mode', ['ring', 'census'])
@pytest.mark.parametrize('exclusive', [False, True])
def test_actual_prepare_projection_owner_is_admitted(mode, exclusive):
    module, bindings, call, raw, event, descriptor, config = fixture(mode, exclusive)
    assert call['participation_active'] == event['participation_active'] != 'True'
    result = TracePlan(module, bindings, {'main': [call]}, config).finish(descriptor)
    dictionary = result['triton_trace']['dictionary']
    domain, = dictionary['execution_domains']['entries']
    assert dictionary['execution_domains']['schema'] == 'flagmega.trace-execution-domains/v1'
    assert domain['ctas'] == [0] and domain['function'] == 'main' and domain['call'] == raw['call']
    assert domain['site_key_sha256'] == digest(domain['site_key'])
    assert domain['proof']['helper_predicate'] != 'True'
    assert domain['proof']['prepared_predicate'] == domain['proof']['projected_predicate']
    if mode == 'ring':
        assert domain['record_mode'] == 'caller_pair' and domain['site_id'] == event['trace_site']
    else:
        assert domain['record_mode'] == 'admission_only' and domain['site_id'] is None
        assert len(dictionary['occurrences']) == 2
        assert all(row['kind'] == 'barrier' for row in dictionary['occurrences'])
        assert 'trace_site' not in event

@pytest.mark.parametrize('hierarchy', [(3,), (2, 3), (2, 2, 3)])
def test_grid_domain_is_derived_without_hardcoded_geometry(hierarchy):
    module, bindings, call, _, _, descriptor, config = fixture(hierarchy=hierarchy, levels='b' * len(hierarchy))
    result = TracePlan(module, bindings, {'main': [call]}, config).finish(descriptor)
    domain, = result['triton_trace']['dictionary']['execution_domains']['entries']
    assert domain['ctas'] == [0]
    assert domain['grid_mapping']['grid'] == [prod(hierarchy), 1, 1]

def test_caller_markers_and_callee_share_guard_outer_barriers_unchanged():
    module, bindings, call, raw, event, descriptor, config = fixture()
    TracePlan(module, bindings, {'main': [call]}, config).finish(descriptor)
    context = {**descriptor, 'pipeline_schedule': None, 'entry_events': descriptor['pipeline_schedule']['consumer_events'],
               'symbol': 'run', 'signature': call['signature']}
    source = TritonTemplateRegistry().render('entrypoints/call_graph.py.jinja', context)
    tree = ast.parse(source)
    entry, = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    guard, = [node for node in entry.body if isinstance(node, ast.If)]
    assert len(guard.body) == 3
    names = [ast.unparse(node.value.func) for node in guard.body]
    assert names == ['tle.gpu.trace_event', call['symbol'], 'tle.gpu.trace_event']
    assert [guard.body[index].value.args[0].value for index in (0, 2)] == [272, 273]
    barriers = [node for node in entry.body if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and ast.unparse(node.value.func) == 'tle.distributed_barrier']
    from pathlib import Path
    from triton.flagmega.codegen.triton import templates
    before = TritonTemplateRegistry(root=Path(templates.__file__).parent).render('entrypoints/call_graph.py.jinja', context)
    old_entry, = [node for node in ast.parse(before).body if isinstance(node, ast.FunctionDef)]
    old_barriers = [node for node in old_entry.body if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                    and ast.unparse(node.value.func) == 'tle.distributed_barrier']
    assert len(barriers) == 2 and barriers[0].lineno < guard.lineno < barriers[-1].lineno
    assert [ast.dump(node) for node in barriers] == [ast.dump(node) for node in old_barriers]
    assert ast.dump(guard.body[1].value.args[0]) == ast.dump(ast.parse(call['arguments'][0], mode='eval').body)

@pytest.mark.parametrize('mode', ['ring', 'census'])
@pytest.mark.parametrize('mutation', ['predicate', 'symbol', 'arguments', 'writer', 'call',
    'forged_domain', 'dynamic_grid', 'grid_mapping', 'dynamic_chip', 'empty_E', 'private', 'partial', 'scratch',
    'prepared_math', 'prepared_arguments'])
def test_unproven_or_forged_owner_continues_reject(mode, mutation):
    module, bindings, call, raw, event, descriptor, config = fixture(mode)
    if mutation == 'predicate': event['participation_active'] = '(shard_x == 1)'
    elif mutation == 'symbol': event['symbol'] = 'unrelated'
    elif mutation == 'arguments': event['arguments'] = 'unrelated_pointer'
    elif mutation == 'writer': call['writer_active'] = 'True'
    elif mutation == 'call': event['call'] = 'forged'
    elif mutation == 'forged_domain': event['trace_execution_domain'] = {'ctas': [0]}
    elif mutation == 'dynamic_grid': descriptor['dynamic_execution'] = True
    elif mutation == 'grid_mapping': descriptor['mesh_hierarchy'] = [16, 8]
    elif mutation == 'dynamic_chip':
        module, bindings, call, raw, event, descriptor, config = fixture(mode, hierarchy=(2, 8), levels='cb')
    elif mutation == 'empty_E':
        _binding(raw, 'inputs')['distributed_type']['exclusive'] = {'axes': [0], 'owner_coordinates': [1]}
    elif mutation == 'private': _binding(raw, 'inputs')['memory_sharing_scope'] = 'block'
    elif mutation == 'partial': _binding(raw, 'inputs')['distributed_type']['partial'] = {'kind': 'sum'}
    elif mutation == 'scratch': raw['workspaces'] = [{'name': 'operation_scratch'}]
    elif mutation == 'prepared_math': call['reduction_block_size'] += 1
    elif mutation == 'prepared_arguments':
        call['signature'] = event['arguments'] = 'forged_pointer'
    with pytest.raises(CodegenError):
        TracePlan(module, bindings, {'main': [call]}, config).finish(descriptor)

def test_uniform_and_tracing_absent_template_bytes_are_unchanged():
    from pathlib import Path
    from triton.flagmega.codegen.triton import templates
    _, _, call, _, event, descriptor, _ = fixture()
    context = {**descriptor, 'pipeline_schedule': None, 'entry_events': [deepcopy(event)],
               'symbol': 'run', 'signature': call['signature']}
    current, baseline = TritonTemplateRegistry(), TritonTemplateRegistry(root=Path(templates.__file__).parent)
    assert current.render('entrypoints/call_graph.py.jinja', context) == baseline.render('entrypoints/call_graph.py.jinja', context)
    context['entry_events'][0].update(participation_active='True', trace_site=1, trace_layer='0')
    assert current.render('entrypoints/call_graph.py.jinja', context) == baseline.render('entrypoints/call_graph.py.jinja', context)

@pytest.mark.parametrize('mode', ['ring', 'census'])
def test_unit_extent_chip_context_is_proved_from_mapping(mode):
    module, bindings, call, _, _, descriptor, config = fixture(mode, hierarchy=(1, 2, 3), levels='cbb')
    result = TracePlan(module, bindings, {'main': [call]}, config).finish(descriptor)
    domain, = result['triton_trace']['dictionary']['execution_domains']['entries']
    assert domain['ctas'] == [0]
    assert domain['invocation_context']['chip_node_coordinates'] == [{'axis': 0, 'level': 'c', 'value': 0}]

def test_domain_precedes_unchanged_recorder_selection_and_replay_is_deterministic():
    def run():
        module, bindings, call, _, _, descriptor, config = fixture()
        config['typed_ctas'] = [1]
        return TracePlan(module, bindings, {'main': [call]}, config).finish(descriptor)
    first, second = run(), run()
    domain, = first['triton_trace']['dictionary']['execution_domains']['entries']
    assert domain['ctas'] == [0] and first['triton_trace']['native_contract']['typed_ctas'] == [1]
    assert first['triton_trace']['dictionary_sha256'] == second['triton_trace']['dictionary_sha256']

def nested_fixture(monkeypatch, mode):
    from triton.flagmega.codegen.triton import function_programs
    # Constant lookup is a separately tested owning dependency. Real full-model
    # typed constants are also exercised by the task's nonpatched CPU probe.
    monkeypatch.setattr(function_programs, '_constant', lambda module, key: {'zero': 0}[key])
    module, bindings, _, raw, _, descriptor, config = fixture(mode)
    call, = prepare_kernel_calls([raw], function_name='decoder')
    event = _kernel_source_event(call, {}, {}, [], (), memory_dependencies=())
    child = descriptor['pipeline_schedule']
    child['consumer_events'][1] = event
    invoke = {'kind': 'function_call', 'call': 'invoke0', 'callee': 'decoder',
              'argument_expressions': ('tl.full((), 0, tl.int32)',), 'participation_active': 'True'}
    descriptor['pipeline_schedule'] = {'consumer_events': [deepcopy(invoke)],
                                      'producer_events': [deepcopy(invoke)], 'stages': []}
    descriptor['device_functions'] = [{'function': 'decoder', 'signature_arguments': ['layer'], 'schedule': child}]
    config['layer_bindings']['decoder'] = {'formal': 'layer_id'}
    if mode == 'census': config['layer_bindings']['decoder']['values'] = [0]
    typed_call = fm.PrimFunctionCall('invoke0', 'decoder', arguments=(fm.PrimCallBinding('layer_id', 'zero'),))
    module.execution_function_map = {'main': SimpleNamespace(body=fm.Sequential((typed_call,))),
                                    'decoder': SimpleNamespace(body=fm.Sequential())}
    bindings['decoder'] = {'arguments': [{'buffer': 'layer_id', 'name': 'layer',
                                         'runtime_value_kind': 'scalar', 'scalar_dtype': 'int32'}],
                           'call_abi': {'kernel_calls': [raw]}}
    return module, bindings, {'decoder': [call]}, descriptor, config

@pytest.mark.parametrize('mode', ['ring', 'census'])
def test_reusable_owner_binds_actual_root_formal_and_static_projection(monkeypatch, mode):
    module, bindings, calls, descriptor, config = nested_fixture(monkeypatch, mode)
    result = TracePlan(module, bindings, calls, config).finish(descriptor)
    domain, = result['triton_trace']['dictionary']['execution_domains']['entries']
    invocation, = domain['invocation_context']['invocations']
    assert invocation['call'] == 'invoke0' and invocation['layer'] == 0
    assert invocation['formal_bindings'] == [{'formal': 'layer_id', 'actual': 'zero', 'value': 0, 'dtype': 'int32'}]
    assert invocation['source_roles'] == ['consumer', 'producer']

@pytest.mark.parametrize('mode', ['ring', 'census'])
@pytest.mark.parametrize('mutation', ['changed_projection', 'missing_invocation', 'nonuniform_invocation',
                                    'dynamic_value', 'float_value', 'missing_formal'])
def test_reusable_unbound_dynamic_or_changed_context_rejects(monkeypatch, mode, mutation):
    from triton.flagmega.codegen.triton import function_programs
    module, bindings, calls, descriptor, config = nested_fixture(monkeypatch, mode)
    events = descriptor['pipeline_schedule']['consumer_events']
    if mutation == 'changed_projection': events[0]['argument_expressions'] = ('tl.full((), 1, tl.int32)',)
    elif mutation == 'missing_invocation': events.clear()
    elif mutation == 'nonuniform_invocation': events[0]['participation_active'] = 'dynamic'
    elif mutation == 'dynamic_value': monkeypatch.setattr(function_programs, '_constant', lambda *args: 'dynamic')
    elif mutation == 'float_value': monkeypatch.setattr(function_programs, '_constant', lambda *args: 0.0)
    elif mutation == 'missing_formal': descriptor['device_functions'][0]['signature_arguments'] = []
    with pytest.raises(CodegenError): TracePlan(module, bindings, calls, config).finish(descriptor)

@pytest.mark.parametrize('mode', ['ring', 'census'])
@pytest.mark.parametrize('value', [True, False, 1, 0.0])
def test_static_bool_state_actual_is_typed_and_not_an_integer_alias(monkeypatch, mode, value):
    from triton.flagmega.codegen.triton import function_programs
    module, bindings, calls, descriptor, config = nested_fixture(monkeypatch, mode)
    bindings['decoder']['arguments'].append({'buffer': 'advance', 'name': 'advance',
                                           'runtime_value_kind': 'scalar', 'scalar_dtype': 'bool'})
    typed_call = fm.PrimFunctionCall('invoke0', 'decoder', arguments=(
        fm.PrimCallBinding('layer_id', 'zero'), fm.PrimCallBinding('advance', 'advance_value')))
    module.execution_function_map['main'].body = fm.Sequential((typed_call,))
    descriptor['device_functions'][0]['signature_arguments'].append('advance')
    monkeypatch.setattr(function_programs, '_constant', lambda module, key: 0 if key == 'zero' else value)
    for role in ('consumer', 'producer'):
        descriptor['pipeline_schedule'][role + '_events'][0]['argument_expressions'] = (
            'tl.full((), 0, tl.int32)', f'tl.full((), {value}, tl.int1)')
    if type(value) is not bool:
        with pytest.raises(CodegenError): TracePlan(module, bindings, calls, config).finish(descriptor)
    else:
        result = TracePlan(module, bindings, calls, config).finish(descriptor)
        domain, = result['triton_trace']['dictionary']['execution_domains']['entries']
        bound = domain['invocation_context']['invocations'][0]['formal_bindings'][-1]
        assert bound['dtype'] == 'bool' and bound['value'] is value

@pytest.mark.parametrize('mode', ['ring', 'census'])
def test_normal_call_ordinal_and_callee_identity_are_reprepared_in_context(mode):
    module, bindings, _, raw, _, descriptor, config = fixture(mode)
    first = deepcopy(raw)
    first['call'] = 'preceding'
    bindings['main']['call_abi']['kernel_calls'] = [first, raw]
    calls = prepare_kernel_calls([first, raw], function_name='main')
    descriptor['pipeline_schedule']['consumer_events'][1] = _kernel_source_event(calls[1], {}, {}, [], (), memory_dependencies=())
    result = TracePlan(module, bindings, {'main': calls}, config).finish(descriptor)
    domain, = result['triton_trace']['dictionary']['execution_domains']['entries']
    assert domain['call'] == raw['call'] and domain['ctas'] == [0]

@pytest.mark.parametrize('mode', ['ring', 'census'])
def test_exclusive_domains_are_normalized_and_v1_entry_keys_exact(mode):
    module, bindings, _, raw, _, descriptor, config = fixture(mode)
    for section in ('inputs', 'outputs'):
        _binding(raw, section)['distributed_type']['exclusive'] = {'axes': [1, 0], 'owner_coordinates': [0, 0]}
    call, = prepare_kernel_calls([raw], function_name='main')
    descriptor['pipeline_schedule']['consumer_events'][1] = _kernel_source_event(call, {}, {}, [], (), memory_dependencies=())
    result = TracePlan(module, bindings, {'main': [call]}, config).finish(descriptor)
    extension = result['triton_trace']['dictionary']['execution_domains']
    assert set(extension) == {'schema', 'entries'}
    domain, = extension['entries']
    assert set(domain) == {'function', 'call', 'role', 'site_key', 'site_key_sha256', 'site_id',
        'record_mode', 'ctas', 'placement', 'grid_mapping', 'exclusive', 'invocation_context', 'proof'}
    assert all(value == {'axes': [0, 1], 'owner_coordinates': [0, 0]} for value in domain['exclusive'].values())

def test_nonuniform_renderer_markers_without_planner_proof_reject():
    _, _, call, _, event, descriptor, _ = fixture()
    event.update(trace_site=1, trace_layer='0')
    with pytest.raises(CodegenError, match='checked execution domain'):
        TritonTemplateRegistry().render('entrypoints/call_graph.py.jinja', {
            **descriptor, 'pipeline_schedule': None, 'entry_events': [event],
            'symbol': 'run', 'signature': call['signature']})
