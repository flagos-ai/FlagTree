# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Sparse publication diagnostics use the normal schedule and exact slot ABI."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
from triton.flagmega.codegen.triton.trace_config import TracePlan, validate_option
from triton.flagmega.errors import CodegenError


def config():
    return {
        'schema': 'flagmega.triton-census/v1', 'base_semantic_hash': 'a' * 64, 'layer_bindings':
        {'main': {'constant': -1}, 'decoder': {'formal': 'layer_value', 'values':
                                               [0, 1]}}, 'service_calls': [], 'expected_occurrences': 6, 'ctas': 128
    }


def test_census_configuration_is_a_separate_explicit_normal_option():
    validate_option(config(), target='nvidia-sm90')


@pytest.mark.parametrize('change,message', [({'ctas': False}, 'CTA'), ({'expected_occurrences': 0}, 'occurrence'),
                                            ({'service_calls': 'all'}, 'service'),
                                            ({'base_semantic_hash': 'x'}, 'semantic')])
def test_census_rejects_untyped_configuration(change, message):
    options = config()
    options.update(change)
    with pytest.raises(CodegenError, match=message):
        validate_option(options, target='nvidia-sm90')


def nested_plan(monkeypatch, *, values=(0, 1), dtype='int32', services=False):
    from triton.flagmega import ir as fm
    from triton.flagmega.codegen.triton import function_programs
    # Constant lookup is already owned/tested by function_programs; these typed
    # edges exercise census call-domain proof and schedule projection separately.
    monkeypatch.setattr(function_programs, '_constant', lambda module, key: {'zero': 0, 'one': 1}[key])
    calls = tuple(
        fm.PrimFunctionCall(f'call{i}', 'decoder', arguments=(fm.PrimCallBinding('layer_value', ('zero',
                                                                                                 'one')[value]), ))
        for i, value in enumerate(values))
    module = SimpleNamespace(
        semantic_hash='a' * 64, entry='main', execution_function_map={
            'main': SimpleNamespace(body=fm.Sequential(calls)), 'decoder': SimpleNamespace(body=fm.Sequential())
        })
    bindings = {
        'main': {'arguments': []}, 'decoder': {
            'arguments':
            [{'buffer': 'layer_value', 'name': 'actual_layer', 'runtime_value_kind': 'scalar', 'scalar_dtype': dtype}]
        }
    }
    options = config()
    encoded = {}
    if services:
        options['expected_occurrences'] += 4
        options['service_calls'] = [{'function': 'decoder', 'call': 'glu'}]
        encoded = {
            'decoder': [{
                'call': 'glu', 'symbol': 'glu', 'family': 'dense_matmul_glu', 'variant':
                'packed_tensor_descriptor_smem_pipeline_gemv', 'paired_weight_fields': True, 'transfer_pipeline': True,
                'pipeline_channels': [{}], 'arguments': ['data', 'descriptor'], 'descriptor_parameters': ['descriptor']
            }]
        }
    return TracePlan(module, bindings, encoded, options)


def descriptor(*, services=False):

    def barrier():
        return {
            'kind': 'barrier', 'scope': 'grid', 'axis_group_axes': [], 'after': ['writer'], 'before': 'reader',
            'ranges': []
        }

    root = {
        role + '_events': [
            barrier(), *[{
                'kind': 'function_call', 'call': f'call{i}', 'callee': 'decoder', 'argument_expressions':
                ('endpoint', f'tl.full((), {i}, tl.int32)')
            }
                         for i in range(2)]
        ]
        for role in ('consumer', 'producer')
    }
    child = {role + '_events': [barrier()] for role in ('consumer', 'producer')}
    calls = []
    if services:
        for role in ('consumer', 'producer'):
            endpoint = 'reader' if role == 'consumer' else 'writer'
            child[role + '_events'] += [{
                'kind': 'pipeline_kernel_call', 'call': 'glu', 'census_endpoint': endpoint,
                'pipeline_commits_per_invocation': 6
            }, {'kind': 'drain', 'endpoints': [endpoint]}]
        child['stages'] = [{
            'channels': [{
                'reader_name': 'reader', 'writer_name': 'writer', 'capacity': 2, 'fields': [{'name': 'gate'},
                                                                                            {'name': 'up'}]
            }]
        }]
        calls = [{'call': 'glu', 'symbol': 'glu'}]
    return {
        'platform': 'nvidia', 'target_architecture': 'sm90', 'grid': [128, 1,
                                                                      1], 'pipeline_schedule': root, 'render_calls':
        calls, 'device_functions': [{'function': 'decoder', 'signature_arguments': ['actual_layer'], 'schedule': child}]
    }


def test_typed_domain_and_unique_fixed_slots(monkeypatch):
    result = nested_plan(monkeypatch, services=True).finish(descriptor(services=True))
    metadata = result['triton_trace']
    rows = metadata['dictionary']['occurrences']
    assert [r['occurrence'] for r in rows] == list(range(10))
    written = [(r['occurrence'] * 128 + cta) * 4 + point for r in rows for cta in range(128) for point in r['points']]
    assert len(written) == len(set(written)) == 128 * (6 * 3 + 4 * 4)
    unused = {(r['occurrence'] * 128 + cta) * 4 + 3 for r in rows if r['kind'] == 'barrier' for cta in range(128)}
    assert not unused.intersection(written)
    context = result['render_calls'][0]['census']
    assert context == {'layer_parameter': '_flagmega_census_layer_id', 'consumer_base': 6, 'producer_base': 8}
    events = result['device_functions'][0]['schedule']['consumer_events']
    assert events[-1]['kind'] == 'drain' and events[-1][
        'census_occurrence'] == '(actual_layer + tl.full((), 6, tl.int32))'
    assert metadata['native_contract']['logical_bytes'] == 64 + 10 * 128 * 4 * 8 + 2048


@pytest.mark.parametrize('values', [(0, 0), (1, ), (0, 1, 1)])
def test_domain_is_proved_from_typed_actuals_not_claimed_by_config(monkeypatch, values):
    with pytest.raises(CodegenError, match='exactly once'):
        nested_plan(monkeypatch, values=values)


def test_no_implicit_layer_narrowing(monkeypatch):
    with pytest.raises(CodegenError, match='scalar int32'):
        nested_plan(monkeypatch, dtype='int64')


@pytest.mark.parametrize('mutation,message', [
    ('wrong_actual', 'ABI projection'),
    ('duplicate_call', 'each typed invocation'),
    ('loop', 'structured-loop'),
    ('predicate', 'uniform'),
    ('grid', 'CTA grid'),
    ('unrelated_drain', 'normal drain'),
    ('delay_drain', 'normal drain'),
    ('epoch', 'fixed epochs'),
    ('capacity', 'two-slot'),
    ('duplicate_definition', 'outlined'),
    ('count', 'occurrence count'),
])
def test_unproved_occurrence_or_service_contract_is_rejected(monkeypatch, mutation, message):
    plan = nested_plan(monkeypatch, services=True)
    data = descriptor(services=True)
    events = data['pipeline_schedule']['consumer_events']
    child = data['device_functions'][0]['schedule']
    if mutation == 'wrong_actual': events[1]['argument_expressions'] = ('endpoint', 'tl.full((), 1, tl.int32)')
    elif mutation == 'duplicate_call': events.append(deepcopy(events[1]))
    elif mutation == 'loop': events.append({'kind': 'loop'})
    elif mutation == 'predicate': events[1]['participation_active'] = 'selected'
    elif mutation == 'grid': data['grid'] = [64, 1, 1]
    elif mutation == 'unrelated_drain': child['consumer_events'][-1]['endpoints'] = ['unrelated']
    elif mutation == 'delay_drain': child['consumer_events'].insert(2, {'kind': 'barrier', 'scope': 'block'})
    elif mutation == 'epoch': child['consumer_events'][1]['pipeline_commits_per_invocation'] = 5
    elif mutation == 'capacity': child['stages'][0]['channels'][0]['capacity'] = 3
    elif mutation == 'duplicate_definition': data['device_functions'] *= 2
    elif mutation == 'count': plan.config['expected_occurrences'] = 11
    with pytest.raises(CodegenError, match=message):
        plan.finish(data)


def test_empty_actual_pipeline_census_is_rejected(packed_glu_paired_full_lhs_descriptor_pipeline_module, tmp_path):
    from triton.flagmega.codegen.triton import describe_tir_package, render_triton_package
    module = packed_glu_paired_full_lhs_descriptor_pipeline_module
    base = describe_tir_package(module)
    assert not any(e['kind'] == 'barrier' and e.get('scope') == 'grid'
                   for role in ('consumer', 'producer')
                   for e in base['pipeline_schedule'][role + '_events'])
    options = {
        'schema': 'flagmega.triton-census/v1', 'base_semantic_hash': module.semantic_hash, 'layer_bindings':
        {'main': {'constant':
                  -1}}, 'service_calls': [], 'expected_occurrences': 1, 'ctas': __import__('math').prod(base['grid'])
    }
    with pytest.raises(CodegenError, match='occurrence count'):
        render_triton_package(module, tmp_path / 'census', triton_trace=options)


def test_generic_scratch_metadata_uses_native_64_byte_alignment(monkeypatch):
    plan = nested_plan(monkeypatch)
    plan.config['expected_occurrences'] = 4
    data = descriptor()
    for role in ('consumer', 'producer'):
        data['pipeline_schedule'][role + '_events'].pop(0)
    contract = plan.finish(data)['triton_trace']['native_contract']
    assert contract['logical_bytes'] == 64 + 4 * 128 * 4 * 8 + 2048
    assert contract['scratch_alignment_bytes'] == 64
    assert contract['scratch_bytes_per_cta'] == 192  # 128B rounding would wrongly yield256.
    assert contract['allocated_bytes'] == 24576
