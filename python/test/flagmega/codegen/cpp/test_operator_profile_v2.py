# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Versioned flat schedule profiling, trace geometry, and lifetime validation."""
import copy
import struct

import pytest

from triton.flagmega.codegen.cpp.operator_profile import configure_descriptor, validate_option
from triton.flagmega.runtime.operator_profile import ProfileSession, digest


def call(name, family='elementwise'):
    return {'call': name, 'symbol': 'kernel_' + name, 'family': family,
            'variant': 'local', 'execution_kind': 'ordinary',
            'inputs': [{'formal': 'value', 'buffers': [
                {'actual': 'buffer_' + name, 'formal': 'value',
                 'abi': {'schema': 'flagmega.local-buffer-abi/v2', 'buffer': 'buffer_' + name,
                         'storage': 'global'},
                 'runtime_value_kind': 'pointer', 'runtime_argument': 'argument_' + name}]}]}


def leaf(value):
    return {**{key: value[key] for key in ('call', 'symbol', 'family', 'variant', 'execution_kind')},
            'kind': 'kernel_call', 'participation_active': 'True', 'barrier_before': False}


def invoke(name, ordinal):
    return {'kind': 'function_call', 'call': f'call_{name}_{ordinal}', 'callee': name,
            'symbol': name + '_impl__consumer', 'participation_active': 'True', 'barrier_before': False}


def descriptor(grid=32, counts=(3, 2)):
    a, b, c = call('entry_leaf'), call('first_leaf'), call('second_leaf')
    functions = [
        {'function': 'first', 'symbol': 'first_impl',
         'schedule': {'consumer_events': [leaf(b), {'kind': 'barrier', 'scope': 'grid'}]}},
        {'function': 'second', 'symbol': 'second_impl',
         'schedule': {'consumer_events': [{'kind': 'barrier', 'scope': 'cta'}, leaf(c)]}},
    ]
    main = [leaf(a)]
    for name, count in zip(('first', 'second'), counts):
        main.extend(invoke(name, i) for i in range(count))
    return {'platform': 'nvidia', 'target_architecture': 'sm120', 'grid': [grid, 1, 1],
            'grid_total_blocks': grid, 'block_threads': 128, 'warp_size': 32,
            'distributed_entry': True, 'symbol': 'entry_impl',
            'entry_events': main, 'device_functions': functions, 'render_calls': [a, b, c]}


def profile(value=None):
    return configure_descriptor(descriptor() if value is None else value, 'cta-globaltimer-v2')['operator_profile']


def rehash(contract):
    for name in ('site_inventory', 'sync_site_inventory', 'invocation_manifest', 'sync_manifest'):
        contract[name + '_sha256'] = digest(contract[name])
    return contract


def trace(contract, epoch=1):
    return b''.join(struct.pack('<QQQQ', event * 10 + cta, event * 10 + cta + 5, epoch, event)
                    for event in range(contract['event_count']) for cta in range(contract['grid_total']))


@pytest.mark.parametrize('grid,counts', [(1, (1, 1)), (32, (3, 2)), (96, (7, 3))])
def test_geometry_and_multiplicity_come_from_static_schedule(grid, counts):
    original = descriptor(grid, counts)
    before = copy.deepcopy(original)
    output = configure_descriptor(original, 'cta-globaltimer-v2')
    p = output['operator_profile']
    assert original == before
    assert p['event_count'] == 1 + 2 * sum(counts)
    assert p['grid_total'] == grid
    assert p['extra_trace_shape_u64'] == [p['event_count'], grid, 4]
    assert p['trace_nbytes'] == p['event_count'] * grid * 32
    assert p['container_multiplicities'] == dict(zip(('first', 'second'), counts))
    assert [row['expected_invocations_per_token'] for row in p['site_inventory']] == [1, *counts]
    assert len(p['invocation_manifest']) == 1 + sum(counts)
    assert len(p['sync_manifest']) == sum(counts)
    assert p['extra_CTA_barriers_per_CTA_per_token'] == 2 * (1 + sum(counts))
    session = ProfileSession(p, grid=original['grid'], block_threads=128)
    session.reserve(123, pointer=4096, capacity=p['trace_nbytes'])
    assert session.complete(trace(p)) == trace(p)


def test_entry_only_schedule_and_implicit_barrier():
    d = descriptor()
    d['device_functions'] = []
    d['render_calls'] = d['render_calls'][:1]
    d['entry_events'] = d['entry_events'][:1]
    d['entry_events'][0].update(barrier_before=True, barrier_scope='grid')
    result = configure_descriptor(d, 'cta-globaltimer-v2')
    p = result['operator_profile']
    assert p['event_count'] == 2
    assert p['sync_manifest'][0]['event_index'] == 0
    assert p['invocation_manifest'][0]['event_index'] == 1
    assert p['container_multiplicities'] == {}
    ProfileSession(p, grid=[32, 1, 1])


@pytest.mark.parametrize('mutation', [
    lambda d: d.update(grid=[32, 2, 1]),
    lambda d: d.update(grid=[0, 1, 1]),
    lambda d: d.update(grid=[True, 1, 1]),
    lambda d: d.update(grid_total_blocks=31),
    lambda d: d.update(block_threads=33),
    lambda d: d.update(warp_size=64),
    lambda d: d.update(distributed_entry=False),
    lambda d: d.update(programs=[{}]),
    lambda d: d.update(pipeline_schedule={'roles': ['producer']}),
    lambda d: d['device_functions'][0]['schedule'].update(producer_events=[{}]),
    lambda d: d['device_functions'][0]['schedule'].update(auxiliary_events=[{}]),
    lambda d: d['device_functions'][0]['schedule']['consumer_events'].append(invoke('first', 0)),
    lambda d: d['device_functions'][0]['schedule']['consumer_events'].append(invoke('second', 0)),
    lambda d: d['entry_events'].append({'kind': 'control.for'}),
    lambda d: d['entry_events'][0].update(participation_active='shard_x == 0'),
    lambda d: d['entry_events'][1].update(callee='unbound'),
    lambda d: d['entry_events'][1].update(symbol='wrong_symbol'),
    lambda d: d['entry_events'][0].update(symbol='wrong_kernel'),
    lambda d: d['render_calls'].append(call('unexecuted')),
    lambda d: d['device_functions'].append(copy.deepcopy(d['device_functions'][0])),
    lambda d: d['device_functions'][0]['schedule']['consumer_events'][1].update(scope='unknown'),
])
def test_unsupported_or_inconsistent_schedules_fail_closed(mutation):
    d = descriptor()
    mutation(d)
    with pytest.raises(ValueError):
        configure_descriptor(d, 'cta-globaltimer-v2')


@pytest.mark.parametrize('mutation', [
    lambda c: c.update(event_count=0),
    lambda c: c.update(event_count=True),
    lambda c: c.update(event_count=2**63),
    lambda c: c.update(grid_total=0),
    lambda c: c.update(launch_grid=[32, 2, 1]),
    lambda c: c.update(block_threads=33),
    lambda c: c.update(trace_nbytes=c['trace_nbytes'] - 8),
    lambda c: c.update(extra_trace_shape_u64=[c['event_count'], 31, 4]),
    lambda c: c.update(scalar_strides=[128, 4, True]),
    lambda c: c.update(scalar_strides=[4, 128, 1]),
    lambda c: c.update(trace_alignment_bytes=4),
    lambda c: c.update(schema='flagmega.cpp-operator-profile/v1'),
    lambda c: c.update(accepted_timing_eligible=True),
    lambda c: c['invocation_manifest'][0].update(expected_tag='1'),
    lambda c: c['invocation_manifest'][0].update(event_index=2),
    lambda c: c['invocation_manifest'][0].update(site_id='unknown'),
    lambda c: c['site_inventory'][0].update(expected_invocations_per_token=2),
    lambda c: c.update(extra_CTA_barriers_per_CTA_per_token=0),
])
def test_trace_contract_requires_consistent_geometry_and_complete_census(mutation):
    contract = profile()
    mutation(contract)
    rehash(contract)
    with pytest.raises(ValueError):
        ProfileSession(contract)


def test_trace_contract_binds_manifest_and_is_frozen_on_prepare():
    contract = profile()
    with pytest.raises(ValueError, match='manifest'):
        ProfileSession(contract, grid=[96, 1, 1])
    with pytest.raises(ValueError, match='manifest'):
        ProfileSession(contract, block_threads=256)
    session = ProfileSession(contract, grid=[32, 1, 1], block_threads=128)
    contract['grid_total'] = 1
    assert session.contract['grid_total'] == 32


@pytest.mark.parametrize('field,value', [(0, 7), (1, 0), (2, 0), (3, 3)])
def test_actual_trace_cells_reject_bad_timestamps_epoch_or_tag(field, value):
    c = profile()
    session = ProfileSession(c)
    session.reserve(1, pointer=4096, capacity=c['trace_nbytes'])
    raw = bytearray(trace(c))
    struct.pack_into('<Q', raw, 32 + field * 8, value)
    with pytest.raises(ValueError):
        session.complete(raw)
    assert not session.completed


def test_actual_trace_rejects_cross_event_time_reversal_and_wrong_size():
    c = profile()
    s = ProfileSession(c)
    s.reserve(1, pointer=4096, capacity=c['trace_nbytes'])
    with pytest.raises(ValueError, match='Incomplete'):
        s.complete(trace(c)[:-8])
    raw = bytearray(trace(c))
    struct.pack_into('<Q', raw, c['grid_total'] * 32, 1)
    with pytest.raises(ValueError, match='per-CTA'):
        s.complete(raw)


def test_eager_lifetime_contract_still_applies_to_v2():
    c = profile()
    s = ProfileSession(c)
    with pytest.raises(ValueError, match='eager'):
        s.reserve(1, pointer=4096, capacity=c['trace_nbytes'], capturing=True)
    with pytest.raises(ValueError, match='alignment'):
        s.reserve(1, pointer=4097, capacity=c['trace_nbytes'])
    s.reserve(1, pointer=4096, capacity=c['trace_nbytes'])
    with pytest.raises(ValueError, match='Pending'):
        s.reserve(1, pointer=4096, capacity=c['trace_nbytes'])


def legacy_descriptor():
    # V1's reviewed census: 6 entry + 14 linear + 24 attention operator sites,
    # 48 linear containers and16 attention containers with 13/18 sync sites.
    main_calls = [call(f'main_{i}') for i in range(6)]
    linear = [call(f'linear_{i}', 'gdn_recurrent' if i == 0 else 'elementwise') for i in range(14)]
    attention = [call(f'attention_{i}', 'paged_attention_partial' if i == 0 else 'elementwise') for i in range(24)]
    d = descriptor(96)
    d['render_calls'] = main_calls + linear + attention
    d['entry_events'] = [*[leaf(c) for c in main_calls],
                         *[{'kind': 'barrier', 'scope': 'grid'} for _ in range(4)],
                         *[invoke('first', i) for i in range(48)], *[invoke('second', i) for i in range(16)]]
    for f, calls, syncs in zip(d['device_functions'], (linear, attention), (13, 18)):
        f['schedule']['consumer_events'] = [*[leaf(c) for c in calls],
                                            *[{'kind': 'barrier', 'scope': 'grid'} for _ in range(syncs)]]
    return d


def test_v1_keeps_original_geometry_and_census_rejections():
    p = configure_descriptor(legacy_descriptor(), 'cta-globaltimer-v1')['operator_profile']
    assert p['extra_trace_shape_u64'] == [1978, 96, 4]
    assert p['trace_nbytes'] == 6076416
    assert len(p['site_inventory']) == 44
    assert len(p['invocation_manifest']) == 1062
    assert len(p['sync_manifest']) == 916
    s = ProfileSession(p)
    s.reserve(1, pointer=4096, capacity=p['trace_nbytes'])
    s.complete(trace(p))
    with pytest.raises(ValueError, match='96'):
        configure_descriptor(descriptor(32), 'cta-globaltimer-v1')
    d = legacy_descriptor()
    d['entry_events'].pop()
    with pytest.raises(ValueError, match='census'):
        configure_descriptor(d, 'cta-globaltimer-v1')


def test_option_is_versioned_and_nvidia_cpp_executable_only():
    for mode in ('cta-globaltimer-v1', 'cta-globaltimer-v2'):
        validate_option(mode, target='nvidia-sm120')
        for bad in ({'target': 'nvidia-sm90'}, {'codegen_backend': 'triton'}, {'emit_executable': False}):
            with pytest.raises(ValueError):
                validate_option(mode, **{'target': 'nvidia-sm120', **bad})
    with pytest.raises(ValueError):
        validate_option('cta-globaltimer-v3', target='nvidia-sm120')


def test_cli_accepts_versioned_diagnostic_mode():
    from triton.flagmega.cli import _parser
    arguments = _parser().parse_args(['compile', '--input', 'model.py', '--output', 'artifact',
                                     '--target', 'nvidia-sm120', '--codegen-backend', 'cpp',
                                     '--emit-executable', '--cpp-operator-profile', 'cta-globaltimer-v2'])
    assert arguments.cpp_operator_profile == 'cta-globaltimer-v2'


@pytest.mark.parametrize('grid', [1, 32, 96])
def test_emitted_trace_helpers_use_the_admitted_grid(grid):
    from triton.flagmega.codegen.cpp.templates import CppTemplateRegistry
    registry = CppTemplateRegistry()
    text = (registry.root / 'entrypoints/call_graph_profile.cpp.jinja').read_text()
    helpers = text.split('{% macro entry_events_body')[0]
    source = registry.environment.from_string(helpers).render(operator_profile={'grid_total': grid})
    assert f'event*{grid}ULL+ftt::arch::block_id()' in source
    assert source.count(f'event*{grid}ULL+') == 2


def test_loaded_identity_supports_normal_unlimited_register_build(tmp_path):
    import hashlib
    import json
    from triton.flagmega.runtime.operator_profile import loaded_identity
    files = {'generated.cpp': b'source', 'library.so': b'library', 'final.py': b'IR'}
    for name, data in files.items():
        (tmp_path / name).write_bytes(data)
    manifest = {'codegen': {'source': 'generated.cpp', 'library': 'library.so',
                            'source_sha256': hashlib.sha256(files['generated.cpp']).hexdigest(),
                            'library_sha256': hashlib.sha256(files['library.so']).hexdigest(),
                            'operator_profile': profile(), 'grid': [32, 1, 1], 'block_threads': 128},
                'final_ir': 'final.py', 'selections': {}, 'abi': {}, 'target': 'nvidia-sm120'}
    (tmp_path / 'artifact.json').write_text(json.dumps(manifest))
    result = loaded_identity(tmp_path, manifest, tmp_path / 'library.so')
    assert result['max_registers'] is None
    assert result['grid'] == [32, 1, 1]


@pytest.mark.parametrize('container', ['entry', 'consumer'])
def test_implicit_block_barrier_matches_explicit_normalization(container):
    d = descriptor()
    events = d['entry_events'] if container == 'entry' else d['device_functions'][0]['schedule']['consumer_events']
    events[0].update(barrier_before=True, barrier_scope='block')
    result = configure_descriptor(d, 'cta-globaltimer-v2')
    output = result['entry_events'] if container == 'entry' else result['device_functions'][0]['schedule']['consumer_events']
    assert output[0]['kind'] == 'barrier' and output[0]['scope'] == 'cta'
    assert output[1]['barrier_before'] is False
    ProfileSession(result['operator_profile'], grid=[32, 1, 1], block_threads=128)


@pytest.mark.parametrize('container', ['entry', 'consumer'])
@pytest.mark.parametrize('scope', ['device', 'unknown'])
def test_unknown_implicit_barrier_rejected_at_configuration(container, scope):
    d = descriptor()
    events = d['entry_events'] if container == 'entry' else d['device_functions'][0]['schedule']['consumer_events']
    events[0].update(barrier_before=True, barrier_scope=scope)
    with pytest.raises(ValueError, match='synchronization scope'):
        configure_descriptor(d, 'cta-globaltimer-v2')


def test_v1_implicit_block_uses_same_normalization_without_census_change():
    d = legacy_descriptor()
    del d['entry_events'][6]  # Replace one existing sync with an implicit one.
    d['entry_events'][0].update(barrier_before=True, barrier_scope='block')
    result = configure_descriptor(d, 'cta-globaltimer-v1')
    assert result['entry_events'][0]['scope'] == 'cta'
    assert result['operator_profile']['extra_trace_shape_u64'] == [1978, 96, 4]
    ProfileSession(result['operator_profile'])
