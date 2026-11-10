# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Variable-grid per-invocation diagnostics; no fixed-grid census reuse."""
from copy import deepcopy
from math import prod
from .program_abi import identity

MODE = 'cta-globaltimer-v3'
SCHEMA = 'flagmega.cpp-program-trace/v1'
RECORD_BYTES = 32


def configure_descriptor(original, mode):
    if mode != MODE or not original.get('cpp_programs'):
        raise ValueError('Native programs require cta-globaltimer-v3 diagnostics.')
    descriptor = deepcopy(original)
    manifest = descriptor['cpp_programs']
    # Event 0/1 are always entry/exit. Local sites are explicit event IDs >=2.
    entry_sites = {}
    for context in descriptor['native_program_contexts']:
        sites = [{'id': 0, 'name': 'entry'}, {'id': 1, 'name': 'exit'}]
        for call in context['render_calls']:
            for site in call.get('cpp_trace_sites', ()):
                if (set(site) != {'id', 'name'} or type(site['id']) is not int or not 2 <= site['id'] < 0x80000000 or
                        not isinstance(site['name'], str) or any(s['id'] == site['id'] for s in sites)):
                    raise ValueError('Program trace event sites must be unique explicit IDs.')
                sites.append(dict(site))
        serial = 0x80000000
        def annotate(events, owner):
            nonlocal serial
            for event in events:
                # Caller event identity includes native entry, owning function and call.
                for edge in ('before', 'after'):
                    event['cpp_trace_' + edge] = serial
                    sites.append({'id': serial, 'name': owner + ':' + str(event.get('call', event['kind'])) + ':' + edge})
                    serial += 1
        annotate(context['entry_events'], context['symbol'])
        for function in context['device_functions']:
            annotate(function['schedule']['consumer_events'], function['function'])
        entry_sites[context['entry_id']] = sites
        context['program_trace'] = True
    ranges, offset = [], 0
    for invocation in manifest['invocations']:
        entry = manifest['entries'][invocation['entry_id']]
        warps = prod(entry['grid']) * entry['block_threads'] // entry['warp_size']
        capacity = warps * len(entry_sites[entry['entry_id']])
        nbytes = 16 + capacity * RECORD_BYTES
        ranges.append(dict(entry_id=entry['entry_id'], invocation=invocation['ordinal'],
            call_path=invocation['call_path'], frame_identity=invocation['frame_identity'],
            grid=entry['grid'], block_threads=entry['block_threads'], launch_kind=entry['launch_kind'],
            warps=warps, sites=entry_sites[entry['entry_id']], offset=offset, capacity=capacity, nbytes=nbytes))
        offset += nbytes
    descriptor['program_profile'] = dict(schema=SCHEMA, mode=MODE, record_bytes=RECORD_BYTES,
        ranges=ranges, trace_nbytes=offset, plan_sha256=manifest['plan_sha256'],
        timestamp='gpu-globaltimer', hardware_module_attribution='unavailable',
        perturbation='unmeasured', timing_eligible=False)
    descriptor['program_profile']['identity'] = identity(descriptor['program_profile'])
    return descriptor


def validate_observation(profile, scope, *, count, dropped, observed_warps, grid, records=None):
    if profile.get('schema') != SCHEMA or tuple(grid) != tuple(scope['grid']):
        raise ValueError('Program trace grid/schema differs from its invocation census.')
    if type(count) is not int or type(dropped) is not int or dropped or count != scope['capacity']:
        raise ValueError('Program trace count/drop/capacity invalid or incomplete.')
    if set(observed_warps) != set(range(scope['warps'])):
        raise ValueError('Program trace lacks complete actual-grid warp coverage.')
    if records is not None:
        expected = {(warp, site['id']) for warp in range(scope['warps']) for site in scope['sites']}
        actual = {(r['warp'], r['event']) for r in records}
        if len(records) != len(expected) or actual != expected or any(
                r['entry_id'] != scope['entry_id'] or r['invocation'] != scope['invocation'] for r in records):
            raise ValueError('Program trace has duplicate/misattributed/missing invocation events.')
