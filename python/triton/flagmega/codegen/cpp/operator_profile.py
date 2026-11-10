# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Normal optional caller-event instrumentation; no model kernel/IR rewrites.

This first protocol accepts only a fixed, flat, uniform SM120 call schedule.
All clock/resources/NUM evidence comes from the separately built diagnostic.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path

MODE = 'cta-globaltimer-v1'
MODE_V2 = 'cta-globaltimer-v2'
MODE_V3 = 'cta-globaltimer-v3'
MODES = (MODE, MODE_V2, MODE_V3)
GRID = 96
RECORD_BYTES = 32
EVENTS = 1978

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def validate_option(mode, *, target, codegen_backend='cpp', emit_executable=True):
    if mode is None:
        return
    if mode not in MODES or target != 'nvidia-sm120' or codegen_backend != 'cpp' or not emit_executable:
        raise ValueError('CPP operator profiling requires explicit cta-globaltimer-v1/v2 NVIDIA SM120 executable output.')

def _uniform(event):
    if event.get('participation_active', 'True') not in ('True', True):
        raise ValueError('Operator profile v1 requires proven full-CTA uniform consumer participation.')

def _events(events):
    result=[]
    for original in events:
        event=copy.deepcopy(original)
        _uniform(event)
        if event.get('kind') not in ('barrier','function_call','kernel_call','tir.kernel_call'):
            raise ValueError('Dynamic/recursive/producer event not admitted by profiling v1.')
        expanded = []
        if event.get('barrier_before'):
            expanded.append({'kind':'barrier','scope':event.get('barrier_scope') or 'grid',
                             'barrier_axis_group_key':event.get('barrier_axis_group_key')})
            event['barrier_before']=False
        expanded.append(event)
        for current in expanded:
            # Synthesized and explicit barriers have the identical scope
            # contract. Never emit an unvalidated implicit synchronization.
            if current['kind']=='barrier' and current.get('scope')=='block':
                current['scope']='cta'
            if current['kind']=='barrier' and current.get('scope') not in ('grid','cta'):
                raise ValueError('Unknown stage synchronization scope.')
            result.append(current)
    return result

def _model_binding_census(render_calls):
    """Bind each model field to its normal typed descriptor and ABI."""
    rows=[]; calls=set(); seen=set()
    for call in render_calls:
        name=call.get('call')
        if not isinstance(name,str) or not name or name in calls:
            raise ValueError('Profile model census requires unique normal calls.')
        calls.add(name)
        for role in ('inputs','outputs','workspaces'):
            for argument in call.get(role,()):
                formal=argument.get('formal'); buffers=argument.get('buffers')
                if not isinstance(formal,str) or not formal or not isinstance(buffers,list):
                    raise ValueError('Profile model census requires typed argument buffers.')
                for binding in buffers:
                    abi=binding.get('abi'); actual=binding.get('actual'); field=binding.get('formal')
                    if (not isinstance(actual,str) or not actual or not isinstance(field,str) or not field
                            or not isinstance(abi,dict) or abi.get('schema')!='flagmega.local-buffer-abi/v2'
                            or abi.get('buffer')!=actual
                            or binding.get('runtime_value_kind') not in (('scalar','immediate') if abi.get('storage')=='scalar' else ('pointer',))
                            or not isinstance(binding.get('runtime_argument'),str) or not binding['runtime_argument']):
                        raise ValueError('Profile model census requires an actual normal local-buffer ABI binding.')
                    key=(name,role,formal,field,actual)
                    if key in seen:
                        raise ValueError('Duplicate profile model field binding.')
                    seen.add(key)
                    rows.append({'call':name,'role':role,'argument_formal':formal,'binding':copy.deepcopy(binding)})
    if not rows:
        raise ValueError('Profile model census has no typed fields.')
    return rows

def configure_descriptor(original, mode):
    if mode is None:
        return original
    if original.get('cpp_programs'):
        from .program_profile import configure_descriptor as programs
        return programs(original, mode)
    if mode == MODE_V2:
        return _configure_v2(original)
    if mode != MODE or original.get('platform')!='nvidia' or original.get('target_architecture')!='sm120':
        raise ValueError('Unsupported profiling timer/target.')
    if (list(original.get('grid',())) != [96,1,1] or original.get('block_threads')!=128
            or original.get('warp_size')!=32 or not original.get('distributed_entry')):
        raise ValueError('Profiling v1 requires the reviewed full96 CTA/128thread/warp32 launch.')
    if original.get('programs') or original.get('pipeline_schedule'):
        raise ValueError('Separate programs or producer/consumer roles require a versioned profiling protocol.')
    result=copy.deepcopy(original)
    functions=result.get('device_functions',[])
    if len(functions)!=2:
        raise ValueError('Profiling v1 requires two nonrecursive consumer containers.')
    aliases={}; schedules={}; sites=[]; sync_sites=[]
    for f in functions:
        schedule=f['schedule']
        if schedule.get('producer_events') or schedule.get('auxiliary_events'):
            raise ValueError('Warp-specialized profiling not supported.')
        e=_events(schedule['consumer_events'])
        if any(x['kind']=='function_call' for x in e):
            raise ValueError('Recursive/nested dynamic profiling not supported.')
        families={x.get('family') for x in e}
        alias='linear' if 'gdn_recurrent' in families else 'attention' if 'paged_attention_partial' in families else None
        if alias is None or alias in aliases.values():
            raise ValueError('Consumer schedule has no unambiguous reviewed namespace.')
        aliases[f['function']]=alias;schedules[f['function']]=e
        f['schedule']['consumer_events']=e
    main=_events(result['entry_events']);result['entry_events']=main
    contexts=[('main',result.get('symbol','flagmega_main'),main,1)]+[
        (aliases[f['function']],f['symbol'],schedules[f['function']],48 if aliases[f['function']]=='linear' else 16) for f in functions]
    for alias,enclosing,events,multiplicity in contexts:
        op=sync=0
        for position,event in enumerate(events):
            event['profile_offset']=position
            if event['kind']=='function_call':
                continue
            if event['kind']=='barrier':
                event['profile_site_id']=f'sync:{alias}:{sync}';sync+=1
                sync_sites.append({'site_id':event['profile_site_id'],'scope':event['scope'],
                                   'expected_invocations_per_token':multiplicity})
            else:
                event['profile_site_id']=f'{alias}:{op}';op+=1
                sites.append({'site_id':event['profile_site_id'],'call':event['call'],
                    'enclosing_function':enclosing,'family':event['family'],'variant':event['variant'],
                    'execution_kind':event['execution_kind'],'primary_leaf':True,
                    'expected_invocations_per_token':multiplicity,
                    'contains_internal_waits':event['family'] in {'gdn_recurrent','greedy_sample','update_paged_attention_kv_cache'},
                    'work_activity':'not_measured; CTA executes wrapper even if no active ordinary writer/work'})
    invocation=[];sync_manifest=[];event_index=0;layer_index=0;kind_indices={'linear':0,'attention':0}
    def append(event, path, layer=None, kind_index=None):
        nonlocal event_index
        sync=event['kind']=='barrier'
        row={'kind':'schedule_barrier' if sync else 'operator','site_id':event['profile_site_id'],
             'event_index':event_index,'expected_tag':str(event_index),'CTA_slot_formula':'event_index*96+cta_index',
             'layer_index':layer,'layer_kind_index':kind_index,'parent_scope_id':'entry' if layer is None else f'layer:{layer}',
             'call_path':path+'/'+event['profile_site_id']}
        if sync:row['scope']=event['scope']
        (sync_manifest if sync else invocation).append(row);event_index+=1
    for event in main:
        if event['kind']=='function_call':
            callee=event['callee']
            if callee not in schedules:raise ValueError('Unbound profile container.')
            alias=aliases[callee];event['profile_base']=event_index
            for child in schedules[callee]:append(child,'main/'+event['call'],layer_index,kind_indices[alias])
            kind_indices[alias]+=1;layer_index+=1
        else:
            event['profile_offset']=event_index
            append(event,'main')
    binding_census=_model_binding_census(result.get('render_calls',()))
    model_fields=len(binding_census)
    model_calls={c['call'] for c in result.get('render_calls',())}
    if (len(sites)!=44 or len(invocation)!=1062 or len(sync_manifest)!=916 or event_index!=EVENTS
            or kind_indices!={'linear':48,'attention':16} or len(model_calls)!=44
            or model_calls!={site['call'] for site in sites}):
        raise ValueError('Normal typed schedule/ABI differs from reviewed profiling v1 census; rebuild/review protocol.')
    metadata={'schema':'flagmega.cpp-operator-profile/v1','profile_mode':MODE,
        'launch_abi':'flagmega_launch_profile/v1','accepted_timing_eligible':False,
        'model_ABI_binding_count':model_fields,
        'model_ABI_binding_manifest':binding_census,'model_ABI_binding_manifest_sha256':digest(binding_census),
        'event_count':event_index,'grid_total':GRID,
        'extra_trace_shape_u64':[EVENTS,GRID,4],'scalar_strides':[GRID*4,4,1],
        'record_bytes':RECORD_BYTES,'trace_alignment_bytes':8,'trace_nbytes':EVENTS*GRID*RECORD_BYTES,
        'extra_CTA_barriers_per_CTA_per_token':2*len(invocation),'runtime_mode':'diagnostic_eager',
        'clock_source':'%globaltimer','counter_bits':64,'ns_per_tick':None,
        'site_inventory':sites,'invocation_manifest':invocation,'sync_manifest':sync_manifest,
        'site_inventory_sha256':digest(sites),'invocation_manifest_sha256':digest(invocation),
        'sync_manifest_sha256':digest(sync_manifest),'resources':None,'resource_admission_receipt':None,
        'NUM_crosscheck_receipt':None,'overhead_comparison_receipt':None,
        'event_launch_abi':'flagmega_launch_profile_events/v1','clock_launch_abi':'flagmega_profile_clock_calibrate/v1',
        'clock_output_shape_u64':[4],'clock_output_nbytes':32,'clock_loop_ladder':[1<<k for k in range(12,27)],
        'trace_fields':['start_tick','end_tick','epoch','tag'],'trace_owners':'one CTA thread0/event, no atomics',
        'trace_effect':'diagnostic auxiliary write-only memory; not a model pool or BufferPlan alias'}
    root=Path(__file__).parent
    paths=[root/'operator_profile.py',root/'templates/entrypoints/call_graph_profile.cpp.jinja',
           root/'templates/entrypoints/_nvidia_profile_launch.cpp.jinja',root/'templates/entrypoints/_nvidia_profile_clock.cpp.jinja',root/'ftt/include/ftt/arch/nvidia/primitive_ops.h']
    metadata['implementation_sources']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    metadata['implementation_sha256']=digest(metadata['implementation_sources'])
    result['operator_profile']=metadata
    result['entry_template']='entrypoints/call_graph_profile.cpp.jinja'
    return result


def _configure_v2(original):
    """Instrument a complete uniform flat schedule without workload constants.

    Only entry -> reusable consumer -> leaf is admitted. Event tags enumerate
    the expanded actual entry schedule; source definitions retain relative
    offsets, with an explicit base on each call. Parent calls are not timed a
    second time, and caller synchronization remains separate from leaf time.
    """
    if original.get('platform') != 'nvidia' or original.get('target_architecture') != 'sm120':
        raise ValueError('Profiling v2 requires NVIDIA SM120.')
    grid = original.get('grid')
    if (not isinstance(grid, (list, tuple)) or len(grid) != 3
            or any(type(n) is not int or n <= 0 for n in grid)
            or grid[0] > (1 << 31) - 1 or list(grid[1:]) != [1, 1]
            or original.get('grid_total_blocks') != grid[0]):
        raise ValueError('Profiling v2 requires a positive flat launch grid and matching total.')
    threads = original.get('block_threads')
    if (type(threads) is not int or threads <= 0 or threads > 1024 or threads % 32
            or original.get('warp_size') != 32 or not original.get('distributed_entry')):
        raise ValueError('Profiling v2 requires a cooperative complete-warp NVIDIA launch.')
    if original.get('programs') or original.get('pipeline_schedule'):
        raise ValueError('Separate programs or producer/consumer roles are not supported by profiling v2.')
    result = copy.deepcopy(original)
    functions = result.get('device_functions', [])
    if not isinstance(functions, (list, tuple)):
        raise ValueError('Invalid reusable consumer function list.')
    schedules = {}
    function_symbols = set()
    for function in functions:
        name, symbol = function.get('function'), function.get('symbol')
        if (not isinstance(name, str) or not name or name in schedules
                or not isinstance(symbol, str) or not symbol or symbol in function_symbols):
            raise ValueError('Profiling requires unique named consumer definitions and symbols.')
        schedule = function.get('schedule')
        if (not isinstance(schedule, dict) or schedule.get('producer_events')
                or schedule.get('auxiliary_events') or schedule.get('warp_specialize')):
            raise ValueError('Warp-specialized or missing consumer schedule is unsupported.')
        events = _events(schedule.get('consumer_events', ()))
        if not events or any(event['kind'] == 'function_call' for event in events):
            raise ValueError('Empty/recursive/nested consumer profiling is unsupported.')
        schedules[name] = events
        function_symbols.add(symbol)
        schedule['consumer_events'] = events
    main = _events(result.get('entry_events', ()))
    if not main:
        raise ValueError('Profiling requires an executable entry schedule.')
    result['entry_events'] = main
    multiplicities = {name: 0 for name in schedules}
    definition_map = {function['function']: function for function in functions}
    for event in main:
        if event['kind'] == 'function_call':
            callee = event.get('callee')
            if not isinstance(event.get('call'), str) or not event['call']:
                raise ValueError('Consumer invocation requires an explicit call identity.')
            if callee not in schedules:
                raise ValueError('Unbound profiling consumer call.')
            if event.get('symbol', '').removesuffix('__consumer') != definition_map[callee]['symbol']:
                raise ValueError('Profile consumer symbol differs from its definition.')
            multiplicities[callee] += 1
    if any(count == 0 for count in multiplicities.values()):
        raise ValueError('Unreachable consumer definitions are not admitted by profiling v2.')
    binding_census = _model_binding_census(result.get('render_calls', ()))
    calls = {call['call']: call for call in result['render_calls']}
    sites, sync_sites = [], []
    contexts = [('entry', result.get('symbol', 'flagmega_main'), main, 1)]
    contexts.extend((f'function:{function["function"]}', function['symbol'],
                     schedules[function['function']], multiplicities[function['function']])
                    for function in functions)
    covered_calls = set()
    for namespace, enclosing, events, count in contexts:
        op_index = sync_index = 0
        for position, event in enumerate(events):
            event['profile_offset'] = position
            if event['kind'] == 'function_call':
                continue
            if event['kind'] == 'barrier':
                event['profile_site_id'] = f'sync:{namespace}:{sync_index}'
                sync_index += 1
                sync_sites.append({'site_id': event['profile_site_id'], 'scope': event['scope'],
                                   'expected_invocations_per_token': count})
                continue
            call = calls.get(event.get('call'))
            if call is None or any(event.get(key) != call.get(key)
                                   for key in ('symbol', 'family', 'variant', 'execution_kind')):
                raise ValueError('Profile leaf event differs from its normal typed call descriptor.')
            covered_calls.add(event['call'])
            event['profile_site_id'] = f'{namespace}:{op_index}'
            op_index += 1
            sites.append({'site_id': event['profile_site_id'], 'call': event['call'],
                          'enclosing_function': enclosing, 'family': event['family'],
                          'variant': event['variant'], 'execution_kind': event['execution_kind'],
                          'primary_leaf': True, 'expected_invocations_per_token': count,
                          'contains_internal_waits': 'included_if_present; inspect selected implementation',
                          'work_activity': 'not_measured; CTA wrapper may have no active ordinary writer/work'})
    if not sites or covered_calls != set(calls):
        raise ValueError('Profile leaf census must cover all normal typed calls.')
    invocations, sync_manifest = [], []
    event_index = container_index = 0
    container_occurrences = {name: 0 for name in schedules}

    def append(event, call_path, parent_scope):
        nonlocal event_index
        synchronization = event['kind'] == 'barrier'
        row = {'kind': 'schedule_barrier' if synchronization else 'operator',
               'site_id': event['profile_site_id'], 'event_index': event_index,
               'expected_tag': str(event_index),
               'CTA_slot_formula': f'event_index*{grid[0]}+cta_index',
               'parent_scope_id': parent_scope,
               'call_path': call_path + '/' + event['profile_site_id']}
        if synchronization:
            row['scope'] = event['scope']
        (sync_manifest if synchronization else invocations).append(row)
        event_index += 1

    for event in main:
        if event['kind'] == 'function_call':
            callee = event['callee']
            event['profile_base'] = event_index
            occurrence = container_occurrences[callee]
            path = f'entry/{event["call"]}[{occurrence}]'
            for child in schedules[callee]:
                append(child, path, f'container:{container_index}')
            container_occurrences[callee] += 1
            container_index += 1
        else:
            event['profile_offset'] = event_index
            append(event, 'entry', 'entry')
    trace_nbytes = event_index * grid[0] * RECORD_BYTES
    if not invocations or trace_nbytes > (1 << 63) - 1:
        raise ValueError('Empty or overflowing diagnostic trace allocation.')
    metadata = {
        'schema': 'flagmega.cpp-operator-profile/v2', 'profile_mode': MODE_V2,
        'launch_abi': 'flagmega_launch_profile/v1', 'accepted_timing_eligible': False,
        'model_ABI_binding_count': len(binding_census),
        'model_ABI_binding_manifest': binding_census,
        'model_ABI_binding_manifest_sha256': digest(binding_census),
        'event_count': event_index, 'grid_total': grid[0], 'launch_grid': list(grid),
        'block_threads': threads, 'warp_size': 32,
        'extra_trace_shape_u64': [event_index, grid[0], 4],
        'scalar_strides': [grid[0] * 4, 4, 1], 'record_bytes': RECORD_BYTES,
        'trace_alignment_bytes': 8, 'trace_nbytes': trace_nbytes,
        'extra_CTA_barriers_per_CTA_per_token': 2 * len(invocations),
        'runtime_mode': 'diagnostic_eager', 'clock_source': '%globaltimer',
        'counter_bits': 64, 'ns_per_tick': None,
        'site_inventory': sites, 'sync_site_inventory': sync_sites,
        'invocation_manifest': invocations, 'sync_manifest': sync_manifest,
        'site_inventory_sha256': digest(sites), 'sync_site_inventory_sha256': digest(sync_sites),
        'invocation_manifest_sha256': digest(invocations), 'sync_manifest_sha256': digest(sync_manifest),
        'resources': None, 'resource_admission_receipt': None, 'NUM_crosscheck_receipt': None,
        'overhead_comparison_receipt': None,
        'event_launch_abi': 'flagmega_launch_profile_events/v1',
        'clock_launch_abi': 'flagmega_profile_clock_calibrate/v1',
        'clock_output_shape_u64': [4], 'clock_output_nbytes': 32,
        'clock_loop_ladder': [1 << k for k in range(12, 27)],
        'trace_fields': ['start_tick', 'end_tick', 'epoch', 'tag'],
        'trace_owners': 'one CTA thread0/event, no atomics',
        'trace_effect': 'diagnostic auxiliary write-only memory; not a model pool or BufferPlan alias',
        'schedule_contract': 'uniform flat entry, optional one-level reusable consumers, no dynamic control or producer roles',
        'container_multiplicities': multiplicities,
    }
    root = Path(__file__).parent
    paths = [root / 'operator_profile.py', root / 'templates/entrypoints/call_graph_profile.cpp.jinja',
             root / 'templates/entrypoints/_nvidia_profile_launch.cpp.jinja',
             root / 'templates/entrypoints/_nvidia_profile_clock.cpp.jinja',
             root / 'ftt/include/ftt/arch/nvidia/primitive_ops.h']
    metadata['implementation_sources'] = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    metadata['implementation_sha256'] = digest(metadata['implementation_sources'])
    result['operator_profile'] = metadata
    result['entry_template'] = 'entrypoints/call_graph_profile.cpp.jinja'
    return result
