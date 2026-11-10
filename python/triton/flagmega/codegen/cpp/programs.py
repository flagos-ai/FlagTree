# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Native definition contexts and serial invocations from verified typed TIR."""
from copy import deepcopy
from dataclasses import replace
import hashlib
from math import prod

from triton.flagmega.errors import CodegenError
from .program_abi import PROGRAM_SCHEMA, ArgTag, SCALAR_TAGS, C_TYPES, canonical_json, identity


def _events(values):
    for value in values:
        yield value
        if value['kind'] == 'loop':
            yield from _events(value['events'])


def _check_controls(program, ordinary):
    for events in [program['entry_events'], *(f['schedule']['consumer_events'] for f in program['device_functions'])]:
        for event in _events(events):
            if event['kind'] not in {'barrier', 'tir.kernel_call', 'function_call', 'loop_copy'}:
                raise CodegenError('Native serial programs require explicitly lowered controls/carry copies.')
            if event['kind'] == 'barrier' and event['scope'] not in {'block', 'grid'}:
                raise CodegenError('Native serial programs do not implement peer-device completion.')
            if ordinary and (event['kind'] == 'barrier' and event['scope'] == 'grid' or
                             event.get('barrier_before') and event.get('barrier_scope', 'grid') != 'block'):
                raise CodegenError('Ordinary native entry cannot contain grid/device waits.')
    if ordinary and any(c.get('internal_grid_barriers') or 'grid_sync' in c.get('facts', {}).get('requires', ())
                        for c in program['render_calls']):
        raise CodegenError('Ordinary native entry has an internal grid wait.')


def _type_from_binding(binding, root_types, root_names):
    from .expressions import cpp_scalar_type
    kind = binding['kind']
    if kind == 'argument':
        return root_types[root_names[binding['source']]]
    if kind == 'pointer':
        return {'type': cpp_scalar_type(binding['dtype']), 'pointer': True}
    if kind in {'immediate', 'scalar_argument'}:
        try:
            SCALAR_TAGS[binding['dtype']]
            return {'type': cpp_scalar_type(binding['dtype']), 'pointer': False}
        except KeyError as error:
            raise CodegenError('Native program scalar type is outside the closed ABI.') from error
    raise CodegenError('Unknown native typed argument binding.')


def describe_cpp_programs(module, descriptor, describe_entry):
    from .renderer import _argument_types
    from triton.flagmega.codegen.triton.runtime_binding import describe_function_runtime_binding
    from triton.flagmega.ir import verify_buffer_plan
    buffer_plan = verify_buffer_plan(module)
    if descriptor['platform'] != 'nvidia':
        raise CodegenError('Native serial programs currently require NVIDIA.')
    if any('stream_index' in launch or 'depends_on' in launch for launch in descriptor['kernels']):
        raise CodegenError('Native serial programs do not implement multistream DAG control.')
    root_types = _argument_types(descriptor, module)
    definitions = {p['symbol']: p for p in descriptor['programs'] if not p.get('entry_alias')}
    launches = descriptor['kernels']
    contexts, entries, symbols, invocations = [], [], {}, []
    for ordinal, launch in enumerate(launches):
        if launch.get('maxnreg') is not None:
            raise CodegenError('Native v3 does not implement per-entry register caps.')
        symbol = launch.get('entry_symbol', launch['symbol'])
        program = deepcopy(definitions[symbol])
        names = program.get('signature_arguments', descriptor['signature_arguments'])
        bindings = launch.get('argument_bindings', [{'kind': 'argument', 'source': i} for i in range(len(names))])
        if launch.get('execution_mesh', {}).get('kind') == 'dynamic':
            needed = {name for call in program['render_calls'] for name in call['arguments']}
            selected = [(n, b) for n, b in zip(names, bindings, strict=True) if n in needed]
            names, bindings = [p[0] for p in selected], [p[1] for p in selected]
            program['scoped_entry_arguments'] = [p for p in program.get('scoped_entry_arguments', ()) if p['name'] in needed]
        if len(bindings) != len(names):
            raise CodegenError('Native program argument count mismatch.')
        native_types = {name: _type_from_binding(b, root_types, descriptor['signature_arguments'])
                        for name, b in zip(names, bindings, strict=True)}
        if symbol not in symbols:
            entry_id = len(entries)
            symbols[symbol] = entry_id
            dynamic = launch.get('execution_mesh', {}).get('kind') == 'dynamic'
            sync = bool(launch['requires_grid_wide_sync'])
            kind = launch.get('program_kind', 'persistent' if sync else 'ordinary')
            ordinary = not sync and (dynamic or not program.get('mesh_context', {}).get('distributed_entry', descriptor.get('distributed_entry')))
            _check_controls(program, ordinary)
            if dynamic and sync:
                raise CodegenError('Dynamic native task entries cannot contain grid waits.')
            recipe = launch.get('launch_recipe')
            if recipe:
                from triton.flagmega.ir.launch_recipe import TaskLaunchRecipe
                typed = TaskLaunchRecipe.from_data(recipe)
                if typed.program_kind.value != 'ordinary':
                    raise CodegenError('Native v3 supports the exact ordinary task grid; persistent recipes need a grid ABI.')
                if tuple(launch['grid']) != typed.grid_upper_bound:
                    raise CodegenError('Native entry grid disagrees with its task recipe.')
            context = deepcopy({**descriptor, **program, **program.get('mesh_context', {}),
                                'grid': launch['grid'], 'programs': [], 'signature_arguments': names,
                                'native_argument_types': native_types, 'cpp_program_entry': True})
            # Only reachable families contribute templates or shared resources.
            used = {(c['family'], c['variant']) for c in context['render_calls']}
            context['kernel_template_specs'] = [s for s in descriptor['kernel_template_specs'] if (s['kernel'], s['variant']) in used]
            # Addressing remains distributed; barrier ABI is a separate fact.
            if dynamic and any(c['family'] != 'elementwise' and c.get('cpp_task_mapping') != 'independent-cta'
                               for c in context['render_calls']):
                raise CodegenError('Native dynamic family must declare an independent CTA task mapping.')
            context['native_task_recipe'] = recipe
            context['cpp_task_loop'] = dynamic and all(c['family'] == 'elementwise' for c in context['render_calls'])
            context['requires_grid_wide_sync'] = sync
            context['barrier_abi'] = not ordinary
            context = describe_entry(module, context)
            context.update(entry_id=entry_id, launch_kind='ordinary' if ordinary else 'cooperative',
                           dynamic_shared_bytes=0, program_kind=kind)
            if ordinary:
                context.update(launch_scratch=None, grid_barrier_counter_slots=0, grid_barrier_group_bases={},
                               grid_barrier_axis_groups=[])
            owner = program.get('owning_function')
            bounds = {}
            if owner is not None or not program.get('scoped_entry_arguments'):
                runtime = describe_function_runtime_binding(module, function_name=owner or module.entry)
                for arg in runtime['arguments']:
                    buffer = buffer_plan.buffer_map[arg['buffer']]
                    if buffer.storage != 'scalar':
                        bounds[arg['name']] = dict(min_nbytes=buffer.mem_span.size.fixed_value,
                                                  alignment=max(buffer.dtype.itemsize, buffer_plan.proven_buffer_alignments[buffer.id]))
                for pool in runtime['pools']:
                    b = next((b for n,b in zip(names,bindings,strict=True) if n == pool['name']), {})
                    stride = b.get('scope_stride_bytes', 0)
                    bounds[pool['name']] = dict(min_nbytes=(stride * (pool.get('scope_count', 1)-1) + pool.get('scope_nbytes',pool['nbytes']))
                                               if stride else pool['nbytes'],
                                               alignment=buffer_plan.function_map[owner or module.entry].memory_pool_map[pool['storage']].alignment
                                               if pool['storage'] in buffer_plan.function_map[owner or module.entry].memory_pool_map
                                               else buffer_plan.memory_space_map[pool['storage']].granularity)
            signature = []
            for name in names:
                info = context['argument_types'][name]
                tag = ArgTag.POINTER if info['pointer'] else next((t for t, c in C_TYPES.items() if c == info['type']), None)
                if tag is None:
                    # Core C spellings can differ from the ABI's fixed width aliases.
                    tag = {'int': ArgTag.I32, 'unsigned int': ArgTag.U32, 'long long': ArgTag.I64,
                           'unsigned long long': ArgTag.U64}.get(info['type'])
                if tag is None:
                    raise CodegenError('Unsupported native program scalar type '+str(info))
                signature.append({'name': name, 'tag': int(tag), 'c_type': info['type'], 'pointer': info['pointer'], **bounds.get(name, {})})
            entry = {k: context[k] for k in ('entry_id', 'symbol', 'block_threads', 'grid', 'warp_size',
                     'launch_kind', 'program_kind', 'dynamic_shared_bytes', 'launch_resource_limits')}
            entry.update(signature=signature, requires_grid_wide_sync=sync,
                scratch_nbytes=0 if ordinary else context['launch_scratch']['nbytes'],
                execution_mesh=launch.get('execution_mesh'), data_placement=launch.get('data_placement'),
                launch_recipe=recipe, scoped_entry_arguments=program.get('scoped_entry_arguments', []),
                definition_sha256=identity(program), calls=launch['calls'])
            context['native_signature'] = signature
            entries.append(entry); contexts.append(context)
        entry_id = symbols[symbol]
        context = contexts[entry_id]
        if any(context['argument_types'][n] != t for n, t in native_types.items()):
            raise CodegenError('Shared native definition changes parameter type between invocations.')
        for b in bindings:
            if b['kind'] == 'pointer' and context['launch_kind'] == 'ordinary' and b.get('scope_stride_bytes', 0):
                raise CodegenError('Ordinary canonical operands cannot inherit fixed CTA-private strides.')
        invocations.append(dict(entry_id=entry_id, ordinal=ordinal, call_path=launch.get('call_path', []),
            argument_bindings=bindings, invocation_kind=launch.get('invocation_kind', 'program'),
            transfer=launch.get('transfer'), frame_identity=identity(bindings)))
    # A companion contains the execution hash it binds, so it cannot be part
    # of that hash itself. All other module metadata remains in the identity;
    # the full IR/artifact semantic hash still includes the companion.
    execution_module = module
    if 'counter_epoch_plan' in module.metadata:
        execution_module = replace(module, metadata={key: value for key, value in module.metadata.items()
                                                     if key != 'counter_epoch_plan'})
    manifest = dict(schema=PROGRAM_SCHEMA, entries=entries, invocations=invocations,
                    program_execution=descriptor.get('program_execution'), semantic_hash=execution_module.semantic_hash)
    manifest['plan_sha256'] = identity(manifest)
    package = {**descriptor, 'schema': 'flagmega.cpp-package-descriptor/v1', 'launch_abi_version': 3,
        'launch_admission': 'cuda-program-resources/v3', 'cpp_programs': manifest,
        'native_program_contexts': contexts, 'block_threads': contexts[0]['block_threads'],
        'argument_types': root_types, 'kernel_templates': (), 'entry_template': '', 'launch_scratch': None}
    if 'counter_epoch_plan' in module.metadata:
        package = configure_counter_epoch_descriptor(package, module.metadata['counter_epoch_plan'])
    return package


def _render_device_programs(descriptor):
    from .renderer import CppTemplateRegistry
    registry = CppTemplateRegistry()
    includes, parts = [], []
    for context in descriptor['native_program_contexts']:
        parts.append('namespace flagmega_entry_'+str(context['entry_id'])+' {')
        for template in context['kernel_templates']:
            source = registry.render(template, context)
            lines = []
            for line in source.splitlines():
                if line.startswith('#include '):
                    if line not in includes:
                        includes.append(line)
                else:
                    lines.append(line)
            parts.append('\n'.join(lines))
        parts.append(registry.render('entrypoints/call_graph.cpp.jinja', context))
        parts.append('}')
    return includes, parts


def device_source_identity(descriptor):
    """Identity of actual selected device text/includes, excluding host launch ABI."""
    includes, parts = _render_device_programs(descriptor)
    return hashlib.sha256(('\n'.join(includes + parts)+'\n').encode()).hexdigest()


def configure_counter_epoch_descriptor(descriptor, plan_data):
    from triton.flagmega.ir.counter_epoch_plan import CounterEpochPlan, verify_counter_epoch_plan
    from triton.flagmega.errors import IRSchemaError, IRVerificationError
    if descriptor.get('program_profile') or descriptor.get('platform') != 'nvidia':
        raise CodegenError('Counter epoch plans require a normal NVIDIA descriptor.')
    try:
        plan = CounterEpochPlan.from_data(plan_data)
        proof = verify_counter_epoch_plan(plan, descriptor['cpp_programs'], device_source_identity(descriptor))
    except (IRSchemaError, IRVerificationError) as error:
        raise CodegenError('Invalid explicit counter epoch plan: '+str(error)) from error
    return {**descriptor, 'cpp_counter_epoch_plan': plan.to_data(), 'counter_epoch_proof': proof}


def render_program_package(descriptor, renderer_version):
    from .renderer import CppTemplateRegistry
    registry = CppTemplateRegistry()
    if 'cpp_counter_epoch_plan' in descriptor:
        # Profiling mutates device bodies after ordinary description; never
        # transfer the ordinary effect proof to a diagnostic instrumentation.
        descriptor = configure_counter_epoch_descriptor(descriptor, descriptor['cpp_counter_epoch_plan'])
    includes, parts = _render_device_programs(descriptor)
    result = ['// Generated by '+renderer_version, '#include <ftt/ftt.h>', '#include <ftt/runtime.h>',
              '#include <ftt/program_abi.h>', '#include <ftt/program_trace.h>', '#include <cstring>', '#include <cstdint>']
    context = {**descriptor, 'native_manifest_literal': canonical_json(canonical_json(descriptor['cpp_programs']))}
    if 'cpp_counter_epoch_plan' in descriptor:
        context['native_counter_epoch_manifest_literal'] = canonical_json(canonical_json(descriptor['cpp_counter_epoch_plan']))
    result += includes + parts
    result.append(registry.render('entrypoints/program_launch.cpp.jinja', context))
    return '\n'.join(result)+'\n'
