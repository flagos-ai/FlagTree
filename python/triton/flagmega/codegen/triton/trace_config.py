# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Explicit diagnostic context on the ordinary typed source schedule."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from math import prod

from triton.flagmega.errors import CodegenError

SCHEMA = 'flagmega.triton-trace/v1'
LAYER_ARGUMENT = '_flagmega_trace_layer_id'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


class _ExecutionDomains:
    """Prove nonuniform pure norm ownership from the normal typed call inputs.

    Predicates are checked provenance, never an evaluator or an admission flag.
    Uniform calls retain their original planner and dictionary path.
    """

    def __init__(self, module, bindings, calls, config):
        self.module, self.bindings, self.calls, self.config = module, bindings, calls, config
        self.entries = {}

    def _invocation_context(self, function, role, descriptor, coordinates):
        from triton.flagmega.codegen.triton.function_programs import _constant
        from triton.flagmega.codegen.triton.physical_access import emit_scalar_immediate
        from triton.flagmega.ir.tir.execution import execution_calls_of

        context = {
            'root': self.module.entry, 'function': function, 'chip_node_coordinates': coordinates, 'invocations': []
        }
        if function == self.module.entry:
            specification = self.config['layer_bindings'][function]
            if set(specification) != {'constant'}:
                raise CodegenError('Nonuniform owner root context is not statically bound.')
            context['invocations'] = [{
                'root_invocation': True, 'layer': specification['constant'], 'formal_bindings': []
            }]
            return context
        specification = self.config['layer_bindings'][function]
        if 'formal' not in specification:
            raise CodegenError('Nonuniform owner requires its actual reusable layer formal.')
        definitions = [d for d in descriptor['device_functions'] if d['function'] == function]
        if len(definitions) != 1:
            raise CodegenError('Nonuniform owner has no unique reusable source definition.')
        definition = definitions[0]
        signature = definition['signature_arguments']
        arguments = self.bindings[function]['arguments']
        layer_arguments = [a for a in arguments if a['buffer'] == specification['formal']]
        if len(layer_arguments) != 1 or layer_arguments[0]['name'] not in signature:
            raise CodegenError('Nonuniform owner layer formal differs from the actual source ABI.')
        for parent, body in self.module.execution_function_map.items():
            for call in execution_calls_of(body):
                if call.callee != function:
                    continue
                if parent != self.module.entry:
                    raise CodegenError('Nonuniform owner context requires direct static root invocations.')
                edges = {edge.formal: edge for edge in call.arguments}
                if len(edges) != len(call.arguments) or any(
                        row.get('call') == call.call_id for row in context['invocations']):
                    raise CodegenError('Nonuniform owner has duplicate typed invocation bindings or identities.')
                values = []
                for argument in arguments:
                    if argument.get('runtime_value_kind') != 'scalar':
                        continue
                    dtype = argument.get('scalar_dtype')
                    if dtype not in {'int32', 'bool'} or argument['buffer'] not in edges:
                        raise CodegenError('Nonuniform owner has an unbound scalar invocation context.')
                    if argument['name'] not in signature:
                        raise CodegenError('Nonuniform owner scalar formal is missing from its source ABI.')
                    edge = edges[argument['buffer']]
                    value = _constant(self.module, edge.actual)
                    if not ((dtype == 'int32' and type(value) is int and -(1 << 31) <= value <
                             (1 << 31)) or (dtype == 'bool' and type(value) is bool)):
                        raise CodegenError('Nonuniform owner context is not an exact typed static scalar.')
                    values.append({'formal': edge.formal, 'actual': edge.actual, 'value': value, 'dtype': dtype})
                layer = next((v['value'] for v in values if v['formal'] == specification['formal']), None)
                if layer is None or ('values' in specification and layer not in specification['values']):
                    raise CodegenError('Nonuniform owner lacks a proven layer invocation value.')
                projected_roles = []
                for invocation_role in ('consumer', 'producer', 'auxiliary'):
                    events = descriptor['pipeline_schedule'].get(invocation_role + '_events', ())
                    projected = [e for e in events if e['kind'] == 'function_call' and e['call'] == call.call_id]
                    if not projected:
                        continue
                    if len(projected) != 1:
                        raise CodegenError('Nonuniform owner context has a repeated source invocation.')
                    event = projected[0]
                    projected_roles.append(invocation_role)
                    actuals = event['argument_expressions']
                    prefix = len(actuals) - len(signature)
                    if (event.get('participation_active', 'True') not in ('True', True) or event['callee'] != function
                            or prefix < 0):
                        raise CodegenError('Nonuniform function invocation context is not supported.')
                    for argument in arguments:
                        if argument.get('runtime_value_kind') != 'scalar':
                            continue
                        bound = next(v for v in values if v['formal'] == argument['buffer'])
                        expression = emit_scalar_immediate({'storage': 'scalar', 'scalar_dtype': bound['dtype']},
                                                           str(bound['value']))
                        if actuals[prefix + signature.index(argument['name'])] != expression:
                            raise CodegenError('Nonuniform owner context changed during source ABI projection.')
                if role not in projected_roles:
                    raise CodegenError('Nonuniform owner has no actual source invocation for its role.')
                context['invocations'].append({
                    'call': call.call_id, 'callee': function, 'layer': layer, 'formal_bindings': values, 'source_roles':
                    projected_roles
                })
        if not context['invocations']:
            raise CodegenError('Nonuniform owner has no actual typed root invocation.')
        context['invocations'].sort(key=lambda row: row['call'])
        return context

    def prove(self, function, role, event, descriptor):
        import ast
        from pathlib import Path
        from triton.flagmega.codegen.triton import kernel_call_renderers as renderers
        from triton.flagmega.codegen.triton.tir_package import _mesh_context, _kernel_source_event

        if event['kind'] not in {'tir.kernel_call', 'kernel_call'} or 'trace_execution_domain' in event:
            raise CodegenError('Nonuniform caller lacks a checked leaf execution-domain proof.')
        raw = [
            c for c in self.bindings.get(function, {}).get('call_abi', {}).get('kernel_calls', ())
            if c['call'] == event['call']
        ]
        prepared = [c for c in self.calls.get(function, ()) if c['call'] == event['call']]
        if len(raw) != 1 or len(prepared) != 1:
            raise CodegenError('Nonuniform caller has no exact raw/prepared identity.')
        raw, prepared = raw[0], prepared[0]
        helper = renderers._canonical_norm_stats_execution_active(raw, prepared)
        if helper is None:
            raise CodegenError('Nonuniform caller is not an eligible complete canonical NormStats owner.')
        fresh = next(c for c in renderers.prepare_kernel_calls(self.bindings[function]['call_abi']['kernel_calls'],
                                                               function_name=function) if c['call'] == event['call'])
        checked = {
            'ordinal', 'call', 'symbol', 'family', 'variant', 'execution_kind', 'arguments', 'signature',
            *renderers._norm_stats_call(raw)
        }
        if any(digest(prepared.get(key)) != digest(fresh[key]) for key in checked):
            raise CodegenError('Nonuniform owner raw/prepared source or arithmetic identity differs.')
        if (helper != fresh['writer_active'] or prepared['writer_active'] != fresh['writer_active']
                or prepared['participation_active'] != fresh['participation_active']):
            raise CodegenError('Nonuniform owner differs from recomputed normal preparation.')
        for name in ('call', 'family', 'variant', 'execution_kind', 'symbol', 'participation_active'):
            if event.get(name) != prepared[name]:
                raise CodegenError('Nonuniform owner projected event differs from its prepared call.')
        binding = self.bindings[function]
        signature = binding.get('signature', ())
        actuals = {name: name for name in signature}
        definitions = [d for d in descriptor['device_functions'] if d['function'] == function]
        if definitions and definitions[0].get('function_pool_abi') is not None:
            from triton.flagmega.codegen.triton.function_pool_abi import (bank_row_strides, function_pool_aliases,
                                                                          plan_function_pool_abis, OFFSET_TABLE)
            from triton.flagmega.ir import verify_buffer_plan
            pool_abis, tables = plan_function_pool_abis(self.bindings,
                                                        bank_row_strides(self.module, verify_buffer_plan(self.module)),
                                                        entry=self.module.entry)
            pool = pool_abis.get(function)
            if digest(pool) != digest(definitions[0]['function_pool_abi']):
                raise CodegenError('Nonuniform owner source aliases differ from the actual typed pool ABI.')
            actuals.update(function_pool_aliases(pool))
            if tables:
                actuals[OFFSET_TABLE] = OFFSET_TABLE
        projected = _kernel_source_event(prepared, actuals, {name: (name, 0)
                                                             for name in signature}, [], (), memory_dependencies=())
        try:
            expected = ast.dump(ast.parse(projected['arguments'], mode='eval'), include_attributes=False)
            actual = ast.dump(ast.parse(event['arguments'], mode='eval'), include_attributes=False)
        except (SyntaxError, TypeError) as error:
            raise CodegenError('Nonuniform owner has invalid projected source arguments.') from error
        if actual != expected:
            raise CodegenError('Nonuniform owner projected arguments differ from its typed source call.')
        abis = [raw[section][0]['buffers'][0]['abi'] for section in ('inputs', 'outputs')]
        placement = deepcopy(abis[0]['distributed_type']['placement'])
        hierarchy = placement['hierarchy']
        levels = placement.get('hierarchy_levels') or 'b' * len(hierarchy)
        placement['hierarchy_levels'] = levels
        if (descriptor.get('distributed_entry') is not True or descriptor.get('dynamic_execution', False) is not False
                or type(descriptor.get('tasks_per_owner', 1)) is not int or descriptor.get('tasks_per_owner', 1) != 1):
            raise CodegenError('Nonuniform owner requires a static placement-to-grid mapping.')
        if any(level != 'b' and extent != 1 for extent, level in zip(hierarchy, levels)):
            raise CodegenError('Nonuniform owner chip/node context is not statically bound.')
        expected_mesh = _mesh_context(placement)
        for name in ('grid_mesh', 'mesh_hierarchy', 'mesh_axis_names', 'mesh_levels', 'mesh_axis_sources',
                     'mesh_coordinate_strides'):
            if digest(descriptor.get(name)) != digest(expected_mesh[name]):
                raise CodegenError('Nonuniform owner placement differs from the actual grid/shard mapping.')
        count = prod(extent for extent, level in zip(hierarchy, levels) if level == 'b')
        if not 0 < count <= 0x7fffffff:
            raise CodegenError('Nonuniform owner CTA grid is outside its native i32 domain.')
        grid = descriptor['grid']
        if (not isinstance(grid, (tuple, list)) or any(type(v) is not int for v in grid)
                or list(grid) != [count, 1, 1]):
            raise CodegenError('Nonuniform owner placement does not cover the actual CTA grid.')
        axis_bindings = []
        for axis, (extent, level) in enumerate(zip(hierarchy, levels)):
            stride = prod(v for v, l in zip(hierarchy[axis + 1:], levels[axis + 1:]) if l == 'b')
            axis_bindings.append({
                'axis': axis, 'extent': extent, 'level': level, 'source': 'CTA' if level == 'b' else 'static', 'stride':
                stride if level == 'b' else 0, 'static_coordinate': None if level == 'b' else 0
            })
        exclusive = {}
        for name, abi in zip(('input', 'result'), abis):
            value = abi['distributed_type'].get('exclusive')
            pairs = [] if value is None else sorted(
                zip(value['axes'],
                    value.get('owner_coordinates') or [0] * len(value['axes'])))
            exclusive[name] = {'axes': [axis for axis, _ in pairs], 'owner_coordinates': [owner for _, owner in pairs]}
        ctas = []
        for cta in range(count):
            coordinates = [(cta // axis['stride']) % axis['extent'] if axis['level'] == 'b' else 0
                           for axis in axis_bindings]
            writer = all(coordinates[i] == 0 for i, level in enumerate(levels) if level == 'b')
            owned = all(coordinates[axis] == owner
                        for value in exclusive.values()
                        for axis, owner in zip(value['axes'], value['owner_coordinates']))
            if writer and owned:
                ctas.append(cta)
        if not ctas:
            raise CodegenError('Nonuniform owner intersection has an empty execution domain.')
        context = self._invocation_context(
            function, role, descriptor,
            [{'axis': i, 'level': level, 'value': 0} for i, level in enumerate(levels) if level != 'b'])
        return {
            'function': function, 'call': event['call'], 'role': role, 'ctas': ctas, 'placement': placement,
            'grid_mapping': {'grid': list(grid), 'axis_bindings': axis_bindings}, 'exclusive': exclusive,
            'invocation_context': context, 'proof': {
                'helper_source_sha256': hashlib.sha256(Path(renderers.__file__).read_bytes()).hexdigest(),
                'base_semantic_hash': self.module.semantic_hash, 'raw_call_sha256': digest(raw), 'prepared_call_sha256':
                digest(prepared), 'projected_event_sha256': digest(event), 'helper_predicate': helper,
                'prepared_predicate': prepared['participation_active'], 'projected_predicate':
                event['participation_active']
            }
        }

    def add(self, proof, event, site_id):
        key = {name: proof[name] for name in ('function', 'role', 'call')}
        key['kind'] = 'device_call_envelope'
        entry = {
            **proof, 'site_key': key, 'site_key_sha256': digest(key), 'site_id': site_id, 'record_mode':
            'admission_only' if site_id is None else 'caller_pair'
        }
        identity = (proof['function'], proof['call'], proof['role'])
        if identity in self.entries:
            raise CodegenError('Duplicate nonuniform execution-domain identity.')
        self.entries[identity] = entry
        event['trace_execution_domain'] = entry['site_key_sha256']

    def attach(self, dictionary):
        if self.entries:
            dictionary['execution_domains'] = {
                'schema': 'flagmega.trace-execution-domains/v1', 'entries':
                [self.entries[key] for key in sorted(self.entries)]
            }


def validate_option(config, *, target, codegen_backend='triton', emit_executable=True):
    if config is None:
        return
    if target != 'nvidia-sm90' or codegen_backend != 'triton' or not emit_executable:
        raise CodegenError('Typed diagnostic tracing requires an explicit NVIDIA SM90 Triton executable.')
    if isinstance(config, Mapping) and config.get('schema') == CENSUS_SCHEMA:
        _validate_census_option(config)
        return
    if not isinstance(config, Mapping) or set(config) != {
            'schema', 'base_semantic_hash', 'layer_bindings', 'detailed_calls', 'typed_ctas', 'capacity', 'coarse_calls'
    } or config.get('schema') != SCHEMA:
        raise CodegenError('Invalid typed diagnostic trace configuration schema.')
    if not isinstance(config['base_semantic_hash'], str) or len(config['base_semantic_hash']) != 64:
        raise CodegenError('Trace configuration requires the complete base semantic hash.')
    if config['coarse_calls'] is not True:
        raise CodegenError('Trace v1 requires complete caller envelopes.')
    if type(config['capacity']) is not int or config['capacity'] <= 0:
        raise CodegenError('Trace capacity must be a positive integer.')
    ctas = config['typed_ctas']
    if (not isinstance(ctas, (list, tuple)) or not ctas
            or any(type(value) is not int or not 0 <= value <= 0x7fffffff for value in ctas)
            or list(ctas) != sorted(set(ctas))):
        raise CodegenError('Trace CTA allowlist must contain unique sorted nonnegative i32 IDs.')
    if not isinstance(config['layer_bindings'], Mapping) or not isinstance(config['detailed_calls'], (tuple, list)):
        raise CodegenError('Trace context and detailed calls must be explicit structured bindings.')


class TracePlan:

    def __new__(cls, module, bindings, encoded_calls, config):
        # Keep the existing trace contract independent of fixed-slot census.
        if cls is TracePlan and isinstance(config, Mapping) and config.get('schema') == CENSUS_SCHEMA:
            return CensusPlan(module, bindings, encoded_calls, config)
        return super().__new__(cls)

    def __init__(self, module, bindings, encoded_calls, config):
        validate_option(config, target='nvidia-sm90')
        self.config = deepcopy(dict(config))
        if config['base_semantic_hash'] != module.semantic_hash:
            raise CodegenError('Trace configuration does not match the base TIR semantic hash.')
        if set(config['layer_bindings']) != set(bindings):
            raise CodegenError('Trace layer bindings must cover exactly the reachable functions.')
        self.entry = module.entry
        self._execution_domains = _ExecutionDomains(module, bindings, encoded_calls, self.config)
        self.layers = {}
        self.sites = {}
        for function, specification in config['layer_bindings'].items():
            if not isinstance(specification, Mapping):
                raise CodegenError('Trace layer binding must be a formal or an explicit int32 constant.')
            if set(specification) == {'constant'}:
                value = specification['constant']
                if type(value) is not int or not -(1 << 31) <= value < (1 << 31):
                    raise CodegenError('Trace constant must be an exact int32 value.')
                self.layers[function] = {'expression': f'tl.full((), {value}, tl.int32)', 'constant': value}
            elif set(specification) == {'formal'}:
                matching = [
                    argument for argument in bindings[function]['arguments']
                    if argument['buffer'] == specification['formal']
                ]
                if (len(matching) != 1 or matching[0].get('runtime_value_kind') != 'scalar'
                        or matching[0].get('scalar_dtype') != 'int32'):
                    raise CodegenError('Trace layer formal must be an existing scalar int32 ABI argument.')
                self.layers[function] = {
                    'expression': matching[0]['name'], 'formal': matching[0]['name'], 'buffer': specification['formal']
                }
            else:
                raise CodegenError('Trace binding must select exactly one formal or constant.')
        selected = set()
        for specification in config['detailed_calls']:
            if not isinstance(specification, Mapping) or set(specification) != {'function', 'call'}:
                raise CodegenError('Detailed tracing requires exact function/call identity.')
            function, identity = specification['function'], specification['call']
            key = (function, identity)
            if key in selected:
                raise CodegenError('Duplicate detailed trace call.')
            selected.add(key)
            matching = [call for call in encoded_calls.get(function, ()) if call['call'] == identity]
            if len(matching) != 1:
                raise CodegenError('Detailed trace call is absent or ambiguous.')
            call = matching[0]
            if (call['family'], call['variant']) != ('dense_matmul_glu', 'packed_tensor_descriptor_smem_pipeline_gemv'):
                raise CodegenError('Trace v1 detailed sites require the implemented packed gate/up pipeline.')
            if not call.get('transfer_pipeline') or len(call.get('pipeline_channels', ())) != 1:
                raise CodegenError('Detailed gate/up trace requires one typed channel.')
            if LAYER_ARGUMENT in call['arguments']:
                raise CodegenError('Diagnostic layer argument collides with the ordinary helper ABI.')
            descriptor_count = len(call.get('descriptor_parameters', ()))
            arguments = list(call['arguments'])
            arguments.insert(len(arguments) - descriptor_count, LAYER_ARGUMENT)
            call['arguments'] = arguments
            call['signature'] = ', '.join(arguments)
            call['trace_layer_actual'] = self.layers[function]['expression']
            fields = ('paired', ) if call['paired_weight_fields'] else ('gate', 'up')
            call['trace_sites'] = {
                f'{action}_{field}':
                self.site(function, 'producer' if action == 'acquire' else 'consumer', identity, 'pipe_boundary',
                          action=action, field=field, family=call['family'], variant=call['variant'],
                          layer_provenance=self.layers[function])
                for action in ('acquire', 'wait', 'release')
                for field in fields
            }

    def site(self, function, role, call, kind, **facts):
        key = {
            'function': function, 'role': role, 'call': call, 'kind': kind,
            **{name: facts[name]
               for name in ('action', 'field') if name in facts}
        }
        full_hash = digest(key)
        identity = int(full_hash[:8], 16) & 0x7fffffff
        previous = self.sites.get(identity)
        if previous is not None and previous['key_sha256'] != full_hash:
            raise CodegenError('Typed trace site hash collision.')
        row = {'site_id': identity, 'key_sha256': full_hash, 'key': key, **facts}
        if previous is not None and previous != row:
            raise CodegenError('Typed trace site has inconsistent semantic provenance.')
        self.sites[identity] = row
        return identity

    def finish(self, descriptor):
        if descriptor['platform'] != 'nvidia' or descriptor['target_architecture'] != 'sm90':
            raise CodegenError('Typed trace v1 requires the NVIDIA SM90 central recorder.')
        if max(self.config['typed_ctas']) >= prod(descriptor['grid']):
            raise CodegenError('Typed trace CTA is outside the actual grid.')
        schedule = descriptor.get('pipeline_schedule')
        if not schedule:
            raise CodegenError('Typed trace v1 requires the ordinary producer/consumer schedule.')
        definitions = {definition['function']: definition for definition in descriptor['device_functions']}
        if len(definitions) != len(descriptor['device_functions']):
            raise CodegenError('Trace v1 does not admit multiple outlined forms of one function.')

        def walk(events, function, role):
            for event in events:
                kind = event['kind']
                if kind == 'loop':
                    raise CodegenError(
                        'Trace v1 requires explicit invocation bindings; structured-loop support is not proven.')
                if kind not in {'function_call', 'pipeline_kernel_call', 'tir.kernel_call', 'kernel_call'}:
                    continue
                owner = None
                if event.get('participation_active', 'True') not in ('True', True):
                    if kind == 'function_call':
                        raise CodegenError('Trace v1 requires uniform caller participation.')
                    owner = self._execution_domains.prove(function, role, event, descriptor)
                layer = self.layers[function]['expression']
                provenance = self.layers[function]
                if kind == 'function_call':
                    callee = event['callee']
                    specification = self.layers[callee]
                    if 'formal' in specification:
                        definition = definitions[callee]
                        signature = definition['signature_arguments']
                        if specification['formal'] not in signature:
                            raise CodegenError('Trace layer formal was removed from the reusable function ABI.')
                        actuals = event['argument_expressions']
                        prefix = len(actuals) - len(signature)
                        if prefix < 0:
                            raise CodegenError('Trace invocation has an invalid argument arity.')
                        layer = actuals[prefix + signature.index(specification['formal'])]
                    else:
                        layer = specification['expression']
                    provenance = {'callee': callee, **specification, 'actual': layer}
                event['trace_site'] = self.site(function, role, event['call'], 'device_call_envelope',
                                                callee=event.get('callee'), family=event.get('family'),
                                                variant=event.get('variant'), layer_provenance=provenance,
                                                completion='return; excludes following drains/barriers')
                event['trace_layer'] = layer
                if owner is not None:
                    self._execution_domains.add(owner, event, event['trace_site'])

        for role in ('consumer', 'producer', 'auxiliary'):
            walk(schedule.get(role + '_events', ()), self.entry, role)
        for function, definition in definitions.items():
            for role in ('consumer', 'producer', 'auxiliary'):
                walk(definition['schedule'].get(role + '_events', ()), function, role)
        sites = {
            'schema': 'flagmega.trace-sites/v1', 'base_semantic_hash': self.config['base_semantic_hash'],
            'config_sha256': digest(self.config), 'sites': sorted(self.sites.values(), key=lambda row: row['site_id'])
        }
        self._execution_domains.attach(sites)
        descriptor['triton_trace'] = {
            'schema': SCHEMA, 'config': self.config, 'config_sha256': digest(self.config), 'dictionary': sites,
            'dictionary_sha256': digest(sites), 'accepted_timing_eligible': False, 'native_contract': {
                'event_kinds': [272, 273], 'record_bytes': 64, 'typed_ctas': list(self.config['typed_ctas']),
                'capacity': self.config['capacity'], 'runtime_stream_filtering': False
            }
        }
        return descriptor


CENSUS_SCHEMA = 'flagmega.triton-census/v1'
CENSUS_LAYER_ARGUMENT = '_flagmega_census_layer_id'


def _validate_census_option(config):
    if set(config) != {
            'schema', 'base_semantic_hash', 'layer_bindings', 'service_calls', 'expected_occurrences', 'ctas'
    }:
        raise CodegenError('Invalid census configuration schema.')
    identity = config['base_semantic_hash']
    if (not isinstance(identity, str) or len(identity) != 64 or any(c not in '0123456789abcdef' for c in identity)):
        raise CodegenError('Census configuration requires the complete base semantic hash.')
    for key, label in (('ctas', 'CTA'), ('expected_occurrences', 'occurrence')):
        if type(config[key]) is not int or not 0 < config[key] <= 65536:
            raise CodegenError(f'Census {label} count must be a positive i32 integer up to65536.')
    if 64 + config['expected_occurrences'] * config['ctas'] * 32 + config['ctas'] * 16 > (1 << 30):
        raise CodegenError('Census fixed buffer exceeds the native 1 GiB limit.')
    if not isinstance(config['layer_bindings'], Mapping) or not config['layer_bindings']:
        raise CodegenError('Census requires explicit layer bindings.')
    for specification in config['layer_bindings'].values():
        if not isinstance(specification, Mapping):
            raise CodegenError('Census layer binding must be structured.')
        if set(specification) == {'constant'}:
            value = specification['constant']
            if type(value) is not int or not -(1 << 31) <= value < (1 << 31):
                raise CodegenError('Census constant must be an exact int32 value.')
        elif set(specification) == {'formal', 'values'}:
            values = specification['values']
            if (not isinstance(specification['formal'], str) or not specification['formal']
                    or not isinstance(values, (list, tuple)) or not values or any(type(v) is not int for v in values)
                    or list(values) != list(range(len(values)))):
                raise CodegenError('Census formal values must be a complete zero-based unique i32 domain.')
        else:
            raise CodegenError('Census binding must select a constant or a formal with its exact value domain.')
    if not isinstance(config['service_calls'], (tuple, list)):
        raise CodegenError('Census service calls must be an explicit structured selection.')
    selected = set()
    for selection in config['service_calls']:
        if (not isinstance(selection, Mapping) or set(selection) != {'function', 'call'}
                or any(not isinstance(v, str) or not v for v in selection.values())):
            raise CodegenError('Census service selection requires exact function/call identities.')
        key = selection['function'], selection['call']
        if key in selected:
            raise CodegenError('Duplicate census service call.')
        selected.add(key)


class CensusPlan:
    """Bind finite fixed-slot observations to existing, uniform schedule events.

    This version admits one root invocation and directly called reusable functions
    whose existing i32 layer argument has a proven constant domain. A repeated or
    nested invocation needs a richer occurrence contract and is rejected here.
    No timestamp buffer or model synchronization is synthesized by this planner;
    the explicit TLE configuration allocates its separate ProfileScratch resource.
    """

    def __init__(self, module, bindings, encoded_calls, config):
        validate_option(config, target='nvidia-sm90')
        self.config = deepcopy(dict(config))
        if config['base_semantic_hash'] != module.semantic_hash:
            raise CodegenError('Census configuration does not match the base TIR semantic hash.')
        if set(config['layer_bindings']) != set(bindings):
            raise CodegenError('Census bindings must cover exactly the reachable functions.')
        self.entry = module.entry
        self._execution_domains = _ExecutionDomains(module, bindings, encoded_calls, self.config)
        self.layers = {}
        self.calls = {}
        for function, specification in config['layer_bindings'].items():
            if 'constant' in specification:
                if function != self.entry:
                    raise CodegenError('Census constants are admitted only for the single root invocation.')
                self.layers[function] = {
                    'expression': f"tl.full((), {specification['constant']}, tl.int32)", 'values': [None]
                }
            else:
                matching = [a for a in bindings[function]['arguments'] if a['buffer'] == specification['formal']]
                if (len(matching) != 1 or matching[0].get('runtime_value_kind') != 'scalar'
                        or matching[0].get('scalar_dtype') != 'int32' or function == self.entry):
                    raise CodegenError('Census layer formal must be an existing non-root scalar int32 ABI argument.')
                self.layers[function] = {
                    'expression': matching[0]['name'], 'formal': matching[0]['name'], 'buffer': specification['formal'],
                    'values': list(specification['values'])
                }
        self.invocations = self._prove_invocations(module)
        for selection in config['service_calls']:
            function, identity = selection['function'], selection['call']
            matching = [call for call in encoded_calls.get(function, ()) if call['call'] == identity]
            if len(matching) != 1:
                raise CodegenError('Census service call is absent or ambiguous.')
            call = matching[0]
            if ((call['family'], call['variant']) != ('dense_matmul_glu', 'packed_tensor_descriptor_smem_pipeline_gemv')
                    or not call.get('paired_weight_fields') or not call.get('transfer_pipeline')
                    or len(call.get('pipeline_channels', ())) != 1 or function == self.entry):
                raise CodegenError('Census service requires one paired gate/up channel in a reusable function.')
            if CENSUS_LAYER_ARGUMENT in call['arguments']:
                raise CodegenError('Census layer argument collides with the ordinary helper ABI.')
            arguments = list(call['arguments'])
            arguments.insert(len(arguments) - len(call.get('descriptor_parameters', ())), CENSUS_LAYER_ARGUMENT)
            call['arguments'] = arguments
            call['signature'] = ', '.join(arguments)
            call['census_layer_actual'] = self.layers[function]['expression']
            self.calls[(function, identity)] = call

    def _prove_invocations(self, module):
        from triton.flagmega.codegen.triton.function_programs import _constant
        from triton.flagmega.ir.tir.execution import execution_calls_of

        invocations = {}
        for function in self.layers:
            for call in execution_calls_of(module.execution_function_map[function]):
                if call.callee not in self.layers:
                    continue
                if function != self.entry or call.callee == self.entry:
                    raise CodegenError('Census v1 requires direct, nonrecursive root invocations.')
                layer = self.layers[call.callee]
                edges = [edge for edge in call.arguments if edge.formal == layer['buffer']]
                if len(edges) != 1:
                    raise CodegenError('Census invocation does not bind its exact layer formal.')
                value = _constant(module, edges[0].actual)
                if type(value) is not int or value not in layer['values']:
                    raise CodegenError('Census invocation has no exact i32 value in its declared domain.')
                if call.call_id in invocations:
                    raise CodegenError('Census invocation identity is repeated.')
                invocations[call.call_id] = {'callee': call.callee, 'value': value}
        for function, layer in self.layers.items():
            if function == self.entry:
                continue
            actual = [v['value'] for v in invocations.values() if v['callee'] == function]
            if sorted(actual) != layer['values']:
                raise CodegenError('Census invocation values must cover their domain exactly once.')
        return invocations

    def _expression(self, function, base):
        layer = self.layers[function]
        if function == self.entry:
            return f'tl.full((), {base}, tl.int32)'
        return f"({layer['expression']} + tl.full((), {base}, tl.int32))"

    def finish(self, descriptor):
        if (descriptor['platform'] != 'nvidia' or descriptor['target_architecture'] != 'sm90'
                or prod(descriptor['grid']) != self.config['ctas']):
            raise CodegenError('Census requires the exact declared SM90 CTA grid.')
        schedule = descriptor.get('pipeline_schedule')
        if not schedule:
            raise CodegenError('Census requires the ordinary producer/consumer schedule.')
        definitions = {d['function']: d for d in descriptor['device_functions']}
        if (len(definitions) != len(descriptor['device_functions'])
                or set(definitions) != set(self.layers) - {self.entry}):
            raise CodegenError('Census requires exactly one outlined definition per reusable function.')
        schedules = {self.entry: schedule, **{name: d['schedule'] for name, d in definitions.items()}}
        rows = []
        for function, current in schedules.items():
            if current.get('auxiliary_events'):
                raise CodegenError('Census v1 does not admit an auxiliary consumer role.')
            for role in ('consumer', 'producer'):
                observed = []
                for index, event in enumerate(current.get(role + '_events', ())):
                    if event['kind'] == 'loop' or event.get('trip_scoped_pipe') or event.get('trip_scoped_dynamic'):
                        raise CodegenError('Census v1 rejects repeated/structured-loop occurrences.')
                    if event.get('participation_active', 'True') not in ('True', True):
                        owner = self._execution_domains.prove(function, role, event, descriptor)
                        self._execution_domains.add(owner, event, None)
                    if event['kind'] == 'function_call':
                        expected = self.invocations.get(event['call'])
                        if function != self.entry or expected is None or event['callee'] != expected['callee']:
                            raise CodegenError('Census schedule invocation differs from the typed call graph.')
                        definition = definitions[event['callee']]
                        formal = self.layers[event['callee']]['formal']
                        signature = definition['signature_arguments']
                        actuals = event['argument_expressions']
                        prefix = len(actuals) - len(signature)
                        if (formal not in signature or prefix < 0 or actuals[prefix + signature.index(formal)]
                                != f"tl.full((), {expected['value']}, tl.int32)"):
                            raise CodegenError('Census layer actual changed during normal ABI projection.')
                        observed.append(event['call'])
                    if event['kind'] != 'barrier' or event['scope'] != 'grid':
                        continue
                    base = len(rows)
                    identity = {
                        'function': function, 'role': role, 'schedule_index': index, **{
                            k: deepcopy(event.get(k))
                            for k in ('scope', 'axis_group_axes', 'after', 'before', 'ranges')
                        }
                    }
                    for layer in self.layers[function]['values']:
                        row = {'kind': 'barrier', **identity, 'layer': layer, 'points': [0, 1, 2]}
                        rows.append({'occurrence': len(rows), 'identity_sha256': digest(row), **row})
                    event['census_occurrence'] = self._expression(function, base)
                if function == self.entry and sorted(observed) != sorted(self.invocations):
                    raise CodegenError('Census must observe each typed invocation once in both roles.')
        for (function, identity), call in self.calls.items():
            context = {'layer_parameter': CENSUS_LAYER_ARGUMENT}
            for role in ('consumer', 'producer'):
                events = schedules[function][role + '_events']
                matching = [(i, e)
                            for i, e in enumerate(events)
                            if e.get('call') == identity and e['kind'] == 'pipeline_kernel_call']
                if len(matching) != 1:
                    raise CodegenError('Census service must have one normal invocation in each role.')
                index, event = matching[0]
                if (index + 1 >= len(events) or events[index + 1]['kind'] != 'drain'
                        or len(events[index + 1]['endpoints']) != 1
                        or events[index + 1]['endpoints'][0] != event.get('census_endpoint')
                        or type(event.get('pipeline_commits_per_invocation')) is not int
                        or event['pipeline_commits_per_invocation'] != 6):
                    raise CodegenError('Census service needs its own immediate complete normal drain and fixed epochs.')
                endpoint_key = 'reader_name' if role == 'consumer' else 'writer_name'
                channels = [
                    channel for stage in schedules[function]['stages'] for channel in stage['channels']
                    if channel[endpoint_key] == event['census_endpoint']
                ]
                if (len(channels) != 1 or channels[0]['capacity'] != 2
                        or [field['name'] for field in channels[0]['fields']] != ['gate', 'up']):
                    raise CodegenError('Census service requires the actual two-slot paired gate/up resource.')
                base = len(rows)
                context[role + '_base'] = base
                for layer in self.layers[function]['values']:
                    row = {
                        'kind': 'service', 'function': function, 'role': role, 'call': identity, 'schedule_index':
                        index, 'layer': layer, 'points': [0, 1, 2, 3], 'epochs':
                        event['pipeline_commits_per_invocation'], 'slots': channels[0]['capacity'],
                        'drain_schedule_index': index + 1, 'drain_endpoints': events[index + 1]['endpoints']
                    }
                    rows.append({'occurrence': len(rows), 'identity_sha256': digest(row), **row})
                events[index + 1]['census_occurrence'] = self._expression(function, base)
            call['census'] = context
            rendered = [
                c for c in descriptor['render_calls'] if c['call'] == identity and c['symbol'] == call['symbol']
            ]
            if len(rendered) != 1:
                raise CodegenError('Census service has no unique normal rendered helper.')
            rendered[0]['census'] = context
        if len(rows) != self.config['expected_occurrences']:
            raise CodegenError('Census actual occurrence count differs from its explicit configuration.')
        count, ctas = len(rows), self.config['ctas']
        timestamps = count * ctas * 4 * 8
        mapping = 2 * ctas * 2 * 4
        logical = 64 + timestamps + mapping
        per_cta = ((logical + ctas - 1) // ctas + 63) // 64 * 64
        dictionary = {
            'schema': 'flagmega.census-sites/v1', 'base_semantic_hash': self.config['base_semantic_hash'],
            'config_sha256': digest(self.config), 'occurrences': rows
        }
        self._execution_domains.attach(dictionary)
        descriptor['triton_trace'] = {
            'schema': CENSUS_SCHEMA, 'config': self.config, 'config_sha256': digest(self.config), 'dictionary':
            dictionary, 'dictionary_sha256': digest(dictionary), 'accepted_timing_eligible': False, 'native_contract': {
                'occurrences': count, 'ctas': ctas, 'points': 4, 'header_bytes': 64, 'timestamps_offset': 64,
                'timestamp_bytes': timestamps, 'mapping_offset': 64 + timestamps, 'mapping_bytes': mapping,
                'logical_bytes': logical, 'scratch_bytes_per_cta': per_cta, 'allocated_bytes': per_cta * ctas,
                'scratch_alignment_bytes': 64, 'expected_timestamp_writes': ctas * sum(len(r['points']) for r in rows),
                'timestamp_sentinel': '18446744073709551615', 'mapping_sentinel': 4294967295, 'leader_only': True,
                'automatic_runtime_records': False, 'clock': 'globaltimer', 'coverage':
                'scheduled grid barriers and selected service; excludes internal native barriers', 'header_fields':
                {'enable_u32': 0, 'version_u32': 4, 'occurrences_u32': 8, 'ctas_u32':
                 12}, 'profile_addressing': 'global allocation base; do not add CTA scratch stride', 'barrier_points':
                ['after_role_prebar_before_arrival', 'after_successful_acquire',
                 'after_role_postbar'], 'service_point3': 'return from original full drain', 'consumer_point2':
                'final release; before final arithmetic and output store'
            }
        }
        return descriptor
