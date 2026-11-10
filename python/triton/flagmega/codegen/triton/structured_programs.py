# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Lower typed structured control to shared code and physical invocations.

The invocation trace is a backend lowering of constant bounded control, not an
IR rewrite. Address recipes come from the verified physical function ABI; source
expressions are used only by ordinary device codegen, never parsed by runtime.
"""
import json
from math import prod

from triton.flagmega.codegen.triton.function_pool_abi import CALL_RECORD, OFFSET_TABLE, ROW_INDEX, pool_parameter
from triton.flagmega.codegen.triton.function_programs import _constant
from triton.flagmega.codegen.triton.kernel_programs import _describe_program_partition, _entry_item_id
from triton.flagmega.codegen.triton.loop_carry import loop_copy
from triton.flagmega.codegen.triton.package_plan import plain_package_value
from triton.flagmega.codegen.triton.physical_access import emit_triton_scalar_type
from triton.flagmega.codegen.triton.runtime_binding import (
    DEVICE_CONTEXT_ARGUMENTS, DEVICE_CONTEXT_HANDLES, DEVICE_CONTEXT_MODE, describe_function_runtime_binding,
)
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import For, synchronization_scope_name
from triton.flagmega.ir.program_execution import CallInvocation, LoopInvocation, ProgramInvocation, verify_program_execution
from triton.flagmega.ir.tir.visitor import iter_tir_children

KERNEL_PROGRAM_STRUCTURED_SCHEMA = 'flagmega.kernel-programs/v6-structured'


def _walk(statement):
    yield statement
    for child in iter_tir_children(statement):
        yield from _walk(child)


def _pointer(base, dtype, offset=0, stride=0):
    if base['kind'] not in {'argument', 'pointer'}:
        raise CodegenError('Structured pointer bindings require a physical root allocation.')
    previous = base.get('scope_stride_bytes', 0)
    if previous and stride and previous != stride:
        raise CodegenError('Structured frames have incompatible physical scope strides.')
    return dict(kind='pointer', source=base['source'], dtype=dtype,
                byte_offset=base.get('byte_offset', 0)+offset, scope_stride_bytes=previous or stride)


def _compose(binding, environment):
    if binding['kind'] == 'immediate':
        return dict(binding)
    base = environment[binding['source']]
    if binding['kind'] == 'argument':
        return dict(base)
    if binding['kind'] == 'pointer':
        return _pointer(base, binding['dtype'], binding['byte_offset'], binding['scope_stride_bytes'])
    if binding['kind'] == 'scalar_argument':
        if base['kind'] == 'immediate':
            return {**base, 'value': base['value']+binding['add'] if binding['add'] else base['value']}
        return dict(kind='scalar_argument', source=base['source'], dtype=binding['dtype'],
                    add=base.get('add', 0)+binding['add'])
    raise CodegenError('Unknown typed structured argument binding.')


class _StructuredPrograms:
    def __init__(self, module, descriptor):
        self.module, self.descriptor = module, descriptor
        self.plan = verify_program_execution(module)
        self.functions = {f.function: f for f in self.plan.functions}
        self.bindings = {name: describe_function_runtime_binding(module, function_name=name)
                         for name in self.functions}
        self.definitions = {d['function']: d for d in descriptor['device_functions']}
        self.programs, self.launches = [], []
        self.templates, self.variants, self.utilities = {}, {}, {}
        self.symbols = {p['symbol'] for p in module.metadata['kernel_program_partition']}
        self.controls, self.program_controls, self.trailing_controls = {}, {}, {}
        self.loops = {name: {v.call_id: v for v in _walk(module.execution_function_map[name].body)
                             if isinstance(v, For)} for name in self.functions}
        self.dynamic = set(descriptor['dynamic_argument_indices'])
        self.root_environment = {name: dict(kind='argument', source=i)
                                 for i, name in enumerate(descriptor['signature_arguments'])}

    def fresh_symbol(self, prefix):
        index = 0
        while f'{prefix}{index}' in self.symbols:
            index += 1
        symbol = f'{prefix}{index}'
        self.symbols.add(symbol)
        return symbol

    def function_descriptor(self, name):
        if name == self.module.entry:
            return self.descriptor
        definition = self.definitions[name]
        names = []
        for value in definition['signature_arguments']:
            context = DEVICE_CONTEXT_ARGUMENTS if self.descriptor['device_context_abi'] == 'runtime-handles/v2' else DEVICE_CONTEXT_HANDLES
            names.extend(context if value == 'device_dptr' else (value,))
        scalars = {v['name']: v['scalar_dtype'] for v in self.bindings[name]['arguments']
                   if v.get('runtime_value_kind') == 'scalar' and v['name'] in names}
        scalars.update({key: 'int64' for key in (CALL_RECORD, ROW_INDEX, *DEVICE_CONTEXT_HANDLES) if key in names})
        typed = [f'{value}: {emit_triton_scalar_type(scalars[value])}' if value in scalars else
                 f'{value}: tl.constexpr' if value == DEVICE_CONTEXT_MODE else value for value in names]
        schedule = definition['schedule']
        return {**self.descriptor, 'signature_arguments': names, 'typed_signature_arguments': typed,
                'signature': ',\n    '.join(typed), 'scalar_arguments': list(scalars),
                'runtime_binding': self.bindings[name], 'entry_events': schedule['consumer_events'],
                'pipeline_schedule': schedule if schedule['stages'] else None}

    def prepare_definitions(self, requests):
        for name in self.functions:
            local = [plain_package_value(p) for p in requests if p['function'] == name]
            descriptor = self.function_descriptor(name)
            owned = {value for p in local for value in p['calls']}
            dynamic_entries = {p['calls'][0]: p['symbol'] for p in local
                               if p.get('execution_mesh', {}).get('kind') == 'dynamic'}
            projected_events, pending = [], []
            for event in descriptor['entry_events']:
                if event['kind'] in {'barrier', 'drain', 'handoff', 'pipe_close'}:
                    pending.append(event)
                    continue
                identity = _entry_item_id(event, name)
                if identity in owned:
                    # A first-call rendezvous is an incoming boundary, rather
                    # than part of the independent dynamic task algorithm.
                    # Preserve true peer completion as its own device entry.
                    if identity in dynamic_entries and all(e['kind'] == 'barrier' for e in pending):
                        self.program_controls[name, dynamic_entries[identity]] = tuple(pending)
                        pending = []
                    projected_events.extend((*pending, event))
                else:
                    self.controls[name, identity] = tuple(pending)
                pending = []
            self.trailing_controls[name] = tuple(pending)
            if not local:
                continue
            if pending and not (local[-1]['symbol'] in dynamic_entries.values()
                                and all(e['kind'] == 'barrier' for e in pending)):
                projected_events.extend(pending)
                self.trailing_controls[name] = ()
            projected = _describe_program_partition(self.module, local,
                {**descriptor, 'entry_events': projected_events}, function_name=name,
                complete_scope=not any(n == name for n, _ in self.controls))
            for program, launch in zip(projected['programs'], projected['kernels'], strict=True):
                self.templates[program['symbol']] = (name, {
                    **program, **{key: descriptor[key] for key in (
                        'signature_arguments', 'typed_signature_arguments', 'signature', 'scalar_arguments')}}, launch)

    def edge_value(self, edge, environment, induction):
        kind, abi = edge['actual_runtime_value_kind'], edge['actual_abi']
        if edge['actual'] in induction:
            return dict(kind='immediate', dtype=edge['formal_abi']['scalar_dtype'], value=induction[edge['actual']])
        if kind == 'immediate':
            return dict(kind='immediate', dtype=edge['formal_abi']['scalar_dtype'],
                        value=_constant(self.module, edge['actual']))
        base = environment[edge['actual_runtime_argument']]
        if kind == 'scalar':
            return _compose(dict(kind='scalar_argument', source=edge['actual_runtime_argument'],
                                 dtype=edge['formal_abi']['scalar_dtype'], add=0), environment)
        if 'view_byte_offset' in abi:
            raise CodegenError('Structured entries require a typed root-addressable view offset.')
        return _pointer(base, abi['scalar_dtype'], int(abi.get('pool_byte_offset', 0)),
                        int(abi.get('pool_scope_stride_bytes', 0)))

    def call_environment(self, caller, call, environment, induction):
        parent, callee = self.bindings[caller], self.bindings[call.callee]
        physical = next(e for e in parent['call_abi']['events']
                        if e['kind'] == 'function_call' and e['call'] == call.call)
        edges = {e['formal']: e for e in (*physical['arguments'], *physical['results'])}
        actuals = {argument['name']: self.edge_value(edges[argument['buffer']], environment, induction)
                   for argument in callee['arguments']}
        pools = {p['storage']: p for p in parent['pools']}
        frames = {f['memory_space']: f for f in physical['memory_pools']}
        for pool in callee['pools']:
            frame = frames.get(pool['storage'])
            parent_pool = pools[pool['storage']]
            stride = int(parent_pool.get('scope_nbytes', 0)) if int(parent_pool.get('scope_count', 1)) > 1 else 0
            actuals[pool['name']] = _pointer(environment[parent_pool['name']], 'uint8',
                                            int(frame['offset']) if frame else 0, stride)
        definition = self.definitions[call.callee]
        abi = definition['function_pool_abi']
        if abi is not None:
            for pool in abi['pools']:
                actuals[pool_parameter(pool['storage'])] = dict(environment[pools[pool['storage']]['name']])
            if abi['record_width']:
                row = next(r for r in abi['calls'] if r['caller'] == caller and r['call'] == call.call)
                actuals[CALL_RECORD] = dict(kind='immediate', dtype='int64', value=row['record_offset'])
        if ROW_INDEX in definition['signature_arguments']:
            if ROW_INDEX in environment:
                actuals[ROW_INDEX] = dict(environment[ROW_INDEX])
            elif induction:
                actuals[ROW_INDEX] = dict(kind='immediate', dtype='int64', value=next(iter(induction.values())))
            else:
                raise CodegenError('A structured parameter-bank call requires its typed loop induction value.')
        for name in (OFFSET_TABLE, *DEVICE_CONTEXT_ARGUMENTS):
            if name in environment:
                actuals[name] = dict(environment[name])
        # Descriptors remain actual typed resource projections, rather than
        # parsed function_call.argument_expressions.
        def events(values):
            for event in values:
                yield event
                if event['kind'] == 'loop':
                    yield from events(event['events'])
        source_events = self.function_descriptor(caller)['entry_events']
        event = next(e for e in events(source_events) if e['kind'] == 'function_call' and e['call'] == call.call)
        actuals.update({key: _compose(value, environment)
                        for key, value in event['descriptor_argument_bindings'].items()})
        return actuals

    def invoke(self, name, program, launch, environment, *, kind='program', transfer=None, path=()):
        bindings = [dict(environment[value]) for value in program['signature_arguments']]
        scoped = tuple((value, b['scope_stride_bytes'], b['dtype'])
                       for value, b in zip(program['signature_arguments'], bindings, strict=True)
                       if name != self.module.entry and b['kind'] == 'pointer' and b['scope_stride_bytes'])
        key = (program['symbol'], scoped)
        symbol = self.variants.get(key)
        if symbol is None:
            index = sum(k[0] == program['symbol'] for k in self.variants)
            symbol = program['symbol'] if not index else self.fresh_symbol(f"{program['symbol']}_abi")
            self.variants[key] = symbol
            self.programs.append({**program, 'symbol': symbol, 'owning_function': name,
                'scoped_entry_arguments': [dict(name=v, stride=s, dtype=emit_triton_scalar_type(d), scalar_dtype=d) for v, s, d in scoped]})
        dynamic = [i for i, binding in enumerate(bindings)
                   if binding['kind'] != 'immediate' and binding['source'] in self.dynamic]
        self.launches.append({**launch, 'symbol': self.fresh_symbol('flagmega_invocation_'),
            'entry_symbol': symbol, 'argument_bindings': bindings, 'dynamic_argument_indices': dynamic,
            'invocation_kind': kind, 'call_path': list(path),
            **({'transfer': list(transfer)} if transfer else {})})

    def copy(self, name, carry, source_field, result_field, environment, path):
        from .tir_package import _mesh_context
        buffers = self.bindings[name]['call_abi']['loop_buffers']
        source, result = (buffers[getattr(carry, field)] for field in (source_field, result_field))
        actuals = {}
        formal = []
        for parameter, value in zip(('_fm_loop_src', '_fm_loop_dst'), (source, result), strict=True):
            edge = {'actual': value['actual'], 'actual_abi': value['abi'], 'formal_abi': value['abi'],
                    'actual_runtime_argument': value['runtime_argument'],
                    'actual_runtime_value_kind': value['runtime_value_kind']}
            actuals[parameter] = self.edge_value(edge, environment, {})
            abi = dict(value['abi'])
            for key in ('view_byte_offset', 'pool_scope_stride_bytes', 'pool_scope_index'):
                if key == 'view_byte_offset' and key in abi:
                    raise CodegenError('Structured loop copies require a physical root-addressable view.')
                abi.pop(key, None)
            abi.update(pool_byte_offset=0, pool_scope_count=1)
            formal.append({**value, 'abi': abi, 'runtime_argument': parameter,
                           'runtime_value_kind': 'scalar' if value['runtime_value_kind'] == 'immediate' else value['runtime_value_kind']})
        event = loop_copy(*formal)
        distributed = result['abi'].get('distributed_type')
        placement = distributed['placement'] if distributed else {'hierarchy': [1], 'name': 'x', 'hierarchy_levels': 'b'}
        mesh = {**_mesh_context(placement), 'distributed_entry': True, 'use_tle': True, 'dynamic_execution': False}
        key = json.dumps([event, mesh], sort_keys=True)
        if key not in self.utilities:
            symbol = self.fresh_symbol('flagmega_loop_transfer_')
            names = ['_fm_loop_src', '_fm_loop_dst']
            dtype = result['abi']['scalar_dtype']
            scalars = ['_fm_loop_src'] if formal[0]['runtime_value_kind'] in {'scalar', 'immediate'} else []
            typed = [f'{v}: {emit_triton_scalar_type(dtype)}' if v in scalars else v for v in names]
            for v in DEVICE_CONTEXT_ARGUMENTS:
                if v in self.root_environment and mesh.get('mesh_requires_device_dptr'):
                    names.append(v)
                    typed.append(f'{v}: tl.constexpr' if v == DEVICE_CONTEXT_MODE else f'{v}: tl.int64')
                    if v != DEVICE_CONTEXT_MODE:
                        scalars.append(v)
            program = dict(symbol=symbol, num_warps=4, entry_events=[event], render_calls=[], device_functions=[],
                           pipeline_schedule=None, mesh_context=mesh, signature_arguments=names,
                           typed_signature_arguments=typed, signature=',\n    '.join(typed), scalar_arguments=scalars,
                           device_context_abi=self.descriptor['device_context_abi']
                           if mesh.get('mesh_requires_device_dptr') else None)
            grid = [prod(v for v, level in zip(placement['hierarchy'], placement['hierarchy_levels'], strict=True)
                         if level == 'b'), 1, 1]
            launch = dict(symbol=symbol, num_warps=4, grid=grid, requires_grid_wide_sync=False, calls=[])
            self.utilities[key] = (program, launch)
        program, launch = self.utilities[key]
        actuals.update({v: environment[v] for v in program['signature_arguments'] if v in environment})
        self.invoke(None, program, launch, actuals, kind='loop_copy', transfer=(source_field, result_field), path=path)

    def device_completion(self, environment, path, event):
        """A true peer rendezvous cannot be replaced by one chip's stream."""
        from .tir_package import _mesh_context
        key = json.dumps(['device_completion', event], sort_keys=True)
        if key not in self.utilities:
            names = [v for v in DEVICE_CONTEXT_ARGUMENTS if v in self.root_environment]
            typed = [f'{v}: tl.constexpr' if v == DEVICE_CONTEXT_MODE else f'{v}: tl.int64' for v in names]
            placement = self.module.metadata['launch_contract']['grid_mesh']
            mesh = {**_mesh_context(placement), 'distributed_entry': True, 'use_tle': True,
                    'dynamic_execution': False}
            program = dict(symbol=self.fresh_symbol('flagmega_structured_device_completion_'), num_warps=4,
                           entry_events=[dict(event)], render_calls=[], device_functions=[],
                           pipeline_schedule=None, mesh_context=mesh, signature_arguments=names,
                           typed_signature_arguments=typed, signature=',\n    '.join(typed),
                           scalar_arguments=[v for v in names if v != DEVICE_CONTEXT_MODE])
            grid = [prod(v for v, level in zip(placement['hierarchy'], placement['hierarchy_levels'], strict=True)
                         if level == 'b'), 1, 1]
            launch = dict(symbol=program['symbol'], num_warps=4, grid=grid,
                          requires_grid_wide_sync=True, calls=[])
            self.utilities[key] = program, launch
        program, launch = self.utilities[key]
        self.invoke(self.module.entry, program, launch, environment, kind='device_completion', path=path)

    def control(self, events, environment, path):
        for event in events:
            if event['kind'] != 'barrier':
                raise CodegenError('Structured kernel boundaries require a complete pipe/handoff lifetime ABI.')
            if event['scope'] == 'device':
                self.device_completion(environment, path, event)
            elif event['scope'] not in {'block', 'grid'}:
                raise CodegenError('Unsupported structured boundary synchronization scope.')
            # Every preceding launch completes on the same caller stream.
            # This orders chip/global writes without an in-kernel grid wait.

    def execute(self, name, environment, path=()):
        for step in self.functions[name].steps:
            if isinstance(step, ProgramInvocation):
                _, program, launch = self.templates[step.symbol]
                self.control(self.program_controls.get((name, step.symbol), ()), environment,
                             (*path, step.symbol, 'incoming'))
                self.invoke(name, program, launch, environment, path=path)
            elif isinstance(step, CallInvocation):
                self.control(self.controls.get((name, step.call), ()), environment, path)
                self.execute(step.callee, self.call_environment(name, step, environment, {}), (*path, step.call))
            elif isinstance(step, LoopInvocation):
                loop = self.loops[name][step.call]
                self.control(self.controls.get((name, f'{name}.{loop.loop_var.name}'), ()), environment, path)
                carry_controls = (dict(kind='barrier', scope=synchronization_scope_name(loop.carry_scope)),) if loop.carry_bindings else ()
                for carry in loop.carry_bindings:
                    self.copy(name, carry, 'initial', 'current', environment, (*path, step.call, 'initial'))
                self.control(carry_controls, environment, (*path, step.call, 'initialized'))
                for iteration in range(step.start, step.stop, step.step):
                    induction = {loop.loop_var.name: iteration}
                    # Physical argument edges use the body induction buffer id.
                    induction[self.module.function_map[step.callee].parameters[0]] = iteration
                    self.execute(step.callee, self.call_environment(name, step, environment, induction),
                                 (*path, step.call, str(iteration)))
                    self.control(carry_controls, environment, (*path, step.call, str(iteration), 'yield_ready'))
                    for carry in loop.carry_bindings:
                        self.copy(name, carry, 'yielded', 'staging', environment, (*path, step.call, str(iteration), 'stage'))
                    self.control(carry_controls, environment, (*path, step.call, str(iteration), 'staged'))
                    for carry in loop.carry_bindings:
                        self.copy(name, carry, 'staging', 'current', environment, (*path, step.call, str(iteration), 'commit'))
                    self.control(carry_controls, environment, (*path, step.call, str(iteration), 'committed'))
                for carry in loop.carry_bindings:
                    self.copy(name, carry, 'current', 'result', environment, (*path, step.call, 'result'))
                self.control(carry_controls, environment, (*path, step.call, 'published'))
        self.control(self.trailing_controls[name], environment, path)


def describe_structured_programs(module, plan, descriptor):
    requests = plan.get('programs')
    if plain_package_value(requests) != plain_package_value(module.metadata['kernel_program_partition']):
        raise CodegenError('Structured program plan differs from its typed partition.')
    builder = _StructuredPrograms(module, descriptor)
    builder.prepare_definitions(requests)
    builder.execute(module.entry, builder.root_environment)
    return {'kernel_program_schema': KERNEL_PROGRAM_STRUCTURED_SCHEMA,
            'program_execution': plain_package_value(module.metadata['kernel_program_execution']),
            'kernels': builder.launches, 'programs': builder.programs}
