# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Project the model's coarse physical call into three serial native roles.

No source parsing and no standalone roots: bufferized caller parameters carry
readonly rows, selected state views and planned workspace frames. The ordinary
structured ABI resolves those parameters at each of the 48 model calls.
"""
from math import prod
from dataclasses import replace
from collections.abc import Mapping

from triton.flagmega.errors import CodegenError
from triton.flagmega.codegen.triton.package_plan import plain_package_value
from triton.flagmega.rules.ntt.r004_native_model import (
    native_model_contract, native_model_request, model_phase_symbol,
)
from triton.flagmega.rules.ntt.r004_gdn_phase_bindings import _same_contract
from triton.flagmega.rules.ntt.r004_gdn_phase_bindings import phase_writer_proof

PROGRAM_SCHEMA_R031 = "flagmega.kernel-programs/v11-native-model-r031"

PROGRAM_SCHEMA_V6 = "flagmega.kernel-programs/v10-native-model-v6"
PROGRAM_SCHEMA = 'flagmega.kernel-programs/v9-native-model'


def encode_model_call(raw):
    from .kernel_call_renderers import _buffer, _pointer, _call_runtime_arguments
    from .physical_access import emit_triton_scalar_type
    from .kernels.gdn_recurrent.r004_contract_v4 import native_arguments
    parameters = raw.get('parameters', {})
    request = parameters.get('native_model_contract')
    versions = {'r004.gdn.native-model/v1': 4, 'r004.gdn.native-model/v2': 6, 'r031.gdn.native-model/v1':31}
    if not isinstance(request, Mapping) or request.get('schema') not in versions:
        raise CodegenError('Invalid native model coarse GDN contract identity.')
    _rows = int(request.get('rows', 1) or 1)
    contract = native_model_contract(versions[request['schema']], rows=_rows)
    from triton.flagmega.rules.ntt.r004_gdn_phase_bindings import phase_plan_for
    plan = phase_plan_for(contract.get("phase_module", "r004_T_full_p5_v6" if contract["schema"] == "r004.gdn.native-model/v2" else "r004_T_full_p5_v4"),
                          rows=int(contract.get('rows', 1) or 1))
    if (not _same_contract(parameters.get('native_model_contract'), contract)
            or not _same_contract(parameters.get('phase_plan'), plan)
            or raw.get('implementation') != plan['implementation']
            or raw.get('variant') != plan['implementation'].rsplit('.', 1)[-1]
            or raw.get('facts', {}).get('internal_grid_barriers') != 0):
        raise CodegenError('Invalid native model coarse GDN definition.')
    from .kernel_call_renderers import _r004_gdn_phase_call
    # Share the frozen arithmetic and typed input inventory validator without
    # presenting the model itself as a standalone module or constructor.
    _r004_gdn_phase_call({**raw, 'arguments':list(native_arguments(plan)['prep']),
                         'parameters':{**{k:v for k,v in parameters.items() if k != 'native_model_contract'},
                                       'r004_phase':'prep', 'program_grid':next(e['grid'] for e in plan['entries'] if e['phase']=='prep'),
                                       'program_owner_mapping':'x-fast'}})
    roots = {}
    # A static token bucket widens only the leading activation row axis. Every
    # widened root must agree on that extent; feature extents, dtype and lanes
    # stay exactly as the single-row CHIP inventory.
    token_rows = None
    for name, formal, index in (
        ('QKV','qkv',0), ('X','projection_input',2), ('BW','b_weight',6),
        ('AW','a_weight',7), ('ALog','a_log',8), ('DT','dt_bias',9),
        ('Z','z',1), ('Norm','norm_weight',10),
    ):
        b = _buffer(raw, 'inputs', formal)
        abi = b['abi']
        root = plan['roots'][index]
        shape = list(map(int, abi['logical_shape']))
        lanes = abi.get('scalar_lane_count', 1)
        budget = list(root['shape'])
        if budget and budget[0] == 1 and len(shape) == len(budget):
            token_rows = shape[0] if token_rows is None else token_rows
            matches = (shape[0] == token_rows
                       and prod(shape[1:]) * lanes == prod(budget[1:]))
        else:
            matches = prod(shape) * lanes == prod(budget)
        if (abi['scalar_dtype'] != root['dtype'] or not matches
                or int(abi.get('pool_scope_stride_bytes', 0))):
            raise CodegenError('Native model input must have the complete typed CHIP inventory.')
        roots[name] = _pointer(b)
    state = _buffer(raw, 'inputs', 'state', 1)
    abi = state['abi']
    if (abi['scalar_dtype'] != 'float32' or list(abi['logical_shape']) != [1,48,128,32]
            or list(abi.get('scalar_lane_shape', ())) != [4]
            or 'view_byte_offset' not in abi or int(abi.get('pool_scope_stride_bytes', 0))):
        raise CodegenError('Native model requires the input-owned selected F32 recurrent layer view.')
    roots['StateParent'] = _pointer(state)
    # The caller's typed state View already rebases the full parent. Passing
    # zero to the unchanged algorithm prevents a second Layer displacement.
    roots['Layer'] = 'tl.full((), 0, tl.int32)'
    y = _buffer(raw, 'outputs', 'result_0')
    y_shape = list(map(int, y['abi']['logical_shape']))
    if (y['abi']['scalar_dtype'] != 'bfloat16' or len(y_shape) != 2 or y_shape[0] != (token_rows or 1)
            or y_shape[1] != 6144 or int(y['abi'].get('pool_scope_stride_bytes', 0))):
        raise CodegenError('Native model Y must publish the original complete BF16 result.')
    roots['Y'] = _pointer(y)
    arena = _buffer(raw, 'workspaces', 'phase_pool')
    abi = arena['abi']
    if (abi['scalar_dtype'] != 'uint8' or list(abi['logical_shape']) != [plan["pool"]["nbytes"]]
            or abi['storage'] != 'workspace' or int(abi.get('pool_scope_stride_bytes', 0))
            or int(abi.get('pool_byte_offset', 0)) % 256):
        raise CodegenError('Native model arena must be a 256-aligned planned CHIP workspace.')
    pool = _pointer(arena)
    # Row stride constants for the batch-axis row addressing. Element strides
    # mirror the single-row inventory; the pool stride is the whole arena row.
    _rows = int(contract.get('rows', 1) or 1)
    _pool_row = plan['pool']['nbytes'] // _rows if _rows > 1 else 0
    def _row_constexprs(phase):
        if _rows == 1:
            return []
        base = phase.split('_r')[0]
        if base == 'partials':
            return [f'X_ROW={5120}', f'POOL_ROW={_pool_row}']
        if base == 'prep':
            return [f'QKV_ROW={10240}', f'X_ROW={5120}', f'POOL_ROW={_pool_row}']
        if base == 'core':
            # The recurrence consumes the token rows sequentially inside one
            # CTA, so it needs the row count as well as the per-row strides.
            return [f'QKV_ROW={10240}', f'POOL_ROW={_pool_row}', f'ROWS={_rows}']
        if base == 'y':
            return [f'Z_ROW={6144}', f'Y_ROW={6144}', f'POOL_ROW={_pool_row}']
        return []
    aliases = {'qn':'QN', 'kn':'KN', 'beta':'Beta', 'decay_log':'G', 'core':'Core', 'partials':'Partials'}
    for view in plan['pool']['views']:
        roots[aliases[view['name']]] = f'(({pool}) + {view["byte_offset"]}).to(tl.pointer_type(tl.float32))'
    return dict(native_model_coarse=True, native_model_contract=contract,
                model_phase_arguments={phase:[roots[n] for n in names] + _row_constexprs(phase)
                                       for phase,names in native_arguments(plan).items()},
                arguments=_call_runtime_arguments(raw),
                signature=', '.join(_call_runtime_arguments(raw)))


def project_model_programs(module, descriptor):
    """Preserve ordinary structured calls and replace each isolated P5 program.

    The coarse graph effect/lifetime stays authoritative. Expansion is a native
    implementation of that effect, with whole-program completion at each edge.
    """
    from .structured_programs import KERNEL_PROGRAM_STRUCTURED_SCHEMA
    from .tir_package import _local_mesh
    contract = native_model_request(module)
    if contract is None:
        return descriptor
    if descriptor.get('kernel_program_schema') != KERNEL_PROGRAM_STRUCTURED_SCHEMA:
        raise CodegenError('Native model requires closed structured caller bindings.')
    is_v6 = contract["schema"] == "r004.gdn.native-model/v2"
    from triton.flagmega.rules.ntt.r004_gdn_phase_bindings import phase_plan_for
    plan = phase_plan_for(contract.get("phase_module", "r004_T_full_p5_v6" if is_v6 else "r004_T_full_p5_v4"),
                          rows=int(contract.get('rows', 1) or 1))
    from triton.flagmega.passes.tir.materialize_kernel_definitions import r004_phase_definitions
    coarse = [d for d in module.kernel_callable_map.values()
              if d.dispatch.parameters.get('native_model_contract') is not None
              and d.dispatch.parameters.get('r004_phase') is None]
    if len(coarse) != 1:
        raise CodegenError('Native model requires one reusable coarse GDN definition.')
    for expected in r004_phase_definitions(coarse[0]):
        actual = module.kernel_callable_map.get(expected.name)
        if (actual is None or actual.dispatch != expected.dispatch or actual.attrs != expected.attrs
                or tuple(replace(p,buffers=(),alignment_bytes=None) for p in actual.parameters) != expected.parameters
                or [(r.storage,r.value.value_type) for r in actual.results.values]
                != [(r.storage,r.value.value_type) for r in expected.results.values]):
            raise CodegenError('Native model typed phase definitions are stale or missing.')
        for parameter in actual.parameters:
            if len(parameter.buffers) != 1:
                raise CodegenError('Native model phase formal must have one typed ABI buffer.')
            buffer = parameter.buffers[0]
            count = prod(d.fixed_value for d in parameter.type.shape)
            width = {'uint8':1,'bfloat16':2,'float32':4,'int32':4}[parameter.type.dtype.value]
            if (buffer.elem_type != parameter.type.dtype
                    or [d.fixed_value for d in buffer.dimensions] != [d.fixed_value for d in parameter.type.shape]
                    or buffer.mem_span.size.fixed_value != count*width):
                raise CodegenError('Native model phase ABI buffer differs from its declared typed span.')
    programs, replacements = [], {}
    for program in descriptor['programs']:
        selected = [c for c in program['render_calls'] if c.get('native_model_coarse')]
        if not selected:
            programs.append(program)
            continue
        if (len(selected) != 1 or len(program['render_calls']) != 1 or program.get('pipeline_schedule')
                or program.get('device_functions')):
            raise CodegenError('Native model P5 must own an isolated CHIP program.')
        call = selected[0]
        events = program['entry_events']
        if (len(events) != 1 or events[0]['kind'] != 'tir.kernel_call'
                or events[0].get('barrier_before')):
            raise CodegenError('Native model P5 incoming completion must precede its isolated program.')
        replacement = []
        for entry in plan['entries']:
            phase = entry['phase']
            symbol = program['symbol'] + '_' + phase
            native_call = {**call, 'symbol':call['symbol']+'_'+phase,
                           'native_model_coarse':False, 'phase':phase,
                           'model_native_arguments':call['model_phase_arguments'][phase]}
            event = {**events[0], 'symbol':native_call['symbol'], 'call':call['call']+'.'+phase}
            native_call['call'] = event['call']
            programs.append({**program, 'symbol':symbol, 'render_calls':[native_call],
                             'entry_events':[event], 'num_warps':4, 'maxnreg':80,
                             'enable_static_shared':entry['enable_static_shared'],
                             'scoped_entry_arguments':[],
                             'mesh_context':{**_local_mesh(), 'use_tle':True, 'distributed_entry':False},
                             'model_native_definition':model_phase_symbol(phase, contract)})
            replacement.append((symbol, entry))
        replacements[program['symbol']] = replacement
    launches, calls = [], []
    for launch in descriptor['kernels']:
        entries = replacements.get(launch['entry_symbol'])
        if entries is None:
            launches.append(launch)
            continue
        previous = None
        calls.append(dict(call_path=launch['call_path'], caller_bindings=launch['argument_bindings']))
        for symbol, entry in entries:
            invocation = launch['symbol'] + '_' + entry['phase']
            launches.append({**launch, 'symbol':invocation, 'entry_symbol':symbol,
                             'grid':entry['grid'], 'num_warps':4, 'maxnreg':80,
                             'enable_static_shared':entry['enable_static_shared'],
                             'requires_grid_wide_sync':False,
                             'program_kind':'ordinary', 'execution_mesh':entry['execution_mesh'],
                             'calls':[c+'.'+entry['phase'] for c in launch['calls']],
                             'depends_on':[previous] if previous else [], 'stream_index':0,
                             'model_native_definition':model_phase_symbol(entry['phase'], contract),
                             'scalar_writer_proof':phase_writer_proof(entry['phase'], plan)})
            previous = invocation
    if len(calls) != 48:
        raise CodegenError('Native model package must bind exactly 48 coarse calls / 144 phase launches.')
    return {**descriptor, 'programs':programs, 'kernels':launches,
            'kernel_program_schema':PROGRAM_SCHEMA_R031 if contract['schema']=='r031.gdn.native-model/v1' else PROGRAM_SCHEMA_V6 if is_v6 else PROGRAM_SCHEMA,
            'native_model_binding':dict(contract=plain_package_value(contract), calls=calls,
                                        native_definitions=[model_phase_symbol(e['phase'], contract) for e in plan['entries']]),
            'kernel_templates':[(('kernels/gdn_recurrent/r031_model_r017.py.jinja' if contract['schema']=='r031.gdn.native-model/v1' else 'kernels/gdn_recurrent/r004_model_v2.py.jinja' if is_v6 else 'kernels/gdn_recurrent/r004_model_v1.py.jinja')
                                 if p.endswith('/r017_split4lut.py.jinja' if contract['schema']=='r031.gdn.native-model/v1' else '/r004_column_v6.py.jinja' if is_v6 else '/r004_column_v4.py.jinja') else p)
                                for p in descriptor['kernel_templates']]}

