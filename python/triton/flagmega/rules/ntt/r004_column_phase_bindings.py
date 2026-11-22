# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""H004 standalone policy. No model request or default selection is enabled."""
from copy import deepcopy
from math import prod

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.execution_mesh import ExecutionMesh
from .r004_gdn_phase_bindings import reference_phase_plan, _same_contract, reference_writer_census

MODULE = 'r004_T_full_p5_v4'
IMPLEMENTATION = 'tir.gdn_recurrent.r004_column_v4'
PLAN_BODY_SHA256 = 'e7789cae00f41a00bc6c829c6cbab09351b445cc03568e3631d91e978a1626ae'
POOL_BYTES = 41472


def column_phase_plan():
    plan = reference_phase_plan()
    plan.update(module=MODULE, implementation=IMPLEMENTATION, plan_body_sha256=PLAN_BODY_SHA256)
    plan['roots'] = plan['roots'][:12]
    plan['root_dynamic_indices'] = list(range(6))
    plan['roots'][11].update(shape=[POOL_BYTES], nbytes=POOL_BYTES)
    # Five typed global fields; reused only after complete Y publication.
    views = []
    for name, offset, shape, writer, reader in (
        ('qn', 0, [16,128], 'prep', 'core'), ('kn', 8192, [16,128], 'prep', 'core'),
        ('beta', 16384, [48], 'prep', 'core'), ('decay_log', 16640, [48], 'prep', 'core'),
        ('core', 16896, [48,128], 'core', 'y'),
    ):
        views.append(dict(name=name, dtype='float32', shape=shape, byte_offset=offset,
                          nbytes=prod(shape)*4, writer=writer, last_reader=reader))
    root = lambda i: dict(kind='argument', source=i)
    scalar = lambda i: dict(kind='scalar_argument', source=i, add=0)
    view = lambda off: dict(kind='pointer', source=11, dtype='float32', byte_offset=off)
    table = (
        ('prep', [48,1,1], 4, [root(0),root(2),root(6),root(7),root(8),root(9),
                              view(0),view(8192),view(16384),view(16640)], [0,1]),
        ('core', [48,1,8], 4, [view(0),view(8192),root(0),view(16384),view(16640),
                              root(3),scalar(4),view(16896)], [2,5,6]),
        ('y', [48,1,1], 4, [view(16896),root(1),root(10),root(5)], [1,3]),
    )
    entries = []
    for phase, grid, warps, bindings, dynamic in table:
        shape, axes = ((grid, ['head','batch','column_tile']) if phase=='core' else
                       (grid, ['batch','vector_tile','row']) if phase=='gather' else
                       ([grid[0]], ['head' if phase in ('prep','y') else 'element_tile']))
        entries.append(dict(phase=phase, symbol='r004_T_'+phase, grid=grid, num_warps=warps,
                            maxnreg=80, enable_static_shared=phase in ('prep','y'),
                            requires_grid_wide_sync=False, program_kind='ordinary', tasks_per_owner=1,
                            depends_on=[entries[-1]['symbol']] if entries else [],
                            argument_bindings=bindings, dynamic_argument_indices=dynamic,
                            condition=None,
                            execution_mesh=ExecutionMesh.fixed(shape,axes=axes,mapping='x-fast').to_data()))
    plan.update(pool=dict(root=11,nbytes=POOL_BYTES,scope='chip',pool_scope_stride_bytes=0,views=views),
                entries=entries,
                variants={'T_COMPLETE': dict(entry_indices=[0,1,2],input_boundary='ready_inputs',
                                            output_boundary='r004_T_y.complete')},
                scenario_weights={name:dict(positions=steps,layers=48,executed_launches=48*3*steps)
                    for name,steps in [('synthetic32',32),('chinese64',64)]})
    return plan


def validate_column_plan(plan):
    expected = column_phase_plan()
    if not _same_contract(plan, expected):
        raise IRSchemaError('H004 phase plan differs from its complete standalone recipe.')
    return deepcopy(expected)


def column_writer_proof(phase):
    sizes = reference_writer_census()
    writes = {'prep':['qn','kn','beta','decay_log'], 'core':['state','core'], 'y':['y']}
    if phase not in writes:
        raise IRSchemaError('Unknown H004 phase writer.')
    return dict(schema=column_phase_plan()['schema'],phase=phase,
                scalar_writes={k:sizes[k] for k in writes[phase]},exact_once=True,
                owner_mapping='x-fast',internal_grid_barriers=0)


def column_tensor_effects(phase):
    plan = column_phase_plan()
    entry = next(e for e in plan['entries'] if e['phase']==phase)
    views = {v['byte_offset']:v for v in plan['pool']['views']}
    accesses = []
    for b in entry['argument_bindings']:
        if b['kind']=='scalar_argument':
            continue
        if b['source']==3:
            accesses.append(dict(source=3,mode='read_write' if phase=='core' else 'read',scope='chip',
                                 shape=[48,128,128],nbytes=3145728,layer_source=4,
                                 layer_stride_bytes=3145728,layer_bounds=[0,48]))
        elif b['kind']=='pointer':
            v = views[b['byte_offset']]
            accesses.append(dict(source=11,mode='write' if v['writer']==phase else 'read',scope='chip',view=v))
        else:
            r = plan['roots'][b['source']]
            accesses.append(dict(source=b['source'],mode='write' if b['source']==5 else 'read',
                                 scope='chip',shape=r['shape'],nbytes=r['nbytes']))
    return dict(phase=phase,accesses=accesses,depends_on=entry['depends_on'],condition=entry['condition'],
                completion='all_ctas_on_same_stream',completion_cta_count=prod(entry['grid']),
                state_effect='initial_clear' if phase=='clear' else 'fused_commit' if phase=='core' else None,
                publishes_region_ready=phase=='y')
