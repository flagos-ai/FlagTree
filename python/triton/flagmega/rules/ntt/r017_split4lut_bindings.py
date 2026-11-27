# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""H004 standalone policy. No model request or default selection is enabled."""
from copy import deepcopy
from math import prod

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.execution_mesh import ExecutionMesh
from .r004_gdn_phase_bindings import reference_phase_plan, _same_contract, reference_writer_census

MODULE = 'r017_T_full_p5_split4lut'
IMPLEMENTATION = 'tir.gdn_recurrent.r017_split4lut'
PLAN_BODY_SHA256 = '22e4b0546539150a21f61215ee5a01df8e1df5f63f6110d2d0d9182da9a93148'
POOL_BYTES = 56832


def column_phase_plan(rows: int = 1):
    """Single-row plan (rows=1) or a token-bucket plan whose row axis is the
    grid's batch dimension. The ABI already carries that axis: `core` declares
    ['head','batch','column_tile'] and `partials` already uses program_id(1).
    Rows therefore need no new mechanism, and the extents stay node attributes
    so Bufferize sees them."""
    if type(rows) is not int or rows < 1:
        raise IRSchemaError('H004 phase plan rows must be a positive integer.')
    plan = reference_phase_plan()
    plan.update(schema="r017.gdn.split-ordered/v1", module=MODULE, implementation=IMPLEMENTATION,
                plan_body_sha256=PLAN_BODY_SHA256, norm_primitive="r004.norm128.comparator-faithful/v6")
    plan['roots'] = plan['roots'][:12]
    plan['root_dynamic_indices'] = list(range(6))
    plan['roots'][11].update(shape=[POOL_BYTES*rows], nbytes=POOL_BYTES*rows)
    # Five typed global fields; reused only after complete Y publication.
    views = []
    ROW_STRIDE = POOL_BYTES
    for name, offset, shape, writer, reader in (
        ('qn', 0, [16,128], 'prep', 'core'), ('kn', 8192, [16,128], 'prep', 'core'),
        ('beta', 16384, [48], 'prep', 'core'), ('decay_log', 16640, [48], 'prep', 'core'),
        ('core', 16896, [48,128], 'core', 'y'),
    ):
        views.append(dict(name=name, dtype='float32',
                          shape=shape if rows == 1 else [rows, *shape],
                          byte_offset=offset,
                          nbytes=prod(shape)*4, writer=writer, last_reader=reader,
                          **({} if rows == 1 else {'row_stride_bytes': ROW_STRIDE,
                                                   'row_count': rows})))
    root = lambda i: dict(kind='argument', source=i)
    scalar = lambda i: dict(kind='scalar_argument', source=i, add=0)
    view = lambda off: dict(kind='pointer', source=11, dtype='float32', byte_offset=off)
    table = (
        ('prep', [48,rows,1], 4, [root(0),root(2),root(6),root(7),root(8),root(9),
                              view(0),view(8192),view(16384),view(16640)], [0,1]),
        # `core` is the recurrence: it must stay one CTA per (head, column_tile)
        # and consume the token rows sequentially inside the kernel, so its grid
        # keeps the single-row batch extent even when the pool is row-widened.
        # `prep` and `y` are row-independent and keep the parallel batch axis.
        ('core', [48,1,8], 4, [view(0),view(8192),root(0),view(16384),view(16640),
                              root(3),scalar(4),view(16896)], [2,5,6]),
        ('y', [48,rows,1], 4, [view(16896),root(1),root(10),root(5)], [1,3]),
    )
    entries = []
    for phase, grid, warps, bindings, dynamic in table:
        shape, axes = ((grid, ['head','batch','column_tile']) if phase=='core' else
                       (grid, ['batch','vector_tile','row']) if phase=='gather' else
                       (grid[:2], ['head','batch']) if (rows != 1 and phase in ('prep','y')) else
                       ([grid[0]], ['head' if phase in ('prep','y') else 'element_tile']))
        entries.append(dict(phase=phase, symbol='r017_T_'+phase+'_split4lut', grid=grid, num_warps=warps,
                            maxnreg=80, enable_static_shared=phase=='prep',
                            requires_grid_wide_sync=False, program_kind='ordinary', tasks_per_owner=1,
                            depends_on=[entries[-1]['symbol']] if entries else [],
                            argument_bindings=bindings, dynamic_argument_indices=dynamic,
                            condition=None,
                            execution_mesh=ExecutionMesh.fixed(shape,axes=axes,mapping='x-fast').to_data()))
    partial=dict(name='partials',dtype='float32',
                 shape=[2,48,40] if rows == 1 else [rows,2,48,40],
                 byte_offset=41472, nbytes=15360,
                 writer='partials', last_reader='prep',
                 **({} if rows == 1 else {'row_stride_bytes': POOL_BYTES,
                                          'row_count': rows}))
    views.append(partial)
    first=dict(phase='partials',symbol='r017_T_partials_split4lut',
               grid=[48,2,10] if rows == 1 else [48,2*rows,10],num_warps=4,
               maxnreg=80,enable_static_shared=True,requires_grid_wide_sync=False,
               program_kind='ordinary',tasks_per_owner=1,depends_on=[],condition=None,
               argument_bindings=[root(2),root(6),root(7),view(41472)],dynamic_argument_indices=[0],
               execution_mesh=ExecutionMesh.fixed([48,2,10] if rows == 1 else [48,2*rows,10],
                                                  axes=['head','bank','k_tile'],mapping='x-fast').to_data())
    entries[0]['argument_bindings'].append(view(41472))
    entries[0]['depends_on']=[first['symbol']]
    entries.insert(0,first)
    plan.update(pool=dict(root=11,nbytes=POOL_BYTES*rows,scope='chip',pool_scope_stride_bytes=0,views=views),
                entries=entries,
                variants={'T_COMPLETE': dict(entry_indices=[0,1,2,3],input_boundary='ready_inputs',
                                            output_boundary='r004_T_y_v6.complete')},
                scenario_weights={name:dict(positions=steps,layers=48,executed_launches=48*4*steps)
                    for name,steps in [('synthetic32',32),('chinese64',64)]})
    import hashlib,json
    plan['plan_body_sha256']=hashlib.sha256(json.dumps(
        {k:v for k,v in plan.items() if k!='plan_body_sha256'},
        sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return plan


def validate_plan(plan):
    # Rows are recoverable from the batch extent of any widened entry; the IR
    # freezes attribute mappings and may drop auxiliary keys.
    from collections.abc import Mapping as _Mapping
    rows = 1
    if isinstance(plan, _Mapping):
        for entry in plan.get('entries', ()):
            grid = list(entry.get('grid', ()))
            if entry.get('phase') in ('prep', 'core', 'y') and len(grid) > 1:
                rows = max(1, int(grid[1])); break
    expected = column_phase_plan(rows)
    if not _same_contract(plan, expected):
        raise IRSchemaError('H004 phase plan differs from its complete standalone recipe.')
    return deepcopy(expected)


def column_writer_proof(phase):
    sizes = {**reference_writer_census(),'projection_partials':3840}
    writes = {'partials':['projection_partials'], 'prep':['qn','kn','beta','decay_log'], 'core':['state','core'], 'y':['y']}
    if phase not in writes:
        raise IRSchemaError('Unknown H004 phase writer.')
    return dict(schema=column_phase_plan()['schema'],phase=phase,
                scalar_writes={k:sizes[k] for k in writes[phase]},exact_once=True,
                owner_mapping='x-fast',internal_grid_barriers=0,
                norm_owner='4*lane+j; four readonly warp replicas' if phase in ('prep','y') else None,
                norm_store_election='warp==0' if phase in ('prep','y') else None)


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
