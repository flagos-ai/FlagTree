# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Frozen standalone C phase ABI owned before distribution, not by emission."""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from functools import lru_cache
from math import prod

from triton.flagmega.errors import IRSchemaError, IRVerificationError
from triton.flagmega.ir.execution_mesh import ExecutionMesh

INTERFACE = 'r004.gdn.column-register/v4'
REQUEST_SCHEMA = 'r004.gdn.standalone/v4'
MODULE = 'r004_C_full_p5_v4'
IMPLEMENTATION = 'tir.gdn_recurrent.r004_reference_v4'
PROGRAM_SCHEMA = 'flagmega.kernel-programs/v8-phase-serial'
PLAN_BODY_SHA256 = 'f59c3539a865d2b74ccdb68a33c3e6f7a5e6a3f8c64587f6ab2edb663309f03d'
DELTA_SHA256 = '9aa64e28d3363d7156d17852a6de34d9c47bb48c70eca1eab5254c8868669f4a'
POOL_BYTES = 3212800
DYNAMIC_ROOTS = [0,1,2,3,4,5,12,13]
SCALAR_ROOTS = {4:'int32',13:'int32',14:'float32',15:'float32'}


def reference_phase_plan():
    roots = []
    for name, dtype, shape, role in (
        ('qkv', 'bfloat16', [1, 10240], 'dynamic'), ('z', 'bfloat16', [1, 6144], 'dynamic'),
        ('projection_input', 'bfloat16', [1, 5120], 'dynamic'),
        ('state_parent', 'float32', [48, 48, 128, 128], 'dynamic'),
        ('gdn_layer', 'int32', [], 'dynamic'), ('y', 'bfloat16', [1, 6144], 'dynamic'),
        ('b_weight', 'uint8', [48, 2720], 'setup'), ('a_weight', 'uint8', [48, 2720], 'setup'),
        ('a_log', 'float32', [48], 'setup'), ('dt_bias', 'float32', [48], 'setup'),
        ('norm_weight', 'float32', [128], 'setup'), ('phase_pool', 'uint8', [POOL_BYTES], 'pool'),
        ('state_selector', 'int32', [1,1,1,1], 'dynamic'),
        ('clear_required', 'int32', [], 'dynamic'),
        ('clear_scale', 'float32', [], 'setup'), ('clear_bias', 'float32', [], 'setup'),
    ):
        width = {'bfloat16': 2, 'float32': 4, 'int32': 4, 'uint8': 1}[dtype]
        roots.append(dict(name=name, dtype=dtype, shape=shape, nbytes=prod(shape)*width, role=role))
    roots[12].update(strides_bytes=[4,4,4,4])
    views = [dict(name=name, dtype='float32', shape=shape, byte_offset=offset,
                  nbytes=prod(shape)*4, writer=writer, last_reader=reader)
             for name, offset, shape, writer, reader in (
                 ('bproj', 0, [48], 'pb', 'beta'), ('aproj', 256, [48], 'pa', 'bias'),
                 ('biased', 512, [48], 'bias', 'gate'), ('beta', 768, [48], 'beta', 'core'),
                 ('decay_log', 1024, [48], 'gate', 'core'), ('qn', 1280, [16,128], 'q', 'core'),
                 ('kn', 9472, [16,128], 'k', 'core'), ('core', 17664, [48,128], 'core', 'rms'),
                 ('weighted_core', 42240, [48,128], 'rms', 'y'),
                 ('gathered_state', 67072, [48,128,128], 'gather', 'core'))]
    views[-1].update(physical_dtype='float32',physical_vector_lanes=[4],
                     physical_shape=[1,48,128,32],scalar_strides_bytes=[65536,512,4],
                     release_after={'symbol':'r004_C_core','completion':'all_ctas','cta_count':384},
                     recycle_after='r004_C_y')
    def root(i):
        return dict(kind='argument', source=i)
    def view(offset):
        return dict(kind='pointer', source=11, dtype='float32', byte_offset=offset)
    entries = []
    scalar = lambda source: dict(kind='scalar_argument',source=source,add=0)
    table = (
        ('clear', [3072,1,1], 8, [root(3),scalar(4),scalar(14),scalar(15)], [0,1]),
        ('gather', [1,768,1], 8, [root(3),scalar(4),root(12),view(67072)], [0,1,2]),
        ('q', [16,1,1], 1, [root(0),view(1280)], [0]),
        ('k', [16,1,1], 1, [root(0),view(9472)], [0]),
        ('pa', [48,1,1], 4, [root(2),root(7),view(256)], [0]),
        ('bias', [1,1,1], 1, [view(256),root(9),view(512)], []),
        ('gate', [1,1,1], 8, [view(512),root(8),view(1024)], []),
        ('pb', [48,1,1], 4, [root(2),root(6),view(0)], [0]),
        ('beta', [1,1,1], 8, [view(0),view(768)], []),
        ('core', [48,1,8], 4, [view(1280),view(9472),root(0),view(768),view(1024),view(67072),root(3),
                              scalar(4),view(17664)], [2,6,7]),
        ('rms', [48,1,1], 1, [view(17664),root(10),view(42240)], []),
        ('y', [24,1,1], 8, [view(42240),root(1),root(5)], [1,2]),
    )
    for phase, grid, warps, bindings, dynamic in table:
        shape, axes = (grid, ['head','batch','column_tile']) if phase == 'core' else ((grid,['batch','vector_tile','row']) if phase=='gather' else ([grid[0]], ['head' if phase in ('pb','pa','q','k','rms') else 'element_tile']))
        entries.append(dict(phase=phase, symbol='r004_C_'+phase, grid=grid, num_warps=warps,
                            maxnreg=80, enable_static_shared=phase in ('pb','pa'),
                            requires_grid_wide_sync=False, program_kind='ordinary', tasks_per_owner=1,
                            depends_on=[entries[-1]['symbol']] if entries else [],
                            argument_bindings=bindings, dynamic_argument_indices=dynamic,
                            condition={'source':13,'equals':1} if phase=='clear' else None,
                            execution_mesh=ExecutionMesh.fixed(shape,axes=axes,mapping='x-fast').to_data()))
    return dict(schema=INTERFACE,module=MODULE,implementation=IMPLEMENTATION,
                plan_body_sha256=PLAN_BODY_SHA256,delta_sha256=DELTA_SHA256,
                root_dynamic_indices=DYNAMIC_ROOTS.copy(),roots=roots,
                variants={name:dict(clear_required=flag,entry_indices=list(range(start,12)),
                                   input_boundary='ready_inputs',output_boundary='r004_C_y.complete')
                          for name,flag,start in [('C_INITIAL',1,0),('C_SUBSEQUENT',0,1)]},
                scenario_weights={name:dict(positions=steps,layers=48,initial_per_layer=1,
                    subsequent_per_layer=steps-1,executed_launches=48*(12+11*(steps-1)))
                    for name,steps in [('synthetic32',32),('chinese64',64)]},
                pool=dict(root=11,nbytes=POOL_BYTES,scope='chip',pool_scope_stride_bytes=0,views=views),entries=entries)


def _plain(value):
    if isinstance(value, Mapping):
        return {k:_plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [_plain(v) for v in value]
    return value


def _same_contract(actual, expected):
    actual=_plain(actual)
    expected=_plain(expected)
    if type(actual) is not type(expected):
        return False
    if isinstance(expected,dict):
        return actual.keys()==expected.keys() and all(_same_contract(actual[k],v) for k,v in expected.items())
    if isinstance(expected,list):
        return len(actual)==len(expected) and all(_same_contract(a,b) for a,b in zip(actual,expected))
    return actual==expected


def validate_phase_plan(plan):
    if isinstance(plan,Mapping) and plan.get('module') == 'r017_T_full_p5_split4lut':
        from .r017_split4lut_bindings import validate_plan
        return validate_plan(plan)
    if isinstance(plan,Mapping) and plan.get('module') == 'r017_T_full_p5_split4':
        from .r017_split4_bindings import validate_plan
        return validate_plan(plan)
    if isinstance(plan,Mapping) and plan.get('module') == 'r017_T_full_p5_split8':
        from .r017_split8_bindings import validate_plan
        return validate_plan(plan)
    if isinstance(plan, Mapping) and plan.get('module') == 'r017_T_full_p5_split':
        from .r017_split_bindings import validate_plan
        return validate_plan(plan)
    if isinstance(plan, Mapping) and str(plan.get('module','')).startswith('r017_T_full_p5_w'):
        from .r017_gdn_bindings import validate_plan
        return validate_plan(plan)
    if isinstance(plan, Mapping) and plan.get('module') == 'r004_T_full_p5_v6':
        from .r004_column_phase_bindings_v6 import validate_column_plan
        return validate_column_plan(plan)
    if isinstance(plan, Mapping) and plan.get('module') == 'r004_T_full_p5_v4':
        from .r004_column_phase_bindings import validate_column_plan
        return validate_column_plan(plan)
    """Check bounds/lifetimes before requiring exact signed C offsets/geometry."""
    if not isinstance(plan,Mapping) or plan.get('schema') != INTERFACE:
        raise IRSchemaError('Unsupported GDN phase interface.')
    expected = reference_phase_plan()
    pool = plan.get('pool', {})
    entries = plan.get('entries', ())
    order = {e.get('phase'):i for i,e in enumerate(entries)}
    views = pool.get('views', ())
    for view in views:
        offset, size = view.get('byte_offset'), view.get('nbytes')
        shape = view.get('shape')
        if (type(offset) is not int or type(size) is not int or offset < 0 or size <= 0
                or offset % 256 or offset + size > pool.get('nbytes',0)
                or not isinstance(shape,(tuple,list)) or any(type(d) is not int or d<=0 for d in shape)
                or prod(shape)*4 != size or view.get('dtype') != 'float32'):
            raise IRSchemaError('GDN phase View span/shape is out of bounds or misaligned.')
        if (view.get('writer') not in order or view.get('last_reader') not in order
                or order[view['writer']] >= order[view['last_reader']]):
            raise IRSchemaError('GDN phase View has an invalid lifetime.')
    for i,left in enumerate(views):
        for right in views[i+1:]:
            if (left['byte_offset'] < right['byte_offset']+right['nbytes']
                    and right['byte_offset'] < left['byte_offset']+left['nbytes']):
                raise IRSchemaError('GDN phase Views alias while live.')
    if not _same_contract(plan,expected):
        raise IRSchemaError('GDN phase roots/options/dependencies differ from frozen interface v4.')
    return deepcopy(expected)


def standalone_request(module):
    request = module.metadata.get('r004_standalone_request')
    if request is None:
        return None
    if (not isinstance(request,Mapping) or set(request) != {'schema','module','readonly_layer'}
            or request.get('schema') != REQUEST_SCHEMA or request.get('module') not in (MODULE, 'r004_T_full_p5_v4', 'r004_T_full_p5_v6', 'r017_T_full_p5_w1', 'r017_T_full_p5_w2', 'r017_T_full_p5_w4', 'r017_T_full_p5_w8', 'r017_T_full_p5_w16', 'r017_T_full_p5_w32', 'r017_T_full_p5_w64', 'r017_T_full_p5_split', 'r017_T_full_p5_split4', 'r017_T_full_p5_split8', 'r017_T_full_p5_split4lut')
            or type(request.get('readonly_layer')) is not int or not 0 <= request['readonly_layer'] < 48):
        raise IRVerificationError('Invalid standalone GDN request/readonly row.',stage=module.stage)
    return request


def configure_standalone_phases(module):
    """Install contract on every reachable coarse node at the owning boundary."""
    request = standalone_request(module)
    if request is None:
        if any(n.attrs.get('phase_plan') is not None for n in module.nodes if n.op=='nn.gdn_recurrent_core'):
            raise IRVerificationError('GDN phase plan requires an explicit standalone request.',stage=module.stage)
        return module
    if module.stage not in ('unused_functions_removed','distribution_constants_frozen','distribution_candidates'):
        raise IRVerificationError('GDN phase policy requires full packing before distribution.',stage=module.stage)
    from triton.flagmega.passes.functions.graph import static_node_invocation_counts
    reachable = static_node_invocation_counts(module)
    nodes, census = [], []
    for node in module.nodes:
        if node.op != 'nn.gdn_recurrent_core' or node.id not in reachable:
            nodes.append(node);continue
        validate_coarse_node(module,node)
        plan = validate_phase_plan(node.attrs['phase_plan']) if node.attrs.get('phase_plan') is not None else phase_plan_for(request['module'])
        if plan['module'] != request['module']:
            raise IRVerificationError('Mixed standalone phase module identities.',node_id=node.id)
        nodes.append(replace(node,attrs={**node.attrs,'phase_plan':plan}))
        census.append(dict(node=node.id,implementation=plan['implementation'],interface=plan['schema'],
                           state_effect=node.effect.to_data(),phase_symbols=[e['symbol'] for e in plan['entries']]))
    if not census:
        raise IRVerificationError('Standalone GDN request has no reachable coarse node.',stage=module.stage)
    return replace(module,nodes=tuple(nodes),metadata={**module.metadata,'r004_phase_census':census})


def validate_coarse_node(module,node):
    import struct
    from triton.flagmega.ir.distributed_inference import tensor_of
    dims = dict(num_key_heads=16,num_value_heads=48,key_head_dim=128,value_head_dim=128)
    if any(node.attrs.get(k)!=v for k,v in dims.items()):
        raise IRVerificationError('Standalone C requires the frozen GDN dimensions.',node_id=node.id)
    if (struct.pack('<f',node.attrs.get('epsilon'))!=struct.pack('<f',1e-6) or node.attrs.get('qk_norm_epsilon',1e-12)!=1e-12
            or node.attrs.get('gate_math_mode','accurate')!='accurate'
            or node.attrs.get('norm_math_mode','fast')!='fast'
            or node.attrs.get('b_weight_encoding')!='IQ4_XS' or node.attrs.get('a_weight_encoding')!='IQ4_XS'
            or node.attrs.get('value_head_order','grouped')!='tiled'
            or node.attrs.get('qk_norm_mode','clamp')!='clamp'
            or node.attrs.get('round_normalized_qk',False) or not node.attrs.get('round_beta',True)
            or node.attrs.get('round_core',False)):
        raise IRVerificationError('Standalone C GDN numerical/weight contract differs from v4.',node_id=node.id)
    tokens = module.metadata.get('tokens_per_call', 1)
    if type(tokens) is not int or tokens < 1:
        raise IRVerificationError('Standalone C requires a positive static token extent.',node_id=node.id)
    for i,root in zip(range(1,9),(0,1,2,6,7,8,9,10),strict=True):
        value = tensor_of(module.node_map[node.inputs[i]].type)
        expected = reference_phase_plan()['roots'][root]
        # Packed vector outer geometry remains legal when scalar inventory agrees.
        from triton.flagmega.ir.types import VectorType
        dtype=value.dtype.elem_type if isinstance(value.dtype,VectorType) else value.dtype
        lanes=prod(value.dtype.lanes) if isinstance(value.dtype,VectorType) else 1
        # A static token bucket may scale the leading token axis only; every
        # head/feature dimension of the root keeps its exact v4 inventory, and
        # tokens == 1 reproduces the reference shape byte for byte.
        budget = list(expected['shape'])
        if budget and budget[0] == 1:
            budget[0] = tokens
        if (dtype.value!=expected['dtype'] or any(not d.is_fixed for d in value.shape)
                or prod(d.fixed_value for d in value.shape)*lanes != prod(budget)):
            raise IRVerificationError('Standalone C tensor scalar inventory differs from v4.',node_id=node.id)


@lru_cache(maxsize=1)
def reference_writer_census():
    """Enumerate every core scalar's CTA/thread/register owner, including election."""
    state=bytearray(786432); core=bytearray(6144)
    for z in range(8):
        for h in range(48):
            for warp in range(4):
                for lane in range(32):
                    column=16*z+4*warp+lane//8
                    for register in range(16):
                        key=8*register+lane%8
                        index=(h*128+column)*128+key
                        state[index]+=1
                    if lane%8==0:
                        core[h*128+column]+=1
    if any(v!=1 for v in state) or any(v!=1 for v in core):
        raise IRSchemaError('GDN core scalar writer proof has gaps or duplicate owners.')
    # Each other entry writes its unique scalar coordinate; padded bias lanes excluded.
    bias=[lane+24*r for lane in range(24) for r in range(2)]
    if sorted(bias)!=list(range(48)):
        raise IRSchemaError('GDN bias scalar writer proof failed.')
    def unique(indices, size):
        seen=bytearray(size)
        for index in indices:
            if not 0<=index<size:
                raise IRSchemaError('GDN scalar writer escapes its field.')
            seen[index]+=1
        if any(v!=1 for v in seen):
            raise IRSchemaError('GDN scalar writer has gaps or replicas.')
    unique((h*128+4*lane+r for h in range(16) for lane in range(32) for r in range(4)),2048)
    unique((h*128+4*lane+r for h in range(48) for lane in range(32) for r in range(4)),6144)
    unique((256*cta+32*warp+lane for cta in range(24) for warp in range(8) for lane in range(32)),6144)
    unique((32*warp+lane for warp in range(8) for lane in range(32) if 32*warp+lane<48),48)
    unique((h for h in range(48)),48)  # projection elected thread zero
    unique((4*(256*cta+32*warp+lane)+t for cta in range(768)
            for warp in range(8) for lane in range(32) for t in range(4)),786432)
    unique((256*cta+32*warp+lane for cta in range(3072)
            for warp in range(8) for lane in range(32)),786432)
    return {'state':len(state),'core':len(core),'qn':2048,'kn':2048,'beta':48,'decay_log':48,
            'bproj':48,'aproj':48,'biased':48,'weighted_core':6144,'y':6144,
            'gathered_state':786432,'initial_clear':786432}


def phase_writer_proof(phase, plan=None):
    if plan is not None and plan.get('module') == 'r017_T_full_p5_split4lut':
        from .r017_split4lut_bindings import column_writer_proof
        return column_writer_proof(phase)
    if plan is not None and plan.get('module') == 'r017_T_full_p5_split4':
        from .r017_split4_bindings import column_writer_proof
        return column_writer_proof(phase)
    if plan is not None and plan.get('module') == 'r017_T_full_p5_split8':
        from .r017_split8_bindings import column_writer_proof
        return column_writer_proof(phase)
    if plan is not None and plan.get('module') == 'r017_T_full_p5_split':
        from .r017_split_bindings import column_writer_proof
        return column_writer_proof(phase)
    if plan is not None and str(plan.get('module','')).startswith('r017_T_full_p5_w'):
        from .r017_gdn_bindings import column_writer_proof
        return column_writer_proof(phase, int(plan['module'].rsplit('w',1)[1]))
    if plan is not None and plan.get('module') == 'r004_T_full_p5_v6':
        from .r004_column_phase_bindings_v6 import column_writer_proof
        return column_writer_proof(phase)
    if plan is not None and plan.get('module') == 'r004_T_full_p5_v4':
        from .r004_column_phase_bindings import column_writer_proof
        return column_writer_proof(phase)
    sizes=reference_writer_census()
    writes={'clear':['initial_clear'],'gather':['gathered_state'],
            'pb':['bproj'],'pa':['aproj'],'beta':['beta'],'bias':['biased'],'gate':['decay_log'],
            'q':['qn'],'k':['kn'],'core':['state','core'],'rms':['weighted_core'],'y':['y']}
    if phase not in writes:
        raise IRSchemaError('Unknown GDN phase writer.')
    return dict(schema=INTERFACE,phase=phase,scalar_writes={k:sizes[k] for k in writes[phase]},
                exact_once=True,owner_mapping='x-fast',internal_grid_barriers=0)


def supports_static_shared(capability):
    """Explicit NVIDIA capability, cross-checked against actual option schema."""
    if 'enable_static_shared' in getattr(capability,'supported_program_compile_options',()):
        return True
    # CUDAOptions is shared by NVIDIA physical targets; other backends cannot opt in.
    if not any(base.__module__.startswith('triton.flagmega.targets.nvidia.') for base in type(capability).__mro__):
        return False
    from triton.backends.nvidia.compiler import CUDAOptions
    return 'enable_static_shared' in CUDAOptions.__dataclass_fields__


def verify_scalar_writer_proof(call, grid):
    from triton.flagmega.errors import CodegenError
    parameters=call.get('parameters',{})
    phase=parameters.get('r004_phase')
    proof=call.get('scalar_writer_proof',call.get('facts',{}).get('scalar_writer_proof'))
    try:
        plan=validate_phase_plan(parameters.get('phase_plan'))
        entry=next(e for e in plan['entries'] if e['phase']==phase)
        if list(grid)!=entry['grid'] or _plain(proof)!=phase_writer_proof(phase,plan):
            raise IRSchemaError('Writer geometry/census differs from its phase.')
    except (IRSchemaError,StopIteration) as error:
        raise CodegenError('Custom x-fast program requires the complete per-scalar writer proof.') from error


def phase_runtime_binding(readonly_layer, readonly_provenance, plan=None):
    """Versioned constructor ABI; statics are still external pointer bindings."""
    plan=validate_phase_plan(plan) if plan is not None else reference_phase_plan()
    if type(readonly_layer) is not int or not 0 <= readonly_layer < 48:
        raise IRSchemaError('GDN readonly layer must be an ordinal in [0,48).')
    arguments=[]
    for i,root in enumerate(plan['roots']):
        if i==11:continue
        arguments.append(dict(root,root_index=i,buffer=root['name'],
                              runtime_value_kind='scalar' if i in SCALAR_ROOTS else 'tensor',
                              **({'scalar_dtype':SCALAR_ROOTS[i]} if i in SCALAR_ROOTS else {})))
    return dict(schema=INTERFACE,arguments=arguments,pools=[dict(name='phase_pool',storage='workspace',
                root_index=11,nbytes=plan['pool']['nbytes'],scope='chip',scope_count=1,pool_scope_stride_bytes=0)],
                signature=[r['name'] for r in plan['roots']],root_dynamic_indices=plan['root_dynamic_indices'].copy(),
                **({'setup_scalar_bits':{'clear_scale':'00000000','clear_bias':'00000000'},
                'selector_contract':{'signed_value':0,'nbytes':4,'shape':[1,1,1,1],
                    'source':'kernel_expert/029-r004_selector_payload/selector-spec.md',
                    'upstream_payload_observed':False,'writer':'setup_only'}} if plan['module']==MODULE else {}),
                phase_plan=plan,readonly_layer=readonly_layer,readonly_provenance=readonly_provenance)


def verify_phase_runtime_binding(binding):
    if not isinstance(binding,Mapping) or binding.get('schema')!=INTERFACE:
        raise IRSchemaError('Missing versioned GDN runtime root projection.')
    plan=validate_phase_plan(binding.get('phase_plan'))
    if not _same_contract(binding.get('root_dynamic_indices'),plan['root_dynamic_indices']):
        raise IRSchemaError('GDN Layer must remain dynamic; root projection must be [0..5].')
    args=binding.get('arguments',())
    external=[(i,r) for i,r in enumerate(plan['roots']) if i!=11]
    if len(args)!=len(external) or [a.get('name') for a in args]!=[r['name'] for i,r in external]:
        raise IRSchemaError('GDN root missing or ordered root ABI differs from v4.')
    for actual,(i,root) in zip(args,external,strict=True):
        if any(not _same_contract(actual.get(k),v) for k,v in root.items()):
            raise IRSchemaError('GDN root shape/dtype/role/span differs from v4.')
        if (actual.get('root_index')!=i or actual.get('runtime_value_kind')!=('scalar' if i in SCALAR_ROOTS else 'tensor')
                or (i in SCALAR_ROOTS and actual.get('scalar_dtype')!=SCALAR_ROOTS[i])):
            raise IRSchemaError('GDN Layer must be a dynamic int32 scalar.')
    pools=binding.get('pools',())
    if (len(pools)!=1 or pools[0].get('name')!='phase_pool' or pools[0].get('nbytes')!=plan['pool']['nbytes']
            or pools[0].get('root_index')!=11
            or pools[0].get('scope')!='chip' or pools[0].get('pool_scope_stride_bytes')!=0
            or pools[0].get('scope_count')!=1 or pools[0].get('storage')!='workspace'
            or _plain(binding.get('signature'))!=[r['name'] for r in plan['roots']]):
        raise IRSchemaError('GDN pool must be the bounded runtime-owned CHIP root 11.')
    expected=phase_runtime_binding(binding.get('readonly_layer'),binding.get('readonly_provenance'),plan)
    if not _same_contract(binding,expected):
        raise IRSchemaError('GDN runtime binding differs from the complete v4 root/scalar/selector contract.')
    layer=binding.get('readonly_layer')
    if type(layer) is not int or not 0 <= layer < 48:
        raise IRSchemaError('GDN readonly bank row is missing or out of range.')
    provenance=binding.get('readonly_provenance')
    if not isinstance(provenance,Mapping) or set(provenance)!={r['name'] for r in plan['roots'][6:11]}:
        raise IRSchemaError('GDN readonly root provenance is missing.')
    for root in plan['roots'][6:11]:
        row=provenance[root['name']]
        digest=row.get('sha256') if isinstance(row,Mapping) else None
        if (not isinstance(row,Mapping) or type(row.get('gdn_ordinal')) is not int or row.get('gdn_ordinal')!=layer
                or type(digest) is not str or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest)):
            raise IRSchemaError('GDN readonly row/hash does not match selected layer.')
    return plan


def verify_phase_external_arguments(binding,roots,*,check_payload=False):
    """Explicit field capacity, readonly row, parent rebasing and alias proof."""
    import hashlib
    from triton.flagmega.errors import RuntimeContractError
    try:
        plan=verify_phase_runtime_binding(binding)
    except IRSchemaError as error:
        raise RuntimeContractError(str(error)) from error
    if len(roots)!=len(plan['roots']):
        raise RuntimeContractError('GDN requires fifteen external roots and its runtime pool at root 11.')
    layer=roots[4]
    if type(layer) is not int or not 0 <= layer < 48 or layer!=binding['readonly_layer']:
        raise RuntimeContractError('GDN dynamic layer is outside parent bounds or mismatches readonly row.')
    spans=[]
    for i,(value,root) in enumerate(zip(roots,plan['roots'],strict=True)):
        if i in SCALAR_ROOTS:
            import struct
            if i in (4,13) and type(value) is not int:
                raise RuntimeContractError('GDN Layer/clear request must be native signed I32 scalars.')
            if i==13 and value not in (0,1):
                raise RuntimeContractError('GDN clear request must select exactly one closure variant.')
            if i in (14,15) and (type(value) is not float or struct.pack('<f',value)!=b'\0\0\0\0'):
                raise RuntimeContractError('GDN clear scalars must be real F32 +0 runtime arguments.')
            continue
        if (not callable(getattr(value,'data_ptr',None)) or not callable(getattr(value,'numel',None))
                or not value.is_contiguous() or str(value.dtype).removeprefix('torch.')!=root['dtype']
                or value.numel()*value.element_size()!=root['nbytes']
                or tuple(value.shape) not in (tuple(root['shape']),(prod(root['shape']),))
                or value.data_ptr()%256 or value.device!=roots[11].device):
            raise RuntimeContractError(f'GDN root {root["name"]} span/shape/dtype/alignment differs from v4.')
        spans.append((root['name'],value.data_ptr(),value.data_ptr()+root['nbytes']))
        if i==12 and any(stride!=1 for stride in value.stride()):
            raise RuntimeContractError('GDN device selector requires the exact singleton I32 stride4.')
        if check_payload and i==12:
            if value.detach().reshape(-1).cpu().tolist()!=[binding['selector_contract']['signed_value']]:
                raise RuntimeContractError('GDN selector device payload differs from the adjudicated singleton row.')
        if check_payload and 6<=i<=10:
            import torch
            payload=value.detach().view(torch.uint8).reshape(-1).cpu().numpy().tobytes()
            if hashlib.sha256(payload).hexdigest()!=binding['readonly_provenance'][root['name']]['sha256']:
                raise RuntimeContractError('GDN setup readonly pointer payload mismatches its bank row/hash.')
    for i,(name,lo,hi) in enumerate(spans):
        for other,begin,end in spans[i+1:]:
            if lo<end and begin<hi:
                raise RuntimeContractError(f'GDN roots {name} and {other} have an undeclared alias.')
    # Check each field, not project_arguments' tail. Exact parent size and
    # layer range ensure neither a selected state view nor double rebase is legal.
    for view in plan['pool']['views']:
        if view['byte_offset']+view['nbytes']>roots[11].numel()*roots[11].element_size():
            raise RuntimeContractError('GDN phase field escapes its pool.')
    return plan


def phase_tensor_effects(phase, plan=None):
    if plan is not None and plan.get('module') == 'r017_T_full_p5_split4lut':
        from .r017_split4lut_bindings import column_tensor_effects
        return column_tensor_effects(phase)
    if plan is not None and plan.get('module') == 'r017_T_full_p5_split4':
        from .r017_split4_bindings import column_tensor_effects
        return column_tensor_effects(phase)
    if plan is not None and plan.get('module') == 'r017_T_full_p5_split8':
        from .r017_split8_bindings import column_tensor_effects
        return column_tensor_effects(phase)
    if plan is not None and plan.get('module') == 'r017_T_full_p5_split':
        from .r017_split_bindings import column_tensor_effects
        return column_tensor_effects(phase)
    if plan is not None and str(plan.get('module','')).startswith('r017_T_full_p5_w'):
        from .r017_gdn_bindings import column_tensor_effects
        return column_tensor_effects(phase, int(plan['module'].rsplit('w',1)[1]))
    if plan is not None and plan.get('module') == 'r004_T_full_p5_v6':
        from .r004_column_phase_bindings_v6 import column_tensor_effects
        return column_tensor_effects(phase)
    if plan is not None and plan.get('module') == 'r004_T_full_p5_v4':
        from .r004_column_phase_bindings import column_tensor_effects
        return column_tensor_effects(phase)
    """Cross-entry CHIP effects and selected-layer state interval are explicit."""
    plan=reference_phase_plan()
    entry=next(e for e in plan['entries'] if e['phase']==phase)
    accesses=[]
    # Write positions are part of the ABI, never inferred from argument order.
    writers={'clear':{0},'gather':{3},'core':{6,8},'pb':{2},'pa':{2},'beta':{1},
             'bias':{2},'gate':{2},'q':{1},'k':{1},'rms':{2},'y':{2}}
    for position,binding in enumerate(entry['argument_bindings']):
        if binding['kind']=='scalar_argument':continue
        mode='write' if position in writers[phase] else 'read'
        if binding.get('source')==3:
            mode='read_write' if phase=='clear' else ('write' if phase=='core' else 'read')
            accesses.append(dict(source=3,mode=mode,scope='chip',shape=[48,128,128],nbytes=3145728,
                                 layer_source=4,layer_stride_bytes=3145728,layer_bounds=[0,48]))
        elif binding['kind']=='pointer':
            view=next(v for v in plan['pool']['views'] if v['byte_offset']==binding['byte_offset'])
            accesses.append(dict(source=11,mode=mode,scope='chip',view=view))
        else:
            root=plan['roots'][binding['source']]
            accesses.append(dict(source=binding['source'],mode=mode,scope='chip',shape=root['shape'],nbytes=root['nbytes']))
    return dict(phase=phase,accesses=accesses,depends_on=entry['depends_on'],condition=entry['condition'],
                completion='all_ctas_on_same_stream',completion_cta_count=prod(entry['grid']),
                state_effect='initial_clear' if phase=='clear' else ('fused_commit' if phase=='core' else None),
                publishes_region_ready=phase=='y')


def project_phase_roots(binding, external, pool):
    """Numbered constructor ABI: pool is in the middle of external roots."""
    from triton.flagmega.errors import RuntimeContractError
    if len(external)!=len(binding['arguments']):
        raise RuntimeContractError('GDN prepare requires 15 ordered external roots (pool is internal root11).')
    roots=[None]*len(binding['signature'])
    for descriptor,value in zip(binding['arguments'],external,strict=True):
        roots[descriptor['root_index']]=value
    roots[11]=pool
    return tuple(roots)


def phase_variant(scenario, token_position, clear_required):
    from triton.flagmega.errors import RuntimeContractError
    steps={'synthetic32':32,'chinese64':64}
    if (scenario not in steps or type(token_position) is not int or not 0<=token_position<steps[scenario]
            or type(clear_required) is not int or clear_required!=int(token_position==0)):
        raise RuntimeContractError('GDN clear request must be 1 iff scenario token_position is zero.')
    return 'C_INITIAL' if clear_required else 'C_SUBSEQUENT'


class StandaloneGdnCandidateProvider:
    """Opt-in policy provider, installed per proposal without changing defaults."""
    op_names=frozenset({'nn.gdn_recurrent_core'})

    def propose(self,node,context):
        from triton.flagmega.codegen.triton.candidates.core import TritonCandidateProposal
        if standalone_request(context.module) is None:
            raise IRVerificationError('Standalone GDN provider requires its explicit request.')
        validate_coarse_node(context.module,node)
        plan=validate_phase_plan(node.attrs.get('phase_plan'))
        implementations=[i for i in context.implementations('gdn_recurrent',standalone_schema=REQUEST_SCHEMA,standalone_module=plan['module']) if i.id==plan['implementation']]
        if len(implementations)!=1:
            raise IRVerificationError('Standalone GDN reference implementation is not registered.')
        candidate=context.configure_implementation(implementations[0],semantic_parameters={'phase_plan':plan})
        return TritonCandidateProposal((candidate,),plan['implementation'])


def propose_standalone_tir(module,target):
    from copy import copy
    policy=copy(target.tir_selection_policy)
    original=policy.registry
    class Registry:
        op_names=original.op_names
        def provider_for(self,op):
            return StandaloneGdnCandidateProvider() if op=='nn.gdn_recurrent_core' else original.provider_for(op)
    policy.registry=Registry()
    result=policy.propose(module,target)
    for node in result.nodes:
        if node.op=='nn.gdn_recurrent_core':
            validate_phase_plan(node.attrs.get('phase_plan'))
            selection=result.selection_map.get('tir.'+node.id)
            if selection is None or selection.candidate_id!=node.attrs['phase_plan']['implementation']:
                raise IRVerificationError('Every standalone GDN selection must be the reference v3 provider.',node_id=node.id)
    return result


def standalone_distribution_target(target,module):
    """Install structural layer-view inference only for the standalone search."""
    if standalone_request(module) is None:
        return target
    from copy import copy
    from triton.flagmega.passes.auto_distributed.inference_providers import TypeInferenceCandidateProvider
    class LayerViewProvider(TypeInferenceCandidateProvider):
        op_names=frozenset({'tir.ref_slice'})
    original=target.distribution_policy
    class Policy:
        def __getattr__(self,name):return getattr(original,name)
        def register_candidate_providers(self,registry):
            original.register_candidate_providers(registry)
            registry.add(LayerViewProvider(frozenset({'tir.ref_slice'})))
    local=copy(target);local.distribution_policy=Policy()
    return local


def phase_plan_for(module_name, rows=1):
    if module_name == 'r017_T_full_p5_split4lut':
        from .r017_split4lut_bindings import column_phase_plan
        return column_phase_plan(rows)
    if module_name == 'r017_T_full_p5_split4':
        from .r017_split4_bindings import column_phase_plan
        return column_phase_plan()
    if module_name == 'r017_T_full_p5_split8':
        from .r017_split8_bindings import column_phase_plan
        return column_phase_plan()
    if module_name == 'r017_T_full_p5_split':
        from .r017_split_bindings import column_phase_plan
        return column_phase_plan()
    if module_name.startswith('r017_T_full_p5_w'):
        from .r017_gdn_bindings import column_phase_plan
        return column_phase_plan(int(module_name.rsplit('w',1)[1]))
    if module_name == 'r004_T_full_p5_v6':
        from .r004_column_phase_bindings_v6 import column_phase_plan
        return column_phase_plan()
    if module_name == MODULE:
        return reference_phase_plan()
    if module_name == 'r004_T_full_p5_v4':
        from .r004_column_phase_bindings import column_phase_plan
        return column_phase_plan()
    raise IRSchemaError('Unknown standalone GDN module identity.')
