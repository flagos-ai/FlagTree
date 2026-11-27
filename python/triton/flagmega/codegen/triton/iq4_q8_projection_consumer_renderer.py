# Copyright 2026 FlagOS Contributors; SPDX-License-Identifier: MIT
"""Qualified T=2 MODE=0 projections with consumer-issued L2 hints."""
from dataclasses import replace
from . import iq4_q8_renderer as accepted
from . import iq4_q8_pipeline_renderer as pipeline
PTX_OPTIONS = '--minnctapersm=1'
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir.launch_recipe import ProjectionConsumerPrefetchLaunchRecipe

VARIANT = 'r050_projection_consumer_prefetch_v1'
SCHEMA = 'flagmega.iq4-q8-projection-consumer-prefetch/r050-v1'
# Shape selects a numerical family. Output and attention output share its recipe.
SHAPES = {'mlp_down': (5120,17408), 'in_proj_qkv': (10240,5120),
          'in_proj_z': (6144,5120), 'out_proj': (5120,6144)}
SELECTED = {}

SELECTED_COLUMNS = {}

def qualify(family,n,k):
    return next((name for name,shape in SHAPES.items()
                 if family == 'iq4_q8_projection' and shape == (n,k)),None)

def checked_config(qualification,config,columns=1):
    if qualification not in SHAPES:
        raise CodegenError('unqualified R050 projection shape')
    n,k=SHAPES[qualification]
    if type(columns) is not int or columns not in (1,4 if qualification=='mlp_down' else 2):
        raise CodegenError('unqualified R050 projection columns')
    grids=tuple(dict.fromkeys((432,n//8,n//4,n//2,n//(8*columns),n//(4*columns),n//(2*columns),288,576,864,1152,1728,2304,n//columns)))
    if (not isinstance(config,(tuple,list)) or len(config)!=4 or any(type(v) is not int for v in config)
            or config[0] not in (4,8) or config[1] not in (1,2,4)
            or config[2] not in (40,48,56,64) or config[3] not in grids or config[3]>n//columns):
        raise CodegenError('unqualified R050 projection configuration')
    return tuple(config)

def recipe(rows=2,qualification='mlp_down',config=None,columns=None):
    if qualification not in SHAPES:
        raise CodegenError('unqualified R050 projection shape')
    if config is None and qualification not in SELECTED:
        raise CodegenError('R050 has no promoted recipe for '+qualification+'; supply a qualified configuration')
    columns=SELECTED_COLUMNS.get(qualification,1) if columns is None else columns
    w,d,b,g=checked_config(qualification,SELECTED[qualification] if config is None else config,columns)
    n,k=SHAPES[qualification]
    return ProjectionConsumerPrefetchLaunchRecipe(n,columns,'persistent',g,(g+35)//36,d,b,rows,w,qualification,columns)

def parameters(family,rows=2,qualification='mlp_down',config=None,columns=None):
    if family!='iq4_q8_projection':
        raise CodegenError('R050 requires a single projection')
    r=recipe(rows,qualification,config,columns)
    return {**accepted.implementation_parameters(family),'num_stages':1,
            'maxnreg':r.register_budget,'num_warps':r.cta_warps,'threads':r.cta_warps*32,
            'consumer_barrier_abi':'','ptx_options':PTX_OPTIONS,'transfer_schema':SCHEMA,
            'no_shared_staging':True,'transport_kind':'consumer_prefetch',
            'prefetch_distance':r.prefetch_distance,
            'prefetch_config':[r.cta_warps,r.prefetch_distance,r.register_budget,r.max_ctas],
            'token_rows':rows,'projection_columns':r.columns,'persistent_row_protocol':r.to_data()}

def implementation(rows,qualification,config=None,columns=None):
    base=next(i for i in pipeline.implementations() if i.family=='iq4_q8_projection')
    return replace(base,id='tir.'+base.family+'.'+VARIANT+'.'+qualification,variant=VARIANT,
                   parameters=parameters(base.family,rows,qualification,config,columns),requires=(),
                   facts={**base.facts,'host_tensor_descriptor':False,'transfer_pipeline':False},
                   shared_workspaces=(),transfer_pipeline=None,
                   contract={**base.contract,'mapping_identity':VARIANT,
                             'row_epoch_recipe':recipe(rows,qualification,config,columns).to_data()})

def implementations(rows=2):
    return tuple(implementation(rows,q) for q in SELECTED)

def validate_raw(raw):
    if raw['variant']!=VARIANT:
        raise CodegenError('wrong R050 projection subject')
    q=qualify(raw['family'],raw['semantic_attrs']['logical_n'],raw['semantic_attrs']['logical_k'])
    if q is None: raise CodegenError('wrong R050 projection shape')
    rows=accepted._shape(accepted._bindings(raw,'inputs','q8')[0]['abi'],'logical_shape')[0]
    columns=raw['parameters'].get('projection_columns',1)
    config=checked_config(q,raw['parameters'].get('prefetch_config',()),columns)
    for key,value in parameters(raw['family'],rows,q,config,columns).items():
        if accepted._canonical(raw['parameters'].get(key))!=accepted._canonical(value):
            raise CodegenError('R050 projection parameter mismatch '+key)
    if accepted._canonical(raw['parameters'].get('task_mapping'))!=accepted._canonical(recipe(rows,q,config,columns).to_data()):
        raise CodegenError('R050 projection task mapping mismatch')
    accepted.validate_raw({**raw,'variant':accepted.VARIANT,
                           'parameters':accepted.implementation_parameters(raw['family'])})
    n,k=SHAPES[q]
    accepted._abi(accepted._bindings(raw,'inputs','weight')[0],dtype='uint8',shape=(n,136*(k//256)),alignment=16)
    if raw.get('transfer_pipeline') is not None or raw.get('shared_workspaces'):
        raise CodegenError('R050 projection has no producer or shared transfer queue')

def materialize(raw):
    validate_raw(raw)
    q=qualify(raw['family'],raw['semantic_attrs']['logical_n'],raw['semantic_attrs']['logical_k'])
    n,k=SHAPES[q]
    columns=raw['parameters']['projection_columns']
    w,d,b,g=checked_config(q,raw['parameters']['prefetch_config'],columns)
    result=accepted.materialize({**raw,'variant':accepted.VARIANT,
                                'parameters':accepted.implementation_parameters(raw['family'])})
    return {**result,'schema':SCHEMA,'jit_module':'','jit_entry':'',
            'grid':[g,1,1],'program_kind':'persistent','constexprs':{**result['constexprs'],'COLS':columns},
            'compile_options':{**result['compile_options'],'maxnreg':b,'num_stages':1,
                               'num_warps':w,'ptx_options':PTX_OPTIONS,'consumer_barrier_abi':''},
            'pipeline_contract':None,'shared_workspaces':(),'num_stages':1,
            'cta_warps':w,'producer_warps':0,'prefetch_distance':d,
            'host_tensor_descriptor_requests':(),'transport_descriptors':(),
            'transport_n':n,'transport_k':k,'transport_mode':0,
            'transport_kind':'consumer_prefetch','no_shared_staging':True,
            'persistent_row_recipe':recipe(2,q,raw['parameters']['prefetch_config'],columns).to_data(),
            'transport_grid_cap':g,'pipeline_commits_per_invocation':0}
