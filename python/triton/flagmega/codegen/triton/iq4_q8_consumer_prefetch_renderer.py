# Copyright 2026 FlagOS Contributors; SPDX-License-Identifier: MIT
"""Exact T=2 GLU with grid-strided, consumer-issued L2 prefetch hints."""
from dataclasses import replace
from . import iq4_q8_renderer as accepted
from . import iq4_q8_pipeline_renderer as pipeline
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir.launch_recipe import ConsumerPrefetchRowLaunchRecipe

VARIANT = 'r049_glu_consumer_prefetch_v1'
SCHEMA = 'flagmega.iq4-q8-consumer-prefetch/r049-v1'
N, K, ROW_BYTES = 17408, 5120, 2720
CONFIGURATIONS = tuple((w,d,b,g) for w in (4,8) for d in (1,2,4,8)
                       for b in (40,48,56,64) for g in (432,1088,2176))
DEFAULT_CONFIG = (4, 1, 64, 2176)
PTX_OPTIONS = '--minnctapersm=1 --opt-level=2'


def checked_config(config):
    if (not isinstance(config, (tuple,list)) or len(config) != 4
            or any(type(v) is not int for v in config) or tuple(config) not in CONFIGURATIONS):
        raise CodegenError('unqualified R049 consumer prefetch configuration')
    return tuple(config)


def recipe(rows=2, qualification='glu', config=DEFAULT_CONFIG):
    if type(rows) is not int or rows != 2 or qualification != 'glu':
        raise CodegenError('R049 qualifies exactly T=2 GLU')
    w,d,b,g = checked_config(config)
    return ConsumerPrefetchRowLaunchRecipe(N,1,'persistent',g,(g+35)//36,d,b,rows,w)


def parameters(family, rows=2, config=DEFAULT_CONFIG):
    if family != 'iq4_q8_glu':
        raise CodegenError('R049 qualifies only GLU')
    w,d,b,g = checked_config(config)
    return {**accepted.implementation_parameters(family), 'num_stages': 1,
            'maxnreg': b, 'num_warps': w, 'threads': w*32,
            'consumer_barrier_abi': '', 'ptx_options': PTX_OPTIONS,
            'transfer_schema': SCHEMA, 'no_shared_staging': True,
            'transport_kind': 'consumer_prefetch', 'prefetch_distance': d,
            'prefetch_config': list(config), 'token_rows': rows,
            'persistent_row_protocol': recipe(rows,config=config).to_data()}


def implementations(rows=2, config=DEFAULT_CONFIG):
    base = next(i for i in pipeline.implementations() if i.family == 'iq4_q8_glu')
    return (replace(base, id='tir.iq4_q8_glu.'+VARIANT, variant=VARIANT,
                    parameters=parameters(base.family,rows,config), requires=(),
                    facts={**base.facts,'host_tensor_descriptor':False,'transfer_pipeline':False},
                    shared_workspaces=(), transfer_pipeline=None,
                    contract={**base.contract,'mapping_identity':VARIANT,
                              'row_epoch_recipe':recipe(rows,config=config).to_data()}),)


def validate_raw(raw):
    if raw['variant'] != VARIANT or raw['family'] != 'iq4_q8_glu':
        raise CodegenError('wrong R049 consumer prefetch subject')
    if (raw['semantic_attrs']['logical_n'],raw['semantic_attrs']['logical_k']) != (N,K):
        raise CodegenError('wrong R049 consumer prefetch shape')
    rows = accepted._shape(accepted._bindings(raw,'inputs','q8')[0]['abi'],'logical_shape')[0]
    config = checked_config(raw['parameters'].get('prefetch_config',()))
    for key,value in parameters(raw['family'],rows,config).items():
        if accepted._canonical(raw['parameters'].get(key)) != accepted._canonical(value):
            raise CodegenError('R049 consumer prefetch parameter mismatch '+key)
    if accepted._canonical(raw['parameters'].get('task_mapping')) != accepted._canonical(recipe(rows,config=config).to_data()):
        raise CodegenError('R049 consumer prefetch task mapping mismatch')
    accepted.validate_raw({**raw,'variant':accepted.VARIANT,
                           'parameters':accepted.implementation_parameters(raw['family'])})
    for name in ('gate_weight','up_weight'):
        accepted._abi(accepted._bindings(raw,'inputs',name)[0],dtype='uint8',shape=(N,ROW_BYTES),alignment=16)
    if raw.get('transfer_pipeline') is not None or raw.get('shared_workspaces'):
        raise CodegenError('R049 consumer prefetch has no producer or shared transfer queue')


def materialize(raw):
    validate_raw(raw)
    w,d,b,g = checked_config(raw['parameters']['prefetch_config'])
    result = accepted.materialize({**raw,'variant':accepted.VARIANT,
                                   'parameters':accepted.implementation_parameters(raw['family'])})
    return {**result,'schema':SCHEMA,'jit_module':'','jit_entry':'',
            'grid':[g,1,1],'program_kind':'persistent',
            'compile_options':{**result['compile_options'],'maxnreg':b,'num_stages':1,
                               'num_warps':w,'ptx_options':PTX_OPTIONS,'consumer_barrier_abi':''},
            'pipeline_contract':None,'shared_workspaces':(),'num_stages':1,
            'cta_warps':w,'producer_warps':0,'prefetch_distance':d,
            'host_tensor_descriptor_requests':(),'transport_descriptors':(),
            'transport_n':N,'transport_k':K,'transport_mode':1,
            'transport_kind':'consumer_prefetch','no_shared_staging':True,
            'persistent_row_recipe':recipe(config=raw['parameters']['prefetch_config']).to_data(),
            'transport_grid_cap':g,'pipeline_commits_per_invocation':0}
