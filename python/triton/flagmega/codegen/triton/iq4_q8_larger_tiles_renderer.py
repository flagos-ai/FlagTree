# Copyright 2026 FlagOS Contributors; SPDX-License-Identifier: MIT
"""Literal multi-column cp.async rings for the exact T=2 GLU experiment."""
from dataclasses import replace

from . import iq4_q8_renderer as accepted
from . import iq4_q8_pipeline_renderer as pipeline
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir.launch_recipe import LargerTileStagingRowLaunchRecipe

VARIANT = 'r048_glu_larger_tiles_v1'
SCHEMA = 'flagmega.iq4-q8-larger-tile-staging/r048-v1'
N, K, ROW_BYTES = 17408, 5120, 2720
CONFIGURATIONS = tuple((c, s, w, b, g) for c in (2, 4, 8, 16)
                       for s in (1, 2, 3, 4) for w in (2, 4, 8)
                       for b in (48, 56, 64) for g in (144, 432, 2176))
DEFAULT_CONFIG = (2, 1, 4, 64, 2176)
PTX_OPTIONS = '--minnctapersm=1 --opt-level=2'


def checked_config(config):
    if (not isinstance(config, (tuple, list)) or len(config) != 5
            or any(type(v) is not int for v in config)
            or tuple(config) not in CONFIGURATIONS):
        raise CodegenError('unqualified R048 staging configuration')
    return tuple(config)


def recipe(rows=2, qualification='glu', config=DEFAULT_CONFIG):
    if rows != 2 or qualification != 'glu':
        raise CodegenError('R048 qualifies exactly T=2 GLU')
    c, s, w, b, g = checked_config(config)
    return LargerTileStagingRowLaunchRecipe(N, 1, 'persistent', g, (g + 35) // 36,
                                            s, b, rows, w, c)


def parameters(family, rows=2, config=DEFAULT_CONFIG):
    if family != 'iq4_q8_glu':
        raise CodegenError('R048 qualifies only GLU')
    c, s, w, b, g = checked_config(config)
    return {**accepted.implementation_parameters(family), 'num_stages': s,
            'maxnreg': b, 'num_warps': w, 'threads': w * 32,
            'consumer_barrier_abi': '', 'ptx_options': PTX_OPTIONS,
            'transfer_schema': SCHEMA, 'no_shared_staging': False,
            'transport_kind': 'cp_async', 'staging_columns': c,
            'staging_config': list(config), 'token_rows': rows,
            'persistent_row_protocol': recipe(rows, config=config).to_data()}


def implementations(rows=2, config=DEFAULT_CONFIG):
    base = next(i for i in pipeline.implementations() if i.family == 'iq4_q8_glu')
    return (replace(base, id='tir.iq4_q8_glu.' + VARIANT, variant=VARIANT,
                    parameters=parameters(base.family, rows, config), requires=(),
                    facts={**base.facts, 'host_tensor_descriptor': False, 'transfer_pipeline': False},
                    shared_workspaces=(), transfer_pipeline=None,
                    contract={**base.contract, 'mapping_identity': VARIANT,
                              'row_epoch_recipe': recipe(rows, config=config).to_data()}),)


def validate_raw(raw):
    if raw['variant'] != VARIANT or raw['family'] != 'iq4_q8_glu':
        raise CodegenError('wrong R048 staging subject')
    if (raw['semantic_attrs']['logical_n'], raw['semantic_attrs']['logical_k']) != (N, K):
        raise CodegenError('wrong R048 staging shape')
    rows = accepted._shape(accepted._bindings(raw, 'inputs', 'q8')[0]['abi'], 'logical_shape')[0]
    config = checked_config(raw['parameters'].get('staging_config', ()))
    for key, value in parameters(raw['family'], rows, config).items():
        if accepted._canonical(raw['parameters'].get(key)) != accepted._canonical(value):
            raise CodegenError('R048 staging parameter mismatch ' + key)
    if accepted._canonical(raw['parameters'].get('task_mapping')) != accepted._canonical(recipe(rows, config=config).to_data()):
        raise CodegenError('R048 staging task mapping mismatch')
    accepted.validate_raw({**raw, 'variant': accepted.VARIANT,
                           'parameters': accepted.implementation_parameters(raw['family'])})
    for name in ('gate_weight', 'up_weight'):
        accepted._abi(accepted._bindings(raw, 'inputs', name)[0], dtype='uint8',
                       shape=(N, ROW_BYTES), alignment=16)
    if raw.get('transfer_pipeline') is not None or raw.get('shared_workspaces'):
        raise CodegenError('R048 owns its CTA ring in the native template')


def materialize(raw):
    validate_raw(raw)
    c, s, w, b, g = checked_config(raw['parameters']['staging_config'])
    result = accepted.materialize({**raw, 'variant': accepted.VARIANT,
                                   'parameters': accepted.implementation_parameters(raw['family'])})
    return {**result, 'schema': SCHEMA, 'jit_module': '', 'jit_entry': '',
            'grid': [g, 1, 1], 'program_kind': 'persistent',
            'compile_options': {**result['compile_options'], 'maxnreg': b, 'num_stages': s,
                                'num_warps': w, 'ptx_options': PTX_OPTIONS, 'consumer_barrier_abi': ''},
            'pipeline_contract': None, 'shared_workspaces': (), 'num_stages': s,
            'cta_warps': w, 'columns': c, 'producer_warps': 0,
            'host_tensor_descriptor_requests': (), 'transport_descriptors': (),
            'transport_n': N, 'transport_k': K, 'transport_mode': 1,
            'transport_kind': 'cp_async', 'no_shared_staging': False,
            'persistent_row_recipe': recipe(config=raw['parameters']['staging_config']).to_data(),
            'transport_grid_cap': g, 'pipeline_commits_per_invocation': 0,
            'pipeline_epoch_contract': {'schema': SCHEMA, 'weight_staging': True,
                'columns_per_stage': c, 'payload_bytes_per_bank': c * ROW_BYTES,
                'carrier_bytes_per_bank': c * 2816, 'stages': s,
                'physical_stages': 4 if s == 3 else s, 'epoch': 'column_tile',
                'commits_per_tile': 2, 'release': 'all columns and token accumulations complete',
                'completion': 'cp.async wait_group 2*(S-1), CTA consume/recycle barriers'}}
