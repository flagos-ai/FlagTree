# Copyright 2026 FlagOS Contributors; SPDX-License-Identifier: MIT
"""Literal, revalidatable full-column transport for the exact T=2 GLU.

The original packet arithmetic and token reduction are preserved. Each channel
owns S full weight columns; acquire/wait/release recycle a slot only after all
tokens have finished reading it. No process environment selects a recipe.
"""
from dataclasses import replace

from . import iq4_q8_renderer as accepted
from . import iq4_q8_pipeline_renderer as pipeline
from .iq4_q8_persistent_renderer import MULTIROW_ABI as ABI
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import tensor_type
from triton.flagmega.ir.launch_recipe import StagingRowLaunchRecipe
from triton.flagmega.ir.tir import TIRSharedWorkspaceDescriptor, TIRTransferPipelineContract, TIRTransferPipelineChannel

VARIANT = 'r046_glu_staging_v1'
SCHEMA = 'flagmega.iq4-q8-full-column-staging/r046-v1'
N, K, ROW_BYTES, CARRIER_VECTORS = 17408, 5120, 2720, 176
CONFIGURATIONS = tuple((transport, stages, registers, grid)
                       for transport in ('tma', 'cp_async')
                       for stages in (2, 3, 4)
                       for registers in (56, 64, 80, 104)
                       for grid in (432, 2176))
DEFAULT_CONFIG = ('cp_async', 2, 56, 432)
PTX_OPTIONS = '--minnctapersm=1 --opt-level=2'
TMA_TILES = ((0, 128), (128, 32), (160, 8), (168, 8))


def checked_config(config):
    if (not isinstance(config, (tuple, list)) or len(config) != 4
            or type(config[0]) is not str or any(type(value) is not int for value in config[1:])
            or tuple(config) not in CONFIGURATIONS):
        raise CodegenError('unqualified R046 staging configuration')
    return tuple(config)


def recipe(rows=2, qualification='glu', config=DEFAULT_CONFIG):
    if rows != 2 or qualification != 'glu':
        raise CodegenError('R046 qualifies exactly T=2 GLU')
    transport, stages, registers, grid = checked_config(config)
    return StagingRowLaunchRecipe(N, 1, 'persistent', grid, (grid + 35) // 36,
                                 stages, transport, registers, rows)


def parameters(family, rows=2, config=DEFAULT_CONFIG):
    if family != 'iq4_q8_glu':
        raise CodegenError('R046 qualifies only GLU')
    transport, stages, registers, grid = checked_config(config)
    return {**pipeline.parameters(family), 'num_stages': stages,
            'maxnreg': registers, 'producer_registers': 32,
            'consumer_barrier_abi': ABI, 'ptx_options': PTX_OPTIONS,
            'transfer_schema': SCHEMA, 'transfer_block_shape': [1, 128, 16],
            'transfer_k256_blocks': 20, 'no_shared_staging': False,
            'transport_kind': transport, 'packet_mode': 'cached_tables',
            'staging_config': list(config), 'token_rows': rows,
            'persistent_row_protocol': recipe(rows, config=config).to_data()}


def implementations(rows=2, config=DEFAULT_CONFIG):
    transport, stages, _, _ = checked_config(config)
    base = next(i for i in pipeline.implementations() if i.family == 'iq4_q8_glu')
    spaces = tuple(TIRSharedWorkspaceDescriptor(
        bank + '_s' + str(tile), tensor_type('uint8', (stages, 1, size, 16)),
        128, matrix_compatible=True) for bank in ('gate', 'up')
        for tile, size in enumerate((128, 32, 8, 8)))
    channels = tuple(TIRTransferPipelineChannel(bank, (argument,), tuple(range(4*i, 4*i+4)), 16)
                     for i, (bank, argument) in enumerate((('gate', 1), ('up', 3))))
    return (replace(base, id='tir.iq4_q8_glu.' + VARIANT, variant=VARIANT,
                    parameters=parameters(base.family, rows, config),
                    requires=('tma', 'warp_specialize') if transport == 'tma' else ('warp_specialize',),
                    facts={**base.facts, 'host_tensor_descriptor': transport == 'tma'},
                    shared_workspaces=spaces,
                    transfer_pipeline=TIRTransferPipelineContract(channels, capacity=stages),
                    contract={**base.contract, 'mapping_identity': VARIANT,
                              'row_epoch_recipe': recipe(rows, config=config).to_data()}),)


def validate_raw(raw):
    if raw['variant'] != VARIANT or raw['family'] != 'iq4_q8_glu':
        raise CodegenError('wrong R046 staging subject')
    if (raw['semantic_attrs']['logical_n'], raw['semantic_attrs']['logical_k']) != (N, K):
        raise CodegenError('wrong R046 staging shape')
    rows = accepted._shape(accepted._bindings(raw, 'inputs', 'q8')[0]['abi'], 'logical_shape')[0]
    config = checked_config(raw['parameters'].get('staging_config', ()))
    for key, value in parameters(raw['family'], rows, config).items():
        if accepted._canonical(raw['parameters'].get(key)) != accepted._canonical(value):
            raise CodegenError('R046 staging parameter mismatch ' + key)
    if accepted._canonical(raw['parameters'].get('task_mapping')) != accepted._canonical(recipe(rows, config=config).to_data()):
        raise CodegenError('R046 staging task mapping mismatch')
    accepted.validate_raw({**raw, 'variant': accepted.VARIANT,
                          'parameters': accepted.implementation_parameters(raw['family'])})
    for name in ('gate_weight', 'up_weight'):
        accepted._abi(accepted._bindings(raw, 'inputs', name)[0], dtype='uint8',
                      shape=(N, ROW_BYTES), alignment=16)
    pipeline.validate_transfer_contract(raw, implementations(rows, config)[0])


def materialize(raw):
    validate_raw(raw)
    transport, stages, registers, grid = checked_config(raw['parameters']['staging_config'])
    result = accepted.materialize({**raw, 'variant': accepted.VARIANT,
                                   'parameters': accepted.implementation_parameters(raw['family'])})
    descriptors = []
    if transport == 'tma':
        for bank, name in (('gate', 'gate_weight'), ('up', 'up_weight')):
            binding = accepted._bindings(raw, 'inputs', name)[0]
            for tile, (_, size) in enumerate(TMA_TILES):
                descriptors.append(dict(parameter=bank + '_map' + str(tile), source=binding['runtime_argument'],
                    kind='single', offset_bytes=binding['abi']['pool_byte_offset'], dtype='uint8',
                    shape=(N, 170, 16), strides=(ROW_BYTES, 16, 1), block_shape=(1, size, 16),
                    source_shape_axes=((), (), ()), padding='zero'))
    return {**result, 'schema': SCHEMA, 'jit_module': '', 'jit_entry': '',
            'grid': [grid, 1, 1], 'program_kind': 'persistent',
            'compile_options': {**result['compile_options'], 'maxnreg': registers, 'num_stages': stages,
                                'ptx_options': PTX_OPTIONS, 'consumer_barrier_abi': ABI},
            'pipeline_contract': dict(raw['transfer_pipeline']),
            'shared_workspaces': tuple(raw['shared_workspaces']),
            'num_stages': stages, 'producer_warps': 1, 'producer_registers': 32,
            'host_tensor_descriptor_requests': tuple(descriptors),
            'transport_descriptors': tuple(d['parameter'] for d in descriptors),
            'transport_n': N, 'transport_k': K, 'transport_mode': 1,
            'transport_kind': transport, 'no_shared_staging': False,
            'persistent_row_recipe': recipe(config=raw['parameters']['staging_config']).to_data(),
            'transport_grid_cap': grid, 'pipeline_commits_per_invocation': 0,
            'pipeline_epoch_contract': {'schema': SCHEMA, 'weight_staging': True,
                'payload_bytes_per_bank': ROW_BYTES, 'carrier_bytes_per_bank': CARRIER_VECTORS * 16,
                'stages': stages, 'epoch': 'row_iteration', 'commits_per_output_column': 1,
                'release': 'both K steps and all token accumulations complete',
                'completion': 'TMA mbarrier' if transport == 'tma' else 'cp.async commit/wait group then pipe mbarrier',
                'partial_recycle': 'count128 barrier after all token partial reads'}}
