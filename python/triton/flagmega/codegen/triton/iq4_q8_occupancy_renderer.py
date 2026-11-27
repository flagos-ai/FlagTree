# Copyright 2026 FlagOS Contributors; SPDX-License-Identifier: MIT
"""Explicit occupancy sweep for exact T=2 GLU byte staging.

Eight physical warps keep the R046 producer/consumer ring and use counted
mbarriers for partial recycling. Smaller CTAs use an asynchronous CTA ring.
All variants preserve the four logical partials and their reduction order.
"""
from dataclasses import replace

from . import iq4_q8_renderer as accepted
from . import iq4_q8_pipeline_renderer as pipeline
from .iq4_q8_persistent_renderer import MULTIROW_ABI as ABI
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import tensor_type
from triton.flagmega.ir.launch_recipe import OccupancyStagingRowLaunchRecipe
from triton.flagmega.ir.tir import TIRSharedWorkspaceDescriptor, TIRTransferPipelineContract, TIRTransferPipelineChannel

VARIANT = 'r047_glu_occupancy_v1'
SCHEMA = 'flagmega.iq4-q8-full-column-staging/r047-v1'
N, K, ROW_BYTES, CARRIER_VECTORS = 17408, 5120, 2720, 176
CONFIGURATIONS = tuple(('cp_async', warps, stages, registers, grid)
                       for warps in (2, 4, 8)
                       for stages in (2, 3, 4)
                       for registers in (32, 40, 48, 56)
                       for grid in (288, 432, 864, 2176))
DEFAULT_CONFIG = ('cp_async', 8, 2, 40, 2176)
PTX_OPTIONS = '--minnctapersm=1 --opt-level=2'


def barrier_abi(warps):
    return ABI if warps == 8 else ''


def checked_config(config):
    if (not isinstance(config, (tuple, list)) or len(config) != 5
            or type(config[0]) is not str or any(type(value) is not int for value in config[1:])
            or tuple(config) not in CONFIGURATIONS):
        raise CodegenError('unqualified R047 staging configuration')
    return tuple(config)


def recipe(rows=2, qualification='glu', config=DEFAULT_CONFIG):
    if rows != 2 or qualification != 'glu':
        raise CodegenError('R047 qualifies exactly T=2 GLU')
    transport, warps, stages, registers, grid = checked_config(config)
    return OccupancyStagingRowLaunchRecipe(N, 1, 'persistent', grid, (grid + 35) // 36,
                                 stages, transport, registers, rows, warps)


def parameters(family, rows=2, config=DEFAULT_CONFIG):
    if family != 'iq4_q8_glu':
        raise CodegenError('R047 qualifies only GLU')
    transport, warps, stages, registers, grid = checked_config(config)
    return {**pipeline.parameters(family), 'num_stages': stages,
            'maxnreg': registers, 'producer_registers': 32,
            'num_warps': 4 if warps == 8 else warps, 'threads': 128 if warps == 8 else warps * 32,
            'consumer_barrier_abi': barrier_abi(warps), 'ptx_options': PTX_OPTIONS,
            'transfer_schema': SCHEMA, 'transfer_block_shape': [1, 128, 16],
            'transfer_k256_blocks': 20, 'no_shared_staging': False,
            'transport_kind': transport, 'packet_mode': 'cached_tables',
            'staging_config': list(config), 'token_rows': rows,
            'persistent_row_protocol': recipe(rows, config=config).to_data()}


def implementations(rows=2, config=DEFAULT_CONFIG):
    transport, warps, stages, _, _ = checked_config(config)
    base = next(i for i in pipeline.implementations() if i.family == 'iq4_q8_glu')
    spaces = tuple(TIRSharedWorkspaceDescriptor(
        bank + '_s' + str(tile), tensor_type('uint8', (stages, 1, size, 16)),
        128, matrix_compatible=True) for bank in ('gate', 'up')
        for tile, size in enumerate((128, 32, 8, 8)))
    channels = tuple(TIRTransferPipelineChannel(bank, (argument,), tuple(range(4*i, 4*i+4)), 16)
                     for i, (bank, argument) in enumerate((('gate', 1), ('up', 3))))
    return (replace(base, id='tir.iq4_q8_glu.' + VARIANT, variant=VARIANT,
                    parameters=parameters(base.family, rows, config),
                    requires=('warp_specialize',) if warps == 8 else (),
                    facts={**base.facts, 'host_tensor_descriptor': False},
                    shared_workspaces=spaces if warps == 8 else (),
                    transfer_pipeline=TIRTransferPipelineContract(channels, capacity=stages) if warps == 8 else None,
                    contract={**base.contract, 'mapping_identity': VARIANT,
                              'row_epoch_recipe': recipe(rows, config=config).to_data()}),)


def validate_raw(raw):
    if raw['variant'] != VARIANT or raw['family'] != 'iq4_q8_glu':
        raise CodegenError('wrong R047 staging subject')
    if (raw['semantic_attrs']['logical_n'], raw['semantic_attrs']['logical_k']) != (N, K):
        raise CodegenError('wrong R047 staging shape')
    rows = accepted._shape(accepted._bindings(raw, 'inputs', 'q8')[0]['abi'], 'logical_shape')[0]
    config = checked_config(raw['parameters'].get('staging_config', ()))
    for key, value in parameters(raw['family'], rows, config).items():
        if accepted._canonical(raw['parameters'].get(key)) != accepted._canonical(value):
            raise CodegenError('R047 staging parameter mismatch ' + key)
    if accepted._canonical(raw['parameters'].get('task_mapping')) != accepted._canonical(recipe(rows, config=config).to_data()):
        raise CodegenError('R047 staging task mapping mismatch')
    accepted.validate_raw({**raw, 'variant': accepted.VARIANT,
                          'parameters': accepted.implementation_parameters(raw['family'])})
    for name in ('gate_weight', 'up_weight'):
        accepted._abi(accepted._bindings(raw, 'inputs', name)[0], dtype='uint8',
                      shape=(N, ROW_BYTES), alignment=16)
    if config[1] == 8:
        pipeline.validate_transfer_contract(raw, implementations(rows, config)[0])
    elif raw.get('transfer_pipeline') is not None or raw.get('shared_workspaces'):
        raise CodegenError('R047 owns its CTA ring in the native template')


def materialize(raw):
    validate_raw(raw)
    transport, warps, stages, registers, grid = checked_config(raw['parameters']['staging_config'])
    result = accepted.materialize({**raw, 'variant': accepted.VARIANT,
                                   'parameters': accepted.implementation_parameters(raw['family'])})
    return {**result, 'schema': SCHEMA, 'jit_module': '', 'jit_entry': '',
            'grid': [grid, 1, 1], 'program_kind': 'persistent',
            'compile_options': {**result['compile_options'], 'maxnreg': registers, 'num_stages': stages, 'num_warps': 4 if warps == 8 else warps,
                                'ptx_options': PTX_OPTIONS, 'consumer_barrier_abi': barrier_abi(warps)},
            'pipeline_contract': dict(raw['transfer_pipeline']) if warps == 8 else None,
            'shared_workspaces': tuple(raw['shared_workspaces']),
            'num_stages': stages, 'cta_warps': warps, 'consumer_warps': 4 if warps == 8 else warps, 'producer_warps': 1 if warps == 8 else 0, 'producer_registers': 32,
            'host_tensor_descriptor_requests': (), 'transport_descriptors': (),
            'staging_execution': 'warp_specialized' if warps == 8 else 'cta_ring',
            'transport_n': N, 'transport_k': K, 'transport_mode': 1,
            'transport_kind': transport, 'no_shared_staging': False,
            'persistent_row_recipe': recipe(config=raw['parameters']['staging_config']).to_data(),
            'transport_grid_cap': grid, 'pipeline_commits_per_invocation': 0,
            'pipeline_epoch_contract': {'schema': SCHEMA, 'weight_staging': True,
                'payload_bytes_per_bank': ROW_BYTES, 'carrier_bytes_per_bank': CARRIER_VECTORS * 16,
                'stages': stages, 'epoch': 'row_iteration', 'commits_per_output_column': 2,
                'release': 'both K steps and all token accumulations complete',
                'completion': 'cp.async commit/wait then pipe mbarrier' if warps == 8 else 'cp.async wait_group 2*(S-1), CTA consume/recycle barriers',
                'partial_recycle': 'count128 mbarrier phases 0/1' if warps == 8 else 'physical CTA-count named barrier'}}
