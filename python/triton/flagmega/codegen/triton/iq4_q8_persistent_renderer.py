# Copyright 2026 FlagOS Contributors; SPDX-License-Identifier: MIT
"""Static persistent GLU with the original single-row schedule preserved.

Multirow uses bounded L2 prefetch and typed column-index credits. Each consumer
loads/decodes a weight packet once and accumulates every token before advancing.
A credit means that the prefetch was issued; it does not promise L2 residency.
"""
from dataclasses import replace

from . import iq4_q8_pipeline_renderer as prior
from . import iq4_q8_renderer as accepted
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import tensor_type
from triton.flagmega.ir.launch_recipe import PersistentRowLaunchRecipe
from triton.flagmega.ir.tir import TIRSharedWorkspaceDescriptor

VARIANT = 'r006_glu_persistent_v1'
SCHEMA = 'flagmega.iq4-q8-persistent-transfer/r006-v1'
ABI = 'H003ConsumerPartial-f32-3x32/bar3-count128/v4-compact-padding'
MULTIROW_ABI = 'H003ConsumerPartial-f32-ROWSx4x32/count128/v1-token-batch'
RECIPE = PersistentRowLaunchRecipe(17408, 1, 'persistent', 36, 1)


def recipe(rows=1):
    return replace(RECIPE, token_rows=rows, max_ctas=36 if rows == 1 else 72,
                   ctas_per_sm=1 if rows == 1 else 2)


def token_rows(raw):
    qb = accepted._bindings(raw, 'inputs', 'q8')[0]
    return accepted._shape(qb['abi'], 'logical_shape')[0]


def parameters(family, rows=1):
    if family != 'iq4_q8_glu':
        raise CodegenError('persistent v1 qualifies only GLU')
    params = {**prior.parameters(family), 'consumer_barrier_abi': ABI,
              'transfer_schema': SCHEMA, 'persistent_row_protocol': recipe(rows).to_data()}
    if rows > 1:
        params.update(consumer_barrier_abi=MULTIROW_ABI, transfer_block_shape=[1],
                      transfer_payload='column-prefetch-credit/int32', transfer_k256_blocks=0,
                      producer_registers=32, maxnreg=104)
    return params


def implementations(rows=1):
    base = next(i for i in prior.implementations() if i.family == 'iq4_q8_glu')
    if rows > 1:
        base = replace(base, requires=('warp_specialize',),
                       facts={**base.facts, 'host_tensor_descriptor': False},
                       shared_workspaces=tuple(TIRSharedWorkspaceDescriptor(
                           name + '_stage', tensor_type('int32', (2, 1)), 128,
                           matrix_compatible=False) for name in ('gate', 'up')))
    return (replace(base, id='tir.iq4_q8_glu.' + VARIANT, variant=VARIANT,
                    parameters=parameters(base.family, rows),
                    contract={**base.contract, 'mapping_identity': VARIANT,
                              'row_epoch_recipe': recipe(rows).to_data()}),)


def old_view(raw):
    return {**raw, 'variant': prior.VARIANT, 'parameters': prior.parameters(raw['family'])}


def validate_raw(raw):
    if raw['variant'] != VARIANT or raw['family'] != 'iq4_q8_glu':
        raise CodegenError('wrong persistent subject')
    if (raw['semantic_attrs']['logical_n'], raw['semantic_attrs']['logical_k']) != (17408, 5120):
        raise CodegenError('wrong persistent shape')
    rows = token_rows(raw)
    for key, value in parameters(raw['family'], rows).items():
        if accepted._canonical(raw['parameters'].get(key)) != accepted._canonical(value):
            raise CodegenError('persistent parameter mismatch ' + key)
    if accepted._canonical(raw['parameters'].get('task_mapping')) != accepted._canonical(recipe(rows).to_data()):
        raise CodegenError('persistent task mapping mismatch')
    if rows == 1:
        prior.validate_raw(old_view(raw))
    else:
        accepted.validate_raw({**raw, 'variant': accepted.VARIANT,
                               'parameters': accepted.implementation_parameters(raw['family'])})
        for name in ('gate_weight', 'up_weight'):
            accepted._abi(accepted._bindings(raw, 'inputs', name)[0], dtype='uint8',
                          shape=(17408, 2720), alignment=16)
        prior.validate_transfer_contract(raw, implementations(rows)[0])


def materialize(raw):
    validate_raw(raw)
    rows = token_rows(raw)
    if rows == 1:
        result = prior.materialize(old_view(raw))
        return {**result, 'schema': SCHEMA, 'grid': [36, 1, 1], 'program_kind': 'persistent',
                'compile_options': {**result['compile_options'], 'consumer_barrier_abi': ABI},
                'persistent_row_recipe': RECIPE.to_data(), 'transport_grid_cap': 36,
                'persistent_commits_per_column': 2,
                'pipeline_epoch_contract': {'schema': 'flagmega.persistent-stage-epochs/v1',
                    'commits_per_row': 2, 'total_commits': '2*cdiv(N-pid,grid)',
                    'maximum_commits': 968, 'epoch': '2*row_iteration+it',
                    'close': '2*cdiv(N-pid,grid)'},
                'pipeline_commits_per_invocation': 2}
    result = accepted.materialize({**raw, 'variant': accepted.VARIANT,
                                   'parameters': accepted.implementation_parameters(raw['family'])})
    if result['constexprs']['ROWS'] != rows:
        raise CodegenError('persistent ROWS must match the checked ABI')
    params = parameters(raw['family'], rows)
    return {**result, 'schema': SCHEMA, 'grid': [72, 1, 1], 'program_kind': 'persistent',
            'jit_module': '', 'jit_entry': '',
            'compile_options': {**result['compile_options'], 'maxnreg': 104,
                                'consumer_barrier_abi': MULTIROW_ABI},
            'pipeline_contract': dict(raw['transfer_pipeline']),
            'shared_workspaces': tuple(raw['shared_workspaces']),
            'num_stages': 2, 'producer_warps': 1,
            'producer_registers': params['producer_registers'],
            'host_tensor_descriptor_requests': (), 'transport_descriptors': (),
            'transport_n': 17408, 'transport_k': 5120, 'transport_mode': 1,
            'persistent_row_recipe': recipe(rows).to_data(), 'transport_grid_cap': 72,
            'logical_static_shared_bytes': rows * 128 * 8,
            'persistent_commits_per_column': 1,
            # Different CTAs own 241 or 242 columns: no uniform invocation
            # stride exists for an enclosing counted loop reusing this pipe.
            'pipeline_commits_per_invocation': 0,
            'pipeline_epoch_contract': {'schema': 'flagmega.persistent-stage-epochs/v2',
                'token_rows': rows, 'commits_per_row': 1,
                'commits_per_output_column': 1, 'commits_consumed_per_token_row': 1,
                'additional_commits_per_token_row': 0, 'numerical_k_steps_per_output_column': 2,
                'total_commits': 'cdiv(N-pid,grid)', 'maximum_commits': 242,
                'epoch': 'row_iteration', 'close': 'cdiv(N-pid,grid)',
                'payload': 'output column index after L2 prefetch issue',
                'cache_residency_guaranteed': False,
                'release': 'after both K steps and all token accumulations',
                'partial_recycle': 'all token reads complete before count128 barrier'}}
