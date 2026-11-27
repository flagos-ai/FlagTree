# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The model recipe must compose with the current normal target catalog."""

from pathlib import Path
import importlib
from copy import deepcopy
from dataclasses import replace

import pytest


@pytest.mark.parametrize('offline_projection_weights', (False, True))
def test_native_tp4_recipe_accepts_preferred_kernels_already_in_base(monkeypatch, offline_projection_weights):
    tutorial = Path(__file__).resolve().parents[3] / 'tutorials/flagmega/04-glm5.3-flash-fp8/nvidia-h800'
    monkeypatch.syspath_prepend(str(tutorial))
    recipe = importlib.import_module('local_optimizations.native_tp4')
    model = recipe.native_implementation_model(offline_projection_weights=offline_projection_weights)
    assert model.implementation('tir.sparse_experts_gate_up.complete_k_wgmma_n32') is not None
    assert model.implementation('tir.partitioned_block_scaled_linear.local') is not None


def test_offline_projection_recipe_rejects_same_id_with_another_contract(monkeypatch):
    tutorial = Path(__file__).resolve().parents[3] / 'tutorials/flagmega/04-glm5.3-flash-fp8/nvidia-h800'
    monkeypatch.syspath_prepend(str(tutorial))
    recipe = importlib.import_module('local_optimizations.offline_projection_weights')
    from triton.flagmega.targets import NvidiaSm90Target
    base = NvidiaSm90Target().triton_implementation_model
    identity = 'tir.dense_matmul.packed_k_major_mma_accumulator'
    conflicted = replace(base, implementations=tuple(
        replace(value, parameters={**value.parameters, 'block_k': 256}) if value.id == identity else value
        for value in base.implementations))
    with pytest.raises(ValueError, match='conflicting contract'):
        recipe.implementation_model(conflicted)


def _replica_role_context(schedule_kind):
    placement = {'hierarchy': (4, 8, 16), 'hierarchy_levels': 'cbb'}
    def binding(identity):
        return {'buffers': [{'abi': {
            'physical_buffer': identity,
            'memory_sharing_scope': 'chip',
            'distributed_type': {
                'placement': placement, 'tensor': {'kind': 'tensor'},
                'exclusive': None, 'partial': None, 'axis_policies': (),
            },
        }}]}
    call = {
        'call': 'independent_projection', 'family': 'dense_matmul',
        'variant': 'packed_k_major_mma_prefetch', 'participation_active': 'True',
        'source': 'published_source', 'weight': 'constant_weight', 'result': 'result',
        'inputs': [binding('source'), binding('weight')],
        'outputs': [binding('result')],
    }
    events = [{'kind': 'tir.kernel_call', 'call': call['call'], 'participation_active': 'True'}]
    context = {
        'render_calls': [call], 'distributed_entry': True,
        'mesh_hierarchy': (4, 8, 16), 'mesh_levels': ('c', 'b', 'b'),
        'grid': (128, 1, 1), 'entry_events': (), 'device_functions': (),
    }
    if schedule_kind == 'entry':
        context['entry_events'] = events
    elif schedule_kind == 'pipeline':
        context['pipeline_schedule'] = {'consumer_events': events}
    else:
        context['device_functions'] = [{'schedule': {'consumer_events': events}}]
    return context


@pytest.mark.parametrize('schedule_kind', ('entry', 'pipeline', 'device_function'))
def test_replica_roles_render_reused_program_schedule_without_mutating_input(monkeypatch, schedule_kind):
    tutorial = Path(__file__).resolve().parents[3] / 'tutorials/flagmega/04-glm5.3-flash-fp8/nvidia-h800'
    monkeypatch.syspath_prepend(str(tutorial))
    importlib.import_module('local_optimizations').activate_base()
    recipe = importlib.import_module('local_optimizations.replica_roles')
    captured = []
    def render(self, name, context):
        captured.append(deepcopy(context))
        return 'rendered'
    monkeypatch.setattr(recipe.TritonTemplateRegistry, 'render', render)
    context = _replica_role_context(schedule_kind)
    original = deepcopy(context)
    with recipe.activate({'independent_projection': ((1, 1), (2, 0))}):
        registry = recipe.TritonTemplateRegistry()
        assert registry.render('module.py.jinja', {**context}) == 'rendered'
        assert registry.render('module.py.jinja', {**context}) == 'rendered'
    assert context == original
    assert captured[0] == captured[1]
    schedules = [captured[0]['entry_events']]
    if 'pipeline_schedule' in captured[0]:
        schedules.append(captured[0]['pipeline_schedule']['consumer_events'])
    schedules.extend(f['schedule']['consumer_events'] for f in captured[0]['device_functions'])
    event, = [event for events in schedules for event in events]
    assert event['participation_active'] == '(shard_coord1 == 1) & (shard_coord2 == 0)'


def test_replica_roles_reject_existing_exclusive_participation(monkeypatch):
    tutorial = Path(__file__).resolve().parents[3] / 'tutorials/flagmega/04-glm5.3-flash-fp8/nvidia-h800'
    monkeypatch.syspath_prepend(str(tutorial))
    importlib.import_module('local_optimizations').activate_base()
    recipe = importlib.import_module('local_optimizations.replica_roles')
    from triton.flagmega.errors import CodegenError
    monkeypatch.setattr(recipe.TritonTemplateRegistry, 'render', lambda *args: 'rendered')
    context = _replica_role_context('entry')
    context['render_calls'][0]['participation_active'] = '(shard_coord1 == 0)'
    with recipe.activate({'independent_projection': ((1, 1), (2, 0))}):
        with pytest.raises(CodegenError, match='No complete physical replica proof'):
            recipe.TritonTemplateRegistry().render('module.py.jinja', context)
