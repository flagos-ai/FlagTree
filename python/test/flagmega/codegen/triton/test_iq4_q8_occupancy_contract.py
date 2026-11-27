# Copyright 2026 FlagOS Contributors; SPDX-License-Identifier: MIT
from copy import deepcopy

import pytest

from triton.flagmega.codegen.triton import iq4_q8_occupancy_renderer as renderer
from triton.flagmega.codegen.triton.iq4_q8_pipeline_renderer import validate_transfer_contract
from triton.flagmega.errors import CodegenError, IRSchemaError
from triton.flagmega.ir.launch_recipe import TaskLaunchRecipe


def exported_contract(implementation):
    p = implementation.transfer_pipeline
    return dict(transfer_pipeline=dict(capacity=p.capacity,
        producer_read_argument_indices=p.producer_read_argument_indices,
        consumer_shared_workspace_indices=p.consumer_shared_workspace_indices,
        auxiliary_consumer=None, channels=tuple(dict(name=c.name,
            source_argument_indices=c.source_argument_indices,
            shared_workspace_indices=c.shared_workspace_indices,
            source_alignment_bytes=c.source_alignment_bytes) for c in p.channels)),
        shared_workspaces=tuple(dict(shape=tuple(d.fixed_value for d in w.type.shape),
            dtype=w.type.dtype.value, matrix_compatible=w.matrix_compatible)
            for w in implementation.shared_workspaces))


@pytest.mark.parametrize('config', renderer.CONFIGURATIONS)
def test_physical_cta_and_staging_contract_roundtrip(config):
    _, warps, stages, budget, grid = config
    data = renderer.recipe(config=config).to_data()
    assert TaskLaunchRecipe.from_data(data).to_data() == data
    for key, value in [('cta_warps', 3), ('partial_sync', 'none'),
                       ('preferred_shared_carveout_percent', 0), ('register_budget', 33)]:
        with pytest.raises(IRSchemaError):
            TaskLaunchRecipe.from_data({**data, key: value})
    implementation = renderer.implementations(config=config)[0]
    if warps == 8:
        raw = exported_contract(implementation)
        validate_transfer_contract(raw, implementation)
        corrupt = deepcopy(raw)
        corrupt['transfer_pipeline']['channels'][1]['source_argument_indices'] = (1,)
        with pytest.raises(CodegenError, match='transfer contract mismatch'):
            validate_transfer_contract(corrupt, implementation)
    else:
        assert implementation.transfer_pipeline is None


def test_environment_cannot_select_occupancy_recipe(monkeypatch):
    before = renderer.parameters('iq4_q8_glu')
    for key in ('TRANSPORT_STAGE', 'TRANSPORT_MAXNREG', 'NO_SHARED_STAGING'):
        monkeypatch.setenv(key, 'invalid')
    assert renderer.parameters('iq4_q8_glu') == before
    with pytest.raises(CodegenError):
        renderer.checked_config(('tma', 8, 2, 40, 2176))
