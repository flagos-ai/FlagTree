# Copyright 2026 FlagOS Contributors; SPDX-License-Identifier: MIT
import ast
from copy import deepcopy
from pathlib import Path

import pytest

from triton.flagmega.codegen.triton import iq4_q8_staging_renderer as renderer
from triton.flagmega.codegen.triton.iq4_q8_pipeline_renderer import validate_transfer_contract
from triton.flagmega.errors import CodegenError, IRSchemaError
from triton.flagmega.ir.launch_recipe import TaskLaunchRecipe
def exported_contract(implementation):
    pipeline = implementation.transfer_pipeline
    return dict(transfer_pipeline=dict(capacity=pipeline.capacity,
        producer_read_argument_indices=pipeline.producer_read_argument_indices,
        consumer_shared_workspace_indices=pipeline.consumer_shared_workspace_indices,
        auxiliary_consumer=None, channels=tuple(dict(name=c.name,
            source_argument_indices=c.source_argument_indices,
            shared_workspace_indices=c.shared_workspace_indices,
            source_alignment_bytes=c.source_alignment_bytes) for c in pipeline.channels)),
        shared_workspaces=tuple(dict(shape=tuple(d.fixed_value for d in w.type.shape),
            dtype=w.type.dtype.value, matrix_compatible=w.matrix_compatible)
            for w in implementation.shared_workspaces))


@pytest.mark.parametrize('config', renderer.CONFIGURATIONS)
def test_staged_payload_contract(config):
    implementation = renderer.implementations(2, config)[0]
    raw = exported_contract(implementation)
    validate_transfer_contract(raw, implementation)
    assert TaskLaunchRecipe.from_data(renderer.recipe(config=config).to_data()) == renderer.recipe(config=config)
    data = renderer.recipe(config=config).to_data()
    for key, value in [('weight_staging', False), ('payload_bytes_per_bank', 0),
                       ('token_rows', 1), ('register_budget', 48), ('capacity', 1)]:
        with pytest.raises(IRSchemaError):
            TaskLaunchRecipe.from_data({**data, key: value})
    assert sum(workspace.type.shape[1].fixed_value * workspace.type.shape[2].fixed_value
               * workspace.type.shape[3].fixed_value for workspace in implementation.shared_workspaces) == 5632
    corrupt = deepcopy(raw)
    corrupt['shared_workspaces'][0]['shape'] = (config[1], 1, 64, 16)
    with pytest.raises(CodegenError, match='staging ABI mismatch'):
        validate_transfer_contract(corrupt, implementation)
    corrupt = deepcopy(raw)
    corrupt['transfer_pipeline']['channels'][1]['source_argument_indices'] = (1,)
    with pytest.raises(CodegenError, match='transfer contract mismatch'):
        validate_transfer_contract(corrupt, implementation)


def test_selected_path_is_static_and_rejects_unqualified_recipes(monkeypatch):
    before = renderer.parameters('iq4_q8_glu')
    for key in ('NO_SHARED_STAGING', 'TRANSPORT_KIND', 'TRANSPORT_STAGE', 'TRANSPORT_MAXNREG'):
        monkeypatch.setenv(key, 'invalid')
    assert renderer.parameters('iq4_q8_glu') == before
    tree = ast.parse(Path(renderer.__file__).read_text())
    assert not any(isinstance(n, ast.Attribute) and n.attr in ('environ', 'getenv') for n in ast.walk(tree))
    for config in [('global_idle', 3, 64, 2176), ('tma', 1, 64, 2176), ('tma', 3, 63, 2176)]:
        with pytest.raises(CodegenError):
            renderer.checked_config(config)
    with pytest.raises(CodegenError):
        renderer.implementations(1)
