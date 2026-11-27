# Copyright 2026 FlagOS Contributors; SPDX-License-Identifier: MIT
from copy import deepcopy
import ast
from pathlib import Path

import pytest

from triton.flagmega.codegen.triton import iq4_q8_global_prefetch_renderer as renderer
from triton.flagmega.codegen.triton.iq4_q8_pipeline_renderer import validate_transfer_contract
from triton.flagmega.errors import CodegenError, IRSchemaError
from triton.flagmega.ir.launch_recipe import TaskLaunchRecipe, STATIC_PREFETCH_QUALIFICATIONS


def exported_contract(implementation):
    pipeline = implementation.transfer_pipeline
    return dict(
        transfer_pipeline=dict(
            capacity=pipeline.capacity,
            producer_read_argument_indices=pipeline.producer_read_argument_indices,
            consumer_shared_workspace_indices=pipeline.consumer_shared_workspace_indices,
            auxiliary_consumer=None,
            channels=tuple(
                dict(
                    name=c.name,
                    source_argument_indices=c.source_argument_indices,
                    shared_workspace_indices=c.shared_workspace_indices,
                    source_alignment_bytes=c.source_alignment_bytes,
                )
                for c in pipeline.channels
            ),
        ),
        shared_workspaces=tuple(
            dict(
                shape=tuple(d.fixed_value for d in workspace.type.shape),
                dtype=workspace.type.dtype.value,
                matrix_compatible=workspace.matrix_compatible,
            )
            for workspace in implementation.shared_workspaces
        ),
    )


@pytest.mark.parametrize("rows", [2, 3, 4, 5])
def test_static_global_recipe_and_role_contract(rows):
    recipe = renderer.recipe(rows)
    assert TaskLaunchRecipe.from_data(recipe.to_data()) == recipe
    implementation = renderer.implementations(rows)[0]
    raw = exported_contract(implementation)
    validate_transfer_contract(raw, implementation)
    invalid = deepcopy(raw)
    invalid["transfer_pipeline"]["channels"][0]["source_argument_indices"] = (3,)
    with pytest.raises(CodegenError, match="transfer contract mismatch"):
        validate_transfer_contract(invalid, implementation)
    data = recipe.to_data()
    data["prefetch_issue_window"] = 4
    with pytest.raises(IRSchemaError):
        TaskLaunchRecipe.from_data(data)


def test_static_settings_ignore_legacy_environment(monkeypatch):
    before = renderer.parameters("iq4_q8_glu", 5)
    for key in (
        "TRANSPORT_STAGE",
        "NO_SHARED_STAGING",
        "TRANSPORT_MAXNREG",
        "PREFETCH_DEPTH",
        "PREFETCH_WINDOW",
        "WARP_GRID",
        "PACKET_MODE",
        "Q8_ROWS_OVERRIDE",
        "CONTINUOUS_STREAM",
        "R27_DROP_MAXNREG",
    ):
        monkeypatch.setenv(key, "invalid-legacy-setting")
    assert renderer.parameters("iq4_q8_glu", 5) == before
    tree = ast.parse(Path(renderer.__file__).read_text())
    assert not any(isinstance(node, ast.Attribute) and node.attr in ("environ", "getenv") for node in ast.walk(tree))
    with pytest.raises(CodegenError, match="ROWS=1 keeps the accepted path"):
        renderer.implementations(1)


@pytest.mark.parametrize("qualification", STATIC_PREFETCH_QUALIFICATIONS)
def test_family_recipe_rejects_cross_family_mapping(qualification):
    family, n, k, mode, grid, per_sm = STATIC_PREFETCH_QUALIFICATIONS[qualification]
    assert renderer.qualify(family, n, k) == qualification
    recipe = renderer.recipe(2, qualification)
    data = recipe.to_data()
    assert TaskLaunchRecipe.from_data(data) == recipe
    assert data['k_steps_per_output_column'] == (k // 256 + 15) // 16
    assert (recipe.work_size, recipe.tile_size, recipe.max_ctas, recipe.ctas_per_sm) == (n, 1, grid, per_sm)
    implementation = renderer.implementation(2, qualification)
    validate_transfer_contract(exported_contract(implementation), implementation)
    for key, value in [('work_size', n + 1), ('max_ctas', grid // 2),
                       ('ctas_per_sm', per_sm + 1), ('token_rows', True),
                       ('k_steps_per_output_column', 99), ('qualification', 'unknown')]:
        invalid = {**data, key: value}
        with pytest.raises(IRSchemaError):
            TaskLaunchRecipe.from_data(invalid)
    assert renderer.qualify(family, n, k + 256) is None
