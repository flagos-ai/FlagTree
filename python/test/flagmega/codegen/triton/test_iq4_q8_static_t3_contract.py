# Copyright 2026 FlagOS Contributors; SPDX-License-Identifier: MIT
"""Reject recipe/row changes and preserve the signed single-row arithmetic."""
import ast
from pathlib import Path

import pytest

from triton.flagmega.codegen.triton import iq4_q8_static_t3_renderer as renderer
from triton.flagmega.codegen.triton import iq4_q8_r031_transport_renderer as signed
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir.launch_recipe import TaskLaunchRecipe


@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5])
def test_static_t3_checked_rows(rows):
    recipe = renderer.recipe(rows)
    assert TaskLaunchRecipe.from_data(recipe.to_data()) == recipe
    assert recipe.to_data()["max_ctas"] == 17408
    implementation = renderer.implementations(rows)[0]
    assert implementation.transfer_pipeline is None
    assert implementation.parameters["token_rows"] == rows


@pytest.mark.parametrize("rows", [True, 0, 6, 1.5])
def test_reject_unqualified_rows(rows):
    with pytest.raises(CodegenError):
        renderer.recipe(rows)


def test_static_recipe_ignores_legacy_environment(monkeypatch):
    expected = renderer.parameters("iq4_q8_glu", 5)
    legacy = signed.parameters("iq4_q8_glu")
    for key in ("TRANSPORT_STAGE", "NO_SHARED_STAGING", "TRANSPORT_MAXNREG",
                "PREFETCH_DEPTH", "WARP_GRID", "PACKET_MODE", "OCCUPANCY_PTX_OPTIONS"):
        monkeypatch.setenv(key, "invalid")
    assert renderer.parameters("iq4_q8_glu", 5) == expected
    assert signed.parameters("iq4_q8_glu") == legacy
    for module in (renderer, signed):
        tree = ast.parse(Path(module.__file__).read_text())
        assert not any(isinstance(n, ast.Attribute) and n.attr in ("environ", "getenv")
                       for n in ast.walk(tree))

@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5])
def test_compact_recipe_is_static_and_versioned(rows, monkeypatch):
    from triton.flagmega.codegen.triton import iq4_q8_static_t3_compact_renderer as compact
    assert compact.VARIANT != renderer.VARIANT
    expected = compact.parameters("iq4_q8_glu", rows)
    monkeypatch.setenv("TRANSPORT_MAXNREG", "999")
    assert compact.parameters("iq4_q8_glu", rows) == expected
    assert compact.implementations(rows)[0].parameters["maxnreg"] == 40
    assert compact.recipe(rows).to_data() == renderer.recipe(rows).to_data()
    tree = ast.parse(Path(compact.__file__).read_text())
    assert not any(isinstance(n, ast.Attribute) and n.attr in ("environ", "getenv")
                   for n in ast.walk(tree))


def test_signed_static_rejects_multirow_abi():
    raw = {"variant": signed.VARIANT, "family": "iq4_q8_glu",
           "semantic_attrs": {"logical_n": 17408, "logical_k": 5120},
           "inputs": [{"formal": "q8", "buffers": [{"actual": "value", "runtime_argument": "Q",
                       "abi": {"logical_shape": [2, 160, 36]}}]}]}
    with pytest.raises(CodegenError, match="one token row"):
        signed.validate_raw(raw)
