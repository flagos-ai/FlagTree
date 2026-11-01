# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Vectorize independent sum rows while retaining each row's reduction order."""

from dataclasses import replace

IMPLEMENTATION = "tir.reduce_sum.tiled_rows"


def implementation_model(model):
    original = model.implementation("tir.reduce_sum.local")
    tiled = replace(original, id=IMPLEMENTATION, variant="tiled_rows",
                    parameters={**original.parameters, "row_tile": 64},
                    facts={**original.facts, "portable_triton": False})
    return replace(model, implementations=(*model.implementations, tiled),
                   preferences={**model.preferences, "reduce_sum": (
                       IMPLEMENTATION, *model.preferences.get("reduce_sum", (original.id,)))})
