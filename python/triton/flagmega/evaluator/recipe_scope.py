# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Select requested outputs and their transitive dependencies in a closed recipe."""

from triton.flagmega.errors import EvaluationError


def recipe_output_scope(recipe, outputs=None):
    """Preserve recipe output order and SSA order without changing its identity."""
    if outputs is None:
        return recipe.outputs, recipe.nodes
    requested = frozenset(str(value) for value in outputs)
    missing = requested - set(recipe.outputs)
    if missing:
        raise EvaluationError(f"Unknown outputs of constant recipe {recipe.id!r}: {', '.join(sorted(missing))}.")
    selected = tuple(output for output in recipe.outputs if output in requested)
    required = set(selected)
    live = []
    for node in reversed(recipe.nodes):
        if node.id in required:
            live.append(node)
            required.update(node.inputs)
    return selected, tuple(reversed(live))
