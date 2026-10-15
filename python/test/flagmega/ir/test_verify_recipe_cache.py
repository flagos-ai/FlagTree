# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Frozen constant recipes are structurally verified once per recipe identity.

Recipes are immutable artifact data: once frozen, every later TIR rewrite
re-publishes the *same* recipe objects on a new module.  Re-walking all recipe
nodes for each of those modules makes verification cost
``O(rewrites * recipe nodes)``, which dominates rewrite passes on real models
where the recipe table is an order of magnitude larger than the main graph.
"""

from dataclasses import replace

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir import verify as verify_impl
from triton.flagmega.passes.constants import FreezeConstantIslandsPass


def _frozen_module() -> fm.IRModule:

    class Graph(fm.Module):
        def forward(self):
            x = self.input("x", fm.tensor_type("float32", (8,)))
            constant = fm.F.builtin.splat_const(x.type, 1.0, name="weight")
            output = fm.F.math.add(x, constant, name="output")
            self.function("main", (x,), (output,))

    return FreezeConstantIslandsPass().run(Graph(dialect="ntt", stage="canonical_constants", entry="main").build())


def test_recipe_index_is_reused_and_rebuilt_after_an_edit():
    recipe = _frozen_module().constant_recipes[0]
    assert recipe.node_map is recipe.node_map
    original = recipe.nodes[0]
    changed = replace(original, attrs={**original.attrs, "value": 2.0})
    edited = replace(recipe, nodes=(changed,))
    assert edited.node_map is not recipe.node_map
    assert edited.node_map[original.id] is changed
    assert recipe.node_map[original.id] is original
    with pytest.raises(TypeError):
        recipe.node_map[original.id] = changed


def _count_recipe_node_verifications(monkeypatch):
    """Count op verifications performed against recipe-local views."""

    counter = {"calls": 0}
    original = verify_impl.get_definition

    class Counting:
        def __init__(self, definition):
            self._definition = definition

        def __getattr__(self, name):
            return getattr(self._definition, name)

        def verify(self, node, module):
            if not module.functions:  # recipe views carry no functions
                counter["calls"] += 1
            return self._definition.verify(node, module)

    monkeypatch.setattr(verify_impl, "get_definition", lambda op: Counting(original(op)))
    return counter


def _reconstructed_recipes(module):
    """Recipe objects with fresh identities but identical content."""

    return tuple(type(recipe)(recipe.id, recipe.nodes, recipe.outputs) for recipe in module.constant_recipes)


def _rewrite_main_graph(module):
    return replace(
        module, nodes=tuple(replace(node, op="math.mul") if node.id == "output" else node for node in module.nodes)
    )


def test_unchanged_recipes_are_not_reverified_for_every_rewritten_module(monkeypatch):
    module = _frozen_module()
    assert module.constant_recipes
    counter = _count_recipe_node_verifications(monkeypatch)

    first_seen = replace(module, constant_recipes=_reconstructed_recipes(module))
    verify_impl.verify_module(first_seen)
    first = counter["calls"]
    assert first > 0

    # A main-graph rewrite republishes the identical recipe objects.
    rewritten = _rewrite_main_graph(first_seen)
    assert rewritten.constant_recipes is first_seen.constant_recipes
    verify_impl.verify_module(rewritten)

    assert counter["calls"] == first


def test_a_new_recipe_identity_is_still_verified(monkeypatch):
    module = _frozen_module()
    counter = _count_recipe_node_verifications(monkeypatch)

    verify_impl.verify_module(replace(module, constant_recipes=_reconstructed_recipes(module)))
    first = counter["calls"]
    assert first > 0

    # Rebuilding the recipe objects yields new identities and must be checked.
    verify_impl.verify_module(replace(module, constant_recipes=_reconstructed_recipes(module)))

    assert counter["calls"] == 2 * first


def test_recipe_caching_does_not_hide_a_broken_asset_binding():
    module = _frozen_module()
    verify_impl.verify_module(module)

    asset = next(node for node in module.nodes if node.op == "builtin.const_asset")
    broken = replace(
        module,
        nodes=tuple(
            replace(node, attrs={**node.attrs, "recipe": "missing"}) if node.id == asset.id else node
            for node in module.nodes
        ),
    )
    with pytest.raises(IRVerificationError, match="no main-graph asset"):
        verify_impl.verify_module(broken)


def test_context_independent_recipe_is_verified_once_across_stage_changes(monkeypatch):
    from triton.flagmega.ir.ops.builtin.splat_const import SplatConst

    module = _frozen_module()
    counter = {"calls": 0}
    original = SplatConst.normalize_attrs

    def count(cls, attrs):
        counter["calls"] += 1
        return original(attrs)

    monkeypatch.setattr(SplatConst, "normalize_attrs", classmethod(count))
    first_seen = replace(module, constant_recipes=_reconstructed_recipes(module))
    verify_impl.verify_module(first_seen)
    assert counter["calls"] > 0
    initial = counter["calls"]
    verify_impl.verify_module(
        replace(
            first_seen,
            stage="lowered_tir",
            dialect="tir",
            entry="renamed",
            functions=tuple(replace(f, name="renamed") for f in first_seen.functions),
        )
    )
    assert counter["calls"] == initial


def test_custom_recipe_verifier_retains_stage_context(monkeypatch):
    from triton.flagmega.ir.ops.builtin.splat_const import SplatConst

    module = _frozen_module()
    original = SplatConst.verify

    def contextual(cls, node, view):
        if view.stage == "invalid_recipe_stage":
            raise IRVerificationError("custom recipe stage check", stage=view.stage)
        return original(node, view)

    monkeypatch.setattr(SplatConst, "verify", classmethod(contextual))
    first_seen = replace(module, constant_recipes=_reconstructed_recipes(module))
    verify_impl.verify_module(first_seen)
    with pytest.raises(IRVerificationError, match="custom recipe stage check"):
        verify_impl.verify_module(replace(first_seen, stage="invalid_recipe_stage"))


def test_verifier_override_cannot_reuse_a_context_independent_recipe_proof(monkeypatch):
    from triton.flagmega.ir.ops.builtin.splat_const import SplatConst

    module = _frozen_module()

    def reject(cls, node, view):
        raise IRVerificationError("new verifier must run", stage=view.stage)

    monkeypatch.setattr(SplatConst, "verify", classmethod(reject))
    with pytest.raises(IRVerificationError, match="new verifier must run"):
        verify_impl.verify_module(replace(module, metadata={**module.metadata, "trial": 2}))
