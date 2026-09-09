# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Read-only provenance for the printer's function-local weights sections.

This is a display partition, not constant folding or a scheduling decision.
Unlike the open-phase ConstnessAnalysis pass, it also accepts frozen recipes
and scalar literals. Op-local const-evaluable/deterministic contracts govern
propagation; no op-name heuristics, weight loading or evaluation are used.
"""

from dataclasses import dataclass
from typing import Mapping

from triton.flagmega.ir.constant_recipe import ConstantRecipe
from triton.flagmega.ir.model import DistributedType, IRModule, NoneType, TensorType, TupleType
from triton.flagmega.ir.ops.core import get_definition
from triton.flagmega.ir.print_symbols import function_node_scopes, is_inline_leaf
from triton.flagmega.ir.types import PointerType


def _immutable_value(value_type) -> bool:
    if isinstance(value_type, DistributedType):
        return _immutable_value(value_type.tensor)
    if isinstance(value_type, TensorType):
        return not isinstance(value_type.dtype, PointerType) and all(d.is_fixed for d in value_type.shape)
    if isinstance(value_type, TupleType):
        return not value_type.is_variadic and all(_immutable_value(field) for field in value_type.fields)
    return isinstance(value_type, NoneType)


@dataclass(frozen=True)
class WeightPrintAnalysis:
    """Prove in the complete module, then reuse for its per-function dump views."""

    owner: IRModule
    values: frozenset[str]
    recipes: Mapping[str, tuple[ConstantRecipe, ...]]

    @classmethod
    def analyze(cls, module: IRModule) -> "WeightPrintAnalysis":
        known = set()
        derivable = []
        for node in module.nodes:
            if not node.effect.is_pure or not _immutable_value(node.type):
                continue
            definition = get_definition(node.op)
            if not node.inputs and (is_inline_leaf(node) or definition.constant_source):
                known.add(node.id)
            elif (node.inputs and definition.const_evaluable and definition.deterministic
                  and not node.op.startswith("tir.")):
                # Physical calls, dispatches and synchronization remain in their
                # execution order, even when they read only readonly storage.
                derivable.append(node)

        invocations = {function.name: [] for function in module.functions if function.name != module.entry}
        for node in module.nodes:
            if node.op == "builtin.call" and node.attrs["callee"] in invocations:
                invocations[node.attrs["callee"]].append(node)
        parameters = [(parameter, tuple(call.inputs[index] for call in calls)) for name, calls in invocations.items()
                      if calls for index, parameter in enumerate(module.function_map[name].parameters)
                      if _immutable_value(module.node_map[parameter].type)]
        while True:
            previous = len(known)
            for parameter, arguments in parameters:
                if all(value in known for value in arguments):
                    known.add(parameter)
            for node in derivable:
                if all(value in known for value in node.inputs):
                    known.add(node.id)
            if len(known) == previous:
                break

        scopes = function_node_scopes(module)
        recipe_ids = {
            name: {
                str(node.attrs["recipe"]) if node.op == "builtin.const_asset" else str(node.metadata["constant_recipe"])
                for node in nodes
                if node.op == "builtin.const_asset" or (node.op == "tir.buffer" and "constant_recipe" in node.metadata)
            }
            for name, nodes in scopes.items()
        }
        # Keep unused recipes inspectable without duplicating them in every
        # callee. Actual shared recipes appear in each consuming function.
        used = set().union(*recipe_ids.values())
        if module.entry in recipe_ids:
            recipe_ids[module.entry].update(recipe.id for recipe in module.constant_recipes if recipe.id not in used)
        return cls(
            module, frozenset(known), {
                name: tuple(recipe
                            for recipe in module.constant_recipes
                            if recipe.id in ids)
                for name, ids in recipe_ids.items()
            })
