# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Analysis and the one-way freeze boundary for compile-time constants."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Mapping

from triton.flagmega.errors import IRVerificationError, StageError
from triton.flagmega.ir.constant_recipe import ConstantPhase, ConstantRecipe
from triton.flagmega.ir.model import (
    DistributedType,
    IRModule,
    Node,
    TensorType,
    TupleType,
)
from triton.flagmega.ir.ops.core import get_definition
from triton.flagmega.ir.verify import verify_module


def constant_phase(module: IRModule) -> ConstantPhase:
    try:
        return ConstantPhase(str(module.metadata.get("constant_phase", ConstantPhase.OPEN.value)))
    except ValueError as error:
        raise IRVerificationError(
            f"Unknown constant phase {module.metadata.get('constant_phase')!r}.", stage=module.stage) from error


def require_constants_open(module: IRModule, owner: str) -> None:
    if constant_phase(module) != ConstantPhase.OPEN:
        raise StageError(
            f"{owner} requires constants_open IR; frozen constant recipes are opaque. "
            "Resume from a checkpoint before FreezeConstantIslands.",
            stage=module.stage,
        )


@dataclass(frozen=True)
class ConstnessResult:
    constants: frozenset[str]

    def is_constant(self, value: str | Node) -> bool:
        return (value.id if isinstance(value, Node) else str(value)) in self.constants


class ConstnessAnalysis:
    """Derive constness without annotating or mutating the graph."""

    name = "constness"

    @staticmethod
    def analyze(module: IRModule) -> ConstnessResult:
        require_constants_open(module, "ConstnessAnalysis")
        constants: set[str] = set()
        for node in module.nodes:
            definition = get_definition(node.op)
            if definition.constant_source:
                if node.inputs or not node.effect.is_pure:
                    raise IRVerificationError(
                        f"Constant source {node.op!r} must be a pure leaf.",
                        stage=module.stage,
                        node_id=node.id,
                    )
                constants.add(node.id)
                continue
            if (
                definition.const_evaluable
                and definition.deterministic
                and node.effect.is_pure
                and node.inputs
                and all(value in constants for value in node.inputs)
            ):
                constants.add(node.id)
        return ConstnessResult(frozenset(constants))


@dataclass(frozen=True)
class ConstantCSEPass:
    name: str = "ConstantCSE"
    preserves: frozenset[str] = frozenset({"types", "effects", "shape"})

    def run(self, module: IRModule) -> IRModule:
        module = verify_module(module)
        require_constants_open(module, self.name)
        constants = ConstnessAnalysis.analyze(module).constants
        aliases: dict[str, str] = {}
        representatives: dict[str, str] = {}
        nodes: list[Node] = []

        def resolve(value: str) -> str:
            while value in aliases:
                value = aliases[value]
            return value

        for original in module.nodes:
            node = replace(original, inputs=tuple(resolve(value) for value in original.inputs))
            if original.id in constants:
                key = _constant_cse_key(node)
                representative = representatives.get(key)
                if representative is not None:
                    aliases[node.id] = representative
                    continue
                representatives[key] = node.id
            nodes.append(node)

        if not aliases:
            return module
        functions = tuple(replace(
            function,
            parameters=tuple(resolve(value) for value in function.parameters),
            outputs=tuple(resolve(value) for value in function.outputs),
        ) for function in module.functions)
        points = tuple(
            replace(point, owner=resolve(point.owner)) if point.owner is not None else point
            for point in module.selection_points
        )
        return verify_module(replace(module, nodes=tuple(nodes), functions=functions, selection_points=points))


@dataclass(frozen=True)
class FreezeConstantIslandsPass:
    name: str = "FreezeConstantIslands"
    preserves: frozenset[str] = frozenset({"types", "effects", "shape"})

    def run(self, module: IRModule) -> IRModule:
        return freeze_constant_islands(module)


def freeze_constant_islands(module: IRModule) -> IRModule:
    module = verify_module(module)
    require_constants_open(module, "FreezeConstantIslands")
    # ``None`` and other symbolic compile-time values may appear inside a
    # recipe, but cannot become standalone const_asset boundaries: a
    # const_asset denotes readonly bytes and requires physical tensor leaves.
    const_ids = set(ConstnessAnalysis.analyze(module).constants)
    node_map = module.node_map
    users: dict[str, list[str]] = {node.id: [] for node in module.nodes}
    for node in module.nodes:
        for input_id in node.inputs:
            users[input_id].append(node.id)
    function_outputs = {value for function in module.functions for value in function.outputs}
    boundaries = {
        node_id
        for node_id in const_ids
        if _is_materializable_constant_type(node_map[node_id].type)
        and (
            node_id in function_outputs
            or any(user not in const_ids for user in users[node_id])
        )
    }

    required: set[str] = set()
    pending = list(boundaries)
    while pending:
        node_id = pending.pop()
        if node_id in required:
            continue
        required.add(node_id)
        pending.extend(value for value in node_map[node_id].inputs if value in const_ids)

    adjacency: dict[str, set[str]] = {value: set() for value in required}
    for node_id in required:
        for input_id in node_map[node_id].inputs:
            if input_id in required:
                adjacency[node_id].add(input_id)
                adjacency[input_id].add(node_id)

    order = {node.id: index for index, node in enumerate(module.nodes)}
    components: list[set[str]] = []
    observed: set[str] = set()
    for node in module.nodes:
        if node.id not in required or node.id in observed:
            continue
        component: set[str] = set()
        stack = [node.id]
        while stack:
            value = stack.pop()
            if value in component:
                continue
            component.add(value)
            stack.extend(adjacency[value] - component)
        observed.update(component)
        components.append(component)

    recipes: list[ConstantRecipe] = []
    asset_by_output: dict[str, Node] = {}
    for index, component in enumerate(components):
        outputs = tuple(sorted(component & boundaries, key=order.__getitem__))
        if not outputs:
            continue
        recipe = ConstantRecipe(
            f"constant_recipe_{index:04d}",
            tuple(node for node in module.nodes if node.id in component),
            outputs,
        )
        recipes.append(recipe)
        for output in outputs:
            source = node_map[output]
            asset_metadata = dict(source.metadata)
            if "rdata_group" not in asset_metadata:
                reachable: set[str] = set()
                pending = [output]
                while pending:
                    node_id = pending.pop()
                    if node_id in reachable:
                        continue
                    reachable.add(node_id)
                    pending.extend(
                        value for value in node_map[node_id].inputs
                        if value in component
                    )
                source_weights = [
                    node_map[node_id]
                    for node_id in reachable
                    if node_map[node_id].op == "builtin.weight"
                ]
                if (
                    len(source_weights) == 1
                    and "rdata_group" in source_weights[0].metadata
                ):
                    asset_metadata["rdata_group"] = source_weights[0].metadata["rdata_group"]
            asset_by_output[output] = Node(
                id=output,
                op="builtin.const_asset",
                inputs=(),
                type=source.type,
                attrs={"recipe": recipe.id, "output": output},
                metadata={**asset_metadata, "frozen_constant": True},
            )

    # AutoVectorize/AutoDistribution choices on a constant island have already
    # been materialized in the normal nodes copied into ConstantRecipe.  Once
    # the island becomes opaque, keeping module-level decision points would be
    # misleading: their owners either disappear into the recipe or become a
    # ``const_asset`` boundary and can no longer be re-selected independently.
    # The editable recipe IR (including its node types/metadata) is the source
    # of truth after this one-way boundary.
    frozen_point_ids = {
        point.id for point in module.selection_points if point.owner in required
    }
    selection_points = tuple(
        point for point in module.selection_points if point.id not in frozen_point_ids
    )
    selections = tuple(
        record for record in module.selections if record.point_id not in frozen_point_ids
    )

    nodes = tuple(
        asset_by_output[node.id] if node.id in asset_by_output else node
        for node in module.nodes
        if node.id not in required or node.id in asset_by_output
    )
    metadata = {**dict(module.metadata), "constant_phase": ConstantPhase.FROZEN.value}
    return verify_module(replace(
        module,
        nodes=nodes,
        metadata=metadata,
        constant_recipes=tuple(recipes),
        selection_points=selection_points,
        selections=selections,
    ))


def _is_materializable_constant_type(value_type) -> bool:
    if isinstance(value_type, (TensorType, DistributedType)):
        return True
    return (
        isinstance(value_type, TupleType)
        and bool(value_type.fields)
        and all(_is_materializable_constant_type(field) for field in value_type.fields)
    )


def thaw_constant_islands(module: IRModule) -> IRModule:
    """Restore frozen recipes for an ABI pass, without restoring decisions.

    Selection points removed by :func:`freeze_constant_islands` remain
    removed.  This operation is intentionally paired with an immediate
    refreeze after a post-selection pass has added a new offline constant
    materialization.
    """

    module = verify_module(module)
    if constant_phase(module) != ConstantPhase.FROZEN:
        raise StageError(
            "thaw_constant_islands requires frozen constant recipes.",
            stage=module.stage,
        )
    buffered = tuple(
        node.id
        for node in module.nodes
        if node.op == "tir.buffer" and "constant_recipe" in node.metadata
    )
    if buffered:
        raise StageError(
            "Cannot thaw bufferized constant assets; resume before Bufferize: "
            f"{buffered}.",
            stage=module.stage,
        )
    assets = {
        (str(node.attrs["recipe"]), str(node.attrs["output"])): node
        for node in module.nodes
        if node.op == "builtin.const_asset"
    }
    recipe_map = {recipe.id: recipe for recipe in module.constant_recipes}
    first_asset: dict[str, str] = {}
    for node in module.nodes:
        if node.op == "builtin.const_asset":
            first_asset.setdefault(str(node.attrs["recipe"]), node.id)

    emitted: set[str] = set()
    nodes: list[Node] = []
    for node in module.nodes:
        if node.op != "builtin.const_asset":
            nodes.append(node)
            continue
        recipe_id = str(node.attrs["recipe"])
        if recipe_id in emitted or first_asset.get(recipe_id) != node.id:
            continue
        recipe = recipe_map[recipe_id]
        for recipe_node in recipe.nodes:
            if recipe_node.id not in recipe.outputs:
                nodes.append(recipe_node)
                continue
            asset = assets[(recipe_id, recipe_node.id)]
            metadata = {
                key: value
                for key, value in asset.metadata.items()
                if key != "frozen_constant"
            }
            nodes.append(replace(
                recipe_node,
                metadata={**dict(recipe_node.metadata), **metadata},
            ))
        emitted.add(recipe_id)
    if emitted != set(recipe_map):
        raise IRVerificationError(
            "Frozen constant recipes without materialized assets: "
            f"{sorted(set(recipe_map) - emitted)}.",
            stage=module.stage,
        )
    return verify_module(replace(
        module,
        nodes=tuple(nodes),
        metadata={
            **dict(module.metadata),
            "constant_phase": ConstantPhase.OPEN.value,
        },
        constant_recipes=(),
    ))


def _constant_cse_key(node: Node) -> str:
    payload = {
        "op": node.op,
        "inputs": list(node.inputs),
        "type": node.type.to_data(),
        "effect": node.effect.to_data(),
        "attrs": _plain(node.attrs),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in sorted(value.items())}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_data"):
        return _plain(value.to_data())
    return value


__all__ = [
    "ConstantCSEPass",
    "ConstnessAnalysis",
    "ConstnessResult",
    "FreezeConstantIslandsPass",
    "constant_phase",
    "freeze_constant_islands",
    "require_constants_open",
    "thaw_constant_islands",
]
