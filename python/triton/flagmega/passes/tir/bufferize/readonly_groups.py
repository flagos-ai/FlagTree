# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Physical readonly groups derived from typed, output-specific constant recipes."""

from dataclasses import replace
from typing import Mapping

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir.constant_recipe import constant_recipe_fingerprint
from triton.flagmega.ir.model import IRModule, Node


class ReadonlyGroupResolver:
    """Separate logical repeated-parameter hints from storage representations.

    A source group can feed several Slice/Pack/Cast/etc. outputs. Each recipe
    representation needs its own contiguous sequence of layer values. This is
    layout grouping, never value deduplication: source keys are normalized only
    for this identity, not for recipe evaluation or materialization caches.
    """

    def __init__(self, module: IRModule) -> None:
        self.module = module
        self.recipes = {recipe.id: recipe for recipe in module.constant_recipes}
        self.node_maps = {}
        self.signatures: dict[tuple[str, str], str | None] = {}

    def name_for(self, node: Node, logical_name: str) -> str:
        if node.op == "builtin.const_asset":
            recipe_id, output = str(node.attrs["recipe"]), str(node.attrs["output"])
        elif node.op == "tir.buffer" and node.metadata.get("bufferized_from") == "builtin.const_asset":
            recipe_id = str(node.metadata["constant_recipe"])
            output = str(node.metadata["constant_output"])
        else:
            return logical_name
        key = (recipe_id, output)
        if key not in self.signatures:
            self.signatures[key] = self._signature(recipe_id, output, node.id)
        signature = self.signatures[key]
        return logical_name if signature is None else f"{logical_name}.repr_{signature}"

    def _signature(self, recipe_id: str, output: str, asset_id: str) -> str | None:
        if recipe_id not in self.recipes:
            raise IRVerificationError(
                f"Readonly group references missing constant recipe {recipe_id!r}.",
                stage=self.module.stage,
                node_id=asset_id,
            )
        if recipe_id not in self.node_maps:
            self.node_maps[recipe_id] = self.recipes[recipe_id].node_map
        node_map = self.node_maps[recipe_id]
        ordered = []
        visited = set()
        pending = [(output, False)]
        # Input-ordered postorder excludes other exported branches and ignores
        # incidental topological ordering of independent nodes in the recipe.
        while pending:
            node_id, ready = pending.pop()
            if node_id not in node_map:
                raise IRVerificationError(
                    f"Readonly group recipe {recipe_id!r} references missing node {node_id!r}.",
                    stage=self.module.stage,
                    node_id=asset_id,
                )
            current = node_map[node_id]
            if ready:
                ordered.append(current)
            elif node_id not in visited:
                visited.add(node_id)
                pending.append((node_id, True))
                pending.extend((value, False) for value in reversed(current.inputs))
        if len(ordered) == 1 and ordered[0].op == "builtin.weight":
            return None
        normalized = []
        for current in ordered:
            group = current.metadata.get("rdata_group")
            if current.op == "builtin.weight" and isinstance(group, Mapping) and group.get("name"):
                current = replace(current, attrs={"rdata_group": str(group["name"])})
            normalized.append(current)
        return constant_recipe_fingerprint(tuple(normalized), (output, ))


__all__ = ["ReadonlyGroupResolver"]
