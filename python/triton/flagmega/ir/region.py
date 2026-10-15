# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Closed SSA regions, indexed alongside functions without cloning their bodies."""

import json
from dataclasses import dataclass

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir.model import PURE, Effect, Function


@dataclass(frozen=True)
class Yield:
    values: tuple[str, ...]

    def __post_init__(self):
        object.__setattr__(self, "values", tuple(self.values))


@dataclass(frozen=True)
class Region(Function):
    """A function-indexed lexical region with explicit block arguments.

    The module's node arena supplies storage, not lexical visibility. The
    region verifier prevents references to another scope's bound variables.
    Loop recurrence is represented by its binding contract, not graph edges.
    """

    @property
    def terminator(self):
        return Yield(self.outputs)

    @property
    def iter_count(self):
        return int(self.attrs["iter_count"])


def is_region(function):
    return function.attrs.get("region_kind") == "for"


def region_effect(module, region):
    from triton.flagmega.passes.functions.graph import function_nodes

    effects = {node.effect for node in function_nodes(module, region) if not node.effect.is_pure}
    return combine_effects(effects)


def combine_effects(effects):
    effects = tuple(effect for effect in effects if not effect.is_pure)
    if not effects:
        return PURE
    resources = sorted({effect.resource for effect in effects})
    kinds = {effect.kind.value for effect in effects}
    kind = next(iter(kinds)) if len(kinds) == 1 else "read_write"
    resource = resources[0] if len(resources) == 1 else "region:" + json.dumps(resources, separators=(",", ":"))
    return Effect(kind, resource)


def verify_regions(module):
    from triton.flagmega.passes.functions.graph import function_nodes

    for region in module.functions:
        if not is_region(region):
            continue
        count = region.attrs.get("iter_count")
        if type(count) is not int or count < 0 or len(region.parameters) < 1 + count:
            raise IRVerificationError("For Region has an invalid iter argument count.")
        parameters = set(region.parameters)
        for node in function_nodes(module, region):
            if node.op == "builtin.var" and node.id not in parameters:
                raise IRVerificationError(
                    f"Region @{region.name} is not closed: implicit capture {node.id!r}.", node_id=node.id)
        expected = tuple(module.node_map[value].type for value in region.parameters[1:1 + count])
        yielded = tuple(module.node_map[value].type for value in region.outputs)
        if expected != yielded:
            raise IRVerificationError(f"Region @{region.name} Yield types differ from its iter arguments.")
    if is_region(module.function_map[module.entry]) \
            and module.metadata.get("_dump_function_fragment") is not True:
        # A per-function dump view names its single function as the entry, so a
        # Region's own checkpoint legitimately looks like a Region-entry module.
        # The merged dump is verified as a whole, where this rule still holds.
        raise IRVerificationError("A lexical Region cannot be a module entry.")


def region_fingerprint(module, region):
    """Alpha-equivalent bodies have the same structural fingerprint."""
    import hashlib

    from triton.flagmega.passes.functions.graph import function_nodes

    positions = {value: index for index, value in enumerate(region.parameters)}
    rows = []
    for node in function_nodes(module, region):
        if node.id in positions:
            continue
        attrs = dict(node.to_data()["attrs"])
        attrs.pop("name", None)
        positions[node.id] = len(positions)
        rows.append((node.op, tuple(positions[value] for value in node.inputs), node.type.to_data(),
                     node.effect.to_data(), attrs))
    data = (tuple(module.node_map[value].type.to_data() for value in region.parameters), rows,
            tuple(positions[value] for value in region.outputs), int(region.attrs["iter_count"]))
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


__all__ = ["Region", "Yield", "is_region", "region_fingerprint"]
