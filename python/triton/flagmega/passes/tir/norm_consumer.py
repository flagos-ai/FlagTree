# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Legality analysis for RMSNorm apply into consumer-local staging."""

from __future__ import annotations

from dataclasses import dataclass

from triton.flagmega.ir import IRModule, Node, TensorType, logical_type


@dataclass(frozen=True)
class NormConsumerMatch:
    producer: str
    adapters: tuple[str, ...]
    consumer: str
    input_index: int
    weight: str
    epsilon: float
    weight_bias: float
    stats: str | None = None


_TRANSPARENT_ADAPTERS = frozenset({
    "distributed.boxing",
    "distributed.sharded_view",
})
_SUPPORTED_CONSUMERS = frozenset({
    "nn.dense_matmul_glu",
    "nn.packed_dense_matmul_glu",
})


def find_norm_consumer_matches(module: IRModule) -> dict[str, NormConsumerMatch]:
    """Find single-use ``rms_norm -> staged consumer`` boundaries."""

    users: dict[str, list[tuple[Node, int]]] = {node.id: [] for node in module.nodes}
    for node in module.nodes:
        for index, input_id in enumerate(node.inputs):
            users.setdefault(input_id, []).append((node, index))
    matches: dict[str, NormConsumerMatch] = {}
    for norm in module.nodes:
        norm_type = logical_type(norm.type)
        contract = _rms_contract(norm, module)
        if contract is None or not isinstance(norm_type, TensorType):
            continue
        current = norm
        adapters: list[str] = []
        while True:
            current_users = users.get(current.id, ())
            if len(current_users) != 1:
                break
            candidate, input_index = current_users[0]
            if candidate.op not in _TRANSPARENT_ADAPTERS:
                break
            if input_index != 0 or logical_type(candidate.type) != logical_type(current.type):
                break
            adapters.append(candidate.id)
            current = candidate
        if len(current_users) != 1:
            continue
        consumer, input_index = current_users[0]
        if (
            consumer.op not in _SUPPORTED_CONSUMERS
            or input_index != 0
            or logical_type(module.node_map[consumer.inputs[0]].type) != norm_type
        ):
            continue
        matches[consumer.id] = NormConsumerMatch(
            producer=norm.id,
            adapters=tuple(adapters),
            consumer=consumer.id,
            input_index=input_index,
            weight=contract["weight"],
            epsilon=contract["epsilon"],
            weight_bias=contract["weight_bias"],
            stats=contract["stats"],
        )
    return matches


def _rms_contract(norm: Node, module: IRModule) -> dict[str, object] | None:
    if norm.op == "nn.rms_norm" and len(norm.inputs) == 2:
        return {
            "weight": norm.inputs[1],
            "epsilon": float(norm.attrs["epsilon"]),
            "weight_bias": float(norm.attrs["weight_bias"]),
            "stats": None,
        }
    if (
        norm.op != "nn.norm_apply"
        or len(norm.inputs) != 4
        or bool(norm.attrs["use_mean"])
        or _constant_splat_value(module.node_map[norm.inputs[3]], module) != 0.0
    ):
        return None
    return {
        "weight": norm.inputs[2],
        "epsilon": float(norm.attrs["epsilon"]),
        "weight_bias": 0.0,
        "stats": norm.inputs[1],
    }


def _constant_splat_value(node: Node, module: IRModule) -> float | None:
    while node.op in _TRANSPARENT_ADAPTERS and len(node.inputs) == 1:
        node = module.node_map[node.inputs[0]]
    if node.op == "builtin.splat_const":
        return float(node.attrs["value"])
    if node.op != "builtin.const_asset":
        return None
    recipe = next(
        (value for value in module.constant_recipes if value.id == str(node.attrs["recipe"])),
        None,
    )
    if recipe is None:
        return None
    nodes = {value.id: value for value in recipe.nodes}
    source = nodes.get(str(node.attrs["output"]))
    observed: set[str] = set()
    while (
        source is not None
        and source.op in _TRANSPARENT_ADAPTERS
        and len(source.inputs) == 1
        and source.id not in observed
    ):
        observed.add(source.id)
        source = nodes.get(source.inputs[0])
    return (
        float(source.attrs["value"])
        if source is not None and source.op == "builtin.splat_const"
        else None
    )


__all__ = ["NormConsumerMatch", "find_norm_consumer_matches"]
