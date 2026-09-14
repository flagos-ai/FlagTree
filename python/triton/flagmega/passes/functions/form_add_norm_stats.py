# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Form AddNormStats before layout search for equal-shaped residual additions."""

from __future__ import annotations

from dataclasses import dataclass, replace

from triton.flagmega.ir import IRModule, Node, PURE, verify_module
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.ops.nn._norm import normalize_axis
from triton.flagmega.ir.ops.ntt.add_norm_stats import AddNormStats
from triton.flagmega.ir.ops.ntt._matmul_promotion import is_projection_promotion
from triton.flagmega.passes.tir.projection_residual_norm import (
    find_projection_residual_norm_matches,
)


_PARTIAL_CAPABLE_PRODUCERS = frozenset({
    "math.matmul",
    "math.packed_dense_matmul",
    "math.block_scaled_matmul",
    "math.packed_block_scaled_matmul",
    "ntt.packed_matmul",
})


@dataclass(frozen=True)
class _Plan:
    add: Node
    producer: Node
    addend: Node
    stats: tuple[Node, ...]
    axis: int
    use_mean: bool
    combine_id: str


def form_add_norm_stats(module: IRModule) -> IRModule:
    """Replace ``NormStats(Add(value, addend))`` with a tuple combine.

    This is the flat-SSA counterpart of nncase's
    ``FormPackedMatMulNormStatsCombinePass``.  Formation happens before
    AutoDistribution so the solver sees both results as one legal relation.
    A projection is preferred as the first operand because that position
    also admits Sum-partial materialization; other equal-shaped additions
    retain their operand order and need no MatMul producer.
    """

    verify_module(module)
    projection_matches = find_projection_residual_norm_matches(module)
    node_map = module.node_map
    occupied = set(node_map)
    stats_by_add: dict[str, list[Node]] = {}
    for node in module.nodes:
        if node.op != "nn.norm_stats":
            continue
        source = node_map[node.inputs[0]]
        if source.op == "math.add" or (
            source.op == "math.vectorized_binary"
            and source.attrs.get("binary_op") == "add"
        ):
            stats_by_add.setdefault(source.id, []).append(node)

    plans: dict[str, _Plan] = {}
    stats_to_plan: dict[str, _Plan] = {}
    for add_id, stats_nodes in stats_by_add.items():
        add = node_map[add_id]
        producers = tuple(
            node_map[input_id]
            for input_id in add.inputs
            if _projection_source(node_map[input_id], node_map).op in _PARTIAL_CAPABLE_PRODUCERS
        )
        producer = producers[0] if len(producers) == 1 else node_map[add.inputs[0]]
        addend = node_map[add.inputs[0] if add.inputs[1] == producer.id else add.inputs[1]]
        if producer.type != add.type or addend.type != add.type:
            continue
        contracts = set()
        valid = True
        for stats in stats_nodes:
            axis = normalize_axis(int(stats.attrs["axis"]), tensor_of(add.type).rank)
            if axis != tensor_of(add.type).rank - 1:
                valid = False
                break
            contracts.add((axis, bool(stats.attrs["use_mean"])))
        if not valid or len(contracts) != 1:
            continue
        axis, use_mean = next(iter(contracts))
        combine_id = _fresh_id(f"{add.id}.norm_stats_combine", occupied)
        occupied.add(combine_id)
        plan = _Plan(
            add,
            producer,
            addend,
            tuple(stats_nodes),
            axis,
            use_mean,
            combine_id,
        )
        plans[add.id] = plan
        stats_to_plan.update((node.id, plan) for node in stats_nodes)

    if not plans:
        return module

    nodes: list[Node] = []
    for node in module.nodes:
        plan = plans.get(node.id)
        if plan is not None:
            attrs = {"axis": plan.axis, "use_mean": plan.use_mean}
            combine_type = AddNormStats.infer_type(
                (plan.producer, plan.addend), attrs)
            combine_metadata = {
                **dict(plan.add.metadata),
                "introduced_by": "FormAddNormStats",
                "residual_add": plan.add.id,
                "residual_input": plan.addend.id,
                "norm_stats_consumers": tuple(value.id for value in plan.stats),
            }
            projection = _projection_source(plan.producer, node_map)
            if projection.op in _PARTIAL_CAPABLE_PRODUCERS:
                combine_metadata["matmul_producer"] = projection.id
            fusion_match = projection_matches.get(plan.producer.id)
            if fusion_match is not None and fusion_match.residual_add == plan.add.id:
                combine_metadata.update({
                    "norm_consumer": fusion_match.norm_consumer,
                    "projection_adapters": fusion_match.adapters,
                })
            nodes.append(Node(
                plan.combine_id,
                "ntt.add_norm_stats",
                (plan.producer.id, plan.addend.id),
                combine_type,
                PURE,
                attrs,
                combine_metadata,
            ))
            nodes.append(Node(
                node.id,
                "builtin.get_item",
                (plan.combine_id,),
                node.type,
                PURE,
                {"index": 0},
                {
                    **dict(node.metadata),
                    "introduced_by": "FormAddNormStats",
                    "combine": plan.combine_id,
                },
            ))
            continue
        plan = stats_to_plan.get(node.id)
        if plan is not None:
            nodes.append(Node(
                node.id,
                "builtin.get_item",
                (plan.combine_id,),
                node.type,
                PURE,
                {"index": 1},
                {
                    **dict(node.metadata),
                    "introduced_by": "FormAddNormStats",
                    "combine": plan.combine_id,
                },
            ))
            continue
        nodes.append(node)

    return verify_module(replace(module, nodes=tuple(nodes)))


def _fresh_id(stem: str, occupied: set[str]) -> str:
    if stem not in occupied:
        return stem
    index = 1
    while f"{stem}.{index}" in occupied:
        index += 1
    return f"{stem}.{index}"


def _projection_source(node: Node, nodes) -> Node:
    # The combine still consumes the promoted value; formation itself never
    # moves rounding or layout conversion across a projection/collective.
    return nodes[node.inputs[0]] if is_projection_promotion(node, nodes) else node


@dataclass(frozen=True)
class FormAddNormStatsPass:
    name: str = "FormAddNormStats"
    preserves: frozenset[str] = frozenset()

    def run(self, module: IRModule) -> IRModule:
        return form_add_norm_stats(module)


__all__ = ["FormAddNormStatsPass", "form_add_norm_stats"]
