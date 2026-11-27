# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Joint costs for private candidate regions realized by ordinary graph rules."""

from dataclasses import dataclass, replace

from triton.flagmega.ir import Node, get_definition
from triton.flagmega.rules.ntt.lower_add_norm_stats import lower_add_norm_stats_rule


@dataclass(frozen=True)
class CandidateFusion:
    members: tuple[tuple[str, int], ...]
    operation: str
    operation_cost: int
    standalone_cost: int
    invocation_count: int


def candidate_fusions(graph):
    """Use the same typed rewrite legality as post-distribution lowering.

    Costed regions are direct private projection/epilogue pairs. No cost is
    attached to a merely possible fusion across a reshard, an escaping value,
    or a shared producer. Non-analytic candidate prices remain authoritative.
    """
    nodes = graph.module.node_map
    users = {}
    for node in graph.module.nodes:
        for value in set(node.inputs):
            users.setdefault(value, []).append(node.id)
    exported = {value for function in graph.module.functions for value in function.outputs}
    rule = lower_add_norm_stats_rule()
    result = []
    for consumer in graph.module.nodes:
        if consumer.op != "ntt.add_norm_stats":
            continue
        producer = nodes[consumer.inputs[0]]
        if (producer.op != "ntt.packed_matmul" or producer.id in exported
                or producer.id in graph.constant_ids or consumer.id in graph.constant_ids
                or users.get(producer.id) != [consumer.id]):
            continue
        invocations = graph.invocation_counts.get(consumer.id, 1)
        if graph.invocation_counts.get(producer.id, 1) != invocations:
            continue
        for pi, pc in enumerate(graph.bucket_map[producer.id].candidates):
            for ci, cc in enumerate(graph.bucket_map[consumer.id].candidates):
                if pc.return_type != cc.input_types[0]:
                    continue
                candidates = ((producer, pc), (consumer, cc))
                if any(c.objective_kind != "analytic" or c.objective_model != graph.operation_cost_model.identity
                       or c.target_op not in (None, node.op) for node, c in candidates):
                    continue
                # Each use gets its own typed operand: one logical argument
                # can be Broadcast for lhs and Split for the residual.
                operands = tuple(Node(f"<fusion:{i}>", "builtin.var", (), value_type)
                                 for i, value_type in enumerate((*pc.input_types, cc.input_types[1])))
                typed_producer = replace(producer, inputs=tuple(n.id for n in operands[:-1]),
                                         type=pc.return_type, attrs=pc.target_attrs or producer.attrs)
                typed_consumer = replace(consumer, inputs=(producer.id, operands[-1].id),
                                         type=cc.return_type, attrs=cc.target_attrs or consumer.attrs)
                region = replace(graph.module, nodes=(*operands, typed_producer, typed_consumer))
                fused = rule.apply(typed_consumer, region)
                if not isinstance(fused, Node) or fused.op != "ntt.matmul_norm_stats":
                    continue
                definition = get_definition(fused.op)
                factors = definition.cost_factors(tuple(region.node_map[n] for n in fused.inputs),
                                                  fused.attrs, fused.type)
                if factors is None:
                    continue
                # An edited/measured standalone price must not be silently
                # replaced with the target's default analytic model.
                canonical = True
                for typed, candidate in ((typed_producer, pc), (typed_consumer, cc)):
                    own = get_definition(typed.op).cost_factors(
                        tuple(region.node_map[n] for n in typed.inputs), typed.attrs, typed.type)
                    if own is None or graph.operation_cost_model.get_latency(own, typed.type) != candidate.operation_cost:
                        canonical = False
                        break
                if canonical:
                    result.append(CandidateFusion(
                        ((producer.id, pi), (consumer.id, ci)), fused.op,
                        graph.operation_cost_model.get_latency(factors, fused.type),
                        pc.operation_cost + cc.operation_cost, invocations,
                    ))
    return (*result, *_expert_candidate_fusions(graph, users, exported))


def _expert_candidate_fusions(graph, users, exported):
    from triton.flagmega.errors import IRSchemaError
    from triton.flagmega.ir import DistributedType
    from triton.flagmega.ir.op_fusion import has_ops
    from triton.flagmega.ir.ops.ntt.sparse_experts import DispatchedExpertsGateUp, SparseExpertsDownCombine
    from triton.flagmega.ir.ops.nn.sparse_experts_combine import SparseExpertsWeightedSum

    nodes = graph.module.node_map
    pairs = {"nn.sparse_experts_gate_up": "nn.sparse_experts_dispatch",
             "nn.sparse_experts_combine": "nn.sparse_experts_down"}
    for consumer in graph.module.nodes:
        if consumer.op not in pairs or has_ops(consumer.attrs):
            continue
        producer = nodes[consumer.inputs[0]]
        if (producer.op != pairs[consumer.op] or producer.id in exported or has_ops(producer.attrs)
                or producer.id in graph.constant_ids or consumer.id in graph.constant_ids
                or users.get(producer.id) != [consumer.id]):
            continue
        invocations = graph.invocation_counts.get(consumer.id, 1)
        if graph.invocation_counts.get(producer.id, 1) != invocations:
            continue
        for pi, pc in enumerate(graph.bucket_map[producer.id].candidates):
            for ci, cc in enumerate(graph.bucket_map[consumer.id].candidates):
                if pc.return_type != cc.input_types[0]:
                    continue
                if any(c.objective_kind != "analytic" or c.objective_model != graph.operation_cost_model.identity
                       or c.target_op not in (None, n.op) for n, c in ((producer, pc), (consumer, cc))):
                    continue
                pinputs = tuple(Node(f"p{i}", "builtin.var", (), t) for i, t in enumerate(pc.input_types))
                typed = replace(producer, type=pc.return_type, inputs=tuple(n.id for n in pinputs))
                cinputs = (typed, *(Node(f"c{i}", "builtin.var", (), t) for i, t in enumerate(cc.input_types[1:])))
                pf = get_definition(producer.op).cost_factors(pinputs, producer.attrs, pc.return_type)
                cf = get_definition(consumer.op).cost_factors(cinputs, consumer.attrs, cc.return_type)
                if pf is None or cf is None or any(graph.operation_cost_model.get_latency(f, t) != c.operation_cost
                                                   for f, t, c in ((pf, pc.return_type, pc), (cf, cc.return_type, cc))):
                    continue
                try:
                    if consumer.op == "nn.sparse_experts_gate_up":
                        if producer.inputs[1] != consumer.inputs[1] or pc.input_types[1] != cc.input_types[1]:
                            continue
                        definition = DispatchedExpertsGateUp
                        arguments, attrs = (pinputs[0], *cinputs[1:]), consumer.attrs
                        prepared = definition.prepare(arguments, attrs)
                        if prepared.result_type != cc.return_type:
                            continue
                        factors = definition.cost_factors(arguments, prepared.attrs, prepared.result_type)
                    else:
                        definition = SparseExpertsDownCombine
                        arguments = (*pinputs, cinputs[1])
                        attrs = {**producer.attrs, **consumer.attrs}
                        prepared = definition.prepare(arguments, attrs)
                        materialized = not isinstance(prepared.result_type, DistributedType) or prepared.result_type.partial is None
                        if materialized:
                            prepared = definition.prepare(arguments, {**attrs, "cast_output": True})
                        fused = definition.cost_factors(arguments, prepared.attrs, prepared.result_type)
                        local = SparseExpertsWeightedSum.cost_factors(cinputs, consumer.attrs, prepared.result_type)
                        if fused is None or local is None:
                            continue
                        # Preserve the independently priced owner sum and final cast.
                        factors = fused if materialized else replace(fused, **{
                            name: getattr(fused, name) + getattr(cf, name) - getattr(local, name)
                            for name in fused.__dataclass_fields__})
                except IRSchemaError:
                    continue
                if factors is not None:
                    yield CandidateFusion(((producer.id, pi), (consumer.id, ci)), definition.op_name,
                                          graph.operation_cost_model.get_latency(factors, cc.return_type),
                                          pc.operation_cost + cc.operation_cost, invocations)


__all__ = ["CandidateFusion", "candidate_fusions"]
