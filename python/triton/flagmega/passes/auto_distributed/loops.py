# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Loop interface choices and exact SSA binding/backedge relations."""

from dataclasses import replace
from itertools import product

from triton.flagmega.ir import TensorType, TupleType, logical_type
from triton.flagmega.ir.distributed_inference import broadcast_ir_type
from triton.flagmega.passes.auto_distributed.candidate_identity import distributed_candidate_id
from triton.flagmega.passes.auto_distributed.candidates import DistributedCandidate, DistributedCandidateContext


def loop_candidates(module, node, placement, available, split_provider):
    count = int(node.attrs["iter_count"])
    choices = []
    context = DistributedCandidateContext(module, node, placement, (), split_provider)
    for position, node_id in enumerate(node.inputs):
        original = module.node_map[node_id].type
        if position < 3:
            values = (original,)
        else:
            values = tuple(candidate.return_type for candidate in available[node_id])
            if position < 3 + count and isinstance(logical_type(original), TensorType):
                values = (*values, broadcast_ir_type(original, placement),
                          *context.leaf_candidate_types(logical_type(original)))
        choices.append(tuple(dict.fromkeys(values)))
    return tuple(DistributedCandidate(
        distributed_candidate_id(node.id, "for-interface", TupleType(tuple(types[3:3 + count])), tuple(types)),
        TupleType(tuple(types[3:3 + count])), tuple(types), 0, "for-iter-argument-fixed-point",
        objective_model="flagmega.loop-interface/v1",
    ) for types in product(*choices))


def add_binding_candidates(module, buckets):
    """Bound variables share the selected loop interface, including captures."""
    by_id = {bucket.node_id: bucket for bucket in buckets}
    for node in module.nodes:
        if node.op != "control.for":
            continue
        region = module.function_map[str(node.attrs["callee"])]
        for position, parameter in enumerate(region.parameters):
            bucket = by_id[parameter]
            values = list(bucket.candidates)
            for candidate in by_id[node.id].candidates:
                target = module.node_map[parameter].type if position == 0 else candidate.input_types[position + 2]
                if not any(value.return_type == target for value in values):
                    values.append(DistributedCandidate(
                        distributed_candidate_id(parameter, "for-bound-argument", target, ()),
                        target, (), 0, "for-bound-argument", objective_model="flagmega.loop-interface/v1"))
            by_id[parameter] = replace(bucket, candidates=tuple(values))
    return [by_id[bucket.node_id] for bucket in buckets]


def loop_edges(module):
    """Yield targets the shared iter parameter, separately from DAG inputs."""
    seen = set()
    for node in module.nodes:
        if node.op != "control.for":
            continue
        region = module.function_map[str(node.attrs["callee"])]
        if region.name in seen:
            continue
        seen.add(region.name)
        for index, output in enumerate(region.outputs):
            yield region.parameters[1 + index], output, index


def binding_relations(module, buckets):
    for node in module.nodes:
        if node.op != "control.for":
            continue
        region = module.function_map[str(node.attrs["callee"])]
        for position, parameter in enumerate(region.parameters):
            pairs = []
            for choice, candidate in enumerate(buckets[node.id].candidates):
                target = module.node_map[parameter].type if position == 0 else candidate.input_types[position + 2]
                pairs.extend((source, choice) for source, value in enumerate(buckets[parameter].candidates)
                             if value.return_type == target)
            yield parameter, node.id, tuple(pairs)


def add_solver_constraints(model, candidate_model, graph, domains, variables, site_map):
    buckets = graph.bucket_map
    for parameter, loop, pairs in binding_relations(graph.module, buckets):
        for choice in domains[loop]:
            supports = [source for source, target in pairs if target == choice and source in domains[parameter]]
            condition = variables[loop, choice] <= sum(variables[parameter, source] for source in supports)
            model.Add(condition)
            candidate_model.Add(condition)
    for parameter, output, yield_index in loop_edges(graph.module):
        for choice in domains[parameter]:
            target = buckets[parameter].candidates[choice].return_type
            supports = [source for source in domains[output]
                        if buckets[output].candidates[source].return_type == target
                        or (output, source, parameter, choice, yield_index) in site_map]
            condition = variables[parameter, choice] <= sum(variables[output, source] for source in supports)
            model.Add(condition)
            candidate_model.Add(condition)


__all__ = ["add_binding_candidates", "add_solver_constraints", "binding_relations", "loop_candidates", "loop_edges"]
