# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Monotone producer-layout and consumer-demand closure before CP-SAT.

Only declared provider relations create candidates. Missing producer layouts
still require the ordinary explicit reshard edges; propagation is not a
selection, a reinterpretation, or permission to change a function ABI.
"""

from collections import deque

from triton.flagmega.ir import TupleType, logical_type

from .candidate_identity import distributed_candidate_id
from .candidates import DistributedCandidate, DistributedCandidateContext


def complete_candidate_relations(module, buckets, placement, registry, reshard_cost_model,
                                operation_cost_model, type_inference_memo):
    from .search import CandidateBucket, _provider_candidates, function_output_type

    by_id = {bucket.node_id: bucket for bucket in buckets}
    nodes = module.node_map
    users = {name: [] for name in nodes}
    demands = {name: {} for name in nodes}
    pending = deque(nodes)
    queued = set(nodes)
    processed = {}

    def enqueue(name):
        if name not in queued:
            pending.append(name)
            queued.add(name)

    def demand(name, value):
        if value not in demands[name]:
            demands[name][value] = None
            enqueue(name)

    for node in module.nodes:
        for source in node.inputs:
            users[source].append(node.id)
        for candidate in by_id[node.id].candidates:
            for source, value in zip(node.inputs, candidate.input_types, strict=True):
                demand(source, value)
    for function in module.functions:
        for source in function.outputs:
            demand(source, function_output_type(module, function.name, source, placement))

    while pending:
        name = pending.popleft()
        queued.remove(name)
        node, bucket = nodes[name], by_id[name]
        available = tuple(tuple(dict.fromkeys(c.return_type for c in by_id[source].candidates))
                          for source in node.inputs)
        requested = tuple(demands[name])
        key = (available, requested)
        if processed.get(name) == key:
            continue
        processed[name] = key
        provider = registry.try_get(node.op)
        if provider is not None:
            context = DistributedCandidateContext(
                module, node, placement, available, registry.split_candidate_provider,
                reshard_cost_model, operation_cost_model, type_inference_memo=type_inference_memo)
            candidates = _provider_candidates(provider, context, node.type, requested)
        elif node.op == "builtin.tuple":
            candidates = tuple(DistributedCandidate(
                distributed_candidate_id(name, "tuple-structural-provider-demand", value, value.fields),
                value, value.fields, 0, "tuple-structural-provider-demand",
                objective_kind="analytic", objective_model="flagmega.structural-zero/v1",
                objective_evidence=("provider-demanded-tuple-layout", "field-wise-reshard-preserves-producer-contract"))
                for value in requested if isinstance(value, TupleType) and len(value.fields) == len(node.inputs))
        elif node.op == "builtin.get_item":
            index = int(node.attrs["index"])
            tuples = dict.fromkeys(value for value in available[0] if isinstance(value, TupleType))
            for value in available[0]:
                if not isinstance(value, TupleType):
                    continue
                for output in requested:
                    if logical_type(output) == logical_type(value.fields[index]):
                        fields = (*value.fields[:index], output, *value.fields[index + 1:])
                        tuples[TupleType(fields, value.is_variadic)] = None
            candidates = tuple(DistributedCandidate(
                distributed_candidate_id(name, f"tuple-field-{index}", value.fields[index], (value,)),
                value.fields[index], (value,), 0, f"tuple-field-{index}")
                for value in tuples)
        else:
            continue
        known = {(c.return_type, c.input_types) for c in bucket.candidates}
        added = []
        for candidate in candidates:
            relation = (candidate.return_type, candidate.input_types)
            if relation in known:
                continue
            known.add(relation)
            added.append(candidate)
            for source, value in zip(node.inputs, candidate.input_types, strict=True):
                demand(source, value)
        if added:
            by_id[name] = CandidateBucket(name, (*bucket.candidates, *added), bucket.executable)
            if any(c.return_type not in {old.return_type for old in bucket.candidates} for c in added):
                for user in users[name]:
                    enqueue(user)
    return tuple(by_id[bucket.node_id] for bucket in buckets)
