# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Exact compatibility propagation before allocating CP-SAT variables.

Domains retain original candidate indexes. This is arc consistency, not a
cost-based pruning heuristic: every removed choice has lost a required
producer/consumer support, or violates an explicit fixed choice/output ABI.
"""

from collections import deque

from triton.flagmega.errors import IRVerificationError


def fixed_candidate_domains(buckets, fixed_selections, *, stage):
    """Validate hard choices before constructing their adapter relations."""
    domains = {name: set(range(len(bucket.candidates))) for name, bucket in buckets.items()}
    unknown = sorted(set(fixed_selections) - set(buckets))
    if unknown:
        raise IRVerificationError("AutoDistributed fixed selections reference unknown nodes: " + ", ".join(unknown),
                                  stage=stage)
    for name, candidate_id in fixed_selections.items():
        selected = {i for i, value in enumerate(buckets[name].candidates) if value.id == candidate_id}
        if not selected:
            raise IRVerificationError(f"AutoDistributed candidate {candidate_id!r} is not legal for node {name!r}.",
                                      stage=stage, node_id=name)
        domains[name] = selected
    return domains


def propagate_domains(graph, fixed_selections, site_map):
    from .search import function_boundary_id, function_output_type
    from triton.flagmega.ir.region import is_region

    buckets = graph.bucket_map
    domains = fixed_candidate_domains(buckets, fixed_selections, stage=graph.module.stage)
    for function in graph.module.functions:
        if is_region(function):
            continue
        for index, output in enumerate(function.outputs):
            target = function_output_type(graph.module, function.name, output, graph.placement)
            domains[output] &= {
                i
                for i, candidate in enumerate(buckets[output].candidates)
                if candidate.return_type == target or (output, i, function_boundary_id(function.name), None,
                                                       index) in site_map
            }

    # Index reshard-site adapter pairs by edge once. The per-edge adapter set
    # below is otherwise a quadratic scan over every (producer, consumer)
    # candidate pair testing site_map membership -- the dominant cost when a
    # node carries many distribution candidates (coarse block grids, multi-token
    # prefill). Boundary sites (consumer candidate is None) are not edge adapters.
    adapters_by_edge: dict[tuple[object, object, int], set[tuple[int, int]]] = {}
    for producer_id, producer_index, consumer_id, consumer_index, input_index in site_map:
        if consumer_index is None:
            continue
        adapters_by_edge.setdefault(
            (producer_id, consumer_id, input_index), set()
        ).add((producer_index, consumer_index))

    # Identical physical relations share a support template, never a decision
    # variable: distinct weights and distinct graph uses stay independently selectable.
    templates = {}
    neighbors = {name: [] for name in buckets}
    for consumer in graph.module.nodes:
        for input_index, producer in enumerate(consumer.inputs):
            source_types = tuple(value.return_type for value in buckets[producer].candidates)
            target_types = tuple(value.input_types[input_index] for value in buckets[consumer.id].candidates)
            adapters = frozenset(
                (p, c)
                for p, c in adapters_by_edge.get((producer, consumer.id, input_index), ())
                if p < len(source_types) and c < len(target_types)
            )
            key = (source_types, target_types, adapters)
            supports = templates.get(key)
            if supports is None:
                pairs = tuple((p, c)
                              for p, source in enumerate(source_types)
                              for c, target in enumerate(target_types)
                              if source == target or (p, c) in adapters)
                forward = tuple(frozenset(c for p, c in pairs if p == i) for i in range(len(source_types)))
                reverse = tuple(frozenset(p for p, c in pairs if c == i) for i in range(len(target_types)))
                supports = templates[key] = forward, reverse
            neighbors[producer].append((consumer.id, supports[1]))
            neighbors[consumer.id].append((producer, supports[0]))

    from .loops import binding_relations, loop_edges
    def add_relation(producer, consumer, pairs):
        forward = tuple(frozenset(c for p, c in pairs if p == i)
                        for i in range(len(buckets[producer].candidates)))
        reverse = tuple(frozenset(p for p, c in pairs if c == i)
                        for i in range(len(buckets[consumer].candidates)))
        neighbors[producer].append((consumer, reverse))
        neighbors[consumer].append((producer, forward))
    for producer, consumer, pairs in binding_relations(graph.module, buckets):
        add_relation(producer, consumer, pairs)
    for node, output, slot in loop_edges(graph.module, buckets):
        field = slot - len(node.inputs)
        pairs = tuple((p, c) for p, source in enumerate(buckets[output].candidates)
                      for c, target in enumerate(buckets[node.id].candidates)
                      if source.return_type == target.return_type.fields[field]
                      or (output, p, node.id, c, slot) in site_map)
        add_relation(output, node.id, pairs)

    pending = deque(buckets)
    queued = set(buckets)
    while pending:
        name = pending.popleft()
        queued.remove(name)
        if not domains[name]:
            raise IRVerificationError(
                f"AutoDistributed CP-SAT extraction failed: INFEASIBLE (no support for {name!r}).",
                stage=graph.module.stage, node_id=name)
        for other, supports in neighbors[name]:
            remaining = {i for i in domains[other] if supports[i] & domains[name]}
            if remaining != domains[other]:
                domains[other] = remaining
                if other not in queued:
                    pending.append(other)
                    queued.add(other)
    return {name: tuple(sorted(values)) for name, values in domains.items()}


__all__ = ["propagate_domains"]
