# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Share producer completion across read-only views and collective inputs."""

from dataclasses import replace

from triton.flagmega.ir import get_definition
from .realization import (
    DistributedReshardRealization, DistributedReshardRealizationContext,
    DistributedReshardSourceKind, DistributedReshardUsageKind,
)


def publication_origin(graph, node_id, path=()):
    key = (node_id, path)
    if key in graph._publication_origins:
        return graph._publication_origins[key]
    node = graph.module.node_map[node_id]
    if node.op == "builtin.get_item":
        origin = publication_origin(graph, node.inputs[0], (int(node.attrs["index"]), *path))
    elif node.op == "builtin.tuple" and path:
        origin = publication_origin(graph, node.inputs[path[0]], path[1:])
    elif node.inputs and not path:
        definition = get_definition(node.op)
        indexes = set()
        for candidate in graph.bucket_map[node_id].candidates:
            inputs = tuple(replace(graph.module.node_map[value], type=value_type)
                           for value, value_type in zip(node.inputs, candidate.input_types))
            index = definition.zero_copy_input_index(inputs, node.attrs, candidate.return_type)
            if index is not None and not _input_preserves_publication(graph, node.inputs[index], candidate.input_types[index]):
                index = None
            indexes.add(index)
        if len(indexes) == 1 and None not in indexes:
            origin = publication_origin(graph, node.inputs[next(iter(indexes))])
        else:
            origin = node_id
    else:
        # Fields written by one operation share its completion. An assembled
        # tuple is handled above, so independent producers never merge here.
        origin = node_id
    graph._publication_origins[key] = origin
    return origin


def _input_preserves_publication(graph, source_id, required):
    for producer in graph.bucket_map[source_id].candidates:
        source = producer.return_type
        if source == required:
            continue
        context = DistributedReshardRealizationContext(
            source, required, DistributedReshardSourceKind.INTERNAL, DistributedReshardUsageKind.INTERNAL)
        if graph.realization_policy.classify(context) is not DistributedReshardRealization.SHARDED_VIEW:
            return False
        if getattr(source, "exclusive", None) != getattr(required, "exclusive", None):
            return False
    return True


def shared_publication(graph, site, plan):
    """Return a publication group and its per-invocation cost, or None."""
    if (len(plan.step_types) != 1 or site.usage not in {"internal", "function_boundary"}
            or site.producer_id in graph.constant_ids):
        return None
    source_node = graph.module.node_map[site.producer_id]
    if source_node.op in {"builtin.var", "builtin.weight", "builtin.const_asset"}:
        return None
    source_type = graph.bucket_map[site.producer_id].candidates[site.producer_index].return_type
    context = DistributedReshardRealizationContext(
        source_type, plan.step_types[0], DistributedReshardSourceKind.INTERNAL, DistributedReshardUsageKind(site.usage))
    realization = graph.realization_policy.classify(context)
    cost = graph.reshard_cost_model.shared_publication_cost(context, realization)
    if not cost:
        return None
    origin = publication_origin(graph, site.producer_id)
    # The target owns the price; differing scopes/prices are not conflated.
    return (origin, site.invocation_count, cost), cost
