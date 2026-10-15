# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Exact factorization of CP-SAT adapter relations and path costs."""

from collections import defaultdict

from .publication_cost import shared_publication


def _relation_classes(indexes, signature):
    classes, representatives, known = {}, [], {}
    for index in indexes:
        value = signature(index)
        if value not in known:
            known[value] = len(known)
            representatives.append(index)
        classes[index] = known[value]
    return classes, representatives


def add_reshard_relations(model, graph, domains, variables):
    """Factor identical relation rows/columns while retaining candidate choices.

    Two choices share an edge-local relation class only when their complete
    sets of legal paths, costs and publication dependencies are equal for
    every opposing choice. Operation/fusion/hard-selection decisions remain
    distinct. Extraction maps the selected relation option back to the actual
    candidate pair's original path, retaining its precise tensor layout.
    """
    from ortools.sat.python.cp_model import Domain

    from .search import _realized_reshard_plan_cost, _require_cp_sat_int64, function_output_type

    edges = defaultdict(list)
    for site in graph.reshard_sites:
        if site.producer_index in domains[site.producer_id] and (
            site.consumer_index is None or site.consumer_index in domains[site.consumer_id]
        ):
            edges[site.key].append(site)
    indexes = {}

    def selected_class(node_id, classes):
        signature = (node_id, tuple(classes[index] for index in domains[node_id]))
        if signature not in indexes:
            value = model.NewIntVarFromDomain(
                Domain.FromValues(tuple(classes.values())), f"relation_class__{node_id}_{len(indexes)}"
            )
            model.Add(value == sum(classes[index] * variables[(node_id, index)] for index in domains[node_id]))
            indexes[signature] = value
        return indexes[signature]

    paths, originals, costs, publications = {}, {}, [], {}
    estimates = {}
    for ordinal, (key, sites) in enumerate(edges.items()):
        producer_id, consumer_id, input_index = key
        groups, choices = {}, {}
        for site in sites:
            source = graph.bucket_map[producer_id].candidates[site.producer_index].return_type
            estimate_key = (producer_id, source, site.plans, site.usage, site.invocation_count)
            if estimate_key not in estimates:
                best = {}
                for plan_index, plan in enumerate(site.plans):
                    shared = shared_publication(graph, site, plan)
                    cost = _realized_reshard_plan_cost(graph, site, source, plan)
                    previous = best.get(shared)
                    if previous is None or cost < previous[0]:
                        best[shared] = cost, plan_index
                options = []
                for shared, (cost, plan_index) in best.items():
                    net = (cost - (0 if shared is None else shared[1])) * site.invocation_count
                    _require_cp_sat_int64(net, "reshard relation cost")
                    options.append((plan_index, net, None if shared is None else shared[0]))
                estimates[estimate_key] = tuple(options)
            options = estimates[estimate_key]
            choices[(site.producer_index, site.consumer_index)] = options
            for _, _, shared in options:
                if shared is not None:
                    groups.setdefault(shared, len(groups) + 1)
        matrix = {}
        for pair, options in choices.items():
            matrix[pair] = tuple(sorted((0 if shared is None else groups[shared], cost) for _, cost, shared in options))
            originals[(key, *pair)] = {0 if shared is None else groups[shared]: plan for plan, _, shared in options}
        if not groups and all(cost == 0 for options in matrix.values() for _, cost in options):
            continue

        output = sites[0].consumer_index is None
        producer_bucket = graph.bucket_map[producer_id]
        if output:
            target = function_output_type(
                graph.module, consumer_id.removeprefix("@function:"), producer_id, graph.placement
            )
            for producer in domains[producer_id]:
                if (producer, None) not in matrix and producer_bucket.candidates[producer].return_type == target:
                    matrix[(producer, None)] = ((0, 0),)
            consumers = (None,)
        else:
            consumers = domains[consumer_id]
            consumer_bucket = graph.bucket_map[consumer_id]
            for producer in domains[producer_id]:
                source = producer_bucket.candidates[producer].return_type
                for consumer in consumers:
                    if (producer, consumer) not in matrix and source == consumer_bucket.candidates[
                        consumer
                    ].input_types[input_index]:
                        matrix[(producer, consumer)] = ((0, 0),)
        producer_classes, producers = _relation_classes(
            domains[producer_id],
            lambda p, matrix=matrix, consumers=consumers: tuple(matrix.get((p, c), ()) for c in consumers),
        )
        consumer_classes, consumers = _relation_classes(
            consumers, lambda c, matrix=matrix, producers=producers: tuple(matrix.get((p, c), ()) for p in producers)
        )
        rows = []
        for producer in producers:
            for consumer in consumers:
                for option, cost in matrix.get((producer, consumer), ()):
                    row = [producer_classes[producer]]
                    if not output:
                        row.append(consumer_classes[consumer])
                    row.extend((option, cost))
                    row.extend(int(code == option) for code in groups.values())
                    rows.append(tuple(row))
        option_column = 1 if output else 2
        path = model.NewIntVarFromDomain(
            Domain.FromValues(tuple(row[option_column] for row in rows)), f"edge_path__{ordinal}"
        )
        cost = model.NewIntVarFromDomain(
            Domain.FromValues(tuple(row[option_column + 1] for row in rows)), f"edge_cost__{ordinal}"
        )
        columns = [selected_class(producer_id, producer_classes)]
        if not output:
            columns.append(selected_class(consumer_id, consumer_classes))
        columns.extend((path, cost))
        for group, code in groups.items():
            published = model.NewBoolVar(f"edge_publication__{ordinal}_{code}")
            columns.append(published)
            publications.setdefault(group, []).append(published)
        model.AddAllowedAssignments(columns, rows)
        paths[key] = path
        costs.append(cost)
    return paths, originals, costs, publications
