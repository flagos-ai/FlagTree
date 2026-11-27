# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Exact weighted concurrency bounds over a program execution partial order."""

from collections import deque


def maximum_antichain_weight(ancestors, weights):
    """Return the largest summed weight of mutually unordered programs.

    Weighted Dilworth reduction: replace each vertex by weight-many unordered
    copies. A maximum matching of that expanded order has capacity total_weight
    minus the maximum antichain. Capacitated bipartite flow compresses the
    copies, including large byte/register counts, without enumerating states.
    """
    symbols = tuple(ancestors)
    indices = {symbol: index for index, symbol in enumerate(symbols)}
    count = len(symbols)
    source, sink = 2 * count, 2 * count + 1
    network = [[] for _ in range(2 * count + 2)]
    total = sum(weights.values())

    def edge(left, right, capacity):
        forward = [right, len(network[right]), capacity]
        reverse = [left, len(network[left]), 0]
        network[left].append(forward)
        network[right].append(reverse)

    for index, symbol in enumerate(symbols):
        weight = weights[symbol]
        if weight < 0:
            raise ValueError("Program resource weights must be nonnegative.")
        edge(source, index, weight)
        edge(index + count, sink, weight)
        for parent in ancestors[symbol]:
            edge(indices[parent], index + count, total + 1)
    flow = 0
    while True:
        levels = [-1] * len(network)
        levels[source] = 0
        pending = deque((source,))
        while pending:
            node = pending.popleft()
            for target, _, capacity in network[node]:
                if capacity and levels[target] < 0:
                    levels[target] = levels[node] + 1
                    pending.append(target)
        if levels[sink] < 0:
            return total - flow
        current = [0] * len(network)
        while True:
            nodes, path = [source], []
            while nodes and nodes[-1] != sink:
                node = nodes[-1]
                while current[node] < len(network[node]):
                    candidate = network[node][current[node]]
                    if candidate[2] and levels[candidate[0]] == levels[node] + 1:
                        path.append(candidate)
                        nodes.append(candidate[0])
                        break
                    current[node] += 1
                else:
                    levels[node] = -1
                    nodes.pop()
                    if path:
                        path.pop()
                        current[nodes[-1]] += 1
            if not nodes:
                break
            amount = min(item[2] for item in path)
            for item in path:
                item[2] -= amount
                network[item[0]][item[1]][2] += amount
            flow += amount
