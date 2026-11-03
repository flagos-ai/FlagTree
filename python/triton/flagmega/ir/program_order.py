# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Dependency order of explicitly partitioned execution programs."""

from collections.abc import Mapping
from collections import deque
from heapq import heappop, heappush

from triton.flagmega.errors import IRVerificationError


def program_ancestors(programs):
    """Return transitive predecessors, or None for the original serial ABI.

    An explicit DAG is supplied on every program, including roots with an
    empty depends_on tuple. Its stored order is topological and deterministic.
    """
    if programs is None or not any(isinstance(p, Mapping) and "depends_on" in p for p in programs):
        return None
    ancestors = {}
    for program in programs:
        if not isinstance(program, Mapping):
            raise IRVerificationError("Execution program dependency entries must be mappings.")
        symbol, dependencies = program.get("symbol"), program.get("depends_on")
        if (
            not isinstance(symbol, str)
            or not symbol
            or symbol in ancestors
            or not isinstance(dependencies, (tuple, list))
            or any(not isinstance(p, str) or p not in ancestors for p in dependencies)
            or len(set(dependencies)) != len(dependencies)
        ):
            raise IRVerificationError(
                "Execution program dependencies must be explicit, unique and refer to earlier programs."
            )
        ancestors[symbol] = frozenset(dependencies).union(*(ancestors[p] for p in dependencies))
    return ancestors


def program_stream_indices(programs, *, policy="critical_path_v2"):
    """Use a minimum chain cover, with a declared secondary scheduling policy.

    The v2 policy retains the longest dependency path on stable stream chains.
    Serialized artifacts without a policy retain their original v1 mapping;
    codegen and runtime pass that legacy policy explicitly when verifying them.
    """
    if policy not in {"minimum_chains_v1", "critical_path_v2"}:
        raise IRVerificationError("Unknown kernel program stream policy.")
    ancestors = program_ancestors(programs)
    if ancestors is None:
        return None
    if policy == "minimum_chains_v1":
        return _minimum_chain_stream_indices(programs)
    return _critical_path_stream_indices(programs, ancestors)


def _minimum_chain_stream_indices(programs):
    """Cover the partial order with a minimum number of stable stream chains.

    Only already ordered programs share a stream, so a fork after a common
    prefix retains the same parallelism as independent entry roots. Maximum
    bipartite matching on the transitive order gives a minimum chain cover.
    """
    ancestors = program_ancestors(programs)
    if ancestors is None:
        return None
    symbols = tuple(ancestors)
    indices = {symbol: index for index, symbol in enumerate(symbols)}
    successors = [[] for _ in symbols]
    for index, symbol in enumerate(symbols):
        for parent in ancestors[symbol]:
            successors[indices[parent]].append(index)
    left_match, right_match = {}, {}
    for origin in range(len(symbols)):
        pending, visited = deque((origin,)), {origin}
        predecessor = {}
        terminal = None
        while pending and terminal is None:
            left = pending.popleft()
            for right in successors[left]:
                if right in predecessor:
                    continue
                predecessor[right] = left
                if right not in right_match:
                    terminal = right
                    break
                other = right_match[right]
                if other not in visited:
                    visited.add(other)
                    pending.append(other)
        while terminal is not None:
            left = predecessor[terminal]
            previous = left_match.get(left)
            left_match[left] = terminal
            right_match[terminal] = left
            terminal = previous
    streams = {}
    roots = (index for index in range(len(symbols)) if index not in right_match)
    for stream, root in enumerate(roots):
        index = root
        while index is not None:
            streams[symbols[index]] = stream
            index = left_match.get(index)
    return {symbol: streams[symbol] for symbol in symbols}


def _critical_path_stream_indices(programs, verified_ancestors):
    """Maximize cardinality first, then critical and direct adjacencies.

    Matching only transitive dependency edges adds no serialization. Successive
    shortest augmenting paths preserve the minimum stream count; their costs
    resolve ties in favor of the longest path without naming any operators.
    """
    symbols = [p['symbol'] for p in programs]
    indices = {s: i for i, s in enumerate(symbols)}
    parents = [tuple((indices[s] for s in p['depends_on'])) for p in programs]
    ancestors = [{indices[s] for s in verified_ancestors[symbol]} for symbol in symbols]
    depth = []
    for ps in parents:
        depth.append(1 + max((depth[p] for p in ps), default=0))
    successors = [[] for _ in symbols]
    for i, ps in enumerate(parents):
        for p in ps:
            successors[p].append(i)
    tail = [1] * len(symbols)
    for i in reversed(range(len(symbols))):
        tail[i] = 1 + max((tail[s] for s in successors[i]), default=0)
    longest = max(depth, default=0)
    n = len(symbols)
    src, sink = (2 * n, 2 * n + 1)
    graph = [[] for _ in range(2 * n + 2)]

    def edge(a, b, cost):
        forward = [b, 1, cost, len(graph[b])]
        backward = [a, 0, -cost, len(graph[a])]
        graph[a].append(forward)
        graph[b].append(backward)
    for i in range(n):
        edge(src, i, 0)
        edge(n + i, sink, 0)
    for right, ps in enumerate(ancestors):
        for left in sorted(ps):
            critical = depth[left] + tail[right] == longest
            direct = left in parents[right]
            cost = (0 if critical else n + 1) + (0 if direct else 1)
            edge(left, n + right, cost)
    potential = [0] * len(graph)
    while True:
        dist = [float('inf')] * len(graph)
        previous = [None] * len(graph)
        dist[src] = 0
        pending = [(0, src)]
        while pending:
            distance, a = heappop(pending)
            if distance != dist[a]:
                continue
            for j, (b, cap, cost, _) in enumerate(graph[a]):
                if not cap:
                    continue
                candidate = distance + cost + potential[a] - potential[b]
                if candidate < dist[b]:
                    dist[b] = candidate
                    previous[b] = (a, j)
                    heappush(pending, (candidate, b))
        if previous[sink] is None:
            break
        for i, distance in enumerate(dist):
            if distance != float('inf'):
                potential[i] += distance
        b = sink
        while b != src:
            a, j = previous[b]
            e = graph[a][j]
            e[1] -= 1
            graph[b][e[3]][1] += 1
            b = a
    left_match = {left: b - n for left in range(n) for b, cap, _, _ in graph[left] if n <= b < 2 * n and cap == 0}
    right_match = set(left_match.values())
    result = {}
    for stream, root in enumerate((i for i in range(n) if i not in right_match)):
        i = root
        while i is not None:
            result[symbols[i]] = stream
            i = left_match.get(i)
    assert len(result) == n
    return result
