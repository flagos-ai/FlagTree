# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Dependency order of explicitly partitioned execution programs."""

from collections.abc import Mapping
from collections import deque

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


def program_stream_indices(programs):
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
