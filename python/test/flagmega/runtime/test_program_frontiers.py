# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Downstream fork scheduling and dependency-aware residency bounds."""

from types import SimpleNamespace
from itertools import combinations
import random

import pytest

from triton.flagmega.errors import RuntimeContractError
from triton.flagmega.ir.program_order import program_ancestors, program_stream_indices
from triton.flagmega.runtime.prepared import ResourceContract
from triton.flagmega.runtime.program_dag import verify_joint_resources


def prefix_fork():
    return [
        dict(symbol="prefix", depends_on=()),
        dict(symbol="left", depends_on=("prefix",)),
        dict(symbol="left_down", depends_on=("left",)),
        dict(symbol="right", depends_on=("prefix",)),
        dict(symbol="right_down", depends_on=("right",)),
        dict(symbol="join", depends_on=("left_down", "right_down")),
    ]


def test_common_prefix_keeps_independent_descendant_branches_on_distinct_streams():
    streams = program_stream_indices(prefix_fork())
    assert streams["left"] != streams["right"]
    assert streams["left"] == streams["left_down"]
    assert streams["right"] == streams["right_down"]
    assert len(set(streams.values())) == 2


def prepared(registers, shared=0, warps=8):
    return SimpleNamespace(contract=ResourceContract(warps, 1),
        resource_report=dict(registers_per_thread=registers, shared_memory_bytes=shared, num_warps=warps))


def fork_specs():
    return [dict(symbol=name, depends_on=dependencies, stream_index=stream, grid=(128, 1, 1),
                 requires_grid_wide_sync=True)
            for name, dependencies, stream in (("prefix", (), 0), ("left", ("prefix",), 0),
                ("right", ("prefix",), 1), ("join", ("left", "right"), 0))]


def test_serial_prefix_is_not_added_to_the_concurrent_child_resource_peak():
    report = verify_joint_resources(fork_specs(),
        [prepared(224), prepared(64), prepared(64), prepared(32)], 132)
    assert report["registers"] == 224 * 256
    assert report["threads"] == 2 * 256
    assert report["blocks"] == 2


def test_actual_simultaneous_heavy_branches_still_fail_admission():
    with pytest.raises(RuntimeContractError, match="joint SM90 residency"):
        verify_joint_resources(fork_specs(),
            [prepared(32), prepared(224), prepared(224), prepared(32)], 132)


def test_shared_carveout_uses_the_largest_actual_frontier_including_cta_reserve():
    report = verify_joint_resources(fork_specs(),
        [prepared(32, 180000), prepared(32, 40000), prepared(32, 50000), prepared(32)], 132)
    assert report["shared_bytes"] == 180224 + 1024


def test_joint_admission_keeps_register_and_shared_allocation_quantization():
    specs = [dict(symbol=str(i), depends_on=(), stream_index=i, grid=(128, 1, 1),
                  requires_grid_wide_sync=True) for i in range(2)]
    report = verify_joint_resources(specs, [prepared(33, 1, 4), prepared(33, 257, 4)], 132)
    assert report["registers"] == 2 * 40 * 128
    assert report["shared_bytes"] == 256 + 512 + 2 * 1024


def brute_antichain(ancestors, weights):
    symbols = tuple(ancestors)
    best = 0
    for count in range(len(symbols)+1):
        for subset in combinations(symbols, count):
            if all(left not in ancestors[right] and right not in ancestors[left]
                   for left, right in combinations(subset, 2)):
                best = max(best, sum(weights[symbol] for symbol in subset))
    return best


def test_stream_cover_is_minimal_and_never_serializes_unordered_programs():
    generator = random.Random(91383)
    for count in range(1, 9):
        for _ in range(30):
            programs = [dict(symbol=str(i), depends_on=tuple(str(j) for j in range(i) if generator.random() < .3))
                        for i in range(count)]
            ancestors = program_ancestors(programs)
            streams = program_stream_indices(programs)
            assert len(set(streams.values())) == brute_antichain(ancestors, {name: 1 for name in ancestors})
            for left, right in combinations(ancestors, 2):
                if streams[left] == streams[right]:
                    assert left in ancestors[right]
            reverse_edges = [{**program, "depends_on": tuple(reversed(program["depends_on"]))} for program in programs]
            assert program_stream_indices(reverse_edges) == streams


def test_stream_cover_reassigns_a_matching_instead_of_allocating_an_extra_stream():
    programs = [dict(symbol="a", depends_on=()), dict(symbol="b", depends_on=()),
                dict(symbol="c", depends_on=("a", "b")), dict(symbol="d", depends_on=("a",))]
    streams = program_stream_indices(programs)
    assert streams["a"] == streams["d"]
    assert streams["b"] == streams["c"]
    assert len(set(streams.values())) == 2


def test_repeated_prefix_fork_join_reuses_two_streams_without_unrolling_a_body():
    programs = []
    for index in range(45):
        prefix = f"{index}."
        for item in prefix_fork():
            parents = tuple(prefix+parent for parent in item["depends_on"])
            if item["symbol"] == "prefix" and index:
                parents = (f"{index-1}.join",)
            programs.append(dict(symbol=prefix+item["symbol"], depends_on=parents))
    streams = program_stream_indices(programs)
    assert len(set(streams.values())) == 2
    for index in range(45):
        assert streams[f"{index}.left"] != streams[f"{index}.right"]


def test_weighted_frontier_matches_exhaustive_subset_oracle():
    from triton.flagmega.runtime.program_resources import maximum_antichain_weight

    generator = random.Random(81614)
    for count in range(9):
        for _ in range(40):
            programs = [dict(symbol=str(i), depends_on=tuple(str(j) for j in range(i) if generator.random() < .35))
                        for i in range(count)]
            ancestors = program_ancestors(programs) if programs else {}
            weights = {name: generator.choice((0, 1, 7, 233472, 1 << 30)) for name in ancestors}
            assert maximum_antichain_weight(ancestors, weights) == brute_antichain(ancestors, weights)


def test_runtime_frontier_includes_the_order_of_nodes_on_a_shared_stream():
    specs = [dict(symbol=str(i), depends_on=(), stream_index=i % 2, grid=(128, 1, 1),
                  requires_grid_wide_sync=True) for i in range(3)]
    report = verify_joint_resources(specs, [prepared(128), prepared(64), prepared(32)], 132)
    assert report["registers"] == (128+64)*256


def test_joint_residency_requires_the_actual_dependency_contract():
    with pytest.raises(RuntimeContractError, match="explicit dependency DAG"):
        verify_joint_resources([dict(grid=(128, 1, 1), requires_grid_wide_sync=True, stream_index=0)],
                               [prepared(32)], 132)
