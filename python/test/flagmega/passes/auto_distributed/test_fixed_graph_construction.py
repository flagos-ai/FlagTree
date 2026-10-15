# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.passes.auto_distributed import AutoDistributedPass
from triton.flagmega.passes.auto_distributed import search
from triton.flagmega.targets import NvidiaSm90Target

from .helpers import packed_matmul_module


def test_apply_does_not_plan_edges_for_excluded_candidates(monkeypatch):
    target = NvidiaSm90Target()
    calls = []
    original = search._reshard_plans

    def count(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(search, "_reshard_plans", count)
    module = packed_matmul_module()
    graph = AutoDistributedPass._build_graph(module, target)
    full_count = len(calls)
    proposed = AutoDistributedPass._propose_from_graph(module, target, graph)
    calls.clear()
    AutoDistributedPass.apply(proposed, target)
    assert len(calls) < full_count


@pytest.mark.parametrize("output_index", range(3))
@pytest.mark.parametrize("structural", [False, True])
def test_constrained_graph_preserves_surface_and_optimum(output_index, structural):
    target = NvidiaSm90Target()
    module = packed_matmul_module()
    if structural:
        value_type = module.node_map["output"].type
        pair = fm.Node("pair", "builtin.tuple", ("output", "value"), fm.TupleType((value_type, value_type)))
        picked = fm.Node("picked", "builtin.get_item", ("pair",), value_type, attrs={"index": 0})
        module = fm.verify_module(replace(module, nodes=module.nodes + (pair, picked),
                                          functions=(replace(module.functions[0], outputs=("picked",)),)))
    full = AutoDistributedPass._build_graph(module, target)
    candidate = full.bucket_map["output"].candidates[output_index]
    fixed = {"output": candidate.id}
    reduced = AutoDistributedPass._build_graph(module, target, fixed_selections=fixed)
    assert reduced.buckets == full.buckets
    assert len(reduced.reshard_sites) < len(full.reshard_sites)
    expected = search.solve_search_graph(full, fixed_selections=fixed)
    actual = search.solve_search_graph(reduced, fixed_selections=fixed)
    assert actual.selected == expected.selected
    assert actual.selected_reshards == expected.selected_reshards
    assert actual.objective == expected.objective


def test_constrained_graph_cannot_be_solved_after_relaxing_choices():
    target = NvidiaSm90Target()
    module = packed_matmul_module()
    full = AutoDistributedPass._build_graph(module, target)
    candidates = full.bucket_map["output"].candidates
    fixed = {"output": candidates[0].id}
    reduced = AutoDistributedPass._build_graph(module, target, fixed_selections=fixed)
    for replacement in (None, {"output": candidates[1].id}):
        with pytest.raises(IRVerificationError, match="cannot relax"):
            search.solve_search_graph(reduced, fixed_selections=replacement)
    # Additional constraints may safely narrow a graph further.
    fixed["value"] = reduced.bucket_map["value"].candidates[0].id
    search.solve_search_graph(reduced, fixed_selections=fixed)


@pytest.mark.parametrize("fixed, message", [({"missing": "bad"}, "unknown nodes"),
                                           ({"output": "bad"}, "not legal")])
def test_invalid_construction_choices_fail_before_adapter_proofs(monkeypatch, fixed, message):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid hard choice reached adapter planning")

    monkeypatch.setattr(search, "_reshard_plans", forbidden)
    with pytest.raises(IRVerificationError, match=message):
        AutoDistributedPass._build_graph(packed_matmul_module(), NvidiaSm90Target(), fixed_selections=fixed)
