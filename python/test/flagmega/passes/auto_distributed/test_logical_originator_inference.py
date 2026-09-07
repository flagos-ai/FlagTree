# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Infer legal shard-local casts directly from logical function arguments."""

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.passes.auto_distributed import (
    DistributedCandidateContext, DistributedCandidateProviderRegistry,
    NttDistributedReshardRealizationPolicy, build_search_graph,
)
from triton.flagmega.passes.auto_distributed.inference_providers import TypeInferenceCandidateProvider


def cast_chain():
    b = fm.IRBuilder(dialect="ntt", stage="packed")
    wide = fm.tensor_type(fm.vector_type("float32", 4), (1, 512))
    narrow = fm.tensor_type(fm.vector_type("bfloat16", 8), (1, 256))
    value = b.var("value", wide, id="value")
    rounded = b.call("ntt.vectorized_cast", (value,), narrow, id="rounded", attrs={
        "new_type": fm.vector_type("bfloat16", 8).to_data(), "vectorize_axes": (1,)})
    output = b.call("ntt.vectorized_cast", (rounded,), wide, id="output", attrs={
        "new_type": fm.vector_type("float32", 4).to_data(), "vectorize_axes": (1,)})
    b.function("main", (value,), (output,))
    return fm.verify_module(b.build(entry="main"))


@pytest.mark.parametrize("hierarchy", [(8, 16), (2, 4), (4,)])
def test_logical_originator_exposes_shards_and_chain_propagates_them(hierarchy):
    module = cast_chain()
    placement = fm.Placement(hierarchy, "yx"[:len(hierarchy)], "b" * len(hierarchy))
    registry = DistributedCandidateProviderRegistry()
    registry.add(TypeInferenceCandidateProvider(frozenset({"ntt.vectorized_cast"})))
    graph = build_search_graph(module, placement, registry, NttDistributedReshardRealizationPolicy())
    for node_id in ("rounded", "output"):
        value = module.node_map[node_id]
        tensor = value.type
        desired = fm.DistributedType(tensor, (fm.SBP.broadcast(), fm.SBP.split_contiguous(
            tuple(range(placement.rank)), tensor.shape[-1].fixed_value // placement.size)), placement)
        assert any(c.return_type == desired for c in graph.bucket_map[node_id].candidates)
    # Logical input ownership is unchanged; its explicit use edge is resharded.
    assert graph.bucket_map["value"].candidates[0].return_type == module.node_map["value"].type


def test_already_distributed_producer_does_not_invent_new_layouts():
    module = cast_chain()
    placement = fm.Placement((2, 4), "yx", "bb")
    value = module.node_map["value"]
    broadcast = fm.DistributedType(value.type, (fm.SBP.broadcast(),) * 2, placement)
    provider = TypeInferenceCandidateProvider(frozenset({"ntt.vectorized_cast"}))
    candidates = provider.get_candidates(DistributedCandidateContext(
        module, module.node_map["rounded"], placement, ((broadcast,),)))
    assert candidates and all(c.input_types == (broadcast,) for c in candidates)
