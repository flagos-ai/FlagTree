# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The mHC stages are reachable by AutoDistributed.

``nn.mhc_pre`` and ``nn.mhc_post`` landed with the GLM-5.3-Flash importer but
were never registered with a candidate provider, so the full model stopped at
``AutoDistributed has no reviewed candidate provider for compute op
'nn.mhc_pre'``. Both ops declare a broadcast-only distributed contract in their
own ``infer_type`` -- every operand must be broadcast, and a non-broadcast one
is told to "insert explicit Boxing" -- which is precisely what
``BroadcastCandidateProvider`` is for: an explicitly reviewed replicated
candidate whose materialization is still validated by the op's own inference.

The residual streams are token-major and hidden-contracted inside the Sinkhorn
normalization, so no axis of these ops is splittable today; a future split would
have to come with a kernel that reduces across owners.
"""

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.passes.auto_distributed import (
    DistributedCandidateProviderRegistry,
    build_search_graph,
)
from triton.flagmega.passes.auto_distributed.providers import BroadcastCandidateProvider
from triton.flagmega.targets import NvidiaSm90Target

TOKENS, HC, HIDDEN = 3, 4, 16
COEFFICIENTS = (2 + HC) * HC
PRE_ATTRS = {"epsilon": 1e-5, "sinkhorn_iterations": 20, "sinkhorn_epsilon": 1e-6}


def _registry(target):
    registry = DistributedCandidateProviderRegistry()
    target.register_auto_distributed_candidate_providers(registry)
    return registry


def _mhc_module() -> fm.IRModule:
    """One pre stage feeding one post stage, as the decoder emits them."""

    builder = fm.IRBuilder(dialect="high_level", stage="packed")
    streams = builder.var("streams", fm.tensor_type("bfloat16", (TOKENS, HC, HIDDEN)), id="streams")
    projection = builder.var("projection", fm.tensor_type("float32", (COEFFICIENTS, HC * HIDDEN)),
                             id="projection")
    base = builder.var("base", fm.tensor_type("float32", (COEFFICIENTS, )), id="base")
    scale = builder.var("scale", fm.tensor_type("float32", (3, )), id="scale")
    pre = builder.call("nn.mhc_pre", (streams, projection, base, scale),
                       fm.TupleType((
                           fm.tensor_type("float32", (TOKENS, HC)),
                           fm.tensor_type("float32", (TOKENS, HC, HC)),
                           fm.tensor_type("bfloat16", (TOKENS, HIDDEN)),
                       )), attrs=dict(PRE_ATTRS), id="pre")
    # The decoder projects the tuple with ``F.tensors.get_items``, which builds
    # ``builtin.get_item`` nodes; keep the same shape of graph here.
    post = builder.call("builtin.get_item", (pre, ), fm.tensor_type("float32", (TOKENS, HC)),
                        attrs={"index": 0}, id="post")
    comb = builder.call("builtin.get_item", (pre, ), fm.tensor_type("float32", (TOKENS, HC, HC)),
                        attrs={"index": 1}, id="comb")
    value = builder.call("builtin.get_item", (pre, ), fm.tensor_type("bfloat16", (TOKENS, HIDDEN)),
                         attrs={"index": 2}, id="value")
    output = builder.call("nn.mhc_post", (value, streams, post, comb),
                          fm.tensor_type("bfloat16", (TOKENS, HC, HIDDEN)), id="output")
    builder.function("main", (streams, projection, base, scale), (output, ))
    return builder.build(entry="main")


@pytest.mark.parametrize("op_name", ["nn.mhc_pre", "nn.mhc_post"])
def test_the_mhc_stages_have_a_reviewed_provider(op_name):
    """Without this the full GLM-5.3 model cannot leave AutoDistributed."""

    registry = _registry(NvidiaSm90Target())
    assert isinstance(registry.try_get(op_name), BroadcastCandidateProvider)


@pytest.mark.parametrize("mesh", [(4, ), (2, 2)])
def test_a_module_with_both_stages_builds_a_search_graph(mesh):
    """The regression the full model hit: the search graph itself must build."""

    target = NvidiaSm90Target()
    module = _mhc_module()
    placement = fm.Placement(mesh, "xyz"[:len(mesh)], "b" * len(mesh))
    graph = build_search_graph(module, placement, _registry(target),
                               target.distributed_reshard_realization_policy())

    for node_id in ("pre", "output"):
        bucket = graph.bucket_map[node_id]
        assert bucket.executable
        assert bucket.candidates


def test_every_mhc_candidate_is_fully_replicated():
    """The ops reject any non-broadcast operand, so the candidate must match."""

    target = NvidiaSm90Target()
    module = _mhc_module()
    placement = fm.Placement((4, ), "x", "b")
    graph = build_search_graph(module, placement, _registry(target),
                               target.distributed_reshard_realization_policy())

    broadcast = fm.SBP.broadcast()

    def replicated(value):
        if isinstance(value, fm.TupleType):
            return all(replicated(field) for field in value.fields)
        return (isinstance(value, fm.DistributedType) and value.partial is None
                and value.exclusive is None
                and all(policy == broadcast for policy in value.axis_policies))

    for node_id in ("pre", "output"):
        for candidate in graph.bucket_map[node_id].candidates:
            assert replicated(candidate.return_type)
            assert all(replicated(value) for value in candidate.input_types)


@pytest.mark.parametrize("op_name", ["nn.mhc_pre", "nn.mhc_post"])
def test_the_mhc_candidate_carries_real_work(op_name):
    """A zero-cost candidate would make the stage invisible to the solver."""

    target = NvidiaSm90Target()
    module = _mhc_module()
    placement = fm.Placement((4, ), "x", "b")
    graph = build_search_graph(module, placement, _registry(target),
                               target.distributed_reshard_realization_policy())
    node_id = "pre" if op_name == "nn.mhc_pre" else "output"

    for candidate in graph.bucket_map[node_id].candidates:
        assert candidate.operation_cost > 0
        assert candidate.reason == "explicit-broadcast-inference"


@pytest.mark.parametrize("lanes", [(8,), (2, 4)])
def test_packed_collapsed_candidates_preserve_scalar_inputs_and_fp32_coefficients(lanes):
    from python.test.flagmega.passes.functions.post_boundary_pack.test_mhc_producer_packing import MhcConsumers

    module = MhcConsumers(output_lanes=lanes, lanes=(1,)).build()
    target = NvidiaSm90Target()
    placement = fm.Placement((4,), "x", "b")
    graph = build_search_graph(module, placement, _registry(target),
                               target.distributed_reshard_realization_policy())
    for candidate in graph.bucket_map["pre"].candidates:
        for index, field in enumerate(candidate.return_type.fields):
            assert field.axis_policies == tuple(fm.SBP.broadcast() for _ in field.tensor.shape)
            assert field.partial is None
            assert field.tensor.dtype == (fm.vector_type("bfloat16", lanes) if index == 2 else fm.DType.FLOAT32)
        assert all(not isinstance(value.tensor.dtype, fm.VectorType) for value in candidate.input_types)
