# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.passes.auto_distributed.candidates import DistributedCandidateContext
from triton.flagmega.passes.auto_distributed.inference_providers import TypeInferenceCandidateProvider
from triton.flagmega.passes.auto_distributed.providers import BinaryCandidateProvider


def context(op, *, vector=False):
    dtype = fm.vector_type("bfloat16", (8,)) if vector else "bfloat16"
    tensor = fm.tensor_type(dtype, (1, 16, 32))
    mesh = fm.Placement((8, 16), "yx", "bb")
    head = fm.DistributedType(tensor, (fm.SBP.broadcast(), fm.SBP.split_block_cyclic((1,), 3), fm.SBP.broadcast()), mesh)
    broad = fm.DistributedType(tensor, (fm.SBP.broadcast(),) * 3, mesh)
    binary = op in {"math.add", "math.mul", "math.div", "math.vectorized_binary"}
    attrs = {"binary_op": "mul"} if op == "math.vectorized_binary" else (
        {"unary_op": "sigmoid"} if op == "math.vectorized_unary" else {})
    inputs = tuple(fm.Node(f"arg{i}", "builtin.var", (), tensor, attrs={"name": f"arg{i}"})
                   for i in range(2 if binary else 1))
    definition = fm.get_definition(op)
    call = definition.prepare(inputs, attrs)
    node = fm.Node("result", op, tuple(n.id for n in inputs), call.result_type, call.effect, call.attrs)
    module = fm.IRModule("high_level", "packed", (*inputs, node),
                         (fm.Function("main", tuple(n.id for n in inputs), (node.id,)),), "main")
    return DistributedCandidateContext(module, node, mesh, tuple((broad,) for _ in inputs)), head, broad


@pytest.mark.parametrize("op,vector", [("math.add", False), ("math.mul", False), ("math.vectorized_binary", True)])
@pytest.mark.parametrize("operand", [0, 1])
def test_binary_propagates_exact_layout_from_either_input(op, vector, operand):
    ctx, head, broad = context(op, vector=vector)
    choices = [(broad,), (broad,)]
    choices[operand] = (head,)
    ctx = replace(ctx, available_input_types=tuple(choices))
    candidates = BinaryCandidateProvider().get_candidates(ctx)
    assert any(c.return_type == head and c.input_types == (head, head) for c in candidates)


@pytest.mark.parametrize("op,vector", [("math.add", False), ("math.mul", False), ("math.div", False),
                                     ("math.sigmoid", False), ("math.silu", False),
                                     ("math.vectorized_binary", True), ("math.vectorized_unary", True)])
def test_output_contract_can_request_an_exact_input_layout_not_already_available(op, vector):
    ctx, head, _ = context(op, vector=vector)
    provider = TypeInferenceCandidateProvider(frozenset({op}))
    assert head in provider.get_return_candidate_types(ctx, (head,))
    tuples = provider.try_get_input_type_tuples(ctx, head)
    assert tuples and tuples[0].input_types == (head,) * len(ctx.source_call.inputs)
    candidate = provider.create_candidate(ctx, head, tuples[0])
    assert candidate.return_type == head


def test_reverse_inference_does_not_change_dtype_or_invent_partial_pointwise_work():
    ctx, head, _ = context("math.vectorized_binary", vector=True)
    provider = TypeInferenceCandidateProvider(frozenset({ctx.source_call.op}))
    partial = replace(head, partial=fm.SBPPartial((0,)))
    wrong_dtype = replace(head, tensor=replace(head.tensor, dtype=fm.vector_type("float32", (8,))))
    foreign_mesh = replace(head, placement=fm.Placement((8, 16), "xy", "bb"))
    wrong_lanes = replace(head, tensor=replace(head.tensor, dtype=fm.vector_type("bfloat16", (4, 2))))
    for target in (partial, wrong_dtype, foreign_mesh, wrong_lanes):
        assert not provider.try_get_input_type_tuples(ctx, target)


def test_matching_pointwise_types_are_joined_without_a_cartesian_product():
    ctx, head, _ = context("math.vectorized_binary", vector=True)
    types = tuple(replace(head, axis_policies=(fm.SBP.broadcast(), fm.SBP.split_block_cyclic((1,), block),
                                              fm.SBP.broadcast())) for block in range(1, 65))
    ctx = replace(ctx, available_input_types=(types, types))
    candidates = BinaryCandidateProvider().get_candidates(ctx)
    assert set(types) <= {c.return_type for c in candidates}
    assert len(ctx.type_inference_memo) <= len(types) + len(ctx.leaf_candidate_types(head.tensor)) + 1


def test_consumer_output_demand_propagates_through_multiple_pointwise_producers():
    from triton.flagmega.passes.auto_distributed.candidates import (
        DistributedCandidate,
        DistributedCandidateProviderBase,
        DistributedCandidateProviderRegistry,
    )
    from triton.flagmega.passes.auto_distributed.realization import NttDistributedReshardRealizationPolicy
    from triton.flagmega.passes.auto_distributed.search import build_search_graph

    ctx, head, _ = context("math.sigmoid")

    class Graph(fm.Module):
        def forward(self):
            x = self.input("x", head.tensor, id="x")
            first = fm.F.math.sigmoid(x, name="first")
            second = fm.F.math.silu(first, name="second")
            output = fm.F.tensors.bitcast(second, "bfloat16", name="consumer")
            self.function("main", (x,), (output,))

    class Consumer(DistributedCandidateProviderBase):
        op_names = frozenset({"tensors.bitcast"})
        allows_partial_inputs = False
        is_exhaustive = True

        def _enumerate_candidates(self, context):
            return (DistributedCandidate("consumer", head, (head,), 1, "test-exact-consumer-contract"),)

    module = Graph(dialect="high_level", stage="packed", entry="main").build()
    registry = DistributedCandidateProviderRegistry()
    registry.add(TypeInferenceCandidateProvider(frozenset({"math.sigmoid", "math.silu"})))
    registry.add(Consumer())
    graph = build_search_graph(module, ctx.placement, registry, NttDistributedReshardRealizationPolicy())
    for name in ("first", "second"):
        assert any(c.return_type == head and c.input_types == (head,) for c in graph.bucket_map[name].candidates)
