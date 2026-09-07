# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The nncase provider protocol must consume one candidate enumeration."""

from dataclasses import replace

from triton.flagmega import ir as fm
from triton.flagmega.passes.auto_distributed.candidates import (
    DistributedCandidate,
    DistributedCandidateContext,
    DistributedCandidateProviderBase,
)
from triton.flagmega.passes.auto_distributed.search import _provider_candidates


class CountingProvider(DistributedCandidateProviderBase):
    op_names = frozenset({"math.silu"})
    allows_partial_inputs = False
    is_exhaustive = True

    def __init__(self, reason="unit"):
        self.calls = 0
        self.reason = reason

    def _enumerate_candidates(self, context):
        self.calls += 1
        tensor = context.source_call.type
        return tuple(
            DistributedCandidate(
                f"temporary.{index}", value, (value,), 1, self.reason,
            )
            for index, value in enumerate((
                fm.DistributedType(tensor, (fm.SBP.broadcast(),), context.placement),
                fm.DistributedType(tensor, (fm.SBP.split_contiguous((0, 1)),), context.placement),
            ))
        )


def _context():
    builder = fm.IRBuilder(dialect="high_level", stage="packing_applied")
    tensor = fm.tensor_type("float32", (16,))
    value = builder.var("value", tensor, id="value")
    call = builder.call("math.silu", (value,), tensor, id="call")
    builder.function("main", (value,), (call,))
    return DistributedCandidateContext(
        builder.build(entry="main"), call, fm.Placement((2, 2), "yx", "bb"), ((tensor,),),
    )


def test_return_input_and_target_queries_enumerate_each_context_once():
    context = _context()
    provider = CountingProvider()
    candidates = _provider_candidates(provider, context, context.source_call.type)
    assert len(candidates) == 2
    assert all(candidate.target_op == "math.silu" for candidate in candidates)
    assert provider.calls == 1
    assert provider.get_candidates(context) is provider.get_candidates(context)


def test_replaced_context_recomputes_after_agent_changes_type_or_policy():
    context = _context()
    provider = CountingProvider()
    original = provider.get_candidates(context)
    tensor = fm.tensor_type("float32", (32,))
    edited = replace(context, source_call=replace(context.source_call, type=tensor))
    changed = provider.get_candidates(edited)
    assert all(candidate.return_type.tensor == tensor for candidate in changed)
    assert changed != original
    # Equal reconstruction also starts a new search snapshot.
    provider.get_candidates(replace(context))
    assert provider.calls == 3


def test_distinct_provider_instances_do_not_share_context_results():
    context = _context()
    first, second = CountingProvider("first"), CountingProvider("second")
    assert all(candidate.reason == "first" for candidate in first.get_candidates(context))
    assert all(candidate.reason == "second" for candidate in second.get_candidates(context))
    assert first.calls == second.calls == 1
