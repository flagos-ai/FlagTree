# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.ir.bufferization import AliasKind
from triton.flagmega.ir.ops.ntt.matmul_norm_stats_combine import (
    MatMulNormStatsCombine,
)


def _module_with_live_root_behind_a_dead_view():
    tensor = fm.tensor_type("bfloat16", (1, 16))
    builder = fm.IRBuilder(dialect="semantic_tir", stage="packaged_tir")
    lhs = builder.var("lhs", tensor, id="lhs")
    rhs = builder.var("rhs", tensor, id="rhs")
    increment = builder.var("increment", tensor, id="increment")
    source = builder.call("math.add", (lhs, rhs), tensor, id="source")
    view = builder.call(
        "tir.buffer_view",
        (source,),
        tensor,
        id="source_view",
        attrs={"alias_kind": "reshape"},
    )
    updated = builder.call(
        "math.add", (view, increment), tensor, id="updated"
    )
    result = builder.call(
        "math.add", (source, updated), tensor, id="result"
    )
    builder.function("main", (lhs, rhs, increment), (result,))
    return builder.build(entry="main")


def test_dead_view_cannot_authorize_overwrite_of_a_live_alias_root():
    plan = fm.make_buffer_plan(_module_with_live_root_behind_a_dead_view())
    source = plan.buffer_map["source"]
    view = plan.buffer_map["source_view"]
    updated = plan.buffer_map["updated"]

    assert view.alias is not None
    assert view.alias.kind is AliasKind.VIEW
    assert source.mem_span.must_alias(view.mem_span)
    assert not source.mem_span.may_alias(updated.mem_span)
    assert updated.alias is None


def test_sat_allocator_keeps_live_root_and_destructive_candidate_disjoint():
    plan = fm.make_buffer_plan(_module_with_live_root_behind_a_dead_view())
    source = plan.buffer_map["source"]
    updated = plan.buffer_map["updated"]

    assert source.live_end is not None
    assert updated.live_start is not None
    assert source.live_end > updated.live_start
    assert not source.mem_span.may_alias(updated.mem_span)


def _module_with_live_forwarded_tuple_field():
    tensor = fm.tensor_type("bfloat16", (1, 16))
    builder = fm.IRBuilder(dialect="semantic_tir", stage="packaged_tir")
    lhs = builder.var("lhs", tensor, id="lhs")
    rhs = builder.var("rhs", tensor, id="rhs")
    projection = builder.var("projection", tensor, id="projection")
    increment = builder.var("increment", tensor, id="increment")
    source = builder.call("math.add", (lhs, rhs), tensor, id="source")
    prepared = MatMulNormStatsCombine.prepare(
        (projection, source), {"axis": 1, "use_mean": False}
    )
    combined = builder.call(
        MatMulNormStatsCombine.op_name,
        prepared.inputs,
        prepared.result_type,
        id="combined",
        attrs=prepared.attrs,
    )
    value = builder.call(
        "builtin.get_item", (combined,), tensor,
        id="value", attrs={"index": 0},
    )
    updated = builder.call(
        "math.add", (value, increment), tensor, id="updated"
    )
    result = builder.call(
        "math.add", (value, updated), tensor, id="result"
    )
    builder.function("decode", (lhs, rhs, projection, increment), (result,))
    return builder.build(entry="decode")


def test_forwarded_tuple_field_keeps_its_shared_binding_live():
    plan = fm.make_buffer_plan(_module_with_live_forwarded_tuple_field())
    source = plan.buffer_map["source"]
    combined = plan.buffer_map["combined.0"]
    updated = plan.buffer_map["updated"]

    # The combine legally consumes source in place. Its get_item forwards the
    # same binding, so a later add cannot consume that field while another SSA
    # use is still live.
    assert source.mem_span.must_alias(combined.mem_span)
    assert not combined.mem_span.may_alias(updated.mem_span)
