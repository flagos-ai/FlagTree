# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""An escaping local alias must refine storage geometry as well as ownership."""

import pytest

from triton.flagmega import ir as fm


@pytest.mark.parametrize("axis", [0, 1])
def test_repeated_function_calls_share_canonical_view_result_abi(axis):
    placement = fm.Placement((2, 2), "yx", "bb")
    tensor = fm.tensor_type(fm.vector_type("float32", 4), (8, 16))
    policies = [fm.SBP.broadcast()] * 2
    policies[axis] = fm.SBP.split_contiguous((0, 1), tensor.shape[axis].fixed_value // 4)
    split = fm.DistributedType(tensor, tuple(policies), placement)
    b = fm.IRBuilder(dialect="semantic_tir", stage="packaged_tir")
    p = b.var("p", split, id="p")
    temporary = b.call("math.silu", (p,), split, id="temporary")
    view = b.call("tir.buffer_view", (temporary,), split, id="view", attrs={"alias_kind": "reshape"})
    x = b.var("x", split, id="x")
    first = b.call("builtin.call", (x,), split, id="first", attrs={"callee": "worker"})
    second = b.call("builtin.call", (first,), split, id="second", attrs={"callee": "worker"})
    b.function("worker", (p,), (view,), attrs={"reusable": True, "noinline": True})
    b.function("main", (x,), (second,))
    plan = fm.make_buffer_plan(b.build(entry="main"))
    source, result, formal = (plan.buffer_map[key] for key in ("temporary", "view", "p"))
    assert source.mem_span.must_alias(result.mem_span)
    assert source.storage == result.storage == "return"
    assert source.distributed_storage_kind is fm.DistributedBufferStorageKind.CANONICAL_GLOBAL
    assert result.strides == formal.strides == (16, 1)
    assert result.nbytes == formal.nbytes == 8 * 16 * 16
    actual = plan.buffer_map[dict(plan.call_map["second"].arguments)["p"]]
    assert actual.distributed_storage_kind is formal.distributed_storage_kind
    assert actual.strides == formal.strides
