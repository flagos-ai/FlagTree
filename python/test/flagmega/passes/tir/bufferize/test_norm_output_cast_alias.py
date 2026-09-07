# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Optional NormApply in-place candidates must respect physical element type."""

import pytest

from triton.flagmega import ir as fm


@pytest.mark.parametrize("source,target", [("float32", "bfloat16"), ("bfloat16", "float32"),
                                           ("float32", "float32")])
def test_reusable_function_norm_result_alias_requires_compatible_dtype(source, target):
    b = fm.IRBuilder(dialect="semantic_tir", stage="packaged_tir")
    value_type = fm.tensor_type(source, (1, 64))
    x = b.var("x", value_type, id="x")
    stats = b.var("stats", fm.tensor_type("float32", (1, 1, 1)), id="stats")
    scale = b.var("scale", fm.tensor_type(source, (64,)), id="scale")
    bias = b.var("bias", scale.type, id="bias")
    value = b.call("math.silu", (x,), value_type, id="temporary")
    output = b.call("nn.norm_apply", (value, stats, scale, bias), fm.tensor_type(target, (1, 64)),
        id="normalized", attrs={"axis": 1, "epsilon": 1e-6, "use_mean": False, "output_dtype": target})
    params = (x, stats, scale, bias)
    args = tuple(b.var(f"arg_{p.id}", p.type, id=f"arg_{p.id}") for p in params)
    call = b.call("builtin.call", args, output.type, id="call", attrs={"callee": "worker"})
    b.function("worker", params, (output,), attrs={"reusable": True, "noinline": True})
    b.function("main", args, (call,))
    plan = fm.make_buffer_plan(b.build(entry="main"))
    temporary = plan.buffer_map["temporary"]
    normalized = plan.buffer_map["normalized"]
    assert temporary.mem_span.must_alias(normalized.mem_span) is (source == target)
